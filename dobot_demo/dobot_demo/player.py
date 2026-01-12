#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
数据播放节点 - 机器人演示数据回放与执行
==============================================================================

功能概述:
    本节点用于播放录制的机器人演示数据，实现轨迹重放：
    1. 从HDF5文件读取录制的轨迹数据
    2. 按照原始时间戳播放机械臂和夹爪运动
    3. 可选：同步显示录制的图像用于验证
    
工作流程:
    1. 加载HDF5文件，解析轨迹数据
    2. 连接机械臂控制服务
    3. 使能机械臂
    4. 按时间戳顺序发送关节角度和夹爪指令
    5. 播放完成后回到起始位置

发布话题:
    - /playback/image: 播放当前帧的RGB图像（可选）
    - /playback/status: 播放状态（当前帧、总帧数、进度）

调用服务:
    - /dobot_bringup_v3/srv/EnableRobot: 使能机械臂
    - /dobot_bringup_v3/srv/ServoJ: 伺服关节控制（主要方式）
    - /gripper/control: 夹爪控制（需根据实际接口调整）

参数:
    - file_path: HDF5文件路径（必需）
    - playback_speed: 播放速度倍率（默认1.0）
    - enable_visualization: 是否发布图像（默认False）
    - loop: 是否循环播放（默认False）
==============================================================================
"""

# =============================================================================
# 导入必要的库
# =============================================================================
import rclpy
from rclpy.node import Node
import h5py
import numpy as np
import time
import os
from cv_bridge import CvBridge

# ROS消息类型
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String, Float32
from dobot_msgs_v3.srv import EnableRobot, ServoJ, ServoP, SpeedFactor
from scipy.interpolate import interp1d

class DataPlayer(Node):
    """
    数据播放节点类
    
    功能职责：
        1. 从HDF5文件加载轨迹数据
        2. 初始化机械臂连接和使能
        3. 按时间戳回放关节角度
        4. 控制夹爪同步运动
        5. 可选发布图像用于验证
    """
    
    def __init__(self):
        """初始化播放节点"""
        super().__init__('data_player_node')
        
        # ========== 参数声明 ==========
        self.declare_parameter('file_path', '')
        self.declare_parameter('playback_speed', 1.0)
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('loop', False)
        
        # 获取参数
        self.file_path = self.get_parameter('file_path').value
        self.playback_speed = self.get_parameter('playback_speed').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.loop_playback = self.get_parameter('loop').value
        
        # 验证文件路径
        if not self.file_path:
            self.get_logger().error("No file_path specified! Use: --ros-args -p file_path:=./data/episode_0.hdf5")
            raise ValueError("file_path parameter is required")
        
        if not os.path.exists(self.file_path):
            self.get_logger().error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # ========== 数据加载 ==========
        self.trajectory_data = None
        self.load_trajectory()
        
        # ========== ROS接口初始化 ==========
        # 图像转换器
        self.bridge = CvBridge()
        
        # 发布器：可视化和状态
        if self.enable_visualization:
            self.image_pub = self.create_publisher(Image, '/playback/image', 10)
        
        self.status_pub = self.create_publisher(String, '/playback/status', 10)
        self.progress_pub = self.create_publisher(Float32, '/playback/progress', 10)
        
        # 服务客户端：机械臂控制
        self.enable_robot_client = self.create_client(EnableRobot, '/dobot_bringup_v3/srv/EnableRobot')
        self.servo_j_client = self.create_client(ServoJ, '/dobot_bringup_v3/srv/ServoJ')
        self.servo_p_client = self.create_client(ServoP, '/dobot_bringup_v3/srv/ServoP')
        self.speed_factor_client = self.create_client(SpeedFactor, '/dobot_bringup_v3/srv/SpeedFactor')
        
        # 等待服务可用
        self.get_logger().info("Waiting for Dobot services...")
        while not self.enable_robot_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('EnableRobot service not available, waiting...')
        
        while not self.servo_j_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('ServoJ service not available, waiting...')
        
        while not self.servo_p_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('ServoP service not available, waiting...')
        
        self.get_logger().info("✓ All services connected")
        
        # ========== 使能机械臂 ==========
        self.enable_robot()
        
        # ========== 播放状态 ==========
        self.is_playing = False
        self.current_frame = 0
        
        self.get_logger().info(f"Player initialized. Ready to play {len(self.trajectory_data['qpos'])} frames")
        self.get_logger().info(f"Playback speed: {self.playback_speed}x")
        
    def load_trajectory(self):
        """
        从HDF5文件加载轨迹数据
        
        加载内容：
            - qpos: 关节角度序列
            - gripper_pos: 夹爪位置序列
            - timestamp: 时间戳序列
            - images: 图像序列（可选）
        """
        self.get_logger().info(f"Loading trajectory from: {self.file_path}")
        
        try:
            with h5py.File(self.file_path, 'r') as f:
                # 读取基本信息
                total_frames = f.attrs.get('total_frames', 0)
                self.get_logger().info(f"Total frames: {total_frames}")
                
                # 加载轨迹数据
                self.trajectory_data = {
                    'qpos': np.array(f['observations/qpos']),          # (N, 6) 弧度
                    'gripper_pos': np.array(f['observations/gripper_pos']),  # (N, 1)
                    'ee_pose': np.array(f['observations/ee_pose']),    # (N, 7) [x,y,z,qx,qy,qz,qw]
                    'timestamp': np.array(f['timestamp']),             # (N,)
                }
                
                # 可选：加载图像（如果需要可视化）
                if self.enable_visualization:
                    if 'observations/images/color' in f:
                        self.trajectory_data['images'] = np.array(f['observations/images/color'])
                        self.get_logger().info(f"Loaded images: {self.trajectory_data['images'].shape}")
                    else:
                        self.get_logger().warn("No images found in dataset")
                
                # 计算时间间隔（用于播放速率）
                timestamps = self.trajectory_data['timestamp']
                self.dt_array = np.diff(timestamps)  # 帧间时间差
                self.dt_array = np.append(self.dt_array, self.dt_array[-1])  # 补充最后一帧
                
                # 生成插值轨迹（提高平滑度）
                self.interpolate_trajectory()
                
                self.get_logger().info(f"✓ Trajectory loaded successfully")
                self.get_logger().info(f"  - Original frames: {len(self.trajectory_data['qpos'])}")
                self.get_logger().info(f"  - Interpolated frames: {len(self.interpolated_trajectory['ee_pose'])}")
                self.get_logger().info(f"  - Duration: {timestamps[-1] - timestamps[0]:.2f}s")
                self.get_logger().info(f"  - Avg FPS: {len(timestamps) / (timestamps[-1] - timestamps[0]):.1f}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to load trajectory: {e}")
            raise
    
    def interpolate_trajectory(self):
        """
        对轨迹进行插值，生成更密集的控制点
        
        策略：
            1. 使用100Hz控制频率作为目标
            2. 对末端位姿（ee_pose）进行线性插值
            3. 对夹爪位置进行插值
            4. 生成新的时间戳序列
        
        插值方法：
            - 位置（x,y,z）：线性插值
            - 姿态（四元数）：简单线性插值（适合小角度变化）
            - 夹爪：线性插值
        """
        self.get_logger().info("Interpolating trajectory for smooth playback...")
        
        # 原始时间戳和数据
        t_original = self.trajectory_data['timestamp']
        ee_pose_original = self.trajectory_data['ee_pose']  # (N, 7)
        gripper_original = self.trajectory_data['gripper_pos']  # (N, 1)
        
        # 计算总时长和目标帧数（100Hz）
        duration = t_original[-1] - t_original[0]
        target_fps = 100.0
        n_interpolated = int(duration * target_fps)
        
        # 生成新的时间戳（均匀分布）
        t_new = np.linspace(t_original[0], t_original[-1], n_interpolated)
        
        # 对每个维度进行线性插值
        ee_pose_new = np.zeros((n_interpolated, 7))
        for i in range(7):
            interp_func = interp1d(t_original, ee_pose_original[:, i], 
                                   kind='linear', fill_value='extrapolate')
            ee_pose_new[:, i] = interp_func(t_new)
        
        # 夹爪插值
        gripper_new = np.zeros((n_interpolated, 1))
        interp_func = interp1d(t_original, gripper_original[:, 0], 
                               kind='linear', fill_value='extrapolate')
        gripper_new[:, 0] = interp_func(t_new)
        
        # 保存插值后的轨迹
        self.interpolated_trajectory = {
            'ee_pose': ee_pose_new,
            'gripper_pos': gripper_new,
            'timestamp': t_new
        }
        
        self.get_logger().info(f"Interpolation complete: {len(t_original)} -> {n_interpolated} frames")
    
    def enable_robot(self):
        """使能机械臂并设置速度"""
        self.get_logger().info("Enabling robot...")
        
        # 使能机械臂
        request = EnableRobot.Request()
        future = self.enable_robot_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() is not None:
            self.get_logger().info("✓ Robot enabled successfully")
        else:
            self.get_logger().error("✗ Failed to enable robot")
            raise RuntimeError("Failed to enable robot")
        
        # 设置速度因子为100%
        speed_req = SpeedFactor.Request()
        speed_req.ratio = 100
        speed_future = self.speed_factor_client.call_async(speed_req)
        rclpy.spin_until_future_complete(self, speed_future, timeout_sec=2.0)
        self.get_logger().info("✓ Speed factor set to 100%")
    
    def quaternion_to_euler(self, qx, qy, qz, qw):
        """
        将四元数转换为欧拉角（ZYX顺序，即Yaw-Pitch-Roll）
        
        参数:
            qx, qy, qz, qw: 四元数分量
        
        返回:
            tuple: (rx, ry, rz) 欧拉角，单位：度
                   rx - 绕X轴旋转角（Roll）
                   ry - 绕Y轴旋转角（Pitch）
                   rz - 绕Z轴旋转角（Yaw）
        
        说明:
            使用ZYX欧拉角顺序（外旋），这是DOBOT机器人常用的表示方式
            转换公式来源于标准的四元数到欧拉角转换算法
        """
        # 计算旋转矩阵的各个元素（用于欧拉角提取）
        # Roll (rx) - 绕X轴旋转
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        rx = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (ry) - 绕Y轴旋转
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            ry = np.copysign(np.pi / 2, sinp)  # 使用90度，处理万向节锁
        else:
            ry = np.arcsin(sinp)
        
        # Yaw (rz) - 绕Z轴旋转
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        rz = np.arctan2(siny_cosp, cosy_cosp)
        
        # 转换为角度（DOBOT API使用角度制）
        rx_deg = np.rad2deg(rx)
        ry_deg = np.rad2deg(ry)
        rz_deg = np.rad2deg(rz)
        
        return rx_deg, ry_deg, rz_deg
    
    def send_cartesian_command(self, ee_pose):
        """
        发送笛卡尔空间位姿指令（ServoP - 平滑控制）
        
        参数:
            ee_pose (np.ndarray): 末端位姿 [x, y, z, qx, qy, qz, qw]
        
        说明:
            - 使用ServoP进行实时笛卡尔空间控制
            - 异步非阻塞调用，确保高频率控制
            - 从四元数转换为欧拉角（Dobot API使用欧拉角）
        """
        # 提取位置（已经是mm单位）
        x, y, z = ee_pose[0] * 1000, ee_pose[1] * 1000, ee_pose[2] * 1000  # 米转毫米
        
        # 从四元数转换为欧拉角
        # ee_pose存储格式: [x, y, z, qx, qy, qz, qw]
        qx, qy, qz, qw = ee_pose[3:7]
        
        # 使用quaternion_to_euler方法进行转换
        rx, ry, rz = self.quaternion_to_euler(qx, qy, qz, qw)
        
        request = ServoP.Request()
        request.x = float(x)
        request.y = float(y)
        request.z = float(z)
        request.rx = float(rx)
        request.ry = float(ry)
        request.rz = float(rz)
        
        # 异步非阻塞调用（参考data_collector4的实现）
        if self.servo_p_client.service_is_ready():
            self.servo_p_client.call_async(request)
    
    def send_joint_command(self, joint_angles, dt=0.2):
        """
        发送关节角度指令（备用方法）
        
        参数:
            joint_angles (np.ndarray): 6个关节角度（弧度）
            dt (float): 到达时间（秒）
        """
        # 将弧度转换为角度（Dobot API通常使用角度）
        joint_degrees = np.rad2deg(joint_angles)
        
        request = ServoJ.Request()
        request.j1 = float(joint_degrees[0])
        request.j2 = float(joint_degrees[1])
        request.j3 = float(joint_degrees[2])
        request.j4 = float(joint_degrees[3])
        request.j5 = float(joint_degrees[4])
        request.j6 = float(joint_degrees[5])
        request.t = float(dt)  # 运动时间
        
        # 异步调用，不阻塞
        future = self.servo_j_client.call_async(request)
        
        # 可选：如果需要确认命令已执行，取消注释下面的行
        # rclpy.spin_until_future_complete(self, future, timeout_sec=0.1)
    
    def publish_image(self, frame_idx):
        """发布当前帧的图像（用于可视化）"""
        if not self.enable_visualization or 'images' not in self.trajectory_data:
            return
        
        try:
            img_array = self.trajectory_data['images'][frame_idx]
            # 转换为ROS Image消息
            img_msg = self.bridge.cv2_to_imgmsg(img_array, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'playback'
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish image: {e}")
    
    def publish_status(self, frame_idx, total_frames):
        """发布播放状态"""
        status_msg = String()
        status_msg.data = f"Playing: {frame_idx}/{total_frames}"
        self.status_pub.publish(status_msg)
        
        progress_msg = Float32()
        progress_msg.data = float(frame_idx) / float(total_frames) if total_frames > 0 else 0.0
        self.progress_pub.publish(progress_msg)
    
    def play_trajectory(self):
        """
        播放轨迹的主循环 - 使用高频率ServoP控制实现平滑运动
        
        执行流程：
            1. 使用插值后的轨迹（100Hz密度）
            2. 以固定频率（100Hz）发送ServoP指令
            3. 每次发送一个微小的位姿增量
            4. 异步非阻塞调用，确保实时性
            5. 可选发布图像和状态
        
        关键改进：
            - 使用ServoP（笛卡尔空间）而不是ServoJ（关节空间）
            - 高频率控制（100Hz），参考data_collector4
            - 轨迹插值生成密集控制点
            - 异步非阻塞，避免延迟累积
        """
        self.is_playing = True
        
        # 使用插值后的轨迹
        total_frames = len(self.interpolated_trajectory['ee_pose'])
        timestamps = self.interpolated_trajectory['timestamp']
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("▶ Starting smooth trajectory playback (100Hz ServoP)...")
        self.get_logger().info("=" * 50)
        
        # 控制循环参数（参考data_collector4）
        LOOP_RATE = 100.0  # Hz
        dt = 1.0 / LOOP_RATE / self.playback_speed  # 考虑播放速度
        
        try:
            while self.is_playing:
                start_time = time.time()
                
                for i in range(total_frames):
                    if not self.is_playing:
                        break
                    
                    loop_start = time.time()
                    
                    # 获取当前帧数据
                    ee_pose = self.interpolated_trajectory['ee_pose'][i]
                    gripper_pos = self.interpolated_trajectory['gripper_pos'][i]
                    
                    # 发送笛卡尔空间指令（ServoP，平滑控制）
                    self.send_cartesian_command(ee_pose)
                    
                    # TODO: 发送夹爪指令（需要根据实际夹爪接口实现）
                    # 参考data_collector4的GripperManager实现
                    # self.send_gripper_command(gripper_pos)
                    
                    # 发布可视化和状态（降低频率避免过载）
                    if i % 10 == 0:  # 每10帧发布一次（10Hz）
                        # 计算原始帧索引用于图像显示
                        original_idx = int(i * len(self.trajectory_data['qpos']) / total_frames)
                        self.publish_image(original_idx)
                        self.publish_status(i, total_frames)
                    
                    # 进度输出
                    if i % 100 == 0 or i == total_frames - 1:
                        progress = (i + 1) / total_frames * 100
                        elapsed = time.time() - start_time
                        self.get_logger().info(
                            f"Progress: {i+1}/{total_frames} ({progress:.1f}%) | "
                            f"Elapsed: {elapsed:.1f}s"
                        )
                    
                    # 精确时间控制（补偿处理时间）
                    loop_elapsed = time.time() - loop_start
                    sleep_time = dt - loop_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif sleep_time < -0.001:  # 如果延迟超过1ms，警告
                        if i % 100 == 0:
                            self.get_logger().warn(
                                f"Loop rate falling behind: {-sleep_time*1000:.1f}ms"
                            )
                
                # 播放完成
                total_time = time.time() - start_time
                self.get_logger().info("=" * 50)
                self.get_logger().info(f"✓ Trajectory playback completed in {total_time:.2f}s!")
                self.get_logger().info("=" * 50)
                
                # 检查是否循环播放
                if not self.loop_playback:
                    self.is_playing = False
                    break
                else:
                    self.get_logger().info("↻ Looping playback...")
                    time.sleep(1.0)  # 循环间隔
                    
        except KeyboardInterrupt:
            self.get_logger().info("⏸ Playback interrupted by user")
            self.is_playing = False
        except Exception as e:
            self.get_logger().error(f"✗ Playback error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.is_playing = False
    
    def stop(self):
        """停止播放"""
        self.is_playing = False
        self.get_logger().info("⏹ Playback stopped")


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        player = DataPlayer()
        
        # 启动播放
        player.play_trajectory()
        
        # 保持节点运行（如果需要循环播放）
        # rclpy.spin(player)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'player' in locals():
            player.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
