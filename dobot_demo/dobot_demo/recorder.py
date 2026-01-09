

import rclpy
from rclpy.node import Node
import h5py
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
from collections import deque
import threading
import time

# ROS Messages
from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder_node')
        
        # 参数配置
        self.data_dir = "./data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.bridge = CvBridge()
        
        # 状态控制
        # 0: Idle/Discard, 1: Recording, 2: Save
        self.recording_state = 0
        self.episode_buffer = [] # 暂存当前episode数据
        self.mutex = threading.Lock()
        
        # === 缓存最新的传感器数据 ===
        self.last_color = None
        self.last_depth = None
        self.last_camera_info = None
        self.last_joint_state = None
        self.last_gripper_state = None
        self.last_ee_pose = None
        
        # === 订阅器 ===
        # 1. 指令 (来自控制脚本)
        self.sub_cmd = self.create_subscription(Int32, "/recorder/command", self.cmd_callback, 10)
        
        # 2. 相机 (假设Orbbec发布到以下Topic，根据Launch文件调整)
        self.sub_color = self.create_subscription(Image, "/camera/color/image_raw", self.color_callback, 10)
        self.sub_depth = self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, "/camera/color/camera_info", self.info_callback, 10)
        
        # 3. 机器人状态
        self.sub_joints = self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.sub_gripper = self.create_subscription(JointState, "/gripper/state", self.gripper_callback, 10)
        self.sub_pose = self.create_subscription(PoseStamped, "/end_effector_pose", self.pose_callback, 10)
        
        self.get_logger().info("Data Recorder Initialized. Press 'O' in controller to start.")

    # === 回调函数：只负责更新最新数据 ===
    
    def cmd_callback(self, msg):
        new_state = msg.data
        with self.mutex:
            if new_state == 1 and self.recording_state != 1:
                self.get_logger().info(">>> START RECORDING")
                self.episode_buffer = [] # 清空缓存
                self.recording_state = 1
                
            elif new_state == 2 and self.recording_state == 1:
                self.get_logger().info(">>> STOP & SAVING...")
                self.recording_state = 2 # 标记为保存中
                self.save_dataset()
                self.recording_state = 0 # 回到空闲
                
            elif new_state == 0:
                if self.recording_state == 1:
                    self.get_logger().info(">>> STOP & DISCARD")
                self.recording_state = 0
                self.episode_buffer = []

    def color_callback(self, msg):
        self.last_color = msg
        # 以RGB图像的到达作为触发信号，执行一次对齐和记录
        self.try_record_frame()

    def depth_callback(self, msg):
        self.last_depth = msg
    
    def info_callback(self, msg):
        # CameraInfo 通常不变，只需要存一次或最新的即可
        self.last_camera_info = msg

    def joint_callback(self, msg):
        self.last_joint_state = msg

    def gripper_callback(self, msg):
        self.last_gripper_state = msg

    def pose_callback(self, msg):
        self.last_ee_pose = msg

    # === 核心逻辑：数据对齐与记录 ===
    
    def try_record_frame(self):
        """
        当收到一张新图像时，尝试打包当前所有传感器数据。
        这是一种"以图像为基准"的软同步策略。
        """
        if self.recording_state != 1:
            return

        with self.mutex:
            # 检查必要数据是否存在
            if self.last_color is None or self.last_joint_state is None or \
               self.last_gripper_state is None or self.last_ee_pose is None:
                # self.get_logger().warn("Waiting for all sensors...", throttle_duration_sec=1.0)
                return

            # 数据打包
            try:
                # 转换图像
                cv_image = self.bridge.imgmsg_to_cv2(self.last_color, desired_encoding='bgr8')
                cv_depth = None
                if self.last_depth:
                    cv_depth = self.bridge.imgmsg_to_cv2(self.last_depth, desired_encoding='passthrough')

                frame_data = {
                    'timestamp': self.last_color.header.stamp.sec + self.last_color.header.stamp.nanosec * 1e-9,
                    'image': cv_image,
                    'depth': cv_depth,
                    'qpos': np.array(self.last_joint_state.position, dtype=np.float32),
                    'gripper_pos': np.array(self.last_gripper_state.position, dtype=np.float32),
                    'ee_pose': np.array([
                        self.last_ee_pose.pose.position.x,
                        self.last_ee_pose.pose.position.y,
                        self.last_ee_pose.pose.position.z,
                        self.last_ee_pose.pose.orientation.x, # 注意: Dobot通常只给Position，Orientation可能需要自己算或为空
                        self.last_ee_pose.pose.orientation.y,
                        self.last_ee_pose.pose.orientation.z,
                        self.last_ee_pose.pose.orientation.w
                    ], dtype=np.float32)
                }
                
                # 如果有相机内参，也记录（通常只需要第一帧，但为了方便全存）
                if self.last_camera_info:
                    frame_data['camera_intrinsics'] = np.array(self.last_camera_info.k, dtype=np.float32).reshape(3, 3)

                self.episode_buffer.append(frame_data)
                
                # 打印进度
                if len(self.episode_buffer) % 30 == 0:
                    print(f"Recording... {len(self.episode_buffer)} frames")
                    
            except Exception as e:
                self.get_logger().error(f"Error packing frame: {e}")

    # === 保存逻辑 ===
    
    def get_next_episode_index(self):
        """遍历data目录，找到下一个可用的编号"""
        existing_files = [f for f in os.listdir(self.data_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
        if not existing_files:
            return 0
        
        indices = []
        for f in existing_files:
            try:
                idx = int(f.split('_')[1].split('.')[0])
                indices.append(idx)
            except: pass
        
        return max(indices) + 1 if indices else 0

    def save_dataset(self):
        if not self.episode_buffer:
            self.get_logger().warn("Buffer empty, nothing to save.")
            return

        idx = self.get_next_episode_index()
        file_path = os.path.join(self.data_dir, f"episode_{idx}.hdf5")
        
        self.get_logger().info(f"Saving {len(self.episode_buffer)} frames to {file_path}...")
        
        try:
            with h5py.File(file_path, 'w') as f:
                # 1. 提取各个数据流
                data_len = len(self.episode_buffer)
                
                # 预分配数据集 (Chunked storage for images)
                # 图像：(N, H, W, 3)
                img_sample = self.episode_buffer[0]['image']
                dset_img = f.create_dataset('observations/images/color', 
                                          (data_len, img_sample.shape[0], img_sample.shape[1], 3), 
                                          dtype='uint8', chunks=(1, img_sample.shape[0], img_sample.shape[1], 3))
                
                if self.episode_buffer[0]['depth'] is not None:
                    depth_sample = self.episode_buffer[0]['depth']
                    dset_depth = f.create_dataset('observations/images/depth', 
                                                (data_len, depth_sample.shape[0], depth_sample.shape[1]), 
                                                dtype='uint16', chunks=(1, depth_sample.shape[0], depth_sample.shape[1]))

                dset_qpos = f.create_dataset('observations/qpos', (data_len, 6), dtype='float32') # Dobot 6 axis
                dset_gpos = f.create_dataset('observations/gripper_pos', (data_len, 1), dtype='float32')
                dset_pose = f.create_dataset('observations/ee_pose', (data_len, 7), dtype='float32')
                dset_time = f.create_dataset('timestamp', (data_len,), dtype='float64')

                # 2. 写入数据
                for i, frame in enumerate(self.episode_buffer):
                    dset_img[i] = frame['image']
                    if frame['depth'] is not None:
                        dset_depth[i] = frame['depth']
                    
                    # 确保维度匹配
                    if len(frame['qpos']) >= 6: dset_qpos[i] = frame['qpos'][:6]
                    dset_gpos[i] = frame['gripper_pos']
                    dset_pose[i] = frame['ee_pose']
                    dset_time[i] = frame['timestamp']
                
                # 保存相机内参 (取第一帧即可)
                if 'camera_intrinsics' in self.episode_buffer[0]:
                    f.create_dataset('camera/intrinsics', data=self.episode_buffer[0]['camera_intrinsics'])
                
                f.attrs['sim'] = False
                f.attrs['total_frames'] = data_len
                
            self.get_logger().info(f"Save successfully! Saved to {file_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save dataset: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()