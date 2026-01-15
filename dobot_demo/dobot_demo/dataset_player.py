#!/usr/bin/env python3
"""
数据集播放器 - 用于重放录制的机械臂轨迹

功能概述：
从HDF5文件读取轨迹数据，控制Dobot机械臂和夹爪重放录制的动作。
同时发布状态话题供评估节点实时分析播放质量（跟踪误差、延迟等）。

主要组件：
1. DobotRosWrapper: ROS接口封装，提供服务客户端和话题发布器
2. GripperComm: 夹爪Modbus通信管理，控制夹爪开合
3. GripperStateFeedback: 夹爪状态反馈线程，使用时间插补将10Hz采样提升到100Hz发布
4. DatasetPlayer: 主播放器类，加载数据集并按时间序列重放

技术特点：
- 高频控制：使用ServoP伺服模式，100Hz控制频率
- 时间插补：夹爪状态使用线性插补，将低频Modbus读取平滑到高频发布
- 精确时序：基于录制时的实际时间间隔，保证播放时序一致性
- 实时监控：监控单帧延迟，超过10ms发出警告

ROS话题发布：
- /robot/target_pose: 机械臂目标姿态（供评估对比）
- /gripper/state_feedback: 夹爪实际位置（100Hz插补）
- /gripper/command_update: 夹爪目标位置（100Hz插补）

使用方法：
python3 dataset_player.py <hdf5_path> [--rate <playback_rate>]

示例：
python3 dataset_player.py ./data/episode_0.hdf5           # 原速播放
python3 dataset_player.py ./data/episode_0.hdf5 --rate 0.5 # 0.5倍速慢放
python3 dataset_player.py ./data/episode_0.hdf5 --rate 2.0 # 2倍速快放
"""

import rclpy
from rclpy.node import Node
import h5py
import numpy as np
import time
import threading
import re
import sys
import os

# ROS消息类型
from geometry_msgs.msg import PointStamped
from dobot_msgs_v3.msg import ToolVectorActual
from dobot_msgs_v3.srv import *

class DobotRosWrapper(Node):
    """
    机械臂ROS接口封装

    提供与Dobot机械臂交互的ROS服务客户端和话题发布器。
    封装了机械臂控制、夹爪通信、状态发布等功能。

    主要功能：
    - 机械臂使能、清除错误、速度设置
    - ServoP伺服运动控制
    - 获取当前位姿
    - Modbus通信（用于夹爪控制）
    - 发布机械臂目标姿态和夹爪状态
    """
    def __init__(self, node_name="dataset_player"):
        """
        初始化ROS节点和服务客户端

        Args:
            node_name: ROS节点名称，默认为"dataset_player"
        """
        super().__init__(node_name)

        # === 机械臂控制服务客户端 ===
        self.cli_enable = self.create_client(EnableRobot, '/dobot_bringup_v3/srv/EnableRobot')  # 使能机械臂
        self.cli_clear_error = self.create_client(ClearError, '/dobot_bringup_v3/srv/ClearError')  # 清除错误
        self.cli_speed_factor = self.create_client(SpeedFactor, '/dobot_bringup_v3/srv/SpeedFactor')  # 设置速度比例
        self.cli_servo_p = self.create_client(ServoP, '/dobot_bringup_v3/srv/ServoP')  # 伺服运动控制
        self.cli_get_pose = self.create_client(GetPose, '/dobot_bringup_v3/srv/GetPose')  # 获取当前位姿

        # === Modbus通信服务客户端（用于夹爪控制）===
        self.cli_modbus_create = self.create_client(ModbusCreate, '/dobot_bringup_v3/srv/ModbusCreate')  # 创建Modbus连接
        self.cli_modbus_close = self.create_client(ModbusClose, '/dobot_bringup_v3/srv/ModbusClose')  # 关闭Modbus连接
        self.cli_set_hold_regs = self.create_client(SetHoldRegs, '/dobot_bringup_v3/srv/SetHoldRegs')  # 写Modbus寄存器
        self.cli_get_hold_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs')  # 读Modbus寄存器

        # === 话题发布器 ===
        # 发布机械臂目标姿态（供评估节点对比）
        self.pub_robot_target = self.create_publisher(ToolVectorActual, "/robot/target_pose", 10)
        # 发布夹爪实际位置反馈（插补后的100Hz）
        self.pub_gripper_state = self.create_publisher(PointStamped, "/gripper/state_feedback", 10)
        # 发布夹爪目标位置（插补后的100Hz）
        self.pub_gripper_update = self.create_publisher(PointStamped, "/gripper/command_update", 10)

        self.get_logger().info("等待Dobot服务...")
        if not self.cli_servo_p.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Dobot服务未就绪")

    def call_service(self, client, request):
        """
        同步调用ROS服务并等待结果

        使用异步调用方式，但在结果返回前阻塞等待。
        适用于需要确认服务执行结果的场景。

        Args:
            client: ROS服务客户端
            request: 服务请求对象

        Returns:
            服务响应结果，如果服务未就绪则返回None
        """
        if not client.service_is_ready():
            return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.001)  # 轮询间隔1ms，避免CPU占用过高
        return future.result()

    def call_service_async_no_wait(self, client, request):
        """
        异步发送服务请求，不等待结果

        用于高频控制场景（如ServoP），避免阻塞主循环。
        发送后立即返回，不关心服务执行结果。

        Args:
            client: ROS服务客户端
            request: 服务请求对象
        """
        if client.service_is_ready():
            client.call_async(request)

class GripperComm:
    """
    夹爪Modbus通信管理类

    通过Modbus RTU协议与夹爪通信，控制夹爪开合和读取状态。
    使用机械臂控制器作为Modbus主站，通过127.0.0.1:60000端口通信。

    关键寄存器地址：
    - 256: 使能寄存器 (1=使能)
    - 257: 力度/速度寄存器 (0-100)
    - 259: 目标位置寄存器 (0-1000，0=全开，1000=全闭)
    - 514: 当前位置反馈寄存器 (0-1000)
    """
    def __init__(self, wrapper):
        """
        初始化夹爪通信

        Args:
            wrapper: DobotRosWrapper实例，用于调用Modbus服务
        """
        self.wrapper = wrapper
        self.id = 0  # Modbus连接ID，由服务端分配
        self.init_connection()

    def init_connection(self):
        """
        初始化Modbus连接

        步骤：
        1. 关闭可能存在的旧连接（索引1-4）
        2. 创建新的Modbus RTU连接
        3. 初始化夹爪寄存器（使能、力度/速度）
        """
        # 关闭旧连接（清理可能残留的连接）
        print("[Gripper] 关闭旧连接...")
        for i in range(1, 5):
            req = ModbusClose.Request()
            req.index = i
            self.wrapper.call_service(self.wrapper.cli_modbus_close, req)

        # 创建新连接
        print("[Gripper] 创建Modbus连接...")
        req = ModbusCreate.Request()
        req.ip = "127.0.0.1"  # 本地回环地址，连接到机械臂控制器
        req.port = 60000  # 标准Modbus端口
        req.slave_id = 1  # 夹爪从站ID
        req.is_rtu = 1  # RTU模式
        res = self.wrapper.call_service(self.wrapper.cli_modbus_create, req)

        if res and res.res == 0:
            # 从响应中提取连接ID
            match = re.search(r'(\d+)', str(res.index))
            self.id = int(match.group(1)) if match else int(res.index)
            print(f"[Gripper] 连接成功, ID: {self.id}")
        else:
            print(f"[Gripper] 连接失败! Response: {res}")
            self.id = 0

        if self.id > 0:
            print("[Gripper] 初始化寄存器...")
            self.write_reg(256, 1, "1", wait=True)  # 使能夹爪
            self.write_reg(257, 1, "60", wait=True) # 设置力度/速度为60%

    def write_reg(self, addr, count, val_str, wait=False):
        """
        写Modbus保持寄存器

        Args:
            addr: 寄存器起始地址
            count: 要写入的寄存器数量
            val_str: 要写入的值（字符串格式，多个值用空格分隔）
            wait: 是否等待写入完成（True=同步，False=异步）

        常用寄存器地址：
        - 256: 使能 (0=禁用，1=使能)
        - 257: 力度/速度 (0-100)
        - 259: 目标位置 (0-1000)
        """
        if self.id <= 0:
            return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_str
        if wait:
            self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)
        else:
            self.wrapper.call_service_async_no_wait(self.wrapper.cli_set_hold_regs, req)

    def read_reg(self, addr):
        """
        读取Modbus保持寄存器

        Args:
            addr: 寄存器地址

        Returns:
            寄存器值（整数），失败返回None

        常用寄存器地址：
        - 514: 当前位置反馈 (0-1000)
        """
        if self.id <= 0:
            return None
        req = GetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = 1
        res = self.wrapper.call_service(self.wrapper.cli_get_hold_regs, req)
        if res and res.res == 0:
            try:
                return int(res.value)
            except:
                pass
        return None

class GripperStateFeedback(threading.Thread):
    """
    夹爪状态反馈线程

    定期读取夹爪实际位置并发布当前状态和目标状态。
    使用时间插补技术将低频Modbus读取（10Hz）插补到高频发布（100Hz）。

    工作原理：
    1. 以10Hz频率读取夹爪实际位置（寄存器514）
    2. 以10Hz频率采样目标位置（从播放器）
    3. 使用线性插补算法，在两次采样之间进行时间插值
    4. 以100Hz频率发布插补后的状态和目标位置

    这样做的优势：
    - 减少Modbus通信频率，避免总线拥塞
    - 提供平滑的高频状态反馈，便于评估和可视化
    - 避免阶跃变化，更符合物理实际
    """
    def __init__(self, wrapper, comm, target_pos_ref):
        """
        初始化夹爪状态反馈线程

        Args:
            wrapper: DobotRosWrapper实例，用于发布话题
            comm: GripperComm实例，用于读取夹爪位置
            target_pos_ref: 目标位置引用（列表），与播放器共享
        """
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.comm = comm
        self.target_pos_ref = target_pos_ref  # 引用目标位置
        self.running = True

        # 频率设置
        self.feedback_rate = 100.0  # 发布频率 100Hz
        self.modbus_read_rate = 10.0  # Modbus读取频率 10Hz
        self.read_interval = 1.0 / self.modbus_read_rate

        # === 插补状态变量 ===
        # 用于跟踪两次采样点，进行线性插补

        # Modbus读取时间
        self.last_read_time = time.time()

        # 实际位置插补变量
        self.last_real_pos = None  # 上一次读取的实际位置
        self.current_real_pos = None  # 当前读取的实际位置
        self.last_real_read_time = None  # 上一次读取的时间戳
        self.current_real_read_time = None  # 当前读取的时间戳

        # 目标位置插补变量
        self.last_target_pos = None  # 上一次的目标位置
        self.current_target_pos = None  # 当前的目标位置
        self.last_target_read_time = None  # 上一次读取的时间戳
        self.current_target_read_time = None  # 当前读取的时间戳

    def _interpolate(self, last_val, current_val, last_time, current_time, now):
        """
        基于时间的线性插补算法

        在两次采样值之间进行线性插值，使低频数据平滑到高频。

        插补公式：
        interpolated = last_val + alpha * (current_val - last_val)
        其中 alpha = (now - last_time) / (current_time - last_time)

        Args:
            last_val: 上一次采样的值
            current_val: 当前采样的值
            last_time: 上一次采样的时间戳
            current_time: 当前采样的时间戳
            now: 当前时刻（要插补的时间点）

        Returns:
            插补后的值（float）

        示例：
            假设在t=0时刻值为100，t=0.1时刻值为200
            在t=0.05时刻（中间点），插补值为150
        """
        if last_val is None or current_val is None:
            return current_val if current_val is not None else 0.0

        if last_time is None or current_time is None:
            return current_val

        time_span = current_time - last_time
        if time_span <= 0:
            return current_val

        elapsed = now - last_time
        alpha = min(1.0, elapsed / time_span)  # 限制在[0, 1]范围内
        return last_val + alpha * (current_val - last_val)

    def run(self):
        """
        核心循环：100Hz发布夹爪状态

        工作流程：
        1. 每0.1秒（10Hz）读取真实值：
           - 从Modbus寄存器514读取夹爪实际位置
           - 从播放器读取目标位置
           - 更新插补状态变量
        2. 每0.01秒（100Hz）发布：
           - 对实际位置和目标位置进行时间插补
           - 发布到 /gripper/state_feedback（实际位置）
           - 发布到 /gripper/command_update（目标位置）
        """
        while self.running:
            try:
                now = time.time()
                timestamp = self.wrapper.get_clock().now().to_msg()

                # === 每0.1秒读取一次真实值（10Hz采样）===
                if now - self.last_read_time >= self.read_interval:
                    self.last_read_time = now

                    # 读取夹爪实际位置（Modbus寄存器514）
                    real_pos = self.comm.read_reg(514)
                    if real_pos is not None:
                        # 更新插补状态：将当前值变为历史值，新值变为当前值
                        self.last_real_pos = self.current_real_pos
                        self.last_real_read_time = self.current_real_read_time
                        self.current_real_pos = float(real_pos)
                        self.current_real_read_time = now

                    # 读取目标位置（从播放器的target_pos_ref）
                    target = self.target_pos_ref[0] if self.target_pos_ref else 0.0
                    self.last_target_pos = self.current_target_pos
                    self.last_target_read_time = self.current_target_read_time
                    self.current_target_pos = target
                    self.current_target_read_time = now

                # === 基于时间的线性插补（每次循环都执行，100Hz）===
                # 计算插补后的实际位置
                interpolated_current = self._interpolate(
                    self.last_real_pos,
                    self.current_real_pos,
                    self.last_real_read_time,
                    self.current_real_read_time,
                    now
                )

                # 计算插补后的目标位置
                interpolated_target = self._interpolate(
                    self.last_target_pos,
                    self.current_target_pos,
                    self.last_target_read_time,
                    self.current_target_read_time,
                    now
                )

                # === 发布插补后的状态（100Hz）===
                # 发布实际位置反馈
                msg_current = PointStamped()
                msg_current.header.stamp = timestamp
                msg_current.header.frame_id = "gripper_feedback"
                msg_current.point.x = float(interpolated_current)
                self.wrapper.pub_gripper_state.publish(msg_current)

                # 发布目标位置
                msg_target = PointStamped()
                msg_target.header.stamp = timestamp
                msg_target.header.frame_id = "gripper_command"
                msg_target.point.x = float(interpolated_target)
                self.wrapper.pub_gripper_update.publish(msg_target)

            except Exception as e:
                print(f"[ERROR] GripperStateFeedback: {e}")

            time.sleep(1.0 / self.feedback_rate)  # 100Hz = 0.01s

class DatasetPlayer:
    """
    数据集播放器主类

    从HDF5数据集文件读取轨迹数据，控制机械臂和夹爪重放录制的动作。
    同时发布状态话题供评估节点分析播放质量。

    主要功能：
    1. 加载HDF5数据集（机械臂轨迹、夹爪轨迹、时间戳）
    2. 初始化机械臂和夹爪通信
    3. 启动夹爪状态反馈线程
    4. 按时间序列播放轨迹
    5. 发布目标姿态供评估对比

    HDF5数据集格式：
    - actions/robot_target: (N, 6) 机械臂目标姿态 [x, y, z, rx, ry, rz]
    - actions/gripper_target: (N, 1) 夹爪目标位置 [0-1000]
    - timestamp: (N,) 时间戳序列
    """
    def __init__(self, hdf5_path, playback_rate=1.0):
        """
        初始化数据集播放器

        Args:
            hdf5_path: HDF5数据集文件路径
            playback_rate: 播放速率倍数（1.0=原速，2.0=2倍速）
        """
        # 初始化ROS
        if not rclpy.ok():
            rclpy.init()

        self.node = DobotRosWrapper()

        # ROS Spin线程（在后台处理ROS回调）
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        # === 初始化机械臂 ===
        self.node.call_service(self.node.cli_speed_factor, SpeedFactor.Request(ratio=100))  # 设置速度为100%
        self.node.call_service(self.node.cli_clear_error, ClearError.Request())  # 清除错误
        self.node.call_service(self.node.cli_enable, EnableRobot.Request())  # 使能机械臂

        # === 初始化夹爪 ===
        self.gripper_comm = GripperComm(self.node)

        # 夹爪目标位置引用（用于状态反馈线程读取）
        # 使用列表以便引用传递，让反馈线程能够读取最新的目标位置
        self.gripper_target_pos = [0.0]

        # === 启动夹爪状态反馈线程 ===
        self.gripper_feedback = GripperStateFeedback(self.node, self.gripper_comm, self.gripper_target_pos)
        self.gripper_feedback.start()

        # 加载数据集
        self.hdf5_path = hdf5_path
        self.playback_rate = playback_rate
        self.load_dataset()

        print("=== 数据集播放器初始化完成 ===")
        print(f"数据集: {hdf5_path}")
        print(f"总帧数: {self.total_frames}")
        print(f"播放速率: {playback_rate}x")
        print(f"控制频率: 100Hz")

    def load_dataset(self):
        """
        加载HDF5数据集

        从HDF5文件读取以下数据：
        - actions/robot_target: 机械臂目标姿态序列 (N, 6)
        - actions/gripper_target: 夹爪目标位置序列 (N, 1)
        - timestamp: 时间戳序列 (N,)

        同时读取并显示元数据（如同步方法、采样频率等）
        """
        if not os.path.exists(self.hdf5_path):
            print(f"错误：文件不存在 {self.hdf5_path}")
            sys.exit(1)

        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                # 读取机械臂和夹爪轨迹
                self.robot_target = np.array(f['actions/robot_target'])  # (N, 6) [x,y,z,rx,ry,rz]
                self.gripper_target = np.array(f['actions/gripper_target'])  # (N, 1) [0-1000]
                self.timestamps = np.array(f['timestamp'])  # (N,) 时间戳

                self.total_frames = len(self.robot_target)

                # 打印元数据信息
                print(f"\n[Dataset Info]")
                print(f"  Total frames: {self.total_frames}")
                if 'sync_method' in f.attrs:
                    print(f"  Sync method: {f.attrs['sync_method']}")
                if 'camera_frequency_hz' in f.attrs:
                    print(f"  Camera freq: {f.attrs['camera_frequency_hz']} Hz")
                if 'robot_frequency_hz' in f.attrs:
                    print(f"  Robot freq: {f.attrs['robot_frequency_hz']} Hz")
                print()

        except Exception as e:
            print(f"错误：加载数据集失败 - {e}")
            sys.exit(1)

    def ServoP(self, x, y, z, rx, ry, rz):
        """
        发送ServoP伺服运动指令

        ServoP是实时伺服控制指令，用于高频（100Hz）运动控制。
        该方法异步发送，不等待执行结果，以保证高频控制不被阻塞。

        Args:
            x, y, z: 目标位置 (mm)
            rx, ry, rz: 目标姿态 (度)
        """
        req = ServoP.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        self.node.call_service_async_no_wait(self.node.cli_servo_p, req)

    def set_gripper(self, position):
        """
        设置夹爪目标位置

        更新夹爪目标位置，并通过Modbus写入夹爪寄存器。
        同时更新target_pos_ref，供状态反馈线程读取。

        Args:
            position: 目标位置，范围0-1000 (0=全开，1000=全闭)
        """
        self.gripper_target_pos[0] = float(position)  # 更新引用，供反馈线程读取
        self.gripper_comm.write_reg(259, 1, str(int(position)), wait=False)  # 写入寄存器259

    def publish_robot_target(self, pose):
        """
        发布机械臂目标姿态

        将当前目标姿态发布到 /robot/target_pose 话题。
        供评估节点订阅，与实际姿态对比分析跟踪误差。

        Args:
            pose: 目标姿态 [x, y, z, rx, ry, rz]
        """
        msg = ToolVectorActual()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.x = float(pose[0])
        msg.y = float(pose[1])
        msg.z = float(pose[2])
        msg.rx = float(pose[3])
        msg.ry = float(pose[4])
        msg.rz = float(pose[5])
        self.node.pub_robot_target.publish(msg)

    def play(self):
        """
        播放数据集主循环

        按时间序列重放轨迹数据：
        1. 计算平均时间间隔（从timestamps）
        2. 遍历所有帧：
           - 发送机械臂ServoP指令
           - 发送夹爪位置指令
           - 发布目标姿态话题
           - 按实际时间间隔sleep，保持播放速率
        3. 监控延迟，如果单帧延迟超过10ms则警告

        时间控制：
        - 使用录制时的实际时间间隔（从timestamps计算）
        - 根据playback_rate调整播放速度
        - 实时监控执行时间，动态调整sleep时间
        """
        print("\n=== 开始播放 ===")
        print("按 Ctrl+C 停止\n")

        # 计算时间间隔
        if self.total_frames > 1:
            # 使用实际录制的时间间隔（相邻帧之间的时间差）
            time_diffs = np.diff(self.timestamps)
            avg_dt = np.mean(time_diffs)
            print(f"平均时间间隔: {avg_dt*1000:.2f} ms ({1.0/avg_dt:.1f} Hz)")
        else:
            avg_dt = 0.01  # 默认100Hz

        # 播放循环
        start_time = time.time()

        try:
            for i in range(self.total_frames):
                loop_start = time.time()

                # === 获取当前帧数据 ===
                robot_pose = self.robot_target[i]  # (6,) [x,y,z,rx,ry,rz]
                gripper_pos = self.gripper_target[i][0]  # 标量 [0-1000]

                # === 发送控制指令 ===
                self.ServoP(*robot_pose)  # 控制机械臂
                self.set_gripper(gripper_pos)  # 控制夹爪

                # === 发布目标状态话题（供评估节点使用）===
                self.publish_robot_target(robot_pose)

                # === 打印进度（每30帧或最后一帧）===
                if i % 30 == 0 or i == self.total_frames - 1:
                    elapsed = time.time() - start_time
                    progress = (i + 1) / self.total_frames * 100
                    print(f"播放进度: {i+1}/{self.total_frames} ({progress:.1f}%) | "
                          f"时间: {elapsed:.1f}s | "
                          f"姿态: [{robot_pose[0]:.1f}, {robot_pose[1]:.1f}, {robot_pose[2]:.1f}] | "
                          f"夹爪: {gripper_pos:.0f}")

                # === 控制播放速率（时间同步）===
                if i < self.total_frames - 1:
                    # 使用实际录制的时间间隔
                    target_dt = time_diffs[i] / self.playback_rate if i < len(time_diffs) else avg_dt
                else:
                    target_dt = avg_dt / self.playback_rate

                # 计算需要sleep的时间
                elapsed_loop = time.time() - loop_start
                sleep_time = target_dt - elapsed_loop

                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.01:  # 如果延迟超过10ms，发出警告
                    print(f"[WARNING] 帧 {i} 延迟 {-sleep_time*1000:.1f} ms")

            print("\n=== 播放完成 ===")
            print(f"总耗时: {time.time() - start_time:.2f}s")

        except KeyboardInterrupt:
            print("\n\n=== 播放被中断 ===")

        # 保持状态反馈线程运行一小段时间，等待夹爪到达最终位置
        print("等待夹爪到达最终位置...")
        time.sleep(1.0)

    def close(self):
        """
        关闭播放器并清理资源

        步骤：
        1. 停止夹爪状态反馈线程
        2. 等待线程结束
        3. 关闭ROS节点
        """
        self.gripper_feedback.running = False
        self.gripper_feedback.join()
        rclpy.shutdown()

def main():
    """
    主函数：命令行入口

    解析命令行参数，创建播放器实例，执行播放并清理资源。

    命令行参数：
    - dataset: HDF5数据集文件路径（必需）
    - --rate: 播放速率倍数（可选，默认1.0）

    使用示例：
    python3 dataset_player.py ./data/episode_0.hdf5
    python3 dataset_player.py ./data/episode_0.hdf5 --rate 0.5  # 0.5倍速
    python3 dataset_player.py ./data/episode_0.hdf5 --rate 2.0  # 2倍速
    """
    import argparse

    parser = argparse.ArgumentParser(description='播放HDF5数据集控制机械臂')
    parser.add_argument('dataset', type=str, help='HDF5数据集路径 (例如: ./data/episode_0.hdf5)')
    parser.add_argument('--rate', type=float, default=1.0, help='播放速率 (默认: 1.0x)')

    args = parser.parse_args()

    # 创建播放器
    player = DatasetPlayer(args.dataset, playback_rate=args.rate)

    # 开始播放
    player.play()

    # 关闭
    player.close()

if __name__ == "__main__":
    main()
