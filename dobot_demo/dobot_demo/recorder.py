#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
数据录制节点 - 机器人演示数据采集与存储
==============================================================================

功能概述:
    本节点用于录制机器人操作演示数据，包括：
    1. 视觉数据：RGB图像、深度图像、相机内参
    2. 状态数据：机械臂关节角度、夹爪位置、末端执行器位姿
    3. 时间戳：用于数据同步和回放
    
    数据以HDF5格式存储，适用于机器人学习和模仿学习任务。

工作流程:
    1. 待命状态：持续更新各传感器的最新数据
    2. 录制状态：接收到开始指令后，同步打包所有传感器数据
    3. 保存状态：录制结束后，将数据写入HDF5文件

订阅话题:
    - /recorder/command: 录制控制指令（0=停止/丢弃, 1=开始录制, 2=停止并保存）
    - /camera/color/image_raw: RGB图像
    - /camera/depth/image_raw: 深度图像
    - /camera/color/camera_info: 相机内参
    - /joint_states: 机械臂关节状态
    - /gripper/state: 夹爪状态
    - /end_effector_pose: 末端执行器位姿

输出文件:
    ./data/episode_X.hdf5 (X为自动递增的序号)
    
文件结构:
    - observations/images/color: RGB图像序列 (N, H, W, 3)
    - observations/images/depth: 深度图像序列 (N, H, W)
    - observations/qpos: 关节角度序列 (N, 6)
    - observations/gripper_pos: 夹爪位置序列 (N, 1)
    - observations/ee_pose: 末端位姿序列 (N, 7) [x,y,z,qx,qy,qz,qw]
    - timestamp: 时间戳序列 (N,)
    - camera/intrinsics: 相机内参矩阵 (3, 3)
    - attrs: 元数据（sim标志、总帧数等）
==============================================================================
"""

# =============================================================================
# 导入必要的库
# =============================================================================
import rclpy                                           # ROS2 Python接口库，用于与ROS2系统交互
from rclpy.node import Node                            # ROS2节点基类，所有ROS2节点都需要继承此类
import h5py                                            # HDF5文件格式的读写库，用于高效存储大型科学数据
import numpy as np                                     # 数值计算库，提供多维数组和矩阵运算功能
import os                                              # 文件系统操作库，用于目录和文件管理
import cv2                                             # OpenCV图像处理库（虽然代码中未直接使用，但可用于后续扩展）
from cv_bridge import CvBridge                         # ROS图像消息转换工具，将ROS Image消息与OpenCV图像格式相互转换
from collections import deque                          # 双端队列数据结构（代码中未使用，可用于未来扩展）
import threading                                       # 线程同步工具，用于线程安全的数据访问（这里使用互斥锁）
import time                                            # 时间处理库（代码中未使用，可用于时间戳生成）

# ROS标准消息类型
from sensor_msgs.msg import Image, JointState, CameraInfo  # 传感器消息：Image(图像)、JointState(关节状态)、CameraInfo(相机参数)
from geometry_msgs.msg import PoseStamped              # 位姿消息，包含位置(position)和姿态(orientation)信息
from std_msgs.msg import Int32                         # 整型消息，用于接收录制控制指令

class DataRecorder(Node):
    """
    数据录制节点类
    
    功能职责：
        1. 订阅机器人各个传感器的数据流
        2. 以RGB图像到达为基准进行数据同步
        3. 将同步的传感器数据缓存到内存
        4. 接收控制指令时执行开始/停止/保存操作
        5. 将缓存的数据写入HDF5文件格式
    
    核心设计思路：
        - 使用"最新数据缓存"模式：所有回调函数只更新最新接收到的数据
        - 以RGB图像到达作为"触发信号"，在此时刻进行数据同步和打包
        - 使用互斥锁保护共享数据，避免多线程访问冲突
        - 采用三态录制状态机：0(空闲) -> 1(录制中) -> 2(保存中) -> 0
    """
    
    def __init__(self):
        """
        初始化ROS2节点，设置参数、缓存变量和订阅器
        
        初始化步骤：
            1. 调用父类初始化，创建节点名称为'data_recorder_node'
            2. 创建数据存储目录(./data)
            3. 初始化图像消息转换器(CvBridge)
            4. 初始化录制状态机和数据缓存
            5. 初始化传感器数据缓存变量
            6. 创建7个订阅器连接到各传感器话题
        """
        super().__init__('data_recorder_node')
        
        # ========== 文件系统配置 ==========
        # 数据存储目录路径，相对于节点启动位置
        self.data_dir = "./data"
        # 如果目录不存在，递归创建目录及其所有父目录
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # ROS消息转换器：将ROS Image消息格式转换为OpenCV Mat/numpy数组格式
        self.bridge = CvBridge()
        
        # ========== 状态控制 ==========
        # 录制状态机，取值含义：
        #   0: 空闲状态（Idle）或丢弃当前数据
        #   1: 录制中（Recording）- 正在收集数据到缓冲区
        #   2: 保存中（Saving）- 正在写入硬盘
        self.recording_state = 0
        
        # 当前episode的数据缓冲区（Python列表）
        # 每个元素是一个字典，包含单帧的所有传感器数据（图像、深度、关节角、位姿等）
        self.episode_buffer = []
        
        # 互斥锁（线程锁），保护共享数据结构的线程安全访问
        # 用于保护：recording_state、episode_buffer、各传感器缓存变量
        # 防止ROS回调函数和主程序并发访问导致的数据竞争
        self.mutex = threading.Lock()
        
        # ========== 传感器数据缓存 ==========
        # 这些变量存储从各个传感器订阅接收到的"最新"数据
        # ROS回调函数接收到新数据时更新这些变量
        # 当try_record_frame()被触发时，打包这些最新数据到episode_buffer
        
        # 最新的RGB彩色图像（ROS Image消息格式）
        # 格式：BGR8或RGB8编码
        self.last_color = None
        
        # 最新的深度图像（ROS Image消息格式）
        # 格式：16位无符号整数(uint16)，单位为毫米或其他深度单位
        self.last_depth = None
        
        # 最新的相机内参信息（ROS CameraInfo消息）
        # 包含：相机焦距fx/fy、主点cx/cy、畸变系数等
        self.last_camera_info = None
        
        # 最新的关节状态（ROS JointState消息）
        # 包含：position(关节角)、velocity(关节速度)、effort(关节力矩)
        self.last_joint_state = None
        
        # 最新的夹爪状态（ROS JointState消息）
        # 通常包含：position(夹爪开度) 等信息
        self.last_gripper_state = None
        
        # 最新的末端执行器位姿（ROS PoseStamped消息）
        # 包含：position(x,y,z) 和 orientation(四元数qx,qy,qz,qw)
        self.last_ee_pose = None
        
        # ========== 话题订阅器 ==========
        # 创建7个订阅器，分别监听不同的ROS话题
        # 参数含义：create_subscription(消息类型, 话题名称, 回调函数, 队列大小)
        # 队列大小(queue_size=10)：保留最近10条消息以防处理跟不上
        
        # 订阅1：录制控制指令话题
        # 消息格式：std_msgs/Int32，取值为 0(停止)、1(开始)、2(保存)
        self.sub_cmd = self.create_subscription(
            Int32, 
            "/recorder/command", 
            self.cmd_callback, 
            10
        )
        
        # 订阅2：RGB彩色图像话题
        # 消息格式：sensor_msgs/Image，编码为RGB8或BGR8
        self.sub_color = self.create_subscription(
            Image, 
            "/camera/color/image_raw", 
            self.color_callback, 
            10
        )
        
        # 订阅3：深度图像话题
        # 消息格式：sensor_msgs/Image，编码为16UC1(16位无符号整数)
        self.sub_depth = self.create_subscription(
            Image, 
            "/camera/depth/image_raw", 
            self.depth_callback, 
            10
        )
        
        # 订阅4：相机内参话题
        # 消息格式：sensor_msgs/CameraInfo，包含内参矩阵K和畸变系数D
        self.sub_info = self.create_subscription(
            CameraInfo, 
            "/camera/color/camera_info", 
            self.info_callback, 
            10
        )
        
        # 订阅5：机械臂关节状态话题
        # 消息格式：sensor_msgs/JointState，包含关节名称、位置、速度、力矩
        # Dobot 6轴机械臂：包含6个关节的状态
        self.sub_joints = self.create_subscription(
            JointState, 
            "/joint_states", 
            self.joint_callback, 
            10
        )
        
        # 订阅6：夹爪状态话题
        # 消息格式：sensor_msgs/JointState
        self.sub_gripper = self.create_subscription(
            JointState, 
            "/gripper/state", 
            self.gripper_callback, 
            10
        )
        
        # 订阅7：末端执行器位姿话题
        # 消息格式：geometry_msgs/PoseStamped，包含位置和姿态（四元数）
        self.sub_pose = self.create_subscription(
            PoseStamped, 
            "/end_effector_pose", 
            self.pose_callback, 
            10
        )
        
        # 输出初始化完成日志
        self.get_logger().info("Data Recorder Initialized. Press 'O' in controller to start.")

    # =============================================================================
    # 回调函数：处理各个话题的消息
    # 设计原则：回调函数只负责更新"最新数据缓存"，不进行复杂的数据处理
    # =============================================================================
    
    def cmd_callback(self, msg):
        """
        处理录制控制指令的回调函数
        
        功能：根据接收到的指令改变录制状态
        
        参数：
            msg (std_msgs/Int32): 控制指令消息，msg.data 取值为：
                - 0: 停止录制，丢弃当前缓冲数据
                - 1: 开始录制，清空缓冲区并设置为录制状态
                - 2: 停止录制并保存，触发save_dataset()函数
        
        状态转移：
            接收到1 -> recording_state变为1 -> 清空episode_buffer -> 开始采集数据
            接收到2 (在状态1时) -> recording_state变为2 -> 调用save_dataset() -> 回到0
            接收到0 -> recording_state变为0 -> 清空缓冲区 -> 丢弃数据
        
        线程安全：使用互斥锁保护共享状态变量的访问
        """
        new_state = msg.data
        with self.mutex:  # 获取互斥锁，确保原子性操作
            if new_state == 1 and self.recording_state != 1:
                # 状态转移：其他状态 -> 录制状态
                self.get_logger().info(">>> START RECORDING")
                self.episode_buffer = []  # 清空旧数据，为新的录制准备缓冲区
                self.recording_state = 1  # 设置为录制状态
                
            elif new_state == 2 and self.recording_state == 1:
                # 状态转移：录制状态 -> 保存状态 -> 空闲状态
                self.get_logger().info(">>> STOP & SAVING...")
                self.recording_state = 2  # 标记为保存中，防止新数据进入缓冲区
                self.save_dataset()  # 执行保存操作，将缓冲区数据写入HDF5文件
                self.recording_state = 0  # 保存完成，回到空闲状态
                
            elif new_state == 0:
                # 状态转移：任何状态 -> 空闲状态
                if self.recording_state == 1:
                    # 如果之前在录制状态，输出"停止并丢弃"日志
                    self.get_logger().info(">>> STOP & DISCARD")
                self.recording_state = 0  # 设置为空闲状态
                self.episode_buffer = []  # 清空缓冲区中的所有数据

    def color_callback(self, msg):
        """
        处理RGB彩色图像消息的回调函数
        
        功能：
            1. 更新最新接收到的RGB图像
            2. 以RGB图像的到达作为数据同步的触发信号
            3. 在图像到达时刻，打包所有其他传感器的最新数据
        
        参数：
            msg (sensor_msgs/Image): RGB彩色图像消息，包含：
                - header.stamp: 消息时间戳
                - encoding: 图像编码格式(通常为'rgb8'或'bgr8')
                - data: 图像数据(一维字节数组)
                - height, width: 图像分辨率
        
        设计原理："以图像为基准的软同步"
            - RGB相机通常工作在固定帧率(如30Hz)，提供规则的时间间隔
            - 使用图像到达作为触发信号，此时打包其他传感器的最新数据
            - 虽然各传感器的到达时间不完全同步，但误差在一个时间步内
            - 这种策略简单有效，适合模仿学习数据采集
        
        执行步骤：
            1. 保存最新的彩色图像消息
            2. 调用try_record_frame()进行数据打包
        """
        self.last_color = msg
        # 以RGB图像的到达作为触发信号，执行一次对齐和记录
        self.try_record_frame()

    def depth_callback(self, msg):
        """
        处理深度图像消息的回调函数
        
        功能：更新最新接收到的深度图像
        
        参数：
            msg (sensor_msgs/Image): 深度图像消息，包含：
                - encoding: 通常为'16UC1'(16位无符号整数)
                - 数值含义：像素值表示深度距离(单位毫米或由相机决定)
        
        说明：
            - 深度图像不作为同步触发信号
            - 在try_record_frame()中使用最新的深度图像和彩色图像配对
        """
        self.last_depth = msg
    
    def info_callback(self, msg):
        """
        处理相机内参消息的回调函数
        
        功能：更新相机的内参信息
        
        参数：
            msg (sensor_msgs/CameraInfo): 相机内参消息，包含：
                - K: 相机内参矩阵(3x3, 行优先存储)，格式：
                    [fx  0 cx]
                    [ 0 fy cy]
                    [ 0  0  1]
                    其中：fx,fy为焦距(像素单位)，cx,cy为主点坐标
                - D: 畸变系数(5个元素：k1,k2,p1,p2,k3)
                - height, width: 图像分辨率
        
        说明：
            - 相机内参通常在标定后不变化，只需存储一次
            - 在这里每次都更新，确保使用最新的参数
            - 在save_dataset()中只将第一帧的内参写入文件
        """
        self.last_camera_info = msg

    def joint_callback(self, msg):
        """
        处理机械臂关节状态消息的回调函数
        
        功能：更新机械臂的关节状态信息
        
        参数：
            msg (sensor_msgs/JointState): 关节状态消息，包含：
                - name: 关节名称列表，如['joint_1', 'joint_2', ..., 'joint_6']
                - position: 关节角度列表(浮点数，单位为弧度)
                - velocity: 关节速度列表(浮点数，单位为弧度/秒)
                - effort: 关节力矩列表(浮点数，单位为牛顿米)
        
        说明：
            - Dobot 6轴机械臂有6个关节，position数组长度为6
            - 在try_record_frame()中只使用position字段
        """
        self.last_joint_state = msg

    def gripper_callback(self, msg):
        """
        处理夹爪状态消息的回调函数
        
        功能：更新夹爪的状态信息
        
        参数：
            msg (sensor_msgs/JointState): 夹爪状态消息，包含：
                - position: 夹爪位置/开度值(通常为单个浮点数或两个浮点数)
                - 其他字段同关节状态
        
        说明：
            - 夹爪通常只有1个DOF(自由度)，position数组长度为1
            - 在save_dataset()中保存为shape (N, 1)的数组
        """
        self.last_gripper_state = msg

    def pose_callback(self, msg):
        """
        处理末端执行器位姿消息的回调函数
        
        功能：更新机械臂末端执行器的位置和姿态
        
        参数：
            msg (geometry_msgs/PoseStamped): 位姿消息，包含：
                - header.stamp: 时间戳
                - pose.position: 末端位置
                    - x, y, z: 笛卡尔坐标(通常单位为米)
                - pose.orientation: 末端姿态(四元数表示)
                    - x, y, z, w: 四元数分量，满足 x^2+y^2+z^2+w^2=1
        
        说明：
            - Dobot标称通常只提供位置(position)，姿态可能需要自己计算或假设默认值
            - 在try_record_frame()中将位置和姿态合并为7维向量
        """
        self.last_ee_pose = msg

    # =============================================================================
    # 核心逻辑：数据对齐与记录
    # =============================================================================
    
    def try_record_frame(self):
        """
        尝试打包当前所有传感器数据成一帧，加入到episode_buffer
        
        功能逻辑：
            1. 检查当前是否处于录制状态(recording_state == 1)
            2. 检查必要的传感器数据是否都已接收
            3. 将各个传感器的最新数据转换为标准格式
            4. 打包成一个帧字典，添加到episode_buffer
        
        设计原理：
            - 以RGB图像到达为触发信号，被color_callback()调用
            - 在被触发时刻，同时打包其他传感器的最新数据
            - 如果某些传感器数据缺失，则跳过本帧(不保存)
            - 这种设计确保每一帧都有完整的多模态数据
        
        调用时机：每当接收到新的RGB图像时被调用
        
        线程安全：使用互斥锁保护共享变量的访问
        
        异常处理：捕获消息转换过程中的异常，防止程序崩溃
        """
        # 首先检查是否处于录制状态
        if self.recording_state != 1:
            return

        with self.mutex:  # 获取互斥锁
            # ========== 数据完整性检查 ==========
            # 检查必要的传感器数据是否都已接收到至少一次
            # 如果缺少任何必要数据，则无法形成完整的帧，跳过本次打包
            if self.last_color is None or self.last_joint_state is None or \
               self.last_gripper_state is None or self.last_ee_pose is None:
                # 数据不完整，跳过本帧
                # (可选：可以取消下行注释进行调试日志)
                # self.get_logger().warn("Waiting for all sensors...", throttle_duration_sec=1.0)
                return

            # ========== 数据打包 ==========
            try:
                # --------- 图像数据转换 ---------
                # 使用CvBridge将ROS Image消息转换为OpenCV格式(numpy数组)
                cv_image = self.bridge.imgmsg_to_cv2(
                    self.last_color, 
                    desired_encoding='bgr8'  # 指定输出格式为BGR8(OpenCV默认格式)
                )
                
                # 深度图像转换(如果存在)
                cv_depth = None
                if self.last_depth:
                    cv_depth = self.bridge.imgmsg_to_cv2(
                        self.last_depth, 
                        desired_encoding='passthrough'  # 保持原始数据格式(通常16位)
                    )

                # --------- 组织帧数据 ---------
                # 创建一个字典，包含单帧的所有传感器数据
                frame_data = {
                    # 时间戳：从消息header中提取，转换为秒的浮点数
                    # 计算：秒数 + 纳秒数(转换为秒)
                    'timestamp': self.last_color.header.stamp.sec + self.last_color.header.stamp.nanosec * 1e-9,
                    
                    # RGB彩色图像：形状为(H, W, 3)，BGR格式，dtype为uint8
                    'image': cv_image,
                    
                    # 深度图像：形状为(H, W)，单位为毫米(取决于相机)，dtype为uint16
                    'depth': cv_depth,
                    
                    # 关节角度：6维向量，对应Dobot 6轴机械臂的6个关节
                    # 单位：弧度，取值范围由各关节的机械限制决定
                    # dtype：float32，用于节省存储空间
                    'qpos': np.array(self.last_joint_state.position, dtype=np.float32),
                    
                    # 夹爪位置：1维向量，表示夹爪的开度或位置
                    # dtype：float32
                    'gripper_pos': np.array(self.last_gripper_state.position, dtype=np.float32),
                    
                    # 末端执行器位姿：7维向量，包含位置和姿态
                    # 前3维：[x, y, z] 笛卡尔坐标(单位：米)
                    # 后4维：[qx, qy, qz, qw] 四元数(单位化)
                    # 总长：7维，dtype：float32
                    'ee_pose': np.array([
                        self.last_ee_pose.pose.position.x,      # 末端X坐标
                        self.last_ee_pose.pose.position.y,      # 末端Y坐标
                        self.last_ee_pose.pose.position.z,      # 末端Z坐标
                        self.last_ee_pose.pose.orientation.x,   # 四元数X分量
                        self.last_ee_pose.pose.orientation.y,   # 四元数Y分量
                        self.last_ee_pose.pose.orientation.z,   # 四元数Z分量
                        self.last_ee_pose.pose.orientation.w    # 四元数W分量(实部)
                    ], dtype=np.float32)
                }
                
                # --------- 保存相机内参(可选) ---------
                # 相机内参矩阵是3x3的矩阵，用于图像坐标和3D坐标的转换
                # CameraInfo.K是9个元素的列表(行优先)，reshape为3x3矩阵
                if self.last_camera_info:
                    frame_data['camera_intrinsics'] = np.array(
                        self.last_camera_info.k, 
                        dtype=np.float32
                    ).reshape(3, 3)

                # ========== 缓冲数据 ==========
                # 将打包好的单帧数据添加到episode_buffer列表
                self.episode_buffer.append(frame_data)
                
                # ========== 进度输出 ==========
                # 每30帧输出一次日志，用于监控录制进度
                if len(self.episode_buffer) % 30 == 0:
                    print(f"Recording... {len(self.episode_buffer)} frames")
                    
            except Exception as e:
                # 如果在数据转换或打包过程中发生异常(如消息格式错误)
                # 捕获异常并输出错误日志，避免节点崩溃
                self.get_logger().error(f"Error packing frame: {e}")

    # =============================================================================
    # 保存逻辑
    # =============================================================================
    
    def get_next_episode_index(self):
        """
        查询./data目录，确定下一个episode的序号
        
        功能：通过扫描已存在的episode_X.hdf5文件，找出最大的序号，返回下一个序号
        
        返回值：
            int: 下一个可用的episode序号(从0开始)
        
        工作流程：
            1. 列出./data目录中所有文件
            2. 筛选出符合'episode_*.hdf5'格式的文件
            3. 提取文件名中的序号数字
            4. 返回最大序号+1(如果没有既有文件，返回0)
        
        示例：
            - 如果目录中有 episode_0.hdf5, episode_1.hdf5，返回2
            - 如果目录为空，返回0
            - 如果存在 episode_5.hdf5，即使中间没有0-4，也返回6
        """
        # 列出./data目录中所有以'episode_'开头且以'.hdf5'结尾的文件
        existing_files = [
            f for f in os.listdir(self.data_dir) 
            if f.startswith('episode_') and f.endswith('.hdf5')
        ]
        
        # 如果没有既有文件，返回0作为首个episode
        if not existing_files:
            return 0
        
        # 从文件名中提取序号
        indices = []
        for f in existing_files:
            try:
                # 文件名格式：episode_X.hdf5
                # 分割：['episode', 'X.hdf5'] -> ['X', 'hdf5'] -> 'X'
                idx = int(f.split('_')[1].split('.')[0])
                indices.append(idx)
            except:
                # 如果文件名格式异常(无法解析)，跳过该文件
                pass
        
        # 返回最大序号+1，或如果列表为空则返回0
        return max(indices) + 1 if indices else 0

    def save_dataset(self):
        """
        将episode_buffer中的所有帧数据保存为HDF5文件
        
        功能概述：
            1. 检查缓冲区是否有数据
            2. 生成输出文件路径(自动递增的episode序号)
            3. 创建HDF5文件并定义数据集结构
            4. 迭代写入每一帧的数据
            5. 保存元数据和相机内参
        
        输出文件格式：HDF5 (.hdf5)
            - 路径：./data/episode_X.hdf5
            - 结构：(详见模块头注释的"文件结构"部分)
        
        异常处理：
            - 如果缓冲区为空，输出警告并返回
            - 如果写入过程出错，输出错误日志
        
        执行流程：
            1. 获取下一个可用的episode序号
            2. 构建文件完整路径
            3. 打开HDF5文件进行写操作
            4. 创建各数据集(预分配空间，设置分块存储)
            5. 逐帧写入数据
            6. 保存元数据
            7. 关闭文件
        """
        # 检查缓冲区是否为空
        if not self.episode_buffer:
            self.get_logger().warn("Buffer empty, nothing to save.")
            return

        # 获取下一个可用的episode序号
        idx = self.get_next_episode_index()
        
        # 构建完整的文件路径：./data/episode_0.hdf5, episode_1.hdf5, ...
        file_path = os.path.join(self.data_dir, f"episode_{idx}.hdf5")
        
        # 输出保存开始的日志
        self.get_logger().info(f"Saving {len(self.episode_buffer)} frames to {file_path}...")
        
        try:
            # ========== 打开HDF5文件 ==========
            # 'w' 模式：写入(如果文件存在则覆盖)
            with h5py.File(file_path, 'w') as f:
                
                # ========== 预分配数据集 ==========
                # 这是HDF5最佳实践：事先分配好所需的空间，而不是动态增长
                # 预分配可以提高写入速度和文件性能
                
                # 获取总帧数
                data_len = len(self.episode_buffer)
                
                # --------- RGB图像数据集 ---------
                # 获取第一帧图像的尺寸
                img_sample = self.episode_buffer[0]['image']
                
                # 创建RGB图像数据集：shape (N, H, W, 3)
                # N: 帧数
                # H, W: 图像高度和宽度
                # 3: RGB三通道
                # dtype='uint8': 每个像素值为0-255的无符号8位整数
                # chunks: 分块存储的块大小，设置为(1, H, W, 3)表示按帧存储
                #         可加快逐帧访问的速度，代价是文件体积略增
                dset_img = f.create_dataset(
                    'observations/images/color', 
                    (data_len, img_sample.shape[0], img_sample.shape[1], 3), 
                    dtype='uint8', 
                    chunks=(1, img_sample.shape[0], img_sample.shape[1], 3)
                )
                
                # --------- 深度图像数据集(可选) ---------
                # 如果第一帧包含深度图像，则创建深度图像数据集
                if self.episode_buffer[0]['depth'] is not None:
                    depth_sample = self.episode_buffer[0]['depth']
                    
                    # 创建深度图像数据集：shape (N, H, W)
                    # N: 帧数
                    # H, W: 图像高度和宽度
                    # dtype='uint16': 16位无符号整数，能表示0-65535的深度值(单位毫米)
                    # chunks: 按帧分块存储
                    dset_depth = f.create_dataset(
                        'observations/images/depth', 
                        (data_len, depth_sample.shape[0], depth_sample.shape[1]), 
                        dtype='uint16', 
                        chunks=(1, depth_sample.shape[0], depth_sample.shape[1])
                    )

                # --------- 关节角度数据集 ---------
                # Dobot 6轴机械臂：6个关节，因此维度为6
                # shape (N, 6): N帧数据，每帧6个关节角度
                # dtype='float32': 32位浮点数，单位为弧度
                dset_qpos = f.create_dataset(
                    'observations/qpos', 
                    (data_len, 6), 
                    dtype='float32'
                )
                
                # --------- 夹爪位置数据集 ---------
                # shape (N, 1): N帧数据，每帧1个夹爪位置值
                # dtype='float32': 32位浮点数
                dset_gpos = f.create_dataset(
                    'observations/gripper_pos', 
                    (data_len, 1), 
                    dtype='float32'
                )
                
                # --------- 末端位姿数据集 ---------
                # shape (N, 7): N帧数据，每帧包含位置(3)+四元数(4)=7维
                # dtype='float32': 32位浮点数
                # 数据布局：[x, y, z, qx, qy, qz, qw]
                dset_pose = f.create_dataset(
                    'observations/ee_pose', 
                    (data_len, 7), 
                    dtype='float32'
                )
                
                # --------- 时间戳数据集 ---------
                # shape (N,): 1维数组，N个时间戳值
                # dtype='float64': 64位浮点数，精度较高(用于时间戳)
                dset_time = f.create_dataset(
                    'timestamp', 
                    (data_len,), 
                    dtype='float64'
                )

                # ========== 逐帧写入数据 ==========
                # 迭代遍历episode_buffer中的每一帧
                for i, frame in enumerate(self.episode_buffer):
                    # 写入RGB图像
                    dset_img[i] = frame['image']
                    
                    # 写入深度图像(如果存在)
                    if frame['depth'] is not None:
                        dset_depth[i] = frame['depth']
                    
                    # 写入关节角度(确保维度匹配，取前6维)
                    if len(frame['qpos']) >= 6:
                        dset_qpos[i] = frame['qpos'][:6]
                    
                    # 写入夹爪位置
                    dset_gpos[i] = frame['gripper_pos']
                    
                    # 写入末端位姿
                    dset_pose[i] = frame['ee_pose']
                    
                    # 写入时间戳
                    dset_time[i] = frame['timestamp']
                
                # ========== 保存相机内参 ==========
                # 相机内参矩阵是固定的，只需要保存一次(从第一帧获取)
                if 'camera_intrinsics' in self.episode_buffer[0]:
                    f.create_dataset(
                        'camera/intrinsics', 
                        data=self.episode_buffer[0]['camera_intrinsics']
                    )
                
                # ========== 保存元数据 ==========
                # 在HDF5文件的根属性中存储元数据
                
                # sim标志：标识数据是否来自仿真(False=真实机器人数据)
                f.attrs['sim'] = False
                
                # 总帧数：方便后续读取时了解数据量
                f.attrs['total_frames'] = data_len
                
            # ========== 保存成功 ==========
            self.get_logger().info(f"Save successfully! Saved to {file_path}")
            
        except Exception as e:
            # ========== 异常处理 ==========
            # 如果保存过程中出现异常(如磁盘满、权限错误等)
            # 输出错误日志并返回，避免程序崩溃
            self.get_logger().error(f"Failed to save dataset: {e}")

def main(args=None):
    """
    ROS2节点的主函数
    
    功能：
        1. 初始化ROS2系统
        2. 创建DataRecorder节点实例
        3. 运行节点的主事件循环
        4. 处理异常退出(如Ctrl+C)
        5. 清理资源
    
    参数：
        args: ROS2命令行参数(默认None)
    
    执行流程：
        1. rclpy.init(): 初始化ROS2 Python接口
        2. 创建DataRecorder节点
        3. rclpy.spin(): 进入事件循环，持续处理各订阅话题的消息
        4. 捕获KeyboardInterrupt异常(Ctrl+C)
        5. 清理资源：销毁节点、关闭ROS2
    """
    # 初始化ROS2 Python接口
    # 必须在使用任何ROS2功能前调用
    rclpy.init(args=args)
    
    # 创建DataRecorder节点实例
    node = DataRecorder()
    
    try:
        # 运行节点的主事件循环
        # spin()会持续阻塞，直到节点被关闭
        # 期间持续接收订阅话题的消息，调用相应的回调函数
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 捕获Ctrl+C信号，允许优雅的中断
        pass
    finally:
        # 销毁节点资源
        node.destroy_node()
        
        # 关闭ROS2系统
        rclpy.shutdown()

if __name__ == "__main__":
    # Python脚本直接运行时的入口点
    main()