#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dobot机械臂反馈数据接收节点
功能说明:
    1. 通过Socket连接到Dobot机械臂的反馈端口(30004)
    2. 实时接收机械臂的状态数据(关节角度、工具坐标等)
    3. 将数据解析后发布到ROS2话题供其他节点使用

发布的话题:
    - /dobot_msgs_v3/msg/ToolVectorActual: 工具坐标系位置(x,y,z,rx,ry,rz)
    - /joint_states_robot: 机械臂关节状态(6个关节角度)
"""

import rclpy                                     # ROS2 Python接口库
from rclpy.node import Node                      # ROS2节点基类
from dobot_msgs_v3.msg import ToolVectorActual   # Dobot工具坐标消息类型
from sensor_msgs.msg import JointState           # 关节状态消息类型(ROS标准消息)
import socket                                    # TCP/IP通信库
import numpy as np                               # 数值计算库,用于解析二进制数据
import os                                        # 操作系统接口,用于读取环境变量
import time                                      # 时间处理库

# =============================================================================
# Dobot机械臂反馈数据结构定义
# =============================================================================
# 该数据类型定义了从Dobot机械臂接收的完整反馈数据包的二进制格式
# 数据包大小: 1440字节
# 数据来源: TCP端口30004/30005
#
# 主要包含以下信息:
#   - 机器人状态(模式、时间戳、速度缩放等)
#   - 关节数据(目标/实际位置、速度、加速度、电流等)
#   - 工具坐标系数据(TCP位置、速度、力矩)
#   - 电机温度、关节模式
#   - 控制比例(速度、加速度、加加速度比例)
#   - 使能、拖拽、运行、错误等状态标志
#   - 六维力传感器数据
#   - 四元数姿态信息
# =============================================================================
MyType = np.dtype([
    # -------------------------------------------------------------------------
    # 基础信息和系统状态 (64位整数和浮点数)
    # -------------------------------------------------------------------------
    ('len', np.int64,),                           # 数据包长度(字节数)
    ('digital_input_bits', np.uint64,),           # 数字输入端口状态位(位掩码)
    ('digital_output_bits', np.uint64,),          # 数字输出端口状态位(位掩码)
    ('robot_mode', np.uint64,),                   # 机器人模式(待机/运行/错误等)
    ('time_stamp', np.uint64,),                   # 时间戳(毫秒)
    ('time_stamp_reserve_bit', np.uint64,),       # 时间戳保留位
    ('test_value', np.uint64,),                   # 测试魔数值(用于数据验证,期望值:0x123456789abcdef)

    # -------------------------------------------------------------------------
    # 系统电气和动力学参数
    # -------------------------------------------------------------------------
    ('test_value_keep_bit', np.float64,),         # 测试值保留位
    ('speed_scaling', np.float64,),               # 速度缩放比例(0.0-1.0)
    ('linear_momentum_norm', np.float64,),        # 线性动量范数
    ('v_main', np.float64,),                      # 主电源电压(V)
    ('v_robot', np.float64,),                     # 机器人系统电压(V)
    ('i_robot', np.float64,),                     # 机器人系统总电流(A)
    ('i_robot_keep_bit1', np.float64,),           # 机器人电流保留位1
    ('i_robot_keep_bit2', np.float64,),           # 机器人电流保留位2

    # -------------------------------------------------------------------------
    # 工具端传感器数据 (3维向量)
    # -------------------------------------------------------------------------
    ('tool_accelerometer_values', np.float64, (3,)),  # 工具端加速度计值[x,y,z] (m/s²)
    ('elbow_position', np.float64, (3,)),         # 肘部位置[x,y,z] (mm)
    ('elbow_velocity', np.float64, (3,)),         # 肘部速度[vx,vy,vz] (mm/s)

    # -------------------------------------------------------------------------
    # 关节目标值 (6个关节,J1-J6)
    # -------------------------------------------------------------------------
    ('q_target', np.float64, (6,)),               # 目标关节角度[J1-J6] (度)
    ('qd_target', np.float64, (6,)),              # 目标关节角速度[J1-J6] (度/s)
    ('qdd_target', np.float64, (6,)),             # 目标关节角加速度[J1-J6] (度/s²)
    ('i_target', np.float64, (6,)),               # 目标关节电流[J1-J6] (A)
    ('m_target', np.float64, (6,)),               # 目标关节力矩[J1-J6] (N·m)

    # -------------------------------------------------------------------------
    # 关节实际值 (6个关节,J1-J6)
    # -------------------------------------------------------------------------
    ('q_actual', np.float64, (6,)),               # 实际关节角度[J1-J6] (度) ★核心数据★
    ('qd_actual', np.float64, (6,)),              # 实际关节角速度[J1-J6] (度/s)
    ('i_actual', np.float64, (6,)),               # 实际关节电流[J1-J6] (A)

    # -------------------------------------------------------------------------
    # TCP(工具中心点)数据 (6维: x,y,z,rx,ry,rz)
    # -------------------------------------------------------------------------
    ('actual_TCP_force', np.float64, (6,)),       # 实际TCP力/力矩[Fx,Fy,Fz,Mx,My,Mz] (N, N·m)
    ('tool_vector_actual', np.float64, (6,)),     # 实际工具坐标[x,y,z,rx,ry,rz] (mm, 度) ★核心数据★
    ('TCP_speed_actual', np.float64, (6,)),       # 实际TCP速度[vx,vy,vz,ωx,ωy,ωz] (mm/s, 度/s)
    ('TCP_force', np.float64, (6,)),              # TCP力/力矩[Fx,Fy,Fz,Mx,My,Mz] (N, N·m)
    ('Tool_vector_target', np.float64, (6,)),     # 目标工具坐标[x,y,z,rx,ry,rz] (mm, 度)
    ('TCP_speed_target', np.float64, (6,)),       # 目标TCP速度[vx,vy,vz,ωx,ωy,ωz] (mm/s, 度/s)

    # -------------------------------------------------------------------------
    # 关节状态监控
    # -------------------------------------------------------------------------
    ('motor_temperatures', np.float64, (6,)),     # 电机温度[J1-J6] (℃)
    ('joint_modes', np.float64, (6,)),            # 关节模式[J1-J6] (空闲/运行/错误等)
    ('v_actual', np.float64, (6,)),               # 实际关节电压[J1-J6] (V)

    # -------------------------------------------------------------------------
    # 末端执行器和坐标系配置 (字节型)
    # -------------------------------------------------------------------------
    ('hand_type', np.byte, (4,)),                 # 手爪类型标识(4字节)
    ('user', np.byte,),                           # 当前用户坐标系编号(0-9)
    ('tool', np.byte,),                           # 当前工具坐标系编号(0-9)

    # -------------------------------------------------------------------------
    # 运动控制标志
    # -------------------------------------------------------------------------
    ('run_queued_cmd', np.byte,),                 # 运行队列命令标志(0=否, 1=是)
    ('pause_cmd_flag', np.byte,),                 # 暂停命令标志(0=运行, 1=暂停)

    # -------------------------------------------------------------------------
    # 运动参数比例 (0-100的百分比值)
    # -------------------------------------------------------------------------
    ('velocity_ratio', np.int8,),                 # 全局速度比例(%)
    ('acceleration_ratio', np.int8,),             # 全局加速度比例(%)
    ('jerk_ratio', np.int8,),                     # 全局加加速度比例(%)
    ('xyz_velocity_ratio', np.int8,),             # 笛卡尔空间速度比例(%)
    ('r_velocity_ratio', np.int8,),               # 旋转速度比例(%)
    ('xyz_acceleration_ratio', np.int8,),         # 笛卡尔空间加速度比例(%)
    ('r_acceleration_ratio', np.int8,),           # 旋转加速度比例(%)
    ('xyz_jerk_ratio', np.int8,),                 # 笛卡尔空间加加速度比例(%)
    ('r_jerk_ratio', np.int8,),                   # 旋转加加速度比例(%)

    # -------------------------------------------------------------------------
    # 机器人状态标志 (0=否, 1=是 或枚举值)
    # -------------------------------------------------------------------------
    ('brake_status', np.int8,),                   # 刹车状态(0=释放, 1=锁定)
    ('enable_status', np.int8,),                  # 使能状态(0=禁用, 1=使能)
    ('drag_status', np.int8,),                    # 拖拽示教状态(0=关闭, 1=开启)
    ('running_status', np.int8,),                 # 运行状态(0=停止, 1=运行, 2=暂停等)
    ('error_status', np.int8,),                   # 错误状态(0=无错误, >0=错误代码)
    ('jog_status', np.int8,),                     # 点动状态(0=关闭, 1=点动中)
    ('robot_type', np.int8,),                     # 机器人型号(如CR3/CR5/CR10等)

    # -------------------------------------------------------------------------
    # 示教器按钮信号 (0=未按下, 1=按下)
    # -------------------------------------------------------------------------
    ('drag_button_signal', np.int8,),             # 拖拽按钮信号
    ('enable_button_signal', np.int8,),           # 使能按钮信号
    ('record_button_signal', np.int8,),           # 记录按钮信号
    ('reappear_button_signal', np.int8,),         # 重现/回放按钮信号
    ('jaw_button_signal', np.int8,),              # 夹爪按钮信号

    # -------------------------------------------------------------------------
    # 六维力传感器
    # -------------------------------------------------------------------------
    ('six_force_online', np.int8,),               # 六维力传感器在线状态(0=离线, 1=在线)

    # -------------------------------------------------------------------------
    # 保留字段
    # -------------------------------------------------------------------------
    ('reserve2', np.int8, (82,)),                 # 保留字段2(82字节,用于未来扩展)

    # -------------------------------------------------------------------------
    # 动力学和负载信息
    # -------------------------------------------------------------------------
    ('m_actual', np.float64, (6,)),               # 实际关节力矩[J1-J6] (N·m)
    ('load', np.float64,),                        # 末端负载质量(kg)
    ('center_x', np.float64,),                    # 负载重心X坐标(mm,相对于法兰中心)
    ('center_y', np.float64,),                    # 负载重心Y坐标(mm,相对于法兰中心)
    ('center_z', np.float64,),                    # 负载重心Z坐标(mm,相对于法兰中心)

    # -------------------------------------------------------------------------
    # 坐标系定义 (用户坐标系和工具坐标系的位姿)
    # -------------------------------------------------------------------------
    ('user1', np.float64, (6,)),                  # 用户坐标系1的位姿[x,y,z,rx,ry,rz] (mm, 度)
    ('Tool1', np.float64, (6,)),                  # 工具坐标系1的位姿[x,y,z,rx,ry,rz] (mm, 度)

    # -------------------------------------------------------------------------
    # 轨迹和力控信息
    # -------------------------------------------------------------------------
    ('trace_index', np.float64,),                 # 轨迹索引(当前执行的轨迹点序号)
    ('six_force_value', np.float64, (6,)),        # 六维力传感器值[Fx,Fy,Fz,Mx,My,Mz] (N, N·m)

    # -------------------------------------------------------------------------
    # 姿态四元数 (用于精确表示末端姿态,避免万向节锁)
    # -------------------------------------------------------------------------
    ('target_quaternion', np.float64, (4,)),      # 目标姿态四元数[w,x,y,z]
    ('actual_quaternion', np.float64, (4,)),      # 实际姿态四元数[w,x,y,z]

    # -------------------------------------------------------------------------
    # 保留字段
    # -------------------------------------------------------------------------
    ('reserve3', np.int8, (24,))                  # 保留字段3(24字节,用于未来扩展)
])




# =============================================================================
# 反馈数据接收类
# =============================================================================
class fankuis():
    """
    Dobot机械臂反馈数据接收器

    功能:
        通过Socket连接到Dobot机械臂的反馈端口,实时接收机械臂状态数据

    参数:
        ip (str): 机械臂的IP地址
        port (int): 反馈端口号,应为30004或30005

    属性:
        socket_feedback: TCP socket连接对象
    """
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket_feedback = 0

        # 只接受30004或30005端口的连接
        if self.port == 30005 or self.port == 30004:
                self.socket_feedback = socket.socket()
                self.socket_feedback.settimeout(1)  # 设置1秒超时
                self.socket_feedback.connect((self.ip, self.port))
        else:
            print("Connect to feedback server need use port 30003 !")

    def feed(self):
        """
        接收并解析机械臂反馈数据

        返回:
            list: 成功时返回[tool_vector, joint_angles]
                  - tool_vector: 工具坐标系位置 [x,y,z,rx,ry,rz] (6个浮点数)
                  - joint_angles: 关节角度 [j1,j2,j3,j4,j5,j6] (6个浮点数,单位:度)
            list: 失败时返回["NG1"]

        数据验证:
            通过test_value字段验证数据有效性(期望值: 0x123456789abcdef)
        """
        try:
            self.socket_feedback.setblocking(True)  # 设置为阻塞模式
            self.all = self.socket_feedback.recv(10240)  # 接收最多10240字节
            data = self.all[0:1440]  # 提取前1440字节(完整的反馈数据包)

            # 使用NumPy解析二进制数据
            a = np.frombuffer(data, dtype=MyType)

            # 验证数据有效性(检查魔数)
            if hex((a['test_value'][0])) == '0x123456789abcdef':
                tool_v = a['tool_vector_actual'][0]  # 提取工具坐标
                tool_j = a['q_actual'][0]            # 提取关节角度
            return [tool_v,tool_j]
        except:
            return ["NG1"]  # 异常时返回错误标志


# =============================================================================
# ROS2发布者节点
# =============================================================================
class PublisherNode(Node):
    """
    Dobot机械臂反馈数据发布节点

    功能:
        1. 从环境变量读取机械臂IP地址
        2. 连接到机械臂反馈端口(30004)
        3. 以100Hz频率(0.01秒周期)获取机械臂状态数据
        4. 发布工具坐标和关节角度到ROS2话题

    发布话题:
        - /dobot_msgs_v3/msg/ToolVectorActual: 工具坐标系位置(x,y,z,rx,ry,rz)
        - /joint_states_robot: 关节状态(关节角度,已转换为弧度制)

    环境变量:
        IP_address: 机械臂的IP地址(例如: 192.168.5.1)
    """

    def __init__(self, name):
        super().__init__(name)
        # 注意: 也可以使用ROS参数代替环境变量
        # self.declare_parameter('IP', '192.168.5.1')  # 默认值
        # self.IP = self.get_parameter('IP').get_parameter_value().string_value

        # 从环境变量获取机械臂IP地址
        self.IP = str(os.getenv("IP_address"))

        # 建立与机械臂的连接
        self.connect()

        # 创建发布者: 工具坐标
        self.pub = self.create_publisher(ToolVectorActual, "dobot_msgs_v3/msg/ToolVectorActual", 10)

        # 创建发布者: 关节状态
        self.pub2 = self.create_publisher(JointState, "joint_states_robot", 10)

        # 创建定时器: 每0.01秒(100Hz)调用一次回调函数
        self.timer = self.create_timer(0.01, self.timer_callback)

    def connect(self):
        """
        连接到机械臂反馈端口

        连接信息:
            - IP地址: 从环境变量IP_address读取
            - 端口: 30004 (实时反馈端口)
        """
        try:
           self.get_logger().info("connection:30004")
           self.feed_v = fankuis(self.IP, 30004)
           self.get_logger().info("connection succeeded:30004")
        except:
            self.get_logger().info("Connection failed!!!")

    def timer_callback(self):
        """
        定时回调函数 - 获取并发布机械臂状态数据

        执行频率: 100Hz (每0.01秒)

        数据流程:
            1. 从机械臂获取反馈数据 (工具坐标 + 关节角度)
            2. 如果数据有效:
               - 发布工具坐标到ToolVectorActual话题
               - 将关节角度从角度制转换为弧度制
               - 发布关节状态到joint_states_robot话题
            3. 如果数据无效(异常),跳过本次发布
        """
        msg = ToolVectorActual()
        actual = self.feed_v.feed()  # 获取反馈数据
        msg2 = JointState()

        # self.get_logger().info(str(actual))

        # 检查数据是否有效(有效数据长度为2: [tool_vector, joint_angles])
        if len(actual) != 1:
           # 设置关节状态消息的关节名称
           msg2.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

           # 添加时间戳
           msg2.header.stamp = self.get_clock().now().to_msg()
           msg2.header.frame_id = 'joint_states'

           # 获取关节角度(单位:度)
           q_target = actual[1]

           # 将关节角度从度转换为弧度
           joint_a = []
           for ii in q_target:
               joint_a.append(float(ii * 3.14159 / 180))

           print(joint_a)
           msg2.position = joint_a

           # 填充工具坐标消息 (单位: 毫米和度)
           msg.x = actual[0][0]   # X坐标
           msg.y = actual[0][1]   # Y坐标
           msg.z = actual[0][2]   # Z坐标
           msg.rx = actual[0][3]  # 绕X轴旋转
           msg.ry = actual[0][4]  # 绕Y轴旋转
           msg.rz = actual[0][5]  # 绕Z轴旋转

           # 发布消息
           self.pub.publish(msg)
           self.pub2.publish(msg2)


# =============================================================================
# 主函数
# =============================================================================
def main(args=None):
    """
    ROS2节点主入口

    执行流程:
        1. 初始化ROS2
        2. 创建发布者节点
        3. 进入循环,持续接收和发布数据
        4. 退出时清理资源

    参数:
        args: 命令行参数(可选)
    """
    rclpy.init(args=args)                       # 初始化ROS2通信
    node = PublisherNode("dobot_feedback")      # 创建节点实例
    rclpy.spin(node)                            # 进入循环,执行定时回调
    node.destroy_node()                         # 销毁节点
    rclpy.shutdown()                            # 关闭ROS2
