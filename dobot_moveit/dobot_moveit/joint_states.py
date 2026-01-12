#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
关节状态中继节点 - 机械臂与夹爪数据处理
==============================================================================

版本更新说明:
    新版本增加了夹爪运动插值补偿功能，通过模拟物理延迟来实现与实际硬件同步

功能概述:
    1. 机械臂关节状态中继: 接收并转发机械臂6个关节的位置数据
    2. 工具坐标系处理: 接收工具向量数据并转换为ROS标准消息
    3. 夹爪状态管理:
       - 从Modbus寄存器读取夹爪初始位置
       - 接收夹爪控制指令
       - 实现运动插值补偿（模拟延迟+速度限制）
       - 发布平滑的夹爪状态数据

订阅话题:
    - /joint_states_robot: 机械臂关节状态
    - /dobot_msgs_v3/msg/ToolVectorActual: 工具坐标向量
    - /gripper/command_update: 夹爪位置指令(带时间戳)

发布话题:
    - /joint_states: 机械臂关节状态(清理后)
    - /end_effector_pose: 末端执行器位姿(PoseStamped)
    - /end_effector_data: 工具向量数据(中继)
    - /gripper/state: 夹爪状态(模拟物理运动)

服务客户端:
    - /dobot_bringup_v3/srv/GetHoldRegs: 读取Modbus寄存器

运动补偿机制:
    1. 反应延迟: 模拟Modbus通信到电机启动的固有延迟
    2. 速度限制: 限制夹爪运动速度，模拟真实物理运动
    3. 线性插值: 平滑过渡到目标位置
==============================================================================
"""

# =============================================================================
# 导入必要的库
# =============================================================================
import rclpy                                           # ROS2 Python接口库
from rclpy.node import Node                            # ROS2节点基类
from rclpy.callback_groups import ReentrantCallbackGroup  # 可重入回调组(支持并发)
from rclpy.executors import MultiThreadedExecutor      # 多线程执行器
from rclpy.time import Time                            # ROS时间类

# ROS标准消息类型
from sensor_msgs.msg import JointState                 # 关节状态消息
from geometry_msgs.msg import PoseStamped, PointStamped  # 位姿和点消息

# Dobot自定义消息和服务
from dobot_msgs_v3.msg import ToolVectorActual         # 工具坐标向量消息
from dobot_msgs_v3.srv import GetHoldRegs              # 读取Modbus寄存器服务

import math                                            # 数学库
import time                                            # 时间处理库


# =============================================================================
# 关节状态中继节点
# =============================================================================
class JointStateRelay(Node):
    """
    关节状态中继节点

    该节点负责处理机械臂和夹爪的状态数据，并提供运动补偿功能。

    核心功能:
        1. 机械臂数据中继: 清理并转发机械臂关节状态
        2. 夹爪运动模拟: 通过插值算法模拟真实夹爪的物理运动特性
        3. 坐标系转换: 将工具坐标从毫米转换为米

    夹爪运动补偿原理:
        真实夹爪存在以下物理限制:
        - 通信延迟: Modbus指令发送到电机响应需要时间
        - 运动速度: 电机转速有物理上限
        - 惯性延迟: 启动和停止不是瞬时的

        本节点通过以下方式模拟这些特性:
        1. 延迟队列: 指令到达后不立即执行，而是等待GRIPPER_REACTION_DELAY
        2. 速度限制: 每帧最多移动 GRIPPER_SPEED * dt 的距离
        3. 线性插值: 平滑过渡，避免突变

    参数调优指南:
        - GRIPPER_SPEED: 如果模拟运动太慢，增加此值；太快则减小
        - GRIPPER_REACTION_DELAY: 如果数据仍比画面快，增加此延迟
    """

    def __init__(self, name):
        super().__init__(name)

        # 创建可重入回调组,允许多个回调函数并发执行
        self.cb_group = ReentrantCallbackGroup()

        # ==========================================
        # [核心修改] 运动插值补偿超参数
        # ==========================================
        # 1. 夹爪运动速度 (单位/秒)
        #    定义: 夹爪每秒能移动的位置单位数量
        #    范围: 0-1000 (夹爪位置范围为0-1000)
        #    示例: 如果夹爪从完全打开(1000)到完全闭合(0)需要0.5秒
        #          则速度 = 1000 / 0.5 = 2000 单位/秒
        #    调优: 数值越小，运动越慢；数值越大，运动越快
        self.GRIPPER_SPEED = 1000.0

        # 2. 通信死区延迟 (秒)
        #    定义: 从发送Modbus指令到电机开始实际运动的延迟时间
        #    用途: 模拟真实硬件的通信延迟和电机响应延迟
        #    调优: 如果发现模拟数据仍然比实际画面快，可以增加此值
        #          推荐范围: 0.05 - 0.3 秒
        self.GRIPPER_REACTION_DELAY = 0.25
        # ==========================================

        # ==========================================
        # 内部状态变量
        # ==========================================
        self.current_simulated_pos = 0.0  # 当前模拟的夹爪位置 (0-1000)
        self.target_gripper_pos = 0.0     # 目标夹爪位置 (从指令接收)
        self.last_update_time = time.time()  # 上次更新时间戳,用于计算时间差dt
        self.gripper_initialized = False  # 夹爪初始化标志

        # 延迟队列: 存储待执行的指令 [(执行时间戳, 目标位置值), ...]
        # 指令到达后不立即执行,而是在指定时间后才生效
        self.cmd_queue = []

        # ==========================================
        # 1. 机械臂相关订阅和发布
        # ==========================================
        # 订阅机械臂原始关节状态
        self.sub_robot_joints = self.create_subscription(
            JointState,
            "/joint_states_robot",
            self.robot_joint_callback,
            10,
            callback_group=self.cb_group)

        # 发布清理后的关节状态(仅包含6个关节)
        self.pub_joint_states = self.create_publisher(JointState, "joint_states", 10)

        # 订阅工具坐标向量
        self.sub_tool_vector = self.create_subscription(
            ToolVectorActual,
            "/dobot_msgs_v3/msg/ToolVectorActual",
            self.tool_vector_callback,
            10,
            callback_group=self.cb_group)

        # 发布末端执行器位姿(PoseStamped格式,单位转换为米)
        self.pub_end_effector = self.create_publisher(PoseStamped, "end_effector_pose", 10)

        # 中继发布工具向量数据
        self.pub_tool_vector_relay = self.create_publisher(ToolVectorActual, "end_effector_data", 10)

        # ==========================================
        # 2. 夹爪相关逻辑
        # ==========================================
        # 发布夹爪状态
        self.pub_gripper = self.create_publisher(JointState, "gripper/state", 10)

        # 订阅带时间戳的夹爪指令
        self.sub_gripper_update = self.create_subscription(
            PointStamped,
            "/gripper/command_update",
            self.gripper_update_callback,
            10,
            callback_group=self.cb_group
        )

        # 创建服务客户端: 用于读取Modbus寄存器(获取夹爪初始位置)
        self.cli_get_regs = self.create_client(
            GetHoldRegs,
            '/dobot_bringup_v3/srv/GetHoldRegs',
            callback_group=self.cb_group)

        # 创建初始化定时器: 每1秒尝试初始化夹爪位置
        self.init_timer = self.create_timer(1.0, self.try_init_gripper, callback_group=self.cb_group)

        # 创建发布定时器: 以100Hz频率发布夹爪状态(提供平滑的插值效果)
        self.publish_timer = self.create_timer(0.01, self.publish_gripper_state, callback_group=self.cb_group)

        self.get_logger().info(f"JointStateRelay Started. Gripper Speed: {self.GRIPPER_SPEED}, Delay: {self.GRIPPER_REACTION_DELAY}s")

    # =========================================================================
    # 夹爪初始化方法
    # =========================================================================
    def try_init_gripper(self):
        """
        尝试初始化夹爪位置

        流程:
            1. 检查是否已经初始化,如果是则销毁定时器
            2. 检查服务是否就绪
            3. 异步调用GetHoldRegs服务读取夹爪当前位置
            4. 设置回调函数处理响应

        Modbus寄存器信息:
            - 索引(index): 1
            - 地址(addr): 514
            - 数量(count): 1
        """
        if self.gripper_initialized:
            self.init_timer.destroy()  # 已初始化,销毁定时器
            return

        if not self.cli_get_regs.service_is_ready():
            return  # 服务未就绪,下次再试

        # 构造服务请求
        req = GetHoldRegs.Request()
        req.index = 1    # Modbus设备索引
        req.addr = 514   # 夹爪位置寄存器地址
        req.count = 1    # 读取1个寄存器

        # 异步调用服务
        future = self.cli_get_regs.call_async(req)
        future.add_done_callback(self.init_response_callback)

    def init_response_callback(self, future):
        """
        处理夹爪初始化响应

        参数:
            future: 服务调用的Future对象

        流程:
            1. 获取服务响应结果
            2. 检查响应状态码(res == 0 表示成功)
            3. 解析夹爪位置值
            4. 初始化所有相关状态变量
            5. 销毁初始化定时器
        """
        try:
            response = future.result()
            if response.res == 0:  # 成功响应
                # 解析返回值(格式: "{value}",需要去除花括号)
                val_str = str(response.value).strip("{}")
                if not val_str:
                    val_str = "0"
                val = float(val_str)

                # 初始化所有状态为读取到的当前位置
                self.current_simulated_pos = val  # 当前模拟位置
                self.target_gripper_pos = val     # 目标位置
                self.gripper_initialized = True   # 标记为已初始化
                self.get_logger().info(f"Gripper Initialized at pos: {val}")
                self.init_timer.destroy()  # 销毁初始化定时器
        except Exception as e:
            self.get_logger().warn(f"Gripper init failed: {e}")

    # =========================================================================
    # 夹爪运动控制方法
    # =========================================================================
    def gripper_update_callback(self, msg: PointStamped):
        """
        夹爪指令接收回调 - 延迟队列机制

        功能:
            接收到新指令时,不立即更新目标位置,而是将指令放入延迟队列
            这样可以模拟真实硬件的通信延迟

        参数:
            msg (PointStamped): 夹爪指令消息
                - msg.point.x: 目标位置值 (0-1000)
                - msg.header.stamp: 指令时间戳(未使用,因为使用系统时间)

        工作原理:
            1. 记录当前时间
            2. 计算指令应该执行的时间 = 当前时间 + 反应延迟
            3. 将(执行时间, 目标值)加入队列
            4. 在_process_gripper_motion中检查队列并执行到期的指令
        """
        # 记录接收时间 + 需要人为增加的延迟时间
        execution_time = time.time() + self.GRIPPER_REACTION_DELAY
        self.cmd_queue.append((execution_time, msg.point.x))

    def _process_gripper_motion(self):
        """
        夹爪运动物理模拟 - 核心算法

        该方法实现了夹爪运动的物理特性模拟,包括:
            1. 延迟处理: 从队列中取出到期的指令
            2. 速度限制: 限制每帧的最大移动距离
            3. 线性插值: 平滑地从当前位置过渡到目标位置

        算法详解:
            第一步 - 处理延迟队列:
                遍历cmd_queue,如果指令的执行时间已到,则更新target_gripper_pos

            第二步 - 计算运动步长:
                step = GRIPPER_SPEED * dt
                其中 dt 是距离上次更新的时间间隔

            第三步 - 线性插值:
                如果当前位置与目标位置差距很小(<1.0),直接吸附
                否则,按照计算的步长逐步接近目标,防止超调

        物理意义:
            - GRIPPER_SPEED模拟电机转速限制
            - dt确保运动与时间同步
            - 线性插值模拟惯性和加速过程
        """
        now = time.time()
        dt = now - self.last_update_time  # 计算时间差
        self.last_update_time = now

        # 1. 检查队列，是否有指令已经过了延迟期，可以执行了
        while self.cmd_queue and self.cmd_queue[0][0] <= now:
            _, target_val = self.cmd_queue.pop(0)  # 取出队列头部的指令
            self.target_gripper_pos = target_val   # 更新目标位置
            # self.get_logger().info(f"Executing gripper cmd: {target_val}")

        # 2. 线性插值逻辑 (Simulate Physics)
        diff = self.target_gripper_pos - self.current_simulated_pos

        # 如果差距很小(<1.0单位),直接吸附到目标位置
        if abs(diff) < 1.0:
            self.current_simulated_pos = self.target_gripper_pos
        else:
            # 计算这一帧能移动的最大步长
            step = self.GRIPPER_SPEED * dt

            if diff > 0:
                # 目标在前方,向前移动
                self.current_simulated_pos += step
                # 防止超调(超过目标)
                if self.current_simulated_pos > self.target_gripper_pos:
                    self.current_simulated_pos = self.target_gripper_pos
            else:
                # 目标在后方,向后移动
                self.current_simulated_pos -= step
                # 防止超调(低于目标)
                if self.current_simulated_pos < self.target_gripper_pos:
                    self.current_simulated_pos = self.target_gripper_pos

    def publish_gripper_state(self):
        """
        持续发布夹爪状态

        执行频率: 100Hz (每0.01秒)

        流程:
            1. 调用_process_gripper_motion计算新的模拟位置
            2. 构造JointState消息
            3. 发布模拟后的位置(而不是目标位置)

        注意:
            发布的是current_simulated_pos而非target_gripper_pos
            这确保了RViz等可视化工具看到的是模拟的物理运动
        """
        # 计算物理模拟
        self._process_gripper_motion()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gripper_link"
        msg.name = ["gripper_finger_joint"]
        # 发布模拟后的位置，而不是目标位置
        msg.position = [self.current_simulated_pos]
        self.pub_gripper.publish(msg)

    # =========================================================================
    # 机械臂数据处理方法
    # =========================================================================
    def robot_joint_callback(self, msg):
        """
        机械臂关节状态回调 - 数据清理与转发

        功能:
            接收机械臂原始关节状态,清理后重新发布

        参数:
            msg (JointState): 原始关节状态消息

        处理流程:
            1. 创建新的JointState消息
            2. 复制header(保留时间戳)
            3. 设置标准关节名称
            4. 只保留前6个关节的位置数据(过滤多余数据)
            5. 发布清理后的消息

        用途:
            确保发布的数据格式统一,避免数组长度不一致等问题
        """
        try:
            clean_msg = JointState()
            clean_msg.header = msg.header  # 保留原始时间戳
            clean_msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            if len(msg.position) >= 6:
                clean_msg.position = msg.position[:6]  # 只取前6个关节
                self.pub_joint_states.publish(clean_msg)
        except Exception:
            pass  # 静默处理异常

    def tool_vector_callback(self, msg):
        """
        工具坐标向量回调 - 坐标转换与发布

        功能:
            1. 接收Dobot工具坐标数据(ToolVectorActual)
            2. 转发原始数据到中继话题
            3. 转换为ROS标准的PoseStamped消息并发布

        参数:
            msg (ToolVectorActual): 工具坐标向量消息
                - x, y, z: 位置坐标(单位: 毫米)
                - rx, ry, rz: 姿态角度(未在此处使用)

        坐标转换:
            输入: 毫米(mm) -> 输出: 米(m)
            除以1000.0进行单位转换

        发布话题:
            - end_effector_data: 原始ToolVectorActual消息(中继)
            - end_effector_pose: PoseStamped消息(单位转换为米)

        注意:
            姿态信息(rx,ry,rz)未包含在PoseStamped中
            如需姿态,可订阅end_effector_data话题
        """
        current_time = self.get_clock().now().to_msg()

        # 中继发布原始数据
        self.pub_tool_vector_relay.publish(msg)

        # 构造PoseStamped消息
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time
        pose_msg.header.frame_id = "base_link"

        # 单位转换: 毫米 -> 米
        pose_msg.pose.position.x = msg.x / 1000.0
        pose_msg.pose.position.y = msg.y / 1000.0
        pose_msg.pose.position.z = msg.z / 1000.0

        # 发布位姿消息
        self.pub_end_effector.publish(pose_msg)


# =============================================================================
# 主函数
# =============================================================================
def main(args=None):
    """
    ROS2节点主入口

    功能:
        启动关节状态中继节点,使用多线程执行器处理并发回调

    执行流程:
        1. 初始化ROS2
        2. 创建JointStateRelay节点
        3. 创建多线程执行器
        4. 将节点添加到执行器
        5. 启动执行器循环(spin)
        6. 退出时清理资源

    多线程说明:
        使用MultiThreadedExecutor允许多个回调并发执行
        这对于需要同时处理夹爪和机械臂数据的场景很重要

    参数:
        args: 命令行参数(可选)
    """
    rclpy.init(args=args)
    node = JointStateRelay("dobot_joint_states")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()  # 进入循环,处理回调
    except KeyboardInterrupt:
        pass  # 用户按Ctrl+C退出
    node.destroy_node()  # 销毁节点
    rclpy.shutdown()     # 关闭ROS2