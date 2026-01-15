# 新版本：夹爪信息由控制脚本做增量 加入时间戳 运动补偿(夹爪速度和夹爪延迟)

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PointStamped
from dobot_msgs_v3.msg import ToolVectorActual
from dobot_msgs_v3.srv import GetHoldRegs

import math
import time

class JointStateRelay(Node):
    
    def __init__(self, name):
        super().__init__(name)
        
        self.cb_group = ReentrantCallbackGroup()

        # ==========================================
        # [核心修改] 运动插值补偿超参数
        # ==========================================
        # 1. 夹爪运动速度 (单位/秒)。范围0-1000。
        #    如果夹爪完全闭合(1000->0)需要0.5秒，则速度 = 1000/0.5 = 2000
        #    数值越小，数据变化越慢，越能模拟物理延迟。
        self.GRIPPER_SPEED = 1000.0  
        
        # 2. 通信死区延迟 (秒)。
        #    模拟Modbus发送指令到电机开始转动的固有延迟。
        #    如果发现数据还是比画面快，稍微增加这个值 (e.g. 0.05 - 0.1)
        self.GRIPPER_REACTION_DELAY = 0.25 
        # ==========================================

        # 内部状态
        self.current_simulated_pos = 0.0  # 当前发布的模拟位置 (0-1000)
        self.target_gripper_pos = 0.0     # 接收到的目标位置
        self.last_update_time = time.time()
        self.gripper_initialized = False
        
        # 延迟队列: [(timestamp, target_val), ...]
        self.cmd_queue = [] 

        # 1. 机械臂订阅
        self.sub_robot_joints = self.create_subscription(
            JointState, "/joint_states_robot", self.robot_joint_callback, 10, callback_group=self.cb_group)
        self.pub_joint_states = self.create_publisher(JointState, "joint_states", 10)

        self.sub_tool_vector = self.create_subscription(
            ToolVectorActual, "/dobot_msgs_v3/msg/ToolVectorActual", self.tool_vector_callback, 10, callback_group=self.cb_group)
        self.pub_end_effector = self.create_publisher(PoseStamped, "end_effector_pose", 10)
        self.pub_tool_vector_relay = self.create_publisher(ToolVectorActual, "end_effector_data", 10)

        # 2. 夹爪逻辑
        self.pub_gripper = self.create_publisher(JointState, "gripper/state", 10)
        
        # 订阅带时间戳的指令
        self.sub_gripper_update = self.create_subscription(
            PointStamped, 
            "/gripper/command_update", 
            self.gripper_update_callback, 
            10, 
            callback_group=self.cb_group
        )

        self.cli_get_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs', callback_group=self.cb_group)
        self.init_timer = self.create_timer(1.0, self.try_init_gripper, callback_group=self.cb_group)
        
        # 提高发布频率到 100Hz 以获得更平滑的插值效果
        self.publish_timer = self.create_timer(0.01, self.publish_gripper_state, callback_group=self.cb_group)
        
        self.get_logger().info(f"JointStateRelay Started. Gripper Speed: {self.GRIPPER_SPEED}, Delay: {self.GRIPPER_REACTION_DELAY}s")

    def try_init_gripper(self):
        if self.gripper_initialized:
            self.init_timer.destroy()
            return

        if not self.cli_get_regs.service_is_ready():
            return

        req = GetHoldRegs.Request()
        req.index = 1
        req.addr = 514
        req.count = 1
        
        future = self.cli_get_regs.call_async(req)
        future.add_done_callback(self.init_response_callback)

    def init_response_callback(self, future):
        try:
            response = future.result()
            if response.res == 0:
                val_str = str(response.value).strip("{}")
                if not val_str: val_str = "0"
                val = float(val_str)
                
                # 初始化所有状态
                self.current_simulated_pos = val
                self.target_gripper_pos = val
                self.gripper_initialized = True
                self.get_logger().info(f"Gripper Initialized at pos: {val}")
                self.init_timer.destroy()
        except Exception as e:
            self.get_logger().warn(f"Gripper init failed: {e}")

    def gripper_update_callback(self, msg: PointStamped):
        """
        接收到新指令时，不立即更新目标，而是放入延迟队列
        """
        # 记录接收时间 + 需要人为增加的延迟时间
        execution_time = time.time() + self.GRIPPER_REACTION_DELAY
        self.cmd_queue.append((execution_time, msg.point.x))

    def _process_gripper_motion(self):
        """
        计算下一帧的模拟位置
        """
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # 1. 检查队列，是否有指令已经过了延迟期，可以执行了
        while self.cmd_queue and self.cmd_queue[0][0] <= now:
            _, target_val = self.cmd_queue.pop(0)
            self.target_gripper_pos = target_val
            # self.get_logger().info(f"Executing gripper cmd: {target_val}")

        # 2. 线性插值逻辑 (Simulate Physics)
        diff = self.target_gripper_pos - self.current_simulated_pos
        
        # 如果差距很小，直接吸附
        if abs(diff) < 1.0:
            self.current_simulated_pos = self.target_gripper_pos
        else:
            # 计算这一帧能走的最大步长
            step = self.GRIPPER_SPEED * dt
            
            if diff > 0:
                self.current_simulated_pos += step
                # 防止超调
                if self.current_simulated_pos > self.target_gripper_pos:
                    self.current_simulated_pos = self.target_gripper_pos
            else:
                self.current_simulated_pos -= step
                if self.current_simulated_pos < self.target_gripper_pos:
                    self.current_simulated_pos = self.target_gripper_pos

    def publish_gripper_state(self):
        """持续发布夹爪状态"""
        # 计算物理模拟
        self._process_gripper_motion()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gripper_link"
        msg.name = ["gripper_finger_joint"] 
        # 发布模拟后的位置，而不是目标位置
        msg.position = [self.current_simulated_pos]
        self.pub_gripper.publish(msg)

    # ---------------------------------------------------
    # 下面是机械臂相关回调 (保持不变)
    # ---------------------------------------------------
    def robot_joint_callback(self, msg):
        try:
            clean_msg = JointState()
            clean_msg.header = msg.header 
            clean_msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            if len(msg.position) >= 6:
                clean_msg.position = msg.position[:6]
                self.pub_joint_states.publish(clean_msg)
        except Exception: pass

    def tool_vector_callback(self, msg):
        # 保留原始时间戳
        self.pub_tool_vector_relay.publish(msg)
        pose_msg = PoseStamped()
        pose_msg.header = msg.header  # 直接使用原始header（包含时间戳）
        pose_msg.pose.position.x = msg.x / 1000.0
        pose_msg.pose.position.y = msg.y / 1000.0
        pose_msg.pose.position.z = msg.z / 1000.0
        self.pub_end_effector.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStateRelay("dobot_joint_states")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()