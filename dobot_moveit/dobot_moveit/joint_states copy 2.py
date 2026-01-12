# 新版本：夹爪信息由控制脚本做增量 加入时间戳

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PointStamped  # [修改] 使用 PointStamped
from dobot_msgs_v3.msg import ToolVectorActual
from dobot_msgs_v3.srv import GetHoldRegs

class JointStateRelay(Node):
    
    def __init__(self, name):
        super().__init__(name)
        
        self.cb_group = ReentrantCallbackGroup()

        self.current_gripper_pos = 0.0
        self.last_gripper_stamp = self.get_clock().now() # 记录最后一次更新的时间
        self.gripper_initialized = False

        self.sub_robot_joints = self.create_subscription(
            JointState, "/joint_states_robot", self.robot_joint_callback, 10, callback_group=self.cb_group)
        self.pub_joint_states = self.create_publisher(JointState, "joint_states", 10)

        self.sub_tool_vector = self.create_subscription(
            ToolVectorActual, "/dobot_msgs_v3/msg/ToolVectorActual", self.tool_vector_callback, 10, callback_group=self.cb_group)
        self.pub_end_effector = self.create_publisher(PoseStamped, "end_effector_pose", 10)
        self.pub_tool_vector_relay = self.create_publisher(ToolVectorActual, "end_effector_data", 10)

        self.pub_gripper = self.create_publisher(JointState, "gripper/state", 10)
        
        # [修改] 订阅 PointStamped 以获取带时间戳的夹爪指令
        self.sub_gripper_update = self.create_subscription(
            PointStamped, 
            "/gripper/command_update", 
            self.gripper_update_callback, 
            10, 
            callback_group=self.cb_group
        )

        self.cli_get_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs', callback_group=self.cb_group)
        self.init_timer = self.create_timer(1.0, self.try_init_gripper, callback_group=self.cb_group)
        self.publish_timer = self.create_timer(0.02, self.publish_gripper_state, callback_group=self.cb_group)
        
        self.get_logger().info("JointStateRelay Started. Waiting for gripper init...")

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
                self.current_gripper_pos = float(val_str)
                self.gripper_initialized = True
                self.get_logger().info(f"Gripper Initialized at pos: {self.current_gripper_pos}")
                self.init_timer.destroy()
        except Exception as e:
            self.get_logger().warn(f"Gripper init failed: {e}")

    def gripper_update_callback(self, msg: PointStamped):
        """
        接收来自遥操作脚本的带时间戳更新
        """
        self.current_gripper_pos = msg.point.x
        # 可以选择使用 msg.header.stamp 更新内部时间记录，
        # 但为了保证 joint_states 数据流的频率稳定，我们通常在发布时使用当前系统时间，
        # 除非我们要严格重放数据。这里我们只更新数值。

    def publish_gripper_state(self):
        """持续发布夹爪状态"""
        msg = JointState()
        # [关键] 必须加上时间戳，这是数据对齐的依据
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gripper_link"
        msg.name = ["gripper_finger_joint"] 
        msg.position = [self.current_gripper_pos]
        self.pub_gripper.publish(msg)

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
        current_time = self.get_clock().now().to_msg()
        self.pub_tool_vector_relay.publish(msg)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time
        pose_msg.header.frame_id = "base_link"
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