#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础版 DOBOT 机械臂键盘控制系统
简单易用的关节级控制
"""

import math
import os
from typing import Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


class KeyboardControlNode(Node):
    """基础键盘控制节点"""

    def __init__(self) -> None:
        super().__init__("keyboard_control")

        # 机械臂配置
        arm_type = os.getenv("DOBOT_TYPE", "cr5")
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.controller_ns = f"/{arm_type}_group_controller/follow_joint_trajectory"

        # 当前关节状态
        self.current_joints: Optional[list] = None
        self.joint_sub = self.create_subscription(
            JointState, "joint_states", self._joint_cb, 10
        )

        # Action客户端
        self.action_client = ActionClient(
            self, FollowJointTrajectory, self.controller_ns
        )

        # 安全参数
        self.JOINT_STEP_DEG = 1.0   # 关节步长（度）
        self.MAX_DEG = 170.0        # 软限位
        self.SAFE_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 等待控制器就绪
        self.get_logger().info(f"等待控制器动作服务器: {self.controller_ns}")
        self.action_client.wait_for_server()

        self.get_logger().info("=== DOBOT 基础键盘控制系统已就绪 ===")
        self.print_help()

    def _joint_cb(self, msg: JointState) -> None:
        """关节状态回调"""
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        if all(n in name_to_pos for n in self.joint_names):
            self.current_joints = [name_to_pos[n] for n in self.joint_names]

    def _clip_rad(self, rad: float) -> float:
        """角度限幅"""
        deg = math.degrees(rad)
        deg = max(-self.MAX_DEG, min(self.MAX_DEG, deg))
        return math.radians(deg)

    def _ensure_state(self) -> bool:
        """确保关节状态可用"""
        if self.current_joints is None:
            self.get_logger().warn("尚未收到 /joint_states，等待关节状态后再操作。")
            return False
        return True

    def send_joint_goal(self, target: list, duration: float = 1.0) -> bool:
        """发送关节目标"""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start.sec = int(duration)
        goal.trajectory.points.append(point)

        # 异步发送目标
        send_goal_future = self.action_client.send_goal_async(goal)

        # 非阻塞方式处理结果
        def goal_response_callback(future):
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().warn("关节目标被拒绝")
            else:
                self.get_logger().debug("关节目标已接受")

        send_goal_future.add_done_callback(goal_response_callback)
        return True

    def move_joint(self, joint_index: int, delta_deg: float) -> bool:
        """移动单个关节"""
        if not self._ensure_state():
            return False

        new_target = list(self.current_joints)
        new_val = self._clip_rad(new_target[joint_index] + math.radians(delta_deg))
        new_target[joint_index] = new_val

        success = self.send_joint_goal(new_target, 0.5)
        if success:
            self.get_logger().info(
                f"关节 {joint_index + 1} 增量 {delta_deg:+.1f}° → 目标 {math.degrees(new_val):+.1f}°"
            )
        return success

    def go_home(self) -> None:
        """回到初始姿态"""
        self.get_logger().info("回到初始姿态...")
        home_pose = [0.0] * 6
        self.send_joint_goal(home_pose, 3.0)

    def go_safe_pose(self) -> None:
        """回到安全姿态"""
        self.get_logger().info("回到安全姿态...")
        safe_pose = [0.0, math.radians(30), math.radians(-30), 0.0, 0.0, 0.0]
        self.send_joint_goal(safe_pose, 3.0)

    def show_status(self) -> None:
        """显示当前状态"""
        if not self._ensure_state():
            return

        joints = self.current_joints
        print("\n" + "=" * 50)
        print("当前关节状态：")
        print("=" * 50)
        print("弧度: ", [f"{j:+.4f}" for j in joints])
        print("度数: ", [f"{math.degrees(j):+.1f}°" for j in joints])
        print("软限位: ±%.1f°" % self.MAX_DEG)
        print("=" * 50 + "\n")

    def print_help(self) -> None:
        """打印帮助信息"""
        print("\n" + "=" * 50)
        print("DOBOT 基础键盘控制系统")
        print("=" * 50)
        print("关节步长: ±1°（软限位 ±170°）")
        print("")
        print("指令：")
        print("  1-6 : 选择关节")
        print("  +/- : 所选关节 ±1°")
        print("  R   : 回到初始姿态")
        print("  S   : 回到安全姿态")
        print("  G   : 显示当前状态")
        print("  H   : 显示帮助")
        print("  Q   : 退出程序")
        print("=" * 50 + "\n")


def main(args=None) -> None:
    """主函数"""
    rclpy.init(args=args)
    node = KeyboardControlNode()

    current_joint = 0  # 默认选择关节1

    try:
        # 等待关节状态
        print("等待关节状态...")
        for i in range(50):  # 最多等待5秒
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.current_joints is not None:
                break

        if node.current_joints is None:
            print("错误：无法获取关节状态，请确保仿真环境正在运行")
            return

        while rclpy.ok():
            try:
                command = input("输入命令 (H 查看帮助): ").strip().upper()

                if command in ["1", "2", "3", "4", "5", "6"]:
                    current_joint = int(command) - 1
                    print(f"✓ 已选择关节 {command}")
                elif command == "+":
                    node.move_joint(current_joint, node.JOINT_STEP_DEG)
                elif command == "-":
                    node.move_joint(current_joint, -node.JOINT_STEP_DEG)
                elif command == "R":
                    node.go_home()
                elif command == "S":
                    node.go_safe_pose()
                elif command == "G":
                    node.show_status()
                elif command == "H":
                    node.print_help()
                elif command == "Q":
                    print("退出程序...")
                    break
                else:
                    if command:
                        print("未知命令，按 H 查看帮助")

                rclpy.spin_once(node, timeout_sec=0.1)

            except KeyboardInterrupt:
                print("\nCtrl+C 中断")
                break
            except Exception as e:
                print(f"输入错误: {e}")
                rclpy.spin_once(node, timeout_sec=0.1)

    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("✓ 程序已退出")


if __name__ == "__main__":
    main()