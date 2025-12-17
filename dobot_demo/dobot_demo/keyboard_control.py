#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
键盘控制 CR5 机械臂（无 moveit_commander 依赖）
通过 FollowJointTrajectory 直接下发小增量关节指令，适用于仿真和真机。
安全特性：关节小步进、保守软限位、预置安全姿态。
"""

import math
import os
import sys
from typing import Dict, List, Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


class KeyboardControlNode(Node):
    def __init__(self) -> None:
        super().__init__("keyboard_control")

        arm_type = os.getenv("DOBOT_TYPE", "cr5")
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.controller_ns = f"/{arm_type}_group_controller/follow_joint_trajectory"

        # 当前关节状态（订阅 /joint_states 获取）
        self.current_joints: Optional[List[float]] = None
        self.joint_sub = self.create_subscription(
            JointState, "joint_states", self._joint_cb, 10
        )

        # FollowJointTrajectory 动作客户端
        self.action_client = ActionClient(
            self, FollowJointTrajectory, self.controller_ns
        )
        self.get_logger().info(f"等待控制器动作服务器: {self.controller_ns}")
        self.action_client.wait_for_server()

        # 安全参数（保守设置）
        self.JOINT_STEP_DEG = 1.0   # 单次按键关节步长（度）
        self.MAX_DEG = 170.0        # 软限位（绝对值），防止过大角度
        self.SAFE_POSE = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]

        self.get_logger().info("=== CR5 键盘关节控制已就绪（无 MoveIt 依赖） ===")
        self.get_logger().warn("注意：仅支持关节小步进；不提供笛卡尔/规划功能。")
        self.print_help()

    # -------------------- 状态与工具 --------------------
    def _joint_cb(self, msg: JointState) -> None:
        name_to_pos: Dict[str, float] = {n: p for n, p in zip(msg.name, msg.position)}
        if all(n in name_to_pos for n in self.joint_names):
            self.current_joints = [name_to_pos[n] for n in self.joint_names]

    def _clip_rad(self, rad: float) -> float:
        deg = math.degrees(rad)
        deg = max(-self.MAX_DEG, min(self.MAX_DEG, deg))
        return math.radians(deg)

    def _ensure_state(self) -> bool:
        if self.current_joints is None:
            self.get_logger().warn("尚未收到 /joint_states，等待关节状态后再操作。")
            return False
        return True

    # -------------------- 执行动作 --------------------
    def send_joint_goal(self, target: List[float]) -> bool:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start.sec = 1
        goal.trajectory.points.append(point)

        goal_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("动作目标被拒绝，可能控制器未就绪或目标无效。")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().warn("未获取到动作结果。")
            return False

        if result.result.error_code == 0:
            return True
        self.get_logger().warn(f"控制器返回错误码: {result.result.error_code}")
        return False

    def move_joint(self, joint_index: int, delta_deg: float) -> bool:
        if not self._ensure_state():
            return False
        new_target = list(self.current_joints)
        new_val = self._clip_rad(new_target[joint_index] + math.radians(delta_deg))
        new_target[joint_index] = new_val
        ok = self.send_joint_goal(new_target)
        if ok:
            self.get_logger().info(
                f"关节 {joint_index + 1} 增量 {delta_deg:+.1f}° → 目标 {math.degrees(new_val):+.1f}°"
            )
        return ok

    def go_safe_pose(self) -> None:
        self.get_logger().info("回到安全姿态（打开构型）...")
        ok = self.send_joint_goal(self.SAFE_POSE)
        if ok:
            self.get_logger().info("✓ 已到达安全姿态")

    # -------------------- 信息展示 --------------------
    def show_status(self) -> None:
        if not self._ensure_state():
            return
        joints = self.current_joints
        print("\n" + "=" * 46)
        print("当前关节状态：")
        print("=" * 46)
        print("弧度: ", [f"{j:+.4f}" for j in joints])
        print("度数: ", [f"{math.degrees(j):+.1f}°" for j in joints])
        print("软限位: ±%.1f°" % self.MAX_DEG)
        print("安全姿态: ", [f"{math.degrees(j):+.1f}°" for j in self.SAFE_POSE])
        print("=" * 46 + "\n")

    def print_help(self) -> None:
        print("\n========== CR5 关节键盘控制（无 MoveIt）==========")
        print("关节步长: ±1°（保守软限位 ±170°）")
        print("指令：")
        print("  1-6 : 选择关节")
        print("  +/- : 所选关节 ±1°")
        print("  R   : 回安全姿态")
        print("  G   : 显示当前状态")
        print("  H   : 显示帮助")
        print("  QUIT: 退出")
        print("============================================\n")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KeyboardControlNode()

    current_joint = 0  # 默认选择关节1
    try:
        while rclpy.ok():
            command = input("输入命令 (H 查看帮助): ").strip().upper()
            if command in ["1", "2", "3", "4", "5", "6"]:
                current_joint = int(command) - 1
                print(f"✓ 已选择关节 {command}")
            elif command == "+":
                node.move_joint(current_joint, node.JOINT_STEP_DEG)
            elif command == "-":
                node.move_joint(current_joint, -node.JOINT_STEP_DEG)
            elif command == "R":
                node.go_safe_pose()
            elif command == "G":
                node.show_status()
            elif command == "H":
                node.print_help()
            elif command == "QUIT":
                print("退出程序...")
                break
            else:
                if command:
                    print("未知命令，按 H 查看帮助")
    except KeyboardInterrupt:
        print("\nCtrl+C 中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("✓ 程序已退出")


if __name__ == "__main__":
    main()
