#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版DOBOT机械臂键盘控制系统
支持关节控制和末端执行器控制的混合模式
基于示例文件的最佳实践设计
"""

import math
import os
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs


class EnhancedKeyboardControl(Node):
    """增强版DOBOT键盘控制节点"""

    def __init__(self) -> None:
        super().__init__("enhanced_keyboard_control")

        # 机械臂配置
        arm_type = os.getenv("DOBOT_TYPE", "cr5")
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.controller_ns = f"/{arm_type}_group_controller/follow_joint_trajectory"

        # 当前关节状态
        self.current_joints: Optional[List[float]] = None
        self.joint_sub = self.create_subscription(
            JointState, "joint_states", self._joint_cb, 10
        )

        # Action客户端
        self.action_client = ActionClient(
            self, FollowJointTrajectory, self.controller_ns
        )

        # TF转换器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 控制参数
        self.JOINT_STEP_DEG = 2.0        # 关节步长（度）
        self.EE_STEP_MM = 5.0           # 末端步长（毫米）
        self.MAX_DEG = 170.0            # 软限位
        self.CONTROL_FREQUENCY = 20     # 控制频率（Hz）

        # 控制模式
        self.control_mode = "joint"     # "joint" 或 "ee"
        self.target_joints = [0.0] * 6  # 目标关节角度
        self.target_ee_pose = None      # 目标末端位姿

        # 运动学参数（简化版，用于末端控制）
        self.L1 = 0.135  # 大臂长度
        self.L2 = 0.147  # 小臂长度
        self.L3 = 0.083  # 手腕长度

        # 等待控制器就绪
        self.get_logger().info(f"等待控制器动作服务器: {self.controller_ns}")
        self.action_client.wait_for_server()

        # 控制线程
        self.control_thread = None
        self.running = False

        self.get_logger().info("=== DOBOT 增强版键盘控制系统已就绪 ===")
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
            self.get_logger().warn("尚未收到 /joint_states")
            return False
        return True

    def _get_current_ee_pose(self) -> Optional[PoseStamped]:
        """获取当前末端位姿"""
        try:
            # 获取从基座到末端执行器的变换
            transform = self.tf_buffer.lookup_transform(
                "base_link", "tool0", rclpy.time.Time()
            )

            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "base_link"

            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z

            pose.pose.orientation = transform.transform.rotation

            return pose

        except Exception as e:
            self.get_logger().warn(f"无法获取末端位姿: {e}")
            return None

    def _inverse_kinematics(self, x: float, y: float, z: float) -> Optional[List[float]]:
        """
        简化版逆运动学求解
        仅适用于特定配置的DOBOT机械臂
        """
        try:
            # 简化的2D逆运动学（仅考虑x-y平面）
            r = math.sqrt(x**2 + y**2)

            # 检查可达性
            if r > (self.L1 + self.L2) or r < abs(self.L1 - self.L2):
                return None

            # 使用余弦定理计算关节角度
            cos_theta3 = (r**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
            cos_theta3 = max(-1, min(1, cos_theta3))  # 限幅

            theta3 = math.acos(cos_theta3)  # 肘关节角度

            # 计算肩关节角度
            if r > 0:
                alpha = math.atan2(y, x)
                beta = math.atan2(self.L2 * math.sin(theta3),
                                self.L1 + self.L2 * math.cos(theta3))
                theta2 = alpha - beta
            else:
                theta2 = 0

            # 简化处理其他关节（保持水平）
            theta1 = 0      # 基座旋转
            theta4 = 0      # 手腕俯仰
            theta5 = 0      # 手腕旋转
            theta6 = 0      # 手腕偏航

            return [theta1, theta2, theta3, theta4, theta5, theta6]

        except Exception as e:
            self.get_logger().warn(f"逆运动学求解失败: {e}")
            return None

    def send_joint_goal(self, target: List[float], duration: float = 1.0) -> bool:
        """发送关节目标"""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)

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
        self.target_joints = new_target

        return self.send_joint_goal(new_target, 0.5)

    def move_ee(self, dx: float, dy: float, dz: float) -> bool:
        """移动末端执行器"""
        if not self._ensure_state():
            return False

        # 获取当前末端位姿
        current_pose = self._get_current_ee_pose()
        if not current_pose:
            return False

        # 计算新位置
        new_x = current_pose.pose.position.x + dx / 1000  # mm转m
        new_y = current_pose.pose.position.y + dy / 1000
        new_z = current_pose.pose.position.z + dz / 1000

        # 逆运动学求解
        joint_angles = self._inverse_kinematics(new_x, new_y, new_z)
        if joint_angles:
            self.target_joints = joint_angles
            return self.send_joint_goal(joint_angles, 0.5)

        return False

    def control_loop(self):
        """控制循环线程"""
        while self.running and rclpy.ok():
            try:
                if self.control_mode == "joint" and self._ensure_state():
                    # 关节控制模式：平滑过渡到目标关节角度
                    current = self.current_joints
                    diff = [t - c for t, c in zip(self.target_joints, current)]

                    # 如果误差很小，不发送指令
                    if any(abs(d) > 0.01 for d in diff):
                        # 平滑插值
                        alpha = 0.3  # 平滑系数
                        interpolated = [
                            c + alpha * d for c, d in zip(current, diff)
                        ]
                        self.send_joint_goal(interpolated, 0.1)

                time.sleep(1.0 / self.CONTROL_FREQUENCY)

            except Exception as e:
                self.get_logger().error(f"控制循环错误: {e}")
                time.sleep(1.0)

    def start_control_loop(self):
        """启动控制循环"""
        if not self.running:
            self.running = True
            self.control_thread = threading.Thread(target=self.control_loop)
            self.control_thread.daemon = True
            self.control_thread.start()
            self.get_logger().info("控制循环已启动")

    def stop_control_loop(self):
        """停止控制循环"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        self.get_logger().info("控制循环已停止")

    def go_home(self) -> None:
        """回到初始姿态"""
        self.get_logger().info("回到初始姿态...")
        home_pose = [0.0] * 6
        self.target_joints = home_pose
        self.send_joint_goal(home_pose, 3.0)

    def go_safe_pose(self) -> None:
        """回到安全姿态"""
        self.get_logger().info("回到安全姿态...")
        safe_pose = [0.0, math.radians(45), math.radians(-45), 0.0, 0.0, 0.0]
        self.target_joints = safe_pose
        self.send_joint_goal(safe_pose, 3.0)

    def toggle_mode(self):
        """切换控制模式"""
        if self.control_mode == "joint":
            self.control_mode = "ee"
            self.get_logger().info("切换到末端执行器控制模式")
        else:
            self.control_mode = "joint"
            self.get_logger().info("切换到关节控制模式")

    def show_status(self) -> None:
        """显示当前状态"""
        if not self._ensure_state():
            return

        joints = self.current_joints
        print("\n" + "=" * 60)
        print(f"当前控制模式: {self.control_mode}")
        print("=" * 60)

        if self.control_mode == "joint":
            print("当前关节角度:")
            for i, (name, angle) in enumerate(zip(self.joint_names, joints)):
                print(f"  {name:8s}: {math.degrees(angle):+7.2f}°")
        else:
            pose = self._get_current_ee_pose()
            if pose:
                p = pose.pose.position
                print("当前末端位置:")
                print(f"  X: {p.x*1000:+7.2f} mm")
                print(f"  Y: {p.y*1000:+7.2f} mm")
                print(f"  Z: {p.z*1000:+7.2f} mm")

        print("=" * 60 + "\n")

    def print_help(self) -> None:
        """打印帮助信息"""
        print("\n" + "=" * 60)
        print("DOBOT 增强版键盘控制系统")
        print("=" * 60)
        print(f"控制参数: 关节步长={self.JOINT_STEP_DEG}°, 末端步长={self.EE_STEP_MM}mm")
        print("\n模式控制:")
        print("  M        - 切换控制模式 (关节/末端)")
        print("  H        - 显示当前状态")
        print("  R        - 回到初始姿态")
        print("  S        - 回到安全姿态")
        print("  Q        - 退出程序")

        print("\n关节控制模式:")
        print("  1/2      - 选择关节1 (基座旋转)")
        print("  3/4      - 选择关节2 (肩关节)")
        print("  5/6      - 选择关节3 (肘关节)")
        print("  7/8      - 选择关节4 (腕关节1)")
        print("  9/0      - 选择关节5 (腕关节2)")
        print("  -/=      - 选择关节6 (腕关节3)")
        print("  +/-      - 增加/减少选中关节角度")

        print("\n末端执行器控制模式:")
        print("  W/S      - X轴前进/后退")
        print("  A/D      - Y轴左移/右移")
        print("  Z/X      - Z轴上升/下降")

        print("\n快速预设:")
        print("  F1-F4    - 预设姿态1-4")
        print("=" * 60 + "\n")


def keyboard_input_loop(node: EnhancedKeyboardControl):
    """键盘输入循环"""
    current_joint = 0

    while rclpy.ok():
        try:
            command = input("输入命令 (H查看帮助): ").strip().upper()

            # 模式控制
            if command == "M":
                node.toggle_mode()
            elif command == "H":
                node.show_status()
            elif command == "R":
                node.go_home()
            elif command == "S":
                node.go_safe_pose()
            elif command == "Q":
                break

            # 关节选择
            elif command in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
                current_joint = int(command) - 1
                current_joint = min(5, max(0, current_joint))
                print(f"✓ 已选择关节 {current_joint + 1}")

            elif command in ["-", "="]:
                current_joint = 5  # 关节6
                print(f"✓ 已选择关节 6")

            # 关节控制
            elif command == "+":
                if node.control_mode == "joint":
                    node.move_joint(current_joint, node.JOINT_STEP_DEG)
                else:
                    node.move_ee(0, 0, node.EE_STEP_MM)  # Z+
            elif command == "-":
                if node.control_mode == "joint":
                    node.move_joint(current_joint, -node.JOINT_STEP_DEG)
                else:
                    node.move_ee(0, 0, -node.EE_STEP_MM)  # Z-

            # 末端控制
            elif command == "W":
                node.move_ee(node.EE_STEP_MM, 0, 0)   # X+
            elif command == "S":
                node.move_ee(-node.EE_STEP_MM, 0, 0)  # X-
            elif command == "D":
                node.move_ee(0, node.EE_STEP_MM, 0)   # Y+
            elif command == "A":
                node.move_ee(0, -node.EE_STEP_MM, 0)  # Y-
            elif command == "Z":
                node.move_ee(0, 0, node.EE_STEP_MM)   # Z+
            elif command == "X":
                node.move_ee(0, 0, -node.EE_STEP_MM)  # Z-

            # 预设姿态
            elif command == "F1":
                node.send_joint_goal([0, 0, 0, 0, 0, 0], 2.0)
            elif command == "F2":
                node.send_joint_goal([0, math.radians(30), math.radians(-30), 0, 0, 0], 2.0)
            elif command == "F3":
                node.send_joint_goal([0, math.radians(45), math.radians(-45), 0, 0, 0], 2.0)
            elif command == "F4":
                node.send_joint_goal([math.radians(90), math.radians(45), math.radians(-45), 0, 0, 0], 2.0)

            elif command:
                print("未知命令，按 H 查看帮助")

        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            print(f"输入错误: {e}")


def main(args=None) -> None:
    """主函数"""
    rclpy.init(args=args)

    try:
        # 创建控制节点
        node = EnhancedKeyboardControl()

        # 等待关节状态
        print("等待关节状态...")
        for i in range(50):  # 最多等待5秒
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.current_joints is not None:
                break

        if node.current_joints is None:
            print("错误：无法获取关节状态，请确保仿真环境正在运行")
            return

        # 初始化目标位置
        node.target_joints = list(node.current_joints)

        # 启动控制循环
        node.start_control_loop()

        # 启动键盘输入循环
        try:
            keyboard_input_loop(node)
        finally:
            # 停止控制循环
            node.stop_control_loop()

    except Exception as e:
        print(f"程序错误: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()
        print("程序已退出")


if __name__ == "__main__":
    main()