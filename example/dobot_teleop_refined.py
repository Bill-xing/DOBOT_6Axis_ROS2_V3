#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dobot Refined Teleoperation Script (ROS 2)
------------------------------------------
Features:
- Continuous smooth control loop (Topic-based)
- Joint Space Control
- Cartesian Space Control (Geometric IK for position)
- Real-time terminal UI
"""

import sys
import os
import time
import math
import termios
import tty
import select
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, qos_profile_sensor_data

# Robot Link Lengths (meters) - approximate for CR5
# These values should be verified against URDF
L1 = 0.138  # Base to Shoulder
L2 = 0.425  # Upper Arm
L3 = 0.395  # Forearm
L4 = 0.100  # Wrist (Distance from J4 to J5/J6 intersection?)
# Note: In "Level Hand" strategy, we consider the distance from J4 to Tool Tip
# This includes L4 (wrist length) + Gripper Length
L_GRIPPER = 0.15 # Gripper + Flange length (Approximate)
L_TOOL = L4 + L_GRIPPER

class KeyPoller:
    """Non-blocking keyboard input reader"""
    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def poll(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

class DobotTeleop(Node):
    def __init__(self):
        super().__init__('dobot_teleop_refined')
        
        # Parameters
        self.dobot_type = os.getenv("DOBOT_TYPE", "cr5")
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        
        # State
        self.current_joints = [0.0] * 6
        self.target_joints = [0.0] * 6
        self.running = True
        self.mode = "JOINT"  # JOINT or CARTESIAN
        self.joint_step = math.radians(1.5)  # 1.5 deg per loop
        self.cart_step = 0.005  # 5mm per loop
        self.control_rate = 20.0  # Hz
        
        # Auto-discover topics
        self.topic_name = self.find_topic_by_type('trajectory_msgs/msg/JointTrajectory')
        
        if not self.topic_name:
            self.get_logger().warn("自动查找失败: 未找到 JointTrajectory 类型的话题!")
            self.get_logger().info("当前可用话题列表:")
            for name, types in self.get_topic_names_and_types():
                self.get_logger().info(f"  - {name}: {types}")
            
            self.get_logger().warn("将尝试使用默认话题名称 (可能无效)...")
            self.topic_name = f'/{self.dobot_type}_group_controller/joint_trajectory'
        else:
            self.get_logger().info(f"自动发现控制话题: {self.topic_name}")
        
        # Communications
        # IMPORTANT: /joint_states is usually Best Effort (SensorData). 
        # If we subscribe with Reliable, we won't get data.
        self.sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, qos_profile_sensor_data)
        
        # Publisher for control (Reliable is standard for commands)
        self.pub = self.create_publisher(JointTrajectory, self.topic_name, 10)
        
        self.get_logger().info(f"初始化 Dobot 键盘控制程序 ({self.dobot_type})")
        self.get_logger().info(f"发布话题: {self.topic_name}")
        self.get_logger().info("等待 /joint_states 数据 (QoS: SensorData)...")
        
    def find_topic_by_type(self, topic_type):
        """Find the first topic of a given type"""
        # Wait a bit for discovery
        time.sleep(1.0)
        topic_names_and_types = self.get_topic_names_and_types()
        for name, types in topic_names_and_types:
            if topic_type in types:
                return name
        return None

    def joint_cb(self, msg):
        """Update current joint state"""
        # ... (remains same)
        # Map message joints to our order
        temp_joints = {}
        for i, name in enumerate(msg.name):
            temp_joints[name] = msg.position[i]
            
        is_valid = True
        new_joints = []
        
        # Check if we have data for our expected joints
        found_joints = []
        for name in self.joint_names:
            if name in temp_joints:
                new_joints.append(temp_joints[name])
                found_joints.append(name)
            else:
                is_valid = False
                # break # Check all
        
        if len(found_joints) > 0 and not hasattr(self, 'joints_logged'):
            self.get_logger().info(f"收到关节状态: 找到 {len(found_joints)}/{len(self.joint_names)} 个关节")
            self.get_logger().info(f"期望: {self.joint_names}")
            self.get_logger().info(f"实际收到: {msg.name}")
            self.joints_logged = True
            
        if is_valid:
            self.current_joints = new_joints
            # If we haven't started controlling yet, sync target to current
            # This prevents jumping on startup, UNLESS current is near zero (folded)
            if not hasattr(self, 'initialized_target'):
                # Check if robot is folded (all zeros)
                if all(abs(j) < 0.1 for j in self.current_joints):
                    self.get_logger().info("检测到机械臂处于零位。自动应用【测试初始姿态】...")
                    # Set a comfortable starting pose for IK testing
                    # J2=0 (Vertical), J3=90 (Horizontal), J4=-90 (Level Hand)
                    # This puts the arm reaching forward, tool horizontal.
                    self.target_joints = [0.0, 0.0, math.pi/2, -math.pi/2, 0.0, 0.0]
                else:
                    self.target_joints = list(self.current_joints)
                    
                self.initialized_target = True
                self.get_logger().info("目标位置已初始化，准备就绪！")

    def solve_ik(self, x_ee, y_ee, z_ee):
        """
        Calculates J1, J2, J3, J4 to reach (x, y, z) with the End-Effector,
        while maintaining a LEVEL HAND (Horizontal Pitch).
        """
        # Strategy:
        # 1. We want Tool Tip at (x_ee, y_ee, z_ee) with Pitch = 0 (Horizontal)
        # 2. This implies the Wrist Center (J4) is at a specific offset behind the Tool Tip.
        #    Since Pitch=0, the tool points along the horizontal radial vector.
        #    R_wrist = R_ee - L_TOOL
        #    Z_wrist = Z_ee
        
        # 1. Joint 1 (Base Yaw) - Determined by EE XY direction
        # Note: We assume the tool points in the same direction as the arm plane
        theta1 = math.atan2(y_ee, x_ee)
        
        # Calculate horizontal distance to EE
        r_ee = math.sqrt(x_ee**2 + y_ee**2)
        
        # Back-calculate Wrist Center target (r, z)
        # Assuming Level Hand: Wrist is just behind EE horizontally
        r_wrist_target = r_ee - L_TOOL
        z_wrist_target = z_ee
        
        # If the target is too close (inside the body), we might fail
        if r_wrist_target < 0:
            return None

        # --- Solve 3-DOF IK for Wrist Center (Same logic as before) ---
        
        # Target (r, z) in shoulder frame (Z axis is up, Shoulder at L1)
        target_z = z_wrist_target - L1
        target_r = r_wrist_target
        
        # Distance from shoulder to wrist center
        dist_sq = target_r**2 + target_z**2
        dist = math.sqrt(dist_sq)
        
        # Check workspace
        if dist > (L2 + L3) or dist < abs(L2 - L3) or dist == 0:
            return None
            
        # Law of Cosines for elbow angle (theta3)
        cos_angle_elbow = (L2**2 + L3**2 - dist_sq) / (2 * L2 * L3)
        cos_angle_elbow = max(-1.0, min(1.0, cos_angle_elbow))
        angle_elbow = math.acos(cos_angle_elbow)
        
        # Calculate theta2 (Shoulder)
        # beta: Angle of chord (Shoulder->Wrist) from horizon
        beta = math.atan2(target_z, target_r)
        
        # angle_shoulder_internal: Angle between chord and L2
        cos_angle_shoulder_internal = (dist_sq + L2**2 - L3**2) / (2 * dist * L2)
        cos_angle_shoulder_internal = max(-1.0, min(1.0, cos_angle_shoulder_internal))
        angle_shoulder_internal = math.acos(cos_angle_shoulder_internal)
        
        # theta2_geo: Angle of L2 from horizon
        theta2_geo = beta + angle_shoulder_internal
        
        # theta3_geo: Angle of L3 relative to L2 direction
        # Note: Geometry definition typically has elbow 'up' or 'down'. 
        # Here assuming simple elbow-up config usually used.
        # External angle at elbow is PI - internal_angle. 
        # If L2 is up, L3 goes down.
        theta3_geo = -(math.pi - angle_elbow)
        
        # Map to Robot Joint Values (CR5 Conventions)
        # J1 = atan2(y, x)
        j1 = theta1
        
        # J2: 0 is Vertical Up. theta2_geo is angle from Horizon.
        # If L2 is horizontal, theta2_geo=0, J2 should be 90 (pi/2).
        # If L2 is vertical up, theta2_geo=90, J2 should be 0.
        j2 = (math.pi / 2) - theta2_geo
        
        # J3: 0 is aligned with L2? Or vertical?
        # Typically J3 is relative to L2 in serial chain.
        # If J3=0 means straight arm, then J3 = theta3_geo (which is ~0 when straight).
        j3 = theta3_geo 
        
        # --- Solve J4 for Level Hand ---
        # We want Global Pitch of Tool = 0 (Horizontal)
        # Global Pitch = Angle_L2 + Angle_L3_rel + Angle_J4_rel
        # Angle_L2_global = theta2_geo
        # Angle_L3_global = theta2_geo + theta3_geo
        # Tool_Pitch_global = Angle_L3_global + j4_val
        # 0 = theta2_geo + theta3_geo + j4
        # => j4 = -(theta2_geo + theta3_geo)
        
        # Adjust for J4 zero definition.
        # If J4=0 means aligned with L3, then yes.
        # Let's verify J4 limits/definitions if possible. Assuming standard.
        j4 = -(theta2_geo + theta3_geo)
        
        return [j1, j2, j3, j4]


    def get_current_xyz(self):
        """Forward Kinematics to get End-Effector Position"""
        j1, j2, j3, j4 = self.current_joints[0], self.current_joints[1], self.current_joints[2], self.current_joints[3]
        
        # 1. Calculate Wrist Center (Standard FK)
        theta2_geo = (math.pi / 2) - j2
        theta3_geo = j3 
        
        angle_l2 = theta2_geo
        angle_l3 = theta2_geo + theta3_geo
        
        r_wrist = L2 * math.cos(angle_l2) + L3 * math.cos(angle_l3)
        z_wrist = L1 + L2 * math.sin(angle_l2) + L3 * math.sin(angle_l3)
        
        # 2. Calculate Tool Tip from Wrist Center
        # Global angle of J4 (Hand)
        # Assuming J4 is relative pitch
        angle_hand = angle_l3 + j4
        
        # Add tool vector
        r_ee = r_wrist + L_TOOL * math.cos(angle_hand)
        z_ee = z_wrist + L_TOOL * math.sin(angle_hand)
        
        x_ee = r_ee * math.cos(j1)
        y_ee = r_ee * math.sin(j1)
        
        return x_ee, y_ee, z_ee

    def run(self):
        print_help()
        
        with KeyPoller() as key_poller:
            while self.running and rclpy.ok():
                key = key_poller.poll()
                
                # Check shutdown
                if key == 'q':
                    self.running = False
                    break
                    
                # Process Input
                self.handle_input(key)
                
                # Enforce limits
                self.enforce_limits()
                
                # Publish
                self.publish_command()
                
                # UI Update
                self.print_status()
                
                time.sleep(1.0 / self.control_rate)

    def handle_input(self, key):
        if key is None:
            return

        key = key.lower()
        
        # Mode Switch
        if key == 'm':
            self.mode = "CARTESIAN" if self.mode == "JOINT" else "JOINT"
            return
        elif key == 'r':
            # Reset
            self.target_joints = [0.0] * 6
            return

        elif key == 't':
            # Test Pose (Ready for IK)
            self.get_logger().info("移动到测试姿态...")
            # Puts arm forward, horizontal tool
            self.target_joints = [0.0, 0.0, math.pi/2, -math.pi/2, 0.0, 0.0]
            return
        elif key == '[':
            self.get_logger().info("夹爪动作: [关闭]")
            return
        elif key == ']':
            self.get_logger().info("夹爪动作: [打开]")
            return

        if self.mode == "JOINT":
            # Joint Mappings
            # 1/2: J1, 3/4: J2, ...
            idx = -1
            direction = 0
            
            mapping = {
                '1': (0, 1), '2': (0, -1),
                '3': (1, 1), '4': (1, -1),
                '5': (2, 1), '6': (2, -1),
                '7': (3, 1), '8': (3, -1),
                '9': (4, 1), '0': (4, -1),
                '-': (5, 1), '=': (5, -1)
            }
            
            if key in mapping:
                idx, direction = mapping[key]
                self.target_joints[idx] += direction * self.joint_step

        elif self.mode == "CARTESIAN":
            # XYZ Control
            if not hasattr(self, 'initialized_target'):
                return
                
            x, y, z = self.get_current_xyz()
            
            dx, dy, dz = 0, 0, 0
            
            if key == 'w': dx = self.cart_step
            elif key == 's': dx = -self.cart_step
            elif key == 'a': dy = self.cart_step
            elif key == 'd': dy = -self.cart_step
            elif key == 'z': dz = self.cart_step
            elif key == 'x': dz = -self.cart_step # Use 'x' key for down? might conflict with user expectation but simple
            
            if dx != 0 or dy != 0 or dz != 0:
                new_ik = self.solve_ik(x + dx, y + dy, z + dz)
                if new_ik:
                    # Update J1-J4 (J4 is used for level hand)
                    self.target_joints[0] = new_ik[0]
                    self.target_joints[1] = new_ik[1]
                    self.target_joints[2] = new_ik[2]
                    self.target_joints[3] = new_ik[3]

    def enforce_limits(self):
        # Soft limits (approximate)
        limits = [
            (-3.0, 3.0), # J1
            (-1.5, 1.5), # J2
            (-2.5, 2.5), # J3
            (-3.0, 3.0), # J4
            (-3.0, 3.0), # J5
            (-3.14, 3.14) # J6
        ]
        
        for i in range(6):
            min_l, max_l = limits[i]
            self.target_joints[i] = max(min_l, min(max_l, self.target_joints[i]))

    def publish_command(self):
        msg = JointTrajectory()
        # Using 0 stamp means "execute now" and ignores time sync issues
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = "base_link"
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = self.target_joints
        # Small lookahead time for smooth interpolation by the controller
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 200000000 # 0.2s
        
        msg.points.append(point)
        self.pub.publish(msg)

    def print_status(self):
        # Move cursor up to overwrite
        sys.stdout.write("\033[K") # Clear line
        mode_str = "关节控制" if self.mode == "JOINT" else "笛卡尔控制"
        
        # Calculate current XYZ for display
        cx, cy, cz = self.get_current_xyz()
        
        status = f"模式: {mode_str} | 坐标: [{cx:.3f}, {cy:.3f}, {cz:.3f}] | "
        sys.stdout.write(f"\r{status}")
        sys.stdout.flush()

def print_help():
    print("\n" + "="*60)
    print("DOBOT 机械臂键盘遥控程序 (精简版)")
    print("="*60)
    print("模式切换: M (关节控制 <-> 笛卡尔坐标控制)")
    print("复位归零: R  |  测试姿态: T (推荐用于IK测试)")
    print("夹爪控制: [ (关闭) / ] (打开)")
    print("退出程序: Q")
    print("\n关节控制模式按键:")
    print("  1/2: 关节 1 +/-")
    print("  3/4: 关节 2 +/-")
    print("  5/6: 关节 3 +/-")
    print("  7/8: 关节 4 +/-")
    print("  9/0: 关节 5 +/-")
    print("  -/=: 关节 6 +/-")
    print("\n笛卡尔坐标模式按键 (控制末端位置):")
    print("  W/S: X 轴 前/后")
    print("  A/D: Y 轴 左/右")
    print("  Z/X: Z 轴 上/下 (X键为向下)")
    print("="*60)
    print("控制循环运行中... 按 Q 退出")

def main(args=None):
    rclpy.init(args=args)
    
    node = DobotTeleop()
    
    # Spin in a separate thread so callbacks work
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        # Wait for first state
        print("正在等待关节状态数据...")
        time.sleep(1.0) 
        if not hasattr(node, 'initialized_target'):
            print("警告: 尚未收到关节状态。默认为全零位置。")
            node.target_joints = [0.0]*6
            
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
