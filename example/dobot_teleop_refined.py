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
L4 = 0.100  # Wrist

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
            # This prevents jumping on startup
            if not hasattr(self, 'initialized_target'):
                self.target_joints = list(self.current_joints)
                self.initialized_target = True
                self.get_logger().info("目标位置已同步至当前位置，准备就绪！")

    def solve_ik(self, x, y, z):
        """
        Simplified Analytic IK for 3-DOF Positioning (Joint 1, 2, 3)
        Keeps wrist joints (4,5,6) at current values (or fixed relative to arm).
        """
        # 1. Joint 1 (Base Yaw)
        theta1 = math.atan2(y, x)
        
        # 2. Project to 2D plane (r, z)
        # r is horizontal distance from base frame (excluding L1 vertical offset)
        # z_arm is height relative to shoulder axis
        r_ground = math.sqrt(x*x + y*y)
        # r = r_ground
        # But wait, we need to account for L2, L3 reaching to (r, z)
        # Coordinate system transformation
        # Target (r, z) in shoulder frame
        # Z axis is up. Shoulder axis is at height L1.
        
        target_z = z - L1
        target_r = r_ground
        
        # Triangle formed by L2, L3, and the chord C connecting shoulder to wrist center
        # distance from shoulder to target
        dist_sq = target_r**2 + target_z**2
        dist = math.sqrt(dist_sq)
        
        # Check workspace
        if dist > (L2 + L3) or dist < abs(L2 - L3) or dist == 0:
            return None
            
        # Law of Cosines for elbow angle (theta3)
        # dist^2 = L2^2 + L3^2 - 2*L2*L3*cos(pi - theta3)  <-- definition depends on zero config
        # Let's use standard convention:
        # alpha = angle(Shoulder->Target, Shoulder->Elbow)
        # beta = angle(Shoulder->Target, Horizon)
        
        # Internal angle at Elbow
        cos_angle_elbow = (L2**2 + L3**2 - dist_sq) / (2 * L2 * L3)
        # Clamp for safety
        cos_angle_elbow = max(-1.0, min(1.0, cos_angle_elbow))
        angle_elbow = math.acos(cos_angle_elbow)
        
        # For CR5, theta3=0 usually means arm straight up or straight horizontal?
        # Usually elbow=0 is 90 degrees bent or straight. 
        # Let's assume standard geometric IK: 
        # theta3 represents deviation from a straight line or relative angle.
        # Let's calc theta2 (Shoulder) and theta3 (Elbow) relative to horizontal/previous link.
        
        # Angle of chord
        beta = math.atan2(target_z, target_r)
        
        # Angle between chord and L2 using law of cosines
        cos_angle_shoulder_internal = (dist_sq + L2**2 - L3**2) / (2 * dist * L2)
        cos_angle_shoulder_internal = max(-1.0, min(1.0, cos_angle_shoulder_internal))
        angle_shoulder_internal = math.acos(cos_angle_shoulder_internal)
        
        # Resulting angles (standard anthropomorphic arm config, elbow up)
        # Theta2: Angle of L2 from horizon. 
        # CAUTION: Setup depends on robot zero position.
        # User guide says: Home=[0,0,0,0,0,0]. 
        # Typically Joint 2=0 is vertical, Joint 3=0 is vertical relative to L2?
        # We need to tune this offset. 
        # Assuming Joint 2=0 is UP, + is Backward?
        # Let's approximate: 
        # If Joint 2=0 is vertical:
        # theta2 = -(pi/2 - (beta + angle_shoulder_internal)) 
        # This part requires tuning based on URDF or empirical trial.
        # Let's stick to the logic from 'enhanced_keyboard_control.py' which assumed:
        # theta2 = alpha - beta
        # theta3 = acos(...)
        
        # We will iterate incrementally instead of absolute IK if possible, or use the provided simpler logic:
        # Recalculate based on current pose logic
        
        theta2 = beta + angle_shoulder_internal
        # Theta3 is angle of L3 relative to L2.
        # External angle is what matters?
        theta3_internal = angle_elbow
        theta3 = -(math.pi - theta3_internal) # Elbow down/up choice
        
        # Corrections for Dobot CR5 Zero-Frame:
        # Standard: joint1=0 (X+), joint2=0 (Up), joint3=0 (rel Up)
        # Our calculated theta1 is from X axis. Direct match.
        # Our calculated theta2 is from Horizon. 
        # If Joint2=0 is vertical (Z+), then joint_val = pi/2 - theta2.
        # If Joint3=0 is parallel to L2, then joint_val = theta3.
        
        # Apply offsets (Approximation, user can adjust)
        j1 = theta1
        j2 = (math.pi / 2) - theta2 
        j3 = theta3
        
        return [j1, j2, j3]


    def get_current_xyz(self):
        """Forward Kinematics for J1, J2, J3 to get roughly X,Y,Z"""
        j1, j2, j3 = self.current_joints[0], self.current_joints[1], self.current_joints[2]
        
        # Geometric FK
        # Projection on plane
        # angles in geometric formula (theta2 from horizon)
        theta2_geo = (math.pi / 2) - j2
        theta3_geo = j3 # relative
        
        # Global angles
        angle_l2 = theta2_geo
        angle_l3 = theta2_geo + theta3_geo
        
        # R and Z
        r = L2 * math.cos(angle_l2) + L3 * math.cos(angle_l3)
        z = L1 + L2 * math.sin(angle_l2) + L3 * math.sin(angle_l3)
        
        x = r * math.cos(j1)
        y = r * math.sin(j1)
        
        return x, y, z

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
                    # Update J1-J3, keep J4-J6
                    self.target_joints[0] = new_ik[0]
                    self.target_joints[1] = new_ik[1]
                    self.target_joints[2] = new_ik[2]

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
    print("复位归零: R")
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
