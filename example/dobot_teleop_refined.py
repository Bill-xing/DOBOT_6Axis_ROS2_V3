#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dobot 机械臂精细化遥控脚本 (ROS 2)
------------------------------------------
功能特性:
- 基于话题的连续平滑控制循环
- 关节空间控制
- 笛卡尔空间控制 (基于几何逆运动学的位置控制)
- 实时终端用户界面
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

# 机器人连杆长度 (米) - 针对 CR5 的近似值
# 这些值应与 URDF 文件核对
L1 = 0.138  # 基座到肩部
L2 = 0.425  # 大臂
L3 = 0.395  # 小臂
L4 = 0.100  # 手腕 (从 J4 到 J5/J6 交点的距离?)
# 注意: 在 "水平持握 (Level Hand)" 策略中，我们考虑从 J4 到工具尖端 (Tool Tip) 的距离
# 这包括 L4 (手腕长度) + 夹爪长度
L_GRIPPER = 0.15 # 夹爪 + 法兰长度 (近似值)
L_TOOL = L4 + L_GRIPPER

class KeyPoller:
    """非阻塞键盘输入读取器"""
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
        
        # 参数
        self.dobot_type = os.getenv("DOBOT_TYPE", "cr5")
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        
        # 状态
        self.current_joints = [0.0] * 6
        self.target_joints = [0.0] * 6
        self.running = True
        self.mode = "JOINT"  # 关节模式 (JOINT) 或 笛卡尔模式 (CARTESIAN)
        self.joint_step = math.radians(1.5)  # 每次循环约 1.5 度
        self.cart_step = 0.005  # 每次循环 5mm
        self.control_rate = 20.0  # 控制频率 Hz
        
        # 自动发现话题
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
        
        # 通信
        # 重要: /joint_states 通常是 Best Effort (SensorData) QoS。
        # 如果我们使用 Reliable 订阅，可能收不到数据。
        self.sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, qos_profile_sensor_data)
        
        # 控制指令发布器 (Reliable 是指令的标准配置)
        self.pub = self.create_publisher(JointTrajectory, self.topic_name, 10)
        
        self.get_logger().info(f"初始化 Dobot 键盘控制程序 ({self.dobot_type})")
        self.get_logger().info(f"发布话题: {self.topic_name}")
        self.get_logger().info("等待 /joint_states 数据 (QoS: SensorData)...")
        
    def find_topic_by_type(self, topic_type):
        """查找指定类型的第一个话题"""
        # 等待一会以便发现话题
        time.sleep(1.0)
        topic_names_and_types = self.get_topic_names_and_types()
        for name, types in topic_names_and_types:
            if topic_type in types:
                return name
        return None

    def joint_cb(self, msg):
        """更新当前关节状态"""
        # 将消息中的关节映射到我们的顺序
        temp_joints = {}
        for i, name in enumerate(msg.name):
            temp_joints[name] = msg.position[i]
            
        is_valid = True
        new_joints = []
        
        # 检查是否包含我们期望的所有关节
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
            # 如果尚未开始控制，将目标同步为当前位置
            # 这可以防止启动时机械臂跳动，除非当前处于零位（折叠状态）
            if not hasattr(self, 'initialized_target'):
                # 检查机器人是否处于折叠状态 (全零)
                if all(abs(j) < 0.1 for j in self.current_joints):
                    self.get_logger().info("检测到机械臂处于零位。自动应用【测试初始姿态】...")
                    # 设置一个适合 IK 测试的舒适起始姿态
                    # J2=0 (垂直), J3=90 (水平), J4=-90 (水平持握)
                    # 这会让机械臂向前伸出，工具保持水平。
                    self.target_joints = [0.0, 0.0, math.pi/2, -math.pi/2, 0.0, 0.0]
                else:
                    self.target_joints = list(self.current_joints)
                    
                self.initialized_target = True
                self.get_logger().info("目标位置已初始化，准备就绪！")

    def solve_ik(self, x_ee, y_ee, z_ee):
        """
        计算 J1, J2, J3, J4 以使末端执行器到达 (x, y, z)，
        同时保持【水平持握】(Level Hand / Horizontal Pitch)。
        """
        # 策略:
        # 1. 我们希望工具尖端 (Tool Tip) 位于 (x_ee, y_ee, z_ee) 且俯仰角 Pitch = 0 (水平)。
        # 2. 这意味着手腕中心 (J4) 位于工具尖端后方的一个特定偏移处。
        #    因为 Pitch=0，工具沿着水平径向向量指向前方。
        #    R_wrist = R_ee - L_TOOL
        #    Z_wrist = Z_ee
        
        # 1. 关节 1 (基座偏航) - 由末端的 XY 方向决定
        # 注意: 我们假设工具指向与机械臂平面相同的方向
        theta1 = math.atan2(y_ee, x_ee)
        
        # 计算到末端的水平距离
        r_ee = math.sqrt(x_ee**2 + y_ee**2)
        
        # 反推手腕中心的目标位置 (r, z)
        # 假设水平持握: 手腕就在末端后方水平位置
        r_wrist_target = r_ee - L_TOOL
        z_wrist_target = z_ee
        
        # 如果目标太近 (在身体内部)，则无解
        if r_wrist_target < 0:
            return None

        # --- 针对手腕中心求解 3自由度 IK (与之前逻辑相同) ---
        
        # 目标 (r, z) 在肩部坐标系中 (Z 轴向上, 肩部高度为 L1)
        target_z = z_wrist_target - L1
        target_r = r_wrist_target
        
        # 从肩部到手腕中心的距离
        dist_sq = target_r**2 + target_z**2
        dist = math.sqrt(dist_sq)
        
        # 检查工作空间
        if dist > (L2 + L3) or dist < abs(L2 - L3) or dist == 0:
            return None
            
        # 使用余弦定理计算肘部角度 (theta3)
        cos_angle_elbow = (L2**2 + L3**2 - dist_sq) / (2 * L2 * L3)
        cos_angle_elbow = max(-1.0, min(1.0, cos_angle_elbow))
        angle_elbow = math.acos(cos_angle_elbow)
        
        # 计算 theta2 (肩部)
        # beta: 弦 (肩部->手腕) 与水平线的夹角
        beta = math.atan2(target_z, target_r)
        
        # angle_shoulder_internal: 弦与 L2 之间的夹角
        cos_angle_shoulder_internal = (dist_sq + L2**2 - L3**2) / (2 * dist * L2)
        cos_angle_shoulder_internal = max(-1.0, min(1.0, cos_angle_shoulder_internal))
        angle_shoulder_internal = math.acos(cos_angle_shoulder_internal)
        
        # theta2_geo: L2 与水平线的夹角
        theta2_geo = beta + angle_shoulder_internal
        
        # theta3_geo: L3 相对于 L2 方向的夹角
        # 注意: 几何定义通常有“肘部向上”或“肘部向下”。
        # 这里假设通常使用的简单“肘部向上”配置。
        # 肘部的外角 = PI - 内角。
        # 如果 L2 向上，L3 向下。
        theta3_geo = -(math.pi - angle_elbow)
        
        # 映射到机器人关节值 (CR5 约定)
        # J1 = atan2(y, x)
        j1 = theta1
        
        # J2: 0 是垂直向上。theta2_geo 是相对于水平线的角度。
        # 如果 L2 水平，theta2_geo=0，J2 应该是 90 (pi/2)。
        # 如果 L2 垂直向上，theta2_geo=90，J2 应该是 0。
        j2 = (math.pi / 2) - theta2_geo
        
        # J3: 0 是与 L2 对齐? 还是垂直?
        # 通常 J3 是串联链中相对于 L2 的角度。
        # 如果 J3=0 意味着手臂伸直，那么 J3 = theta3_geo (伸直时约为 0)。
        j3 = theta3_geo 
        
        # --- 求解 J4 以保持水平持握 ---
        # 我们希望工具的全局俯仰角 = 0 (水平)
        # Global Pitch = Angle_L2 + Angle_L3_rel + Angle_J4_rel
        # Angle_L2_global = theta2_geo
        # Angle_L3_global = theta2_geo + theta3_geo
        # Tool_Pitch_global = Angle_L3_global + j4_val
        # 0 = theta2_geo + theta3_geo + j4
        # => j4 = -(theta2_geo + theta3_geo)
        
        # 调整 J4 的零位定义。
        # 如果 J4=0 意味着与 L3 对齐，则如下。
        # 假设它是标准的。
        j4 = -(theta2_geo + theta3_geo)
        
        return [j1, j2, j3, j4]


    def get_current_xyz(self):
        """正运动学计算末端执行器位置"""
        j1, j2, j3, j4 = self.current_joints[0], self.current_joints[1], self.current_joints[2], self.current_joints[3]
        
        # 1. 计算手腕中心 (标准 FK)
        theta2_geo = (math.pi / 2) - j2
        theta3_geo = j3 
        
        angle_l2 = theta2_geo
        angle_l3 = theta2_geo + theta3_geo
        
        r_wrist = L2 * math.cos(angle_l2) + L3 * math.cos(angle_l3)
        z_wrist = L1 + L2 * math.sin(angle_l2) + L3 * math.sin(angle_l3)
        
        # 2. 从手腕中心计算工具尖端
        # J4 (手) 的全局角度
        # 假设 J4 是相对俯仰
        angle_hand = angle_l3 + j4
        
        # 加上工具向量
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
                
                # 检查退出
                if key == 'q':
                    self.running = False
                    break
                    
                # 处理输入
                self.handle_input(key)
                
                # 强制限位
                self.enforce_limits()
                
                # 发布指令
                self.publish_command()
                
                # 界面更新
                self.print_status()
                
                time.sleep(1.0 / self.control_rate)

    def handle_input(self, key):
        if key is None:
            return

        key = key.lower()
        
        # 模式切换
        if key == 'm':
            self.mode = "CARTESIAN" if self.mode == "JOINT" else "JOINT"
            return
        elif key == 'r':
            # 复位
            self.target_joints = [0.0] * 6
            return

        elif key == 't':
            # 测试姿态 (准备好进行 IK)
            self.get_logger().info("移动到测试姿态...")
            # 手臂前伸，工具水平
            self.target_joints = [0.0, 0.0, math.pi/2, -math.pi/2, 0.0, 0.0]
            return
        elif key == '[':
            self.get_logger().info("夹爪动作: [关闭]")
            return
        elif key == ']':
            self.get_logger().info("夹爪动作: [打开]")
            return

        if self.mode == "JOINT":
            # 关节映射
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
            # XYZ 控制
            if not hasattr(self, 'initialized_target'):
                return
                
            x, y, z = self.get_current_xyz()
            
            dx, dy, dz = 0, 0, 0
            
            if key == 'w': dx = self.cart_step
            elif key == 's': dx = -self.cart_step
            elif key == 'a': dy = self.cart_step
            elif key == 'd': dy = -self.cart_step
            elif key == 'z': dz = self.cart_step
            elif key == 'x': dz = -self.cart_step # 使用 'x' 键向下? 可能与用户习惯冲突但为了简单
            
            if dx != 0 or dy != 0 or dz != 0:
                new_ik = self.solve_ik(x + dx, y + dy, z + dz)
                if new_ik:
                    # 更新 J1-J4 (J4 用于保持水平)
                    self.target_joints[0] = new_ik[0]
                    self.target_joints[1] = new_ik[1]
                    self.target_joints[2] = new_ik[2]
                    self.target_joints[3] = new_ik[3]

    def enforce_limits(self):
        # 软限位 (近似值)
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
        # 使用 0 时间戳意味着 "立即执行"，忽略时间同步问题
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = "base_link"
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = self.target_joints
        # 给控制器一些前瞻时间以实现平滑插补
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 200000000 # 0.2s
        
        msg.points.append(point)
        self.pub.publish(msg)

    def print_status(self):
        # 上移光标以覆盖前一行
        sys.stdout.write("\033[K") # 清除行
        mode_str = "关节控制" if self.mode == "JOINT" else "笛卡尔控制"
        
        # 计算当前 XYZ 用于显示
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
    
    # 在单独的线程中 Spin 以便回调正常工作
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        # 等待初始状态
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
