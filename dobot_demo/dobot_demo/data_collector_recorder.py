#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import threading
import re
import sys
import os
import collections
from collections import deque
import numpy as np
import h5py
import cv2
from cv_bridge import CvBridge

# 引入dobot_msgs_v3的所有服务
from dobot_msgs_v3.srv import *
# 引入传感器消息
from sensor_msgs.msg import Image

# 引入 pynput
from pynput import keyboard, mouse
import tkinter as tk

# ================= 配置区域 =================
DATASET_DIR = "./collected_data"
TASK_NAME = "dobot_teleop_task"
CAMERA_NAMES = ['camera_front'] # 目前代码只适配了一个相机，可扩展
MAX_TIMESTEPS = 1000            # 最大录制帧数
# ===========================================

class DobotRosWrapper(Node):
    """
    ROS2 Node封装：处理所有Service通信 + 相机订阅
    """
    def __init__(self, node_name="dobot_api_client"):
        super().__init__(node_name)
        
        # --- Service Clients (原有) ---
        self.cli_enable = self.create_client(EnableRobot, '/dobot_bringup_v3/srv/EnableRobot')
        self.cli_clear_error = self.create_client(ClearError, '/dobot_bringup_v3/srv/ClearError')
        self.cli_speed_factor = self.create_client(SpeedFactor, '/dobot_bringup_v3/srv/SpeedFactor')
        self.cli_mov_j = self.create_client(MovJ, '/dobot_bringup_v3/srv/MovJ')
        self.cli_mov_l = self.create_client(MovL, '/dobot_bringup_v3/srv/MovL')
        self.cli_servo_j = self.create_client(ServoJ, '/dobot_bringup_v3/srv/ServoJ')
        self.cli_servo_p = self.create_client(ServoP, '/dobot_bringup_v3/srv/ServoP')
        self.cli_sync = self.create_client(Sync, '/dobot_bringup_v3/srv/Sync')
        self.cli_get_pose = self.create_client(GetPose, '/dobot_bringup_v3/srv/GetPose')
        self.cli_get_angle = self.create_client(GetAngle, '/dobot_bringup_v3/srv/GetAngle')
        self.cli_modbus_create = self.create_client(ModbusCreate, '/dobot_bringup_v3/srv/ModbusCreate')
        self.cli_modbus_close = self.create_client(ModbusClose, '/dobot_bringup_v3/srv/ModbusClose')
        self.cli_set_hold_regs = self.create_client(SetHoldRegs, '/dobot_bringup_v3/srv/SetHoldRegs')
        self.cli_get_hold_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs')

        # --- Image Subscribers (新增) ---
        # 使用 Best Effort QoS 以降低延迟，适应视频流
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.img_color_deque = deque(maxlen=2000)
        self.img_depth_deque = deque(maxlen=2000)
        
        # 根据截图，Topic 为 /camera/color/image_raw 和 /camera/depth/image_raw (Orbbec Wrapper默认)
        self.sub_color = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self._color_callback, 
            qos_profile
        )
        self.sub_depth = self.create_subscription(
            Image, 
            '/camera/depth/image_raw', 
            self._depth_callback, 
            qos_profile
        )

        self.get_logger().info("Waiting for Dobot services...")
        if not self.cli_mov_j.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Dobot services not available yet.")

    def _color_callback(self, msg):
        self.img_color_deque.append(msg)

    def _depth_callback(self, msg):
        self.img_depth_deque.append(msg)

    def call_service(self, client, request):
        if not client.service_is_ready():
            self.get_logger().error(f"Service {client.srv_name} is not ready.")
            return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.005) # 简单的忙等待
        return future.result()


class DataCollector:
    """
    负责数据对齐、缓存和保存
    """
    def __init__(self, node: DobotRosWrapper):
        self.node = node
        self.bridge = CvBridge()
        self.recording = False
        self.episode_idx = 0
        
        # 临时存储当前Episode的数据
        self.data_buffer = {
            'qpos': [],      # 关节角
            'qvel': [],      # 关节速度 (Dobot API 未直接提供实时速度，通常用差分或置0)
            'cartesian': [], # 末端位姿
            'gripper': [],   # 夹爪位置
            'action': [],    # 记录下发的控制指令
            'images': [],    # RGB
            'depths': []     # Depth
        }
        
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
            
        # 查找当前最大的 episode index
        task_dir = os.path.join(DATASET_DIR, TASK_NAME)
        if os.path.exists(task_dir):
            existing_files = [f for f in os.listdir(task_dir) if f.endswith('.hdf5')]
            if existing_files:
                indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
                self.episode_idx = max(indices) + 1
        
        print(f"--- 数据收集器就绪 (Next Episode: {self.episode_idx}) ---")

    def start_recording(self):
        self.recording = True
        self.data_buffer = {k: [] for k in self.data_buffer} # 清空buffer
        print(f"\n>>> 开始录制 Episode {self.episode_idx} >>>")

    def stop_recording(self):
        if not self.recording: return
        self.recording = False
        print(f"\n<<< 停止录制. 正在保存 {len(self.data_buffer['qpos'])} 帧数据... <<<")
        self.save_to_hdf5()
        self.episode_idx += 1

    def process_frame(self, robot_pose, robot_joints, gripper_pos, action_target):
        """
        核心函数：在主循环中调用。
        1. 获取当前时间戳
        2. 从 ROS 队列中找到最近的图像
        3. 将所有数据对齐存入 buffer
        """
        if not self.recording:
            return

        # 获取当前 ROS 时间 (秒)
        now = self.node.get_clock().now()
        timestamp = now.nanoseconds * 1e-9

        # --- 图像同步逻辑 ---
        # 策略：因为 Robot 状态是当前帧获取的，我们去 Image 队列找时间戳 <= 当前时间 且最新的那一帧
        
        img_color = self._get_synced_image(self.node.img_color_deque, timestamp, "rgb8")
        img_depth = self._get_synced_image(self.node.img_depth_deque, timestamp, "16UC1") # 深度图通常是 16UC1

        if img_color is None:
            # print("警告: 未找到同步的 RGB 图像 (队列可能为空或延迟过大)")
            # 为了保持数据完整性，如果丢失图像，这一帧通常选择丢弃，或者复用上一帧(不推荐)
            return 
        
        # 处理深度图 (如果是 None，可以填全0或者丢弃，这里选择填0防止崩)
        if img_depth is None:
            img_depth = np.zeros((img_color.shape[0], img_color.shape[1]), dtype=np.uint16)

        # --- 数据存入 Buffer ---
        self.data_buffer['qpos'].append(robot_joints)
        self.data_buffer['qvel'].append([0.0]*6) # 暂无速度接口，填0
        self.data_buffer['cartesian'].append(robot_pose)
        self.data_buffer['gripper'].append(gripper_pos)
        self.data_buffer['action'].append(action_target) # 记录遥操作的目标值作为 Action
        self.data_buffer['images'].append(img_color)
        self.data_buffer['depths'].append(img_depth)
        
        # 打印进度
        if len(self.data_buffer['qpos']) % 10 == 0:
            sys.stdout.write(f"\r[录制中] 帧数: {len(self.data_buffer['qpos'])}")
            sys.stdout.flush()

    def _get_synced_image(self, deque_buffer, target_time, encoding):
        if len(deque_buffer) == 0:
            return None
        
        # 从右往左找（从最新往旧找），找到第一个时间戳 <= target_time 的
        # 注意：ROS2 msg.header.stamp 需要转换
        chosen_msg = None
        
        # 简单策略：取最新的一帧，只要时间差在允许范围内 (例如 0.1秒)
        # 如果队列里的最新帧都很老，说明相机卡了
        latest_msg = deque_buffer[-1]
        msg_time = latest_msg.header.stamp.sec + latest_msg.header.stamp.nanosec * 1e-9
        
        if abs(msg_time - target_time) < 0.2: # 200ms 容忍度
            chosen_msg = latest_msg
        
        if chosen_msg:
            try:
                # 转换图像
                cv_img = self.bridge.imgmsg_to_cv2(chosen_msg, desired_encoding=encoding)
                if encoding == "passthrough" or encoding == "16UC1":
                    # 深度图可能需要处理一下非数值
                    pass
                return cv_img
            except Exception as e:
                print(f"CV Bridge Error: {e}")
                return None
        return None

    def save_to_hdf5(self):
        task_dir = os.path.join(DATASET_DIR, TASK_NAME)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
            
        file_path = os.path.join(task_dir, f"episode_{self.episode_idx}.hdf5")
        data_len = len(self.data_buffer['qpos'])
        
        if data_len == 0:
            print("数据为空，不保存。")
            return

        try:
            with h5py.File(file_path, 'w') as root:
                root.attrs['sim'] = False
                root.attrs['compress'] = True # 标记压缩
                
                # 创建 Observation 组
                obs_group = root.create_group('observations')
                
                # 图像
                img_group = obs_group.create_group('images')
                # 假设 RGB 是 (H, W, 3)
                h, w, c = self.data_buffer['images'][0].shape
                img_dset = img_group.create_dataset(CAMERA_NAMES[0], (data_len, h, w, c), dtype='uint8', chunks=(1, h, w, c))
                for i, img in enumerate(self.data_buffer['images']):
                    img_dset[i] = img
                    
                # 深度图
                depth_group = obs_group.create_group('images_depth')
                h_d, w_d = self.data_buffer['depths'][0].shape
                depth_dset = depth_group.create_dataset(CAMERA_NAMES[0], (data_len, h_d, w_d), dtype='uint16', chunks=(1, h_d, w_d))
                for i, depth in enumerate(self.data_buffer['depths']):
                    depth_dset[i] = depth

                # 状态数据
                obs_group.create_dataset('qpos', data=np.array(self.data_buffer['qpos']))
                obs_group.create_dataset('qvel', data=np.array(self.data_buffer['qvel']))
                obs_group.create_dataset('cartesian', data=np.array(self.data_buffer['cartesian']))
                obs_group.create_dataset('gripper', data=np.array(self.data_buffer['gripper']))
                
                # Action (这里假设 action 就是下一步的目标关节或末端，这里存的是 target_pose)
                # 实际 Imitation Learning 中 action 通常是移位后的 qpos，这里存当时的 target 命令
                root.create_dataset('action', data=np.array(self.data_buffer['action']))

            print(f"保存成功: {file_path}")
            
        except Exception as e:
            print(f"保存 HDF5 失败: {e}")


# ================= 复用之前的 GripperController 类 =================
class GripperController:
    """DH-Robotics AG系列夹爪控制器"""
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.id = 0
        self.init_gripper_connection()

    def init_gripper_connection(self):
        for i in range(1, 5):
            req = ModbusClose.Request()
            req.index = i
            self.wrapper.call_service(self.wrapper.cli_modbus_close, req)
        
        req = ModbusCreate.Request()
        req.ip = "127.0.0.1"
        req.port = 60000
        req.slave_id = 1
        req.is_rtu = 1
        res = self.wrapper.call_service(self.wrapper.cli_modbus_create, req)
        
        if res and res.res == 0:
            try:
                self.id = int(res.index)
            except:
                match = re.search(r'(\d+)', str(res.index))
                self.id = int(match.group(1)) if match else 0
            print(f"Gripper Connected, Index: {self.id}")
        else:
            print("Gripper ModbusCreate Failed")
            self.id = 0
        
        if self.id > 0:
            self.enable()
            self.set_force(60)

    def enable(self):
        self._write_regs(256, 1, "1")

    def set_force(self, force: int):
        force = max(20, min(100, force))
        self._write_regs(257, 1, str(force))

    def move(self, position: int):
        position = max(0, min(1000, position))
        self._write_regs(259, 1, str(position))

    def get_current_position(self):
        return self._read_regs(514, 1)

    def _write_regs(self, addr, count, val_tab):
        if self.id <= 0: return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_tab
        self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)

    def _read_regs(self, addr, count):
        if self.id <= 0: return None
        req = GetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        res = self.wrapper.call_service(self.wrapper.cli_get_hold_regs, req)
        if res and res.res == 0:
            try:
                return int(res.value)
            except:
                pass
        return None


# ================= 复用并修改 Robot 类 =================
class Robot:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        
        self.node = DobotRosWrapper()
        
        # 启动 ROS 线程
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()
        
        print(">.< 正在连接 ROS 服务... >.<")
        try:
            self.gripper = GripperController(self.node)
        except Exception as e:
            print(f":( 夹爪初始化警告: {e}")

        self.enable_robot()
        self.set_speed_factor(100)
        self.clear_error()
        print(">.< 连接成功! >.<")

    def enable_robot(self):
        req = EnableRobot.Request()
        self.node.call_service(self.node.cli_enable, req)

    def clear_error(self):
        req = ClearError.Request()
        self.node.call_service(self.node.cli_clear_error, req)

    def set_speed_factor(self, ratio):
        req = SpeedFactor.Request()
        req.ratio = ratio
        self.node.call_service(self.node.cli_speed_factor, req)

    def ServoP(self, x, y, z, rx, ry, rz):
        req = ServoP.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        return self.node.call_service(self.node.cli_servo_p, req)

    def get_pose(self):
        req = GetPose.Request()
        res = self.node.call_service(self.node.cli_get_pose, req)
        if res and hasattr(res, 'pose'): 
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", res.pose)
            if len(matches) >= 6:
                return [float(x) for x in matches[:6]]
        return None

    def get_angle(self):
        req = GetAngle.Request()
        res = self.node.call_service(self.node.cli_get_angle, req)
        if res and hasattr(res, 'angle'):
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", res.angle)
            if len(matches) >= 6:
                return [float(x) for x in matches[:6]]
        return None
    
    def close(self):
        rclpy.shutdown()


# ================= 修改后的遥操作控制器 =================
class TeleopController:
    def __init__(self):
        print("正在初始化机器人链接...")
        self.robot = Robot()
        
        # --- 初始化数据收集器 ---
        self.collector = DataCollector(self.robot.node)
        
        # --- 控制参数 ---
        self.LOOP_RATE = 30.0       
        self.MOUSE_SENSITIVITY = 0.1 
        self.SCROLL_SENSITIVITY = 2.0 
        self.KEY_XYZ_STEP = 1.0     
        self.KEY_ROT_STEP = 0.3     
        self.GRIPPER_STEP = 40      

        self.running = True
        self.target_pose = [0.0] * 6  
        self.target_gripper_pos = 1000 
        
        self.keys_pressed = set()
        self.mouse_dx = 0
        self.mouse_dy = 0
        self.scroll_dy = 0
        self.mouse_left_pressed = False
        self.mouse_right_pressed = False

        self._init_pose()

        # Listeners
        self.kb_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move, 
            on_click=self._on_mouse_click, 
            on_scroll=self._on_scroll,
            suppress=True 
        )
        self.kb_listener.start()
        self.mouse_listener.start()

        # Screen setup
        root = tk.Tk()
        self.center_x = root.winfo_screenwidth() // 2
        self.center_y = root.winfo_screenheight() // 2
        root.destroy()
        
        try:
            os.system("xdotool getactivewindow windowactivate")
        except: pass
        
        self.mouse_controller = mouse.Controller()
        self.mouse_controller.position = (self.center_x, self.center_y)

        print("=== 控制已启动 ===")
        print("  [C]         : 开始/停止 数据录制 (收集)")
        print("  [Space/Ctrl]: Z轴升降")
        print("  [WASD]      : 旋转姿态")
        print("  [鼠标]      : XY平面移动 + 滚轮旋转")
        print("  [ESC]       : 退出")

    def _init_pose(self):
        retries = 0
        while retries < 5:
            pose = self.robot.get_pose()
            if pose:
                self.target_pose = pose
                g_pos = self.robot.gripper.get_current_position()
                if g_pos is not None:
                    self.target_gripper_pos = g_pos
                return
            time.sleep(0.5)
            retries += 1
        print("错误：无法获取机械臂初始位置！")
        sys.exit(1)

    def _on_key_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            if hasattr(key, 'char'):
                char = key.char.lower()
                self.keys_pressed.add(char)
                
                # --- 新增按键逻辑：C键录制 ---
                if char == 'c':
                    if self.collector.recording:
                        self.collector.stop_recording()
                    else:
                        self.collector.start_recording()
            else:
                self.keys_pressed.add(key)
        except AttributeError:
            pass

    def _on_key_release(self, key):
        try:
            if hasattr(key, 'char'):
                self.keys_pressed.discard(key.char.lower())
            else:
                self.keys_pressed.discard(key)
        except AttributeError:
            pass

    def _on_mouse_move(self, x, y):
        dx = x - self.center_x
        dy = y - self.center_y
        self.mouse_dx += dx
        self.mouse_dy += dy
        if dx != 0 or dy != 0:
            self.mouse_controller.position = (self.center_x, self.center_y)
    
    def _on_mouse_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            self.mouse_left_pressed = pressed
        elif button == mouse.Button.right:
            self.mouse_right_pressed = pressed

    def _on_scroll(self, x, y, dx, dy):
        self.scroll_dy += dy

    def run(self):
        gripper_tick = 0 

        while self.running:
            start_time = time.time()

            # 1. 鼠标输入处理
            dx = self.mouse_dx
            dy = self.mouse_dy
            self.mouse_dx = 0
            self.mouse_dy = 0

            # 2. 姿态计算
            self.target_pose[0] += dx * self.MOUSE_SENSITIVITY
            self.target_pose[1] -= dy * self.MOUSE_SENSITIVITY

            if keyboard.Key.space in self.keys_pressed:
                self.target_pose[2] += self.KEY_XYZ_STEP
            if keyboard.Key.ctrl in self.keys_pressed or keyboard.Key.ctrl_l in self.keys_pressed:
                self.target_pose[2] -= self.KEY_XYZ_STEP

            if 'a' in self.keys_pressed: self.target_pose[3] += self.KEY_ROT_STEP
            if 'd' in self.keys_pressed: self.target_pose[3] -= self.KEY_ROT_STEP
            if 'w' in self.keys_pressed: self.target_pose[4] -= self.KEY_ROT_STEP
            if 's' in self.keys_pressed: self.target_pose[4] += self.KEY_ROT_STEP

            if self.scroll_dy != 0:
                self.target_pose[5] += self.scroll_dy * self.SCROLL_SENSITIVITY
                self.scroll_dy = 0

            # 3. 发送运动指令
            self.robot.ServoP(*self.target_pose)

            # 4. 夹爪逻辑
            gripper_tick += 1
            if gripper_tick >= 3:
                if self.mouse_left_pressed:
                    self.target_gripper_pos = max(0, self.target_gripper_pos - self.GRIPPER_STEP)
                    self.robot.gripper.move(int(self.target_gripper_pos))
                elif self.mouse_right_pressed:
                    self.target_gripper_pos = min(1000, self.target_gripper_pos + self.GRIPPER_STEP)
                    self.robot.gripper.move(int(self.target_gripper_pos))
                gripper_tick = 0

            # 5. --- 数据收集逻辑 (关键修改) ---
            # 如果正在录制，我们需要主动获取一次真实的 Robot 状态用于保存
            if self.collector.recording:
                # 注意：GetPose 和 GetAngle 是 Service 调用，会增加循环耗时
                # 如果发现卡顿，可以考虑改为每隔N帧采集一次，或者依赖 target_pose
                real_pose = self.robot.get_pose()
                real_joints = self.robot.get_angle()
                
                # 只有当获取到有效数据时才保存
                if real_pose and real_joints:
                    # 使用当前的目标 action 和真实的 state 进行对齐保存
                    self.collector.process_frame(
                        robot_pose=real_pose,
                        robot_joints=real_joints,
                        gripper_pos=self.target_gripper_pos, # 夹爪通常没有实时回读，用目标值
                        action_target=self.target_pose # 记录动作指令
                    )

            # 6. 频率控制
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.LOOP_RATE) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 退出前保存
        if self.collector.recording:
            self.collector.stop_recording()
        self.robot.close()

if __name__ == "__main__":
    controller = TeleopController()
    controller.run()