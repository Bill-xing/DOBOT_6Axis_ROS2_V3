#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import time
import threading
import re
import json

# 引入dobot_msgs_v3的所有服务
from dobot_msgs_v3.srv import *

from dobot_msgs_v3.msg import ToolVectorActual
from sensor_msgs.msg import JointState  
from std_msgs.msg import Float64

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy



class DobotRosWrapper(Node):
    """
    ROS2 Node封装：
    1. Service Client: 用于发送控制指令 (写)
    2. Topic Subscriber: 用于实时获取状态 (读) - 解决卡顿的关键
    """
    def __init__(self, node_name="dobot_api_client"):
        super().__init__(node_name)
        
        # 定义Service Client列表
        self.cli_enable = self.create_client(EnableRobot, '/dobot_bringup_v3/srv/EnableRobot')
        self.cli_clear_error = self.create_client(ClearError, '/dobot_bringup_v3/srv/ClearError')
        self.cli_speed_factor = self.create_client(SpeedFactor, '/dobot_bringup_v3/srv/SpeedFactor')
        
        # 运动指令
        self.cli_mov_j = self.create_client(MovJ, '/dobot_bringup_v3/srv/MovJ')
        self.cli_mov_l = self.create_client(MovL, '/dobot_bringup_v3/srv/MovL') # 对应Windows MovL
        self.cli_servo_j = self.create_client(ServoJ, '/dobot_bringup_v3/srv/ServoJ')
        self.cli_servo_p = self.create_client(ServoP, '/dobot_bringup_v3/srv/ServoP')
        self.cli_sync = self.create_client(Sync, '/dobot_bringup_v3/srv/Sync')
        
        # 获取信息
        self.cli_get_pose = self.create_client(GetPose, '/dobot_bringup_v3/srv/GetPose')
        self.cli_get_angle = self.create_client(GetAngle, '/dobot_bringup_v3/srv/GetAngle')
        
        # 夹爪/Modbus相关
        self.cli_modbus_create = self.create_client(ModbusCreate, '/dobot_bringup_v3/srv/ModbusCreate')
        self.cli_modbus_close = self.create_client(ModbusClose, '/dobot_bringup_v3/srv/ModbusClose')
        self.cli_set_hold_regs = self.create_client(SetHoldRegs, '/dobot_bringup_v3/srv/SetHoldRegs')
        self.cli_get_hold_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs')

        # ================== Subscribers (状态缓存) ==================
        # 缓存变量，初始化为 None
        # self.cache_pose = None
        # self.cache_joints = None
        self.cache_gripper_pos = None


        # # 如果发布者是 Reliable (默认)，订阅者必须也是 Reliable 才能通信
        # # 为了兼容性，我们通常使用 KeepLast(1) + Reliable 即可解决大部分卡顿
        # compatible_qos = QoSProfile(
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=1, # <--- 核心修改：只处理最新数据
        #     reliability=ReliabilityPolicy.RELIABLE
        # )

        # # 订阅机械臂末端位姿 (来自 feedback.py)
        # self.sub_tool = self.create_subscription(
        #     ToolVectorActual, 
        #     '/dobot_msgs_v3/msg/ToolVectorActual', 
        #     self.tool_callback, 
        #     compatible_qos
        # )

        # # 订阅机械臂关节角度
        # self.sub_joints = self.create_subscription(
        #     JointState,
        #     '/joint_states_robot',
        #     self.joint_callback,
        #     compatible_qos
        # )

        self.pub_gripper_update = self.create_publisher(Float64, "/gripper/command_update", 10)

        # 等待服务上线 (简单检查几个关键服务)
        self.get_logger().info("Waiting for Dobot services...")
        if not self.cli_servo_p.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Dobot services not available yet. Ensure driver is running.")

    # # --- 回调函数：只负责更新缓存，极快 ---
    # def tool_callback(self, msg):
    #     # 将 ToolVectorActual 转换为列表 [x, y, z, rx, ry, rz]
    #     self.cache_pose = [msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz]

    # def joint_callback(self, msg):
    #     # 转换为角度列表
    #     self.cache_joints = [p * 180.0 / 3.14159265358979 for p in msg.position]

    def call_service(self, client, request):
        """同步调用服务并返回结果"""
        if not client.service_is_ready():
            self.get_logger().error(f"Service {client.srv_name} is not ready.")
            return None
        
        future = client.call_async(request)
        # 等待结果，这里的wait依赖于外部线程的spin
        # 注意：因为我们有了独立接收线程，这里可以用 future.done() 轮询
        while not future.done():
            time.sleep(0.002) 
        
        return future.result()

    def call_service_async_no_wait(self, client, request):
        """
        [新增] 异步调用服务，不等待结果 (非阻塞)
        适用于高频控制或不需要立即知道结果的场景 (如夹爪动作)
        """
        if not client.service_is_ready():
            # 可以在这里做个简单的日志，或者直接跳过
            return
        # 发送请求后直接返回，不等待 future 完成
        client.call_async(request)

 
class GripperController:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.id = 0
        self.current_target_pos = 1000 # 本地维护一个目标位置缓存
        self.init_gripper_connection()

    def init_gripper_connection(self):
        # ... (初始化 Modbus 连接逻辑保持不变) ...
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
            print(f"Gripper Connected: {self.id}")
        else:
            self.id = 0
        
        if self.id > 0:
            self.enable()
            self.set_force(60)
            # 初始化时，发送一次当前状态给 Relay
            self.sync_state_to_relay() 

    def enable(self):
        self._write_regs(256, 1, "1")

    def set_force(self, force: int):
        force = max(20, min(100, force))
        self._write_regs(257, 1, str(force))

    def move(self, position: int):
        """
        移动夹爪并同步状态到 ROS
        """
        position = max(0, min(1000, position))
        
        # 1. 发送硬件指令 (通过 Service)
        self._write_regs(259, 1, str(position))
        
        # 2. 更新本地缓存
        self.current_target_pos = position
        
        # 3. 发送 ROS 消息通知 JointStateRelay 更新状态
        # msg = Float64()
        # msg.data = float(position)
        # self.wrapper.pub_gripper_update.publish(msg)

    def open(self, speed=None):
        if speed: self.set_force(speed)
        self.move(1000)

    def close(self, speed=None):
        if speed: self.set_force(speed)
        self.move(0)

    def sync_state_to_relay(self):
        """强制发送一次当前位置给 Relay"""
        # 只有在初始化时调用一次读取，为了获取当前真实的物理位置
        val = self._read_regs(514, 1)
        if val is not None:
            self.current_target_pos = val
            msg = Float64()
            msg.data = float(val)
            self.wrapper.pub_gripper_update.publish(msg)

    # === 获取状态：直接返回本地指令记录，不再读硬件 ===
    def get_current_position(self):
        return self.current_target_pos

    def _write_regs(self, addr, count, val_tab):
        if self.id <= 0: return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_tab
        # 使用 call_async 且不等待，进一步降低延迟
        self.wrapper.cli_set_hold_regs.call_async(req)

    def _read_regs(self, addr, count):
        # 仅在初始化使用
        if self.id <= 0: return None
        req = GetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        res = self.wrapper.call_service(self.wrapper.cli_get_hold_regs, req)
        if res and res.res == 0:
            try: return int(res.value)
            except: pass
        return None


class Robot:
    """
    机器人接口类
    """
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        
        self.node = DobotRosWrapper()
        
        # 后台线程运行 ROS Spin，负责接收 Topic 和处理 Service 回调
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()
        
        print(">.< 正在连接 ROS 服务... >.<")
        
        # # 等待第一次数据到达，确保 get_pose 能拿到值
        # self._wait_for_initial_data()

        try:
            self.gripper = GripperController(self.node)
        except Exception as e:
            print(f":( 夹爪初始化警告: {e}")

        self.enable_robot()
        self.set_speed_factor(100)
        self.clear_error()
        print(">.< 连接成功! >.<")

    # def _wait_for_initial_data(self):
    #     print("等待初始位姿数据...")
    #     timeout = 5.0
    #     start = time.time()
    #     while self.node.cache_pose is None:
    #         time.sleep(0.1)
    #         if time.time() - start > timeout:
    #             print("警告: 未收到机械臂位姿反馈，可能是 feedback 节点未启动")
    #             break

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

    # ------------------ 运动封装 (写操作，走 Service) ------------------

    def ServoP(self, x, y, z, rx, ry, rz):
        req = ServoP.Request()
        req.x = float(x)
        req.y = float(y)
        req.z = float(z)
        req.rx = float(rx)
        req.ry = float(ry)
        req.rz = float(rz)
        # ServoP 不需要等待返回结果即可发送下一个，为了流畅性
        # 使用 call_async 但不等待 future，实现“发后即忘”
        self.node.cli_servo_p.call_async(req)

    def MovJ(self, x, y, z, rx, ry, rz, *dynParams):
        req = MovJ.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        if dynParams:
            req.param_value = [str(p) for p in dynParams]
        return self.node.call_service(self.node.cli_mov_j, req)

    def MovL(self, x, y, z, rx, ry, rz, *dynParams):
        req = MovL.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        if dynParams:
            req.param_value = [str(p) for p in dynParams]
        return self.node.call_service(self.node.cli_mov_l, req)
    
    def Sync(self):
        req = Sync.Request()
        return self.node.call_service(self.node.cli_sync, req)

    # ------------------ 读状态 (核心修改：读缓存) ------------------

    # def get_pose(self):
    #     """
    #     获取笛卡尔坐标：直接返回缓存
    #     Return: [x, y, z, rx, ry, rz]
    #     """
    #     return self.node.cache_pose

    # def get_angle(self):
    #     """
    #     获取关节角：直接返回缓存
    #     Return: [j1, j2, j3, j4, j5, j6]
    #     """
    #     return self.node.cache_joints

    def get_pose(self):
        """
        获取笛卡尔坐标 (P)
        Return: [x, y, z, rx, ry, rz]
        """
        req = GetPose.Request()
        res = self.node.call_service(self.node.cli_get_pose, req)
        
        # 尝试解析结果，根据实际srv定义调整
        # 通常 res.res 是一个类似 "{x,y,z,rx,ry,rz}" 的字符串
        if res and hasattr(res, 'res'): 
            pose_str = res.pose
            # 使用正则提取数字
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", pose_str)
            if len(matches) >= 6:
                return [float(x) for x in matches[:6]]
        
        print("GetPose 解析失败，原始返回:", res)
        return None

    def get_angle(self):
        """
        获取关节角 (J)
        Return: [j1, j2, j3, j4, j5, j6]
        """
        req = GetAngle.Request()
        res = self.node.call_service(self.node.cli_get_angle, req)
        
        if res and hasattr(res, 'res'):
            angle_str = res.angle
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", angle_str)
            if len(matches) >= 6:
                return [float(x) for x in matches[:6]]
        
        print("GetAngle 解析失败，原始返回:", res)
        return None

    # ------------------ 其他 ------------------
    def close(self):
        rclpy.shutdown()
        self.Sync()


import time
import threading
import sys
import math

# 引入 pynput 用于监听键盘鼠标
from pynput import keyboard, mouse
import os
import tkinter as tk  # 用于获取屏幕尺寸

class TeleopController:
    def __init__(self):
        # 初始化机器人
        print("正在初始化机器人链接...")
        self.robot = Robot()
        
        # --- 控制参数配置 ---
        self.LOOP_RATE = 100.0       # 控制频率 (Hz), Dobot Servo 建议 30-50Hz
        self.MOUSE_SENSITIVITY = 0.1 # 鼠标移动 -> mm 的比例
        self.SCROLL_SENSITIVITY = 2.0 # 滚轮 -> 角度的比例
        self.KEY_XYZ_STEP = 0.3     # 键盘控制 Z (Space/Ctrl) 的步长 (mm)
        self.KEY_ROT_STEP = 0.3     # 键盘控制旋转 (WASD) 的步长 (度)
        self.GRIPPER_STEP = 40      # 夹爪每次按下的变化量 (0-1000)

        # --- 状态变量 ---
        self.running = True
        self.target_pose = [0.0] * 6  # [x, y, z, rx, ry, rz]
        self.target_gripper_pos = 1000 # 0(闭合) - 1000(张开)
        
        # 输入状态缓存
        self.keys_pressed = set()
        self.mouse_dx = 0
        self.mouse_dy = 0
        self.scroll_dy = 0
        self.mouse_left_pressed = False
        self.mouse_right_pressed = False

        # 初始化当前位置
        self._init_pose()

        # 启动监听线程
        self.kb_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release, suppress=True)
        # self.mouse_listener = mouse.Listener(on_move=self._on_mouse_move, on_click=self._on_mouse_click, on_scroll=self._on_scroll)
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move, 
            on_click=self._on_mouse_click, 
            on_scroll=self._on_scroll,
            suppress=True  # <--- 核心修改：屏蔽物理鼠标对系统的操作
        )
        
        self.kb_listener.start()
        self.mouse_listener.start()

        # --- 新增：获取屏幕中心点 ---
        root = tk.Tk()
        self.center_x = root.winfo_screenwidth() // 2
        self.center_y = root.winfo_screenheight() // 2
        root.destroy()
        
        # --- 新增：尝试将当前终端窗口置顶并激活 (依赖 xdotool) ---
        # 搜索当前激活的窗口并使其保持在最前 (Stack Mode Above)
        try:
            os.system("xdotool getactivewindow windowactivate")
            # 可选：如果你想让它强制在最上层不被覆盖，取消下面这行的注释
            # os.system("xdotool getactivewindow windowstate --add ABOVE") 
        except Exception:
            pass
        
        # 初始化鼠标控制器
        self.mouse_controller = mouse.Controller()
        # 初始时将鼠标瞬移到中心
        self.mouse_controller.position = (self.center_x, self.center_y)

        print("=== 控制已启动 ===")
        print("  [鼠标移动]  : 前后左右平移 (XY)")
        print("  [Space/Ctrl]: 上下平移 (Z)")
        print("  [A / D]     : 旋转 Rx (俯仰)")
        print("  [W / S]     : 旋转 Ry (横滚)")
        print("  [鼠标滚轮]  : 旋转 Rz (最后一维关节)")
        print("  [鼠标左键]  : 夹爪闭合 (长按)")
        print("  [鼠标右键]  : 夹爪张开 (长按)")
        print("  [ESC]       : 退出")

    def _init_pose(self):
        """获取初始姿态"""
        retries = 0
        while retries < 5:
            print(111)
            pose = self.robot.get_pose()
            print(222)
            if pose:
                self.target_pose = pose
                # 读取夹爪当前位置，如果读取失败默认为1000
                g_pos = self.robot.gripper.get_current_position()
                print(333)
                if g_pos is not None:
                    self.target_gripper_pos = g_pos
                return
            time.sleep(0.5)
            retries += 1
        print("错误：无法获取机械臂初始位置！")
        sys.exit(1)

    # --- 输入回调函数 ---
    def _on_key_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            if hasattr(key, 'char'):
                self.keys_pressed.add(key.char.lower())
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

    # def _on_mouse_move(self, x, y):
    #     # 这里的 dx, dy 是相对屏幕中心的偏移，或者是与上一帧的差值
    #     # pynput 的 on_move 给的是绝对坐标，我们需要手动计算差值
    #     # 为了简化，我们在主循环中重置中心点，或者使用全局差值逻辑
    #     # 但在 Linux 无窗口模式下，通常很难锁定鼠标。
    #     # 这里采用 差值累加 的方式，需要配合专门的逻辑（见 run 方法）
    #     pass 
    #     # 注意：pynput 在 Linux 下直接获取的是绝对坐标。
    #     # 为了实现“无限移动”的效果，通常需要 reset pointer。
    #     # 但为了简化代码，我们假设用户是在移动鼠标，我们计算 delta。

    def _on_mouse_move(self, x, y):
        # === 修改位置：在回调中计算位移并回中 ===
        # 计算相对于屏幕中心的位移
        dx = x - self.center_x
        dy = y - self.center_y
        
        # 累加位移量给主循环使用
        self.mouse_dx += dx
        self.mouse_dy += dy
        
        # 如果有位移，手动将“虚拟鼠标”重置回中心
        # 这样做是为了防止内部坐标撞到屏幕边缘导致无法继续旋转
        if dx != 0 or dy != 0:
            self.mouse_controller.position = (self.center_x, self.center_y)
    
    def _on_mouse_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            self.mouse_left_pressed = pressed
        elif button == mouse.Button.right:
            self.mouse_right_pressed = pressed

    def _on_scroll(self, x, y, dx, dy):
        self.scroll_dy += dy

    # --- 主控制循环 ---
    def run(self):
        # 初始化鼠标位置跟踪
        mouse_controller = mouse.Controller()
        last_mouse_pos = mouse_controller.position
        
        # 夹爪控制计时器（降低通信频率）
        gripper_tick = 0 
        time_rate = 1.0

        while self.running:
            start_time = time.time()

            # 1. 计算鼠标位移 (Delta)
            # current_mouse_pos = mouse_controller.position
            # dx = current_mouse_pos[0] - last_mouse_pos[0]
            # dy = current_mouse_pos[1] - last_mouse_pos[1]
            # last_mouse_pos = current_mouse_pos # 更新位置

            # # 1. 获取当前位置
            # current_pos = self.mouse_controller.position
            # # 2. 计算相对于屏幕中心的位移 (Delta)
            # dx = current_pos[0] - self.center_x
            # dy = current_pos[1] - self.center_y
            # # 3. 只有当鼠标真的移动了才执行，避免抖动
            # if dx != 0 or dy != 0:
            #     # 强制将鼠标重置回屏幕中心
            #     self.mouse_controller.position = (self.center_x, self.center_y)

            # 取出累计的位移量
            dx = self.mouse_dx
            dy = self.mouse_dy
            
            # 清零，准备下一帧累加
            self.mouse_dx = 0
            self.mouse_dy = 0

            # --- 姿态计算逻辑 (XYZ - 笛卡尔移动) ---
            
            # 鼠标控制 XY 平面
            # 假设：鼠标向上(dy<0) -> 机械臂向前(Y+) 或 X+，根据实际安装方向调整
            # 这里默认：鼠标上 -> Y+, 鼠标右 -> X+
            self.target_pose[0] += dx * self.MOUSE_SENSITIVITY  # X
            self.target_pose[1] -= dy * self.MOUSE_SENSITIVITY  # Y (屏幕坐标系Y向下，所以取反)

            # 键盘控制 Z 轴
            if keyboard.Key.space in self.keys_pressed:
                self.target_pose[2] += self.KEY_XYZ_STEP * time_rate
            if keyboard.Key.ctrl in self.keys_pressed or \
               keyboard.Key.ctrl_l in self.keys_pressed or \
               keyboard.Key.ctrl_r in self.keys_pressed:
                self.target_pose[2] -= self.KEY_XYZ_STEP * time_rate

            # --- 旋转计算逻辑 (Rx Ry Rz) ---

            # WSAD 控制 Rx / Ry (末端两自由度旋转)
            # W/S -> Rx (Pitch)
            if 'a' in self.keys_pressed:
                self.target_pose[3] += self.KEY_ROT_STEP * time_rate
            if 'd' in self.keys_pressed:
                self.target_pose[3] -= self.KEY_ROT_STEP * time_rate
            
            # A/D -> Ry (Roll)
            if 'w' in self.keys_pressed:
                self.target_pose[4] -= self.KEY_ROT_STEP * time_rate
            if 's' in self.keys_pressed:
                self.target_pose[4] += self.KEY_ROT_STEP * time_rate

            # 鼠标滚轮 -> Rz (最后一维关节旋转 / Yaw)
            if self.scroll_dy != 0:
                self.target_pose[5] += self.scroll_dy * self.SCROLL_SENSITIVITY
                self.scroll_dy = 0 # 消费掉事件

            # --- 发送运动指令 (ServoP) ---
            # 使用 ServoP 实现平滑的笛卡尔空间运动
            # 格式: x, y, z, rx, ry, rz
            self.robot.ServoP(
                self.target_pose[0],
                self.target_pose[1],
                self.target_pose[2],
                self.target_pose[3],
                self.target_pose[4],
                self.target_pose[5]
            )

            # --- 夹爪控制逻辑 ---
            # 夹爪通过 Modbus 通信，频率不宜过高，每 3 帧处理一次 (约 10Hz)
            gripper_tick += 1
            if gripper_tick >= 3:
                gripper_changed = False
                time_rate = min(time_rate, 1.1)
                if self.mouse_left_pressed:
                    # 左键闭合 (数值减小)
                    self.target_gripper_pos = max(0, self.target_gripper_pos - self.GRIPPER_STEP * time_rate)
                    gripper_changed = True
                elif self.mouse_right_pressed:
                    # 右键张开 (数值增加)
                    self.target_gripper_pos = min(1000, self.target_gripper_pos + self.GRIPPER_STEP * time_rate)
                    gripper_changed = True
                
                if gripper_changed:
                    # 发送 Modbus 指令 (非阻塞最好，但这里 wrapper 是阻塞的，所以要降频)
                    self.robot.gripper.move(int(self.target_gripper_pos))
                
                gripper_tick = 0

            # --- 循环频率控制 ---
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.LOOP_RATE) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                time_rate = 1.0
            else:
                time_rate = elapsed / (1.0 / self.LOOP_RATE) + 1

        # 退出清理
        self.robot.close()

if __name__ == "__main__":
    controller = TeleopController()
    controller.run()