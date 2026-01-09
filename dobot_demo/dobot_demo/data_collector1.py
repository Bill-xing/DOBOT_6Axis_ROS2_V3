# 控制脚本baseline 发布夹爪信息（不带时间戳） 配合joint_states新版本

import rclpy
from rclpy.node import Node
import time
import threading
import re
import json
from std_msgs.msg import Float64

# 引入dobot_msgs_v3的所有服务
from dobot_msgs_v3.srv import *

class DobotRosWrapper(Node):
    """
    ROS2 Node封装，用于处理所有Service通信
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


        self.pub_gripper_update = self.create_publisher(Float64, "/gripper/command_update", 2)

        # 等待服务上线 (简单检查几个关键服务)
        self.get_logger().info("Waiting for Dobot services...")
        if not self.cli_mov_j.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Dobot services not available yet. Please ensure driver is running.")

    def call_service(self, client, request):
        """同步调用服务并返回结果"""
        if not client.service_is_ready():
            self.get_logger().error(f"Service {client.srv_name} is not ready.")
            return None
        
        future = client.call_async(request)
        # 等待结果，这里的wait依赖于外部线程的spin，或者我们手动等待
        while not future.done():
            time.sleep(0.001)
        
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
    """
    DH-Robotics AG系列夹爪控制器 (Modbus-RTU)
    适配 ROS 2 Service 架构
    """
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.id = 0
        self.init_gripper_connection()

    def init_gripper_connection(self):
        # 1. 关闭旧连接 (1-4) 防止占用
        for i in range(1, 5):
            req = ModbusClose.Request()
            req.index = i
            self.wrapper.call_service(self.wrapper.cli_modbus_close, req)
        
        # 2. 创建 Modbus 连接 (115200, 8, 1, N)
        req = ModbusCreate.Request()
        req.ip = "127.0.0.1"
        req.port = 60000
        req.slave_id = 1
        req.is_rtu = 1
        
        res = self.wrapper.call_service(self.wrapper.cli_modbus_create, req)
        
        if res and res.res == 0:
            try:
                self.id = int(res.index)
            except (ValueError, TypeError):
                # 兼容可能的字符串返回
                match = re.search(r'(\d+)', str(res.index))
                self.id = int(match.group(1)) if match else 0

            print(f"Gripper Connected, Modbus Index: {self.id}")
        else:
            print(":( Gripper ModbusCreate Failed")
            self.id = 0
        
        if self.id > 0:
            self.enable()     # 初始化使能
            self.set_force(60) # 设置默认力度/速度
            self.sync_state_to_relay()

    # ================= 核心写指令 =================

    def enable(self):
        """初始化夹爪 (0x0100 -> 256)"""
        # 写入 1 进行初始化
        self._write_regs(256, 1, "1")
        # 写入 0xA5 (165) 可以重新标定回零 (如果更换指尖需要用这个)
        # self._write_regs(256, 1, "165") 

    def set_force(self, force: int):
        """
        设置夹持力/速度 (0x0101 -> 257)
        :param force: 20-100 (%)
        """
        if not (20 <= force <= 100):
            print("Force must be between 20 and 100")
            force = max(20, min(100, force))
        self._write_regs(257, 1, str(force))

    def move(self, position: int, wait=False):
        """
        动态控制开合位置 (0x0103 -> 259)
        :param position: 0-1000 (千分比)
                         0 = 完全闭合
                         1000 = 完全张开
        """
        if not (0 <= position <= 1000):
            print("Position must be between 0 and 1000")
            position = max(0, min(1000, position))
        self._write_regs(259, 1, str(position), wait=wait)
        self.current_target_pos = position
        msg = Float64()
        msg.data = float(position)
        self.wrapper.pub_gripper_update.publish(msg)

    def open(self, speed=None, wait=True):
        """完全张开 (或张开到指定上限)"""
        if speed: self.set_force(speed)
        # 用户之前提到800，这里保留灵活性，默认1000
        self.move(1000, wait=wait) 

    def close(self, speed=None, wait=True):
        """完全闭合"""
        if speed: self.set_force(speed)
        self.move(0, wait=wait)

    # ================= 核心读指令 (状态反馈) =================

    def get_run_state(self):
        """
        获取夹持状态 (0x0201 -> 513)
        Return:
            0: 运动中
            1: 到达位置 (未夹到物体)
            2: 夹住物体 (成功抓取)
            3: 物体掉落
            -1: 读取失败
        """
        val = self._read_regs(513, 1)
        return val if val is not None else -1

    def get_current_position(self):
        """
        获取实时位置 (0x0202 -> 514)
        Return: 0-1000
        """
        return self._read_regs(514, 1)

    def is_gripped(self):
        """判断是否成功夹住物体"""
        # 状态2代表夹住物体
        return self.get_run_state() == 2

    def is_moving(self):
        """判断是否正在运动"""
        return self.get_run_state() == 0
    
    def sync(self):
        while self.is_moving():
            time.sleep(0.001)

    def sync_state_to_relay(self):
        """强制发送一次当前位置给 Relay"""
        # 只有在初始化时调用一次读取，为了获取当前真实的物理位置
        val = self._read_regs(514, 1)
        if val is not None:
            self.current_target_pos = val
            msg = Float64()
            msg.data = float(val)
            self.wrapper.pub_gripper_update.publish(msg)

    # ================= 底层 Modbus 封装 =================

    def _write_regs(self, addr, count, val_tab, val_type=None, wait=True):
        if self.id <= 0: return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_tab
        if val_type:
            req.val_type = val_type
        if wait:
            # 只有初始化时需要等待
            self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)
        else:
            # [核心修改] 遥操作运动时，使用不等待的调用
            self.wrapper.call_service_async_no_wait(self.wrapper.cli_set_hold_regs, req)

    def _read_regs(self, addr, count, val_type=None):
        """
        读取保持寄存器并解析返回值
        """
        if self.id <= 0: return None
        
        # Dobot 的 GetHoldRegs 服务通常需要通过 GetInRegs 或者特定的 Read 服务
        # 注意：在 dobot_msgs_v3 中通常用 GetHoldRegs
        req = GetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        if val_type: req.val_type = val_type
        
        res = self.wrapper.call_service(self.wrapper.cli_get_hold_regs, req)
        if res and res.res == 0: # 0 表示成功
            try:
                return int(res.value)
            except:
                pass
        return None

class Robot:
    """
    最外层调用接口，模拟 Windows API 风格
    """
    def __init__(self):
        # 初始化 ROS 2
        if not rclpy.ok():
            rclpy.init()
        
        self.node = DobotRosWrapper()
        
        # 在后台线程运行 ROS 节点，避免阻塞主程序
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()
        
        print(">.< 正在连接 ROS 服务... >.<")
        
        # 初始化夹爪
        try:
            self.gripper = GripperController(self.node)
        except Exception as e:
            print(f":( 夹爪初始化警告: {e}")

        # 机器人初始化流程
        self.enable_robot()
        self.set_speed_factor(100)
        self.clear_error()
        self.position_init()
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

    def position_init(self):
        self.clear_error()
        # 初始位置示例
        self.move_to(-100.0, -250.0, 300.0, 180.0, 0.0, 90.0)

    # ------------------ 核心运动封装 ------------------

    def MovJ(self, x, y, z, rx, ry, rz, *dynParams):
        req = MovJ.Request()
        req.x = float(x)
        req.y = float(y)
        req.z = float(z)
        req.rx = float(rx)
        req.ry = float(ry)
        req.rz = float(rz)
        if dynParams:
            req.param_value = [str(p) for p in dynParams]
        return self.node.call_service(self.node.cli_mov_j, req)

    def MovL(self, x, y, z, rx, ry, rz, *dynParams):
        # Windows API 中有人叫 MovP，但在 PDF 文档中对应直线运动通常是 MovL
        req = MovL.Request()
        req.x = float(x)
        req.y = float(y)
        req.z = float(z)
        req.rx = float(rx)
        req.ry = float(ry)
        req.rz = float(rz)
        if dynParams:
            req.param_value = [str(p) for p in dynParams]
        return self.node.call_service(self.node.cli_mov_l, req)

    def ServoJ(self, j1, j2, j3, j4, j5, j6, t=0.0, param_value=None):
        req = ServoJ.Request()
        req.j1 = float(j1)
        req.j2 = float(j2)
        req.j3 = float(j3)
        req.j4 = float(j4)
        req.j5 = float(j5)
        req.j6 = float(j6)
        req.t = float(t) if t else 0.1
        if param_value:
             req.param_value = param_value if isinstance(param_value, list) else [param_value]
        return self.node.call_service(self.node.cli_servo_j, req)

    def ServoP(self, x, y, z, rx, ry, rz,):
        req = ServoP.Request()
        req.x = float(x)
        req.y = float(y)
        req.z = float(z)
        req.rx = float(rx)
        req.ry = float(ry)
        req.rz = float(rz)
        return self.node.call_service(self.node.cli_servo_p, req)

    def Sync(self):
        req = Sync.Request()
        return self.node.call_service(self.node.cli_sync, req)

    # ------------------ 获取姿态 ------------------

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

    # ------------------ 高级封装 ------------------

    def move_to(self, x, y, z, rx, ry, rz, sync=True):
        self.MovJ(x, y, z, rx, ry, rz)
        if sync:
            self.Sync()

    def pick(self, x, y, z, rx, ry, rz, add_z=0, sync=True):
        # 移动到上方
        self.move_to(x, y, z + 100, rx, ry, rz, sync=sync)
        # 下降取货
        self.move_to(x, y, z + add_z, rx, ry, rz, sync=True)
        self.gripper.close()
        # 抬起
        self.move_to(x, y, z + 100, rx, ry, rz, sync=sync)

    def place(self, x, y, z, rx, ry, rz, sync=True):
        # 移动到上方
        self.move_to(x, y, z + 100, rx, ry, rz, sync=sync)
        # 下降放货
        self.move_to(x, y, z, rx, ry, rz, sync=True)
        self.gripper.open()
        # 抬起
        self.move_to(x, y, z + 100, rx, ry, rz, sync=sync)

    def close(self):
        rclpy.shutdown()
        self.Sync()


# # ================= 使用示例 =================
# if __name__ == '__main__':
#     try:
#         robot = Robot()
        
#         # 获取当前姿态
#         print("Current Pose:", robot.get_pose())
#         print("Current Joints:", robot.get_angle())


#         robot.gripper.sync()
#         robot.gripper.close()
#         robot.gripper.sync()
#         print('close')
        
#         # Servo 测试
#         for _ in range(10):
#             joints = robot.get_angle()
#             robot.ServoJ(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]+1)
#             time.sleep(0.03)
#         print('servoj')

#         robot.gripper.open()
#         robot.gripper.sync()
#         print('open')

#         for _ in range(20):
#             poses = robot.get_pose()
#             robot.ServoP(poses[0], poses[1], poses[2], poses[3]+1, poses[4], poses[5])
#             time.sleep(0.03)
#         print('servop')

#         robot.gripper.close()
#         robot.gripper.sync()

#         for i in range(100):
#             p = robot.gripper.get_current_position()
#             robot.gripper.move(p+10)
#             # robot.gripper.sync()
        
#     except KeyboardInterrupt:
#         print("Exiting...")
#     finally:
#         if 'robot' in locals():
#             robot.close()


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
        self.KEY_XYZ_STEP_FAST = 0.6     # 键盘控制 Z (Space/Ctrl) 的步长 (mm)
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
            pose = self.robot.get_pose()
            if pose:
                self.target_pose = pose
                # 读取夹爪当前位置，如果读取失败默认为1000
                g_pos = self.robot.gripper.get_current_position()
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
                if keyboard.Key.alt in self.keys_pressed or keyboard.Key.alt_r in self.keys_pressed:
                    self.target_pose[2] += (self.KEY_XYZ_STEP_FAST - self.KEY_XYZ_STEP) * time_rate
            if keyboard.Key.ctrl in self.keys_pressed or \
               keyboard.Key.ctrl_l in self.keys_pressed or \
               keyboard.Key.ctrl_r in self.keys_pressed:
                self.target_pose[2] -= self.KEY_XYZ_STEP * time_rate
                if keyboard.Key.alt in self.keys_pressed or keyboard.Key.alt_r in self.keys_pressed:
                    self.target_pose[2] -= (self.KEY_XYZ_STEP_FAST - self.KEY_XYZ_STEP) * time_rate

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