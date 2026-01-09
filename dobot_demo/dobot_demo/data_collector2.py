
import rclpy
from rclpy.node import Node
import time
import threading
import re
import json
import sys
import math
import os
import tkinter as tk

# 引入消息类型
from std_msgs.msg import Float64, Int32  # Int32用于录制指令
from geometry_msgs.msg import PointStamped # 带时间戳的消息替代Float64
from dobot_msgs_v3.srv import *
from pynput import keyboard, mouse

class DobotRosWrapper(Node):
    def __init__(self, node_name="dobot_api_client"):
        super().__init__(node_name)
        
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

        # [修改] 使用 PointStamped 以包含 Header (时间戳)
        self.pub_gripper_update = self.create_publisher(PointStamped, "/gripper/command_update", 15)
        
        # [新增] 数据收集指令发布器
        # 0: Idle/Discard, 1: Start Record, 2: Stop & Save
        self.pub_record_cmd = self.create_publisher(Int32, "/recorder/command", 10)

        self.get_logger().info("Waiting for Dobot services...")
        if not self.cli_mov_j.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Dobot services not available yet.")

    def call_service(self, client, request):
        if not client.service_is_ready():
            self.get_logger().error(f"Service {client.srv_name} is not ready.")
            return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.001)
        return future.result()
    
    def call_service_async_no_wait(self, client, request):
        if client.service_is_ready():
            client.call_async(request)

class GripperController:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.id = 0
        self.current_target_pos = 1000
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
            print(f"Gripper Connected, Modbus Index: {self.id}")
        else:
            print(":( Gripper ModbusCreate Failed")
            self.id = 0
        
        if self.id > 0:
            self.enable()
            self.set_force(60)
            self.sync_state_to_relay()

    def enable(self):
        self._write_regs(256, 1, "1")

    def set_force(self, force: int):
        force = max(20, min(100, force))
        self._write_regs(257, 1, str(force))

    def move(self, position: int, wait=False):
        position = max(0, min(1000, position))
        self._write_regs(259, 1, str(position), wait=wait)
        self.current_target_pos = position
        
        # [修改] 发布带时间戳的消息
        msg = PointStamped()
        msg.header.stamp = self.wrapper.get_clock().now().to_msg()
        msg.header.frame_id = "gripper"
        msg.point.x = float(position) # 使用 x 字段存储数值
        self.wrapper.pub_gripper_update.publish(msg)

    def open(self, speed=None, wait=True):
        if speed: self.set_force(speed)
        self.move(1000, wait=wait)

    def close(self, speed=None, wait=True):
        if speed: self.set_force(speed)
        self.move(0, wait=wait)

    def get_run_state(self):
        val = self._read_regs(513, 1)
        return val if val is not None else -1

    def get_current_position(self):
        return self._read_regs(514, 1)

    def is_gripped(self):
        return self.get_run_state() == 2

    def is_moving(self):
        return self.get_run_state() == 0
    
    def sync(self):
        while self.is_moving():
            time.sleep(0.001)

    def sync_state_to_relay(self):
        val = self._read_regs(514, 1)
        if val is not None:
            self.current_target_pos = val
            # [修改] 同步初始状态也带时间戳
            msg = PointStamped()
            msg.header.stamp = self.wrapper.get_clock().now().to_msg()
            msg.header.frame_id = "gripper"
            msg.point.x = float(val)
            self.wrapper.pub_gripper_update.publish(msg)

    def _write_regs(self, addr, count, val_tab, val_type=None, wait=True):
        if self.id <= 0: return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_tab
        if val_type: req.val_type = val_type
        if wait:
            self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)
        else:
            self.wrapper.call_service_async_no_wait(self.wrapper.cli_set_hold_regs, req)

    def _read_regs(self, addr, count, val_type=None):
        if self.id <= 0: return None
        req = GetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        if val_type: req.val_type = val_type
        res = self.wrapper.call_service(self.wrapper.cli_get_hold_regs, req)
        if res and res.res == 0:
            try:
                return int(res.value)
            except:
                pass
        return None

class Robot:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = DobotRosWrapper()
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
        # 避免启动时动作太大，这里可以根据需要保留或注释
        # self.move_to(-100.0, -250.0, 300.0, 180.0, 0.0, 90.0)

    def MovJ(self, x, y, z, rx, ry, rz, *dynParams):
        req = MovJ.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        if dynParams: req.param_value = [str(p) for p in dynParams]
        return self.node.call_service(self.node.cli_mov_j, req)

    def MovL(self, x, y, z, rx, ry, rz, *dynParams):
        req = MovL.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        if dynParams: req.param_value = [str(p) for p in dynParams]
        return self.node.call_service(self.node.cli_mov_l, req)

    def ServoJ(self, j1, j2, j3, j4, j5, j6, t=0.0, param_value=None):
        req = ServoJ.Request()
        req.j1, req.j2, req.j3 = float(j1), float(j2), float(j3)
        req.j4, req.j5, req.j6 = float(j4), float(j5), float(j6)
        req.t = float(t) if t else 0.1
        if param_value: req.param_value = param_value if isinstance(param_value, list) else [param_value]
        return self.node.call_service(self.node.cli_servo_j, req)

    def ServoP(self, x, y, z, rx, ry, rz,):
        req = ServoP.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        return self.node.call_service(self.node.cli_servo_p, req)

    def Sync(self):
        req = Sync.Request()
        return self.node.call_service(self.node.cli_sync, req)

    def get_pose(self):
        req = GetPose.Request()
        res = self.node.call_service(self.node.cli_get_pose, req)
        if res and hasattr(res, 'res'): 
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", res.pose)
            if len(matches) >= 6: return [float(x) for x in matches[:6]]
        return None

    def get_angle(self):
        req = GetAngle.Request()
        res = self.node.call_service(self.node.cli_get_angle, req)
        if res and hasattr(res, 'res'):
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", res.angle)
            if len(matches) >= 6: return [float(x) for x in matches[:6]]
        return None

    def move_to(self, x, y, z, rx, ry, rz, sync=True):
        self.MovJ(x, y, z, rx, ry, rz)
        if sync: self.Sync()

    def close(self):
        rclpy.shutdown()

class TeleopController:
    def __init__(self):
        print("正在初始化机器人链接...")
        self.robot = Robot()
        
        self.LOOP_RATE = 100.0
        self.MOUSE_SENSITIVITY = 0.1
        self.SCROLL_SENSITIVITY = 2.0
        self.KEY_XYZ_STEP = 0.3
        self.KEY_XYZ_STEP_FAST = 0.6
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

        self.kb_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release, suppress=True)
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move, 
            on_click=self._on_mouse_click, 
            on_scroll=self._on_scroll,
            suppress=True
        )
        self.kb_listener.start()
        self.mouse_listener.start()

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
        print("  [O]         : 开始收集数据 (Start)")
        print("  [P]         : 停止并保存 (Save)")
        print("  [L]         : 停止并丢弃 (Discard)")
        print("  ---------------------------------")
        print("  [鼠标]      : XY平移 / 滚轮Rz")
        print("  [Space/Ctrl]: Z轴升降")
        print("  [WASD]      : 姿态旋转")
        print("  [ESC]       : 退出")

    def _init_pose(self):
        for _ in range(5):
            pose = self.robot.get_pose()
            if pose:
                self.target_pose = pose
                g_pos = self.robot.gripper.get_current_position()
                if g_pos is not None:
                    self.target_gripper_pos = g_pos
                return
            time.sleep(0.5)
        print("错误：无法获取机械臂初始位置！")
        sys.exit(1)

    def _on_key_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            
            # [新增] 数据采集按键逻辑
            char_key = None
            if hasattr(key, 'char'):
                char_key = key.char.lower()
                self.keys_pressed.add(char_key)

                # 发布录制指令
                msg = Int32()
                if char_key == 'o':
                    print("\n[Command] START Recording...")
                    msg.data = 1
                    self.robot.node.pub_record_cmd.publish(msg)
                elif char_key == 'p':
                    print("\n[Command] STOP & SAVE...")
                    msg.data = 2
                    self.robot.node.pub_record_cmd.publish(msg)
                elif char_key == 'l':
                    print("\n[Command] DISCARD Data...")
                    msg.data = 0 # 0 也作为停止/丢弃
                    self.robot.node.pub_record_cmd.publish(msg)
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
        time_rate = 1.0

        while self.running:
            start_time = time.time()
            dx = self.mouse_dx
            dy = self.mouse_dy
            self.mouse_dx = 0
            self.mouse_dy = 0

            self.target_pose[0] += dx * self.MOUSE_SENSITIVITY
            self.target_pose[1] -= dy * self.MOUSE_SENSITIVITY

            if keyboard.Key.space in self.keys_pressed:
                step = self.KEY_XYZ_STEP_FAST if (keyboard.Key.alt in self.keys_pressed or keyboard.Key.alt_r in self.keys_pressed) else self.KEY_XYZ_STEP
                self.target_pose[2] += step * time_rate
            if keyboard.Key.ctrl in self.keys_pressed or keyboard.Key.ctrl_l in self.keys_pressed or keyboard.Key.ctrl_r in self.keys_pressed:
                step = self.KEY_XYZ_STEP_FAST if (keyboard.Key.alt in self.keys_pressed or keyboard.Key.alt_r in self.keys_pressed) else self.KEY_XYZ_STEP
                self.target_pose[2] -= step * time_rate

            if 'a' in self.keys_pressed: self.target_pose[3] += self.KEY_ROT_STEP * time_rate
            if 'd' in self.keys_pressed: self.target_pose[3] -= self.KEY_ROT_STEP * time_rate
            if 'w' in self.keys_pressed: self.target_pose[4] -= self.KEY_ROT_STEP * time_rate
            if 's' in self.keys_pressed: self.target_pose[4] += self.KEY_ROT_STEP * time_rate

            if self.scroll_dy != 0:
                self.target_pose[5] += self.scroll_dy * self.SCROLL_SENSITIVITY
                self.scroll_dy = 0

            self.robot.ServoP(*self.target_pose)

            gripper_tick += 1
            if gripper_tick >= 3:
                gripper_changed = False
                time_rate = min(time_rate, 1.1)
                if self.mouse_left_pressed:
                    self.target_gripper_pos = max(0, self.target_gripper_pos - self.GRIPPER_STEP * time_rate)
                    gripper_changed = True
                elif self.mouse_right_pressed:
                    self.target_gripper_pos = min(1000, self.target_gripper_pos + self.GRIPPER_STEP * time_rate)
                    gripper_changed = True
                
                if gripper_changed:
                    self.robot.gripper.move(int(self.target_gripper_pos))
                
                gripper_tick = 0

            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.LOOP_RATE) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                time_rate = 1.0
            else:
                time_rate = elapsed / (1.0 / self.LOOP_RATE) + 1

        self.robot.close()

if __name__ == "__main__":
    controller = TeleopController()
    controller.run()