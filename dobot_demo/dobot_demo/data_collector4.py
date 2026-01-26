# 新版本：夹爪信息由控制脚本做增量 加入时间戳 夹爪单独开了一个线程
# 夹爪没有时间戳 如果要加快夹爪速度 要改joint_states里的补偿参数
# [新增] 垂直抓取功能 (Z键触发)

import rclpy
from rclpy.node import Node
import time
import threading
import re
import sys
import math
import os
import tkinter as tk
from queue import Queue, Empty

# 引入消息类型
from std_msgs.msg import Int32
from geometry_msgs.msg import PointStamped
from dobot_msgs_v3.srv import *
from dobot_msgs_v3.msg import ToolVectorActual
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

        # 使用 PointStamped 发布夹爪状态（x字段存数值，header存时间戳）
        self.pub_gripper_update = self.create_publisher(PointStamped, "/gripper/command_update", 5)

        # [新增] 发布夹爪实际状态反馈
        self.pub_gripper_state = self.create_publisher(PointStamped, "/gripper/state_feedback", 5)

        # [新增] 发布机械臂目标状态 (用于VLA数据采集)
        self.pub_robot_target = self.create_publisher(ToolVectorActual, "/robot/target_pose", 10)

        # 数据收集指令发布器
        self.pub_record_cmd = self.create_publisher(Int32, "/recorder/command", 10)

        self.get_logger().info("Waiting for Dobot services...")
        if not self.cli_mov_j.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Dobot services not available yet.")

    def call_service(self, client, request):
        if not client.service_is_ready():
            return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.001)
        return future.result()
    
    def call_service_async_no_wait(self, client, request):
        """异步发送不等待结果，用于高频写指令"""
        if client.service_is_ready():
            client.call_async(request)

class GripperComm:
    """
    底层 Modbus 通信类，只负责发协议，不含控制逻辑
    """
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.id = 0
        self.init_connection()

    def init_connection(self):
        # 关闭旧连接
        print("[DEBUG] 正在关闭旧的Modbus连接...")
        for i in range(1, 5):
            req = ModbusClose.Request()
            req.index = i
            self.wrapper.call_service(self.wrapper.cli_modbus_close, req)

        # 创建连接
        print("[DEBUG] 正在创建新的Modbus连接...")
        req = ModbusCreate.Request()
        req.ip = "127.0.0.1"
        req.port = 60000
        req.slave_id = 1
        req.is_rtu = 1
        res = self.wrapper.call_service(self.wrapper.cli_modbus_create, req)

        print(f"[DEBUG] ModbusCreate 响应: res={res}, res.res={res.res if res else 'None'}, res.index={res.index if res else 'None'}")

        if res and res.res == 0:
            match = re.search(r'(\d+)', str(res.index))
            self.id = int(match.group(1)) if match else int(res.index)
            print(f"[SUCCESS] Gripper Connected, ID: {self.id}")
        else:
            print(f"[ERROR] Gripper ModbusCreate Failed! Response: {res}")
            self.id = 0

        if self.id > 0:
            print("[DEBUG] 初始化夹爪寄存器...")
            self.write_reg(256, 1, "1", wait=True) # Enable
            print("[DEBUG] 寄存器256 (Enable) = 1")
            self.write_reg(257, 1, "60", wait=True) # Force/Speed
            print("[DEBUG] 寄存器257 (Force/Speed) = 100")

    def write_reg(self, addr, count, val_str, wait=False):
        if self.id <= 0: return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_str  # 保持字符串类型（ROS消息定义要求）
        if wait:
            self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)
        else:
            self.wrapper.call_service_async_no_wait(self.wrapper.cli_set_hold_regs, req)

    def read_reg(self, addr):
        if self.id <= 0: return None
        req = GetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = 1
        res = self.wrapper.call_service(self.wrapper.cli_get_hold_regs, req)
        if res and res.res == 0:
            try:
                return int(res.value)
            except:
                pass
        return None

class GripperManager(threading.Thread):
    """
    [新架构] 独立的夹爪控制进程（线程）
    完全负责：状态维护、指令发送、真实值回读
    """
    def __init__(self, wrapper, mouse_state):
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.comm = GripperComm(wrapper)
        self.mouse_state = mouse_state  # 引用共享的鼠标状态
        self.running = True
        
        self.current_pos = 1000.0 # 0-1000
        self.GRIPPER_STEP = 80
        
        # 状态机标志
        self.was_pressed = False
        
        # [新增] 自动控制标志位
        self.auto_mode = False
        self.target_auto_pos = 1000.0

        # 初始化同步一次真实位置
        init_val = self.comm.read_reg(514)
        if init_val is not None:
            self.current_pos = float(init_val)
            # 注释掉：现在由 GripperStateFeedback 线程统一发布
            # self.publish_state(self.current_pos)

    def publish_state(self, pos):
        """发布带时间戳的夹爪状态"""
        msg = PointStamped()
        msg.header.stamp = self.wrapper.get_clock().now().to_msg()
        msg.header.frame_id = "gripper"
        msg.point.x = float(pos)
        self.wrapper.pub_gripper_update.publish(msg)

    def set_auto_mode(self, enabled, target_pos=None):
        """主线程调用此方法开启/关闭自动模式"""
        self.auto_mode = enabled
        if target_pos is not None:
            self.target_auto_pos = float(target_pos)
    
    def move(self, place):
        self.current_pos = place
        self.comm.write_reg(259, 1, str(int(self.current_pos)), wait=False)

    def run(self):
        """
        核心循环：
        1. 鼠标按下期间：根据按键改变虚拟值 -> 发布消息 -> 发送 Modbus (非阻塞)
        2. [新增] 自动模式：由脚本逻辑控制位置
        3. 鼠标松开后：进入回读模式，循环读取硬件直到稳定
        """
        read_cnt = 0
        while self.running:
            # [新增] 优先处理自动模式
            if self.auto_mode:
                # 在自动模式下，直接驱动到目标位置
                # 为了防止指令过于密集，检查是否有变化或低频刷新
                self.current_pos = self.target_auto_pos
                self.comm.write_reg(259, 1, str(int(self.current_pos)), wait=False)
                # 注释掉：现在由 GripperStateFeedback 线程统一发布
                # self.publish_state(self.current_pos)
                time.sleep(0.02) # 50Hz 控制频率
                continue

            read_cnt += 1
            # 获取当前按键状态 (线程安全读取)
            left_pressed = self.mouse_state.left_pressed
            right_pressed = self.mouse_state.right_pressed
            is_pressed = left_pressed or right_pressed

            if is_pressed:
                # === 阶段 A: 持续按下，仅发送写指令 ===
                self.was_pressed = True
                changed = False

                # 计算新位置 (开环控制)
                if left_pressed: # 闭合
                    self.current_pos = max(0.0, self.current_pos - self.GRIPPER_STEP) # 稍微降速适配循环频率
                    changed = True
                elif right_pressed: # 张开
                    self.current_pos = min(1000.0, self.current_pos + self.GRIPPER_STEP)
                    changed = True
                
                if changed:
                    # 2. 发送 Modbus 写指令 (非阻塞，不等待结果)
                    self.comm.write_reg(259, 1, str(int(self.current_pos)), wait=True)
                    # 注释掉：现在由 GripperStateFeedback 线程统一发布
                    # self.publish_state(self.current_pos)
                
                # 保持一定的指令频率，约 10Hz (原逻辑) - 这里改为0.05
                time.sleep(0.05)

            else:
                read_cnt = 0
                # === 阶段 B: 松开按键，同步真实状态 ===
                if self.was_pressed:
                    # 刚松开，等待 20ms 后开始读取
                    time.sleep(0.02)
                    
                    # 进入回读循环 (Blocking Read)
                    stable_count = 0
                    last_read_val = -1
                    
                    # 尝试读取直到稳定 (或者最多读 1秒，防止死循环)
                    for t_after in range(50): 
                        # 如果用户又按下了，立即退出回读
                        if self.mouse_state.left_pressed or self.mouse_state.right_pressed or self.auto_mode:
                            break
                        
                        # 实际控制0.1频率
                        if t_after % 5 != 4:
                            break
                        real_val = self.comm.read_reg(514) # 514: 实时位置寄存器
                        if real_val is not None:
                            # 更新真实值到 current_pos
                            self.current_pos = float(real_val)
                            # 注释掉：现在由 GripperStateFeedback 线程统一发布
                            # self.publish_state(self.current_pos)
                            
                            # 检查是否稳定
                            if abs(real_val - last_read_val) < 5: # 误差容忍度
                                stable_count += 1
                            else:
                                stable_count = 0
                            
                            if stable_count > 2: # 连续2次读数一致，认为稳定
                                break
                            
                            last_read_val = real_val
                        
                        time.sleep(0.02)
                    
                    self.was_pressed = False
                
                # 空闲状态，低频休眠
                time.sleep(0.01)

class GripperStateFeedback(threading.Thread):
    """
    [新增] 独立的夹爪状态反馈线程
    专门负责定期读取夹爪实际位置并发布当前状态和目标状态
    使用高频发布(100Hz)+低频采样(10Hz)+基于时间的线性插补
    """
    def __init__(self, wrapper, comm, gripper_manager, controller):
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.comm = comm
        self.gripper_manager = gripper_manager
        self.controller = controller  # 引用主控制器，用于检查 control_enabled
        self.running = True

        # 频率设置
        self.feedback_rate = 100.0  # 发布频率 100Hz（参考 joint_states.py）
        self.modbus_read_rate = 10.0  # Modbus实际读取频率 10Hz
        self.read_interval = 1.0 / self.modbus_read_rate  # 读取间隔 0.1秒

        # 插补所需的状态变量（基于时间戳）
        self.last_read_time = time.time()

        # 当前状态插补
        self.last_real_pos = None
        self.current_real_pos = None
        self.last_real_read_time = None
        self.current_real_read_time = None

        # 目标状态插补
        self.last_target_pos = None
        self.current_target_pos = None
        self.last_target_read_time = None
        self.current_target_read_time = None

    def _interpolate(self, last_val, current_val, last_time, current_time, now):
        """
        基于时间的线性插补
        """
        if last_val is None or current_val is None:
            return current_val if current_val is not None else 0.0

        if last_time is None or current_time is None:
            return current_val

        # 计算插补系数
        time_span = current_time - last_time
        if time_span <= 0:
            return current_val

        elapsed = now - last_time
        alpha = min(1.0, elapsed / time_span)  # 限制在 [0, 1]

        # 线性插补
        return last_val + alpha * (current_val - last_val)

    def run(self):
        """
        核心循环：以100Hz频率发布夹爪状态（当前+目标），使用基于时间的线性插补
        """
        while self.running:
            try:
                now = time.time()
                timestamp = self.wrapper.get_clock().now().to_msg()

                # === 每0.1秒读取一次真实值 ===
                if now - self.last_read_time >= self.read_interval:
                    self.last_read_time = now

                    # 读取夹爪实际位置
                    real_pos = self.comm.read_reg(514)
                    if real_pos is not None:
                        self.last_real_pos = self.current_real_pos
                        self.last_real_read_time = self.current_real_read_time

                        self.current_real_pos = float(real_pos)
                        self.current_real_read_time = now

                    # 读取目标位置
                    if not self.controller.control_enabled and real_pos is not None:
                        # 控制禁用时，目标=实际
                        target = float(real_pos)
                    else:
                        # 控制启用时，目标=控制指令
                        target = self.gripper_manager.current_pos

                    self.last_target_pos = self.current_target_pos
                    self.last_target_read_time = self.current_target_read_time

                    self.current_target_pos = target
                    self.current_target_read_time = now

                # === 基于时间的线性插补计算 ===
                interpolated_current = self._interpolate(
                    self.last_real_pos,
                    self.current_real_pos,
                    self.last_real_read_time,
                    self.current_real_read_time,
                    now
                )

                interpolated_target = self._interpolate(
                    self.last_target_pos,
                    self.current_target_pos,
                    self.last_target_read_time,
                    self.current_target_read_time,
                    now
                )

                # === 发布插补后的状态（100Hz）===
                # 发布当前状态
                msg_current = PointStamped()
                msg_current.header.stamp = timestamp
                msg_current.header.frame_id = "gripper_feedback"
                msg_current.point.x = float(interpolated_current)
                msg_current.point.y = 0.0
                msg_current.point.z = 0.0
                self.wrapper.pub_gripper_state.publish(msg_current)

                # 发布目标状态
                msg_target = PointStamped()
                msg_target.header.stamp = timestamp
                msg_target.header.frame_id = "gripper_command"
                msg_target.point.x = float(interpolated_target)
                msg_target.point.y = 0.0
                msg_target.point.z = 0.0
                self.wrapper.pub_gripper_update.publish(msg_target)

            except Exception as e:
                # 避免异常导致线程崩溃
                print(f"[ERROR] GripperStateFeedback: {e}")

            # 控制循环频率 100Hz
            time.sleep(1.0 / self.feedback_rate)

class SharedMouseState:
    """线程安全的鼠标状态容器"""
    def __init__(self):
        self.dx = 0
        self.dy = 0
        self.scroll_dy = 0
        self.left_pressed = False
        self.right_pressed = False
        self.lock = threading.Lock()

    def update_move(self, dx, dy):
        with self.lock:
            self.dx += dx
            self.dy += dy
    
    def get_and_clear_move(self):
        with self.lock:
            dx, dy = self.dx, self.dy
            self.dx, self.dy = 0, 0
            return dx, dy

    def update_scroll(self, dy):
        with self.lock:
            self.scroll_dy += dy

    def get_and_clear_scroll(self):
        with self.lock:
            dy = self.scroll_dy
            self.scroll_dy = 0
            return dy

class Robot:
    def __init__(self):
        if not rclpy.ok(): rclpy.init()
        self.node = DobotRosWrapper()
        
        # ROS Spin 线程
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()
        
        
        self.node.call_service(self.node.cli_speed_factor, SpeedFactor.Request(ratio=100))
        self.node.call_service(self.node.cli_clear_error, ClearError.Request())
        self.enable_robot()
        
        print(">.< 机器人服务已连接 >.<")

    def enable_robot(self):
        self.node.call_service(self.node.cli_enable, EnableRobot.Request())

    def ServoP(self, x, y, z, rx, ry, rz):
        req = ServoP.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        self.node.call_service(self.node.cli_servo_p, req)

    def publish_target_pose(self, x, y, z, rx, ry, rz):
        """发布机械臂目标姿态 (用于VLA数据采集)"""
        msg = ToolVectorActual()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.rx = float(rx)
        msg.ry = float(ry)
        msg.rz = float(rz)
        self.node.pub_robot_target.publish(msg)

    def get_pose(self):
        res = self.node.call_service(self.node.cli_get_pose, GetPose.Request())
        if res and hasattr(res, 'res'): 
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", res.pose)
            if len(matches) >= 6: return [float(x) for x in matches[:6]]
        return None

    def close(self):
        rclpy.shutdown()

class TeleopController:
    def __init__(self):
        self.robot = Robot()
        self.mouse_state = SharedMouseState()

        # 参数
        self.LOOP_RATE = 100.0
        self.MOUSE_SENSITIVITY = 0.1
        # self.SCROLL_SENSITIVITY = 2.0
        self.SCROLL_SENSITIVITY = 5.0
        self.KEY_XYZ_STEP = 0.6
        self.KEY_XYZ_STEP_FAST = 0.8
        self.KEY_ROT_STEP = 0.3

        # [新增] 垂直抓取 (Vertical Grab) 参数
        # 请根据实际环境修改这些值！
        self.V_G_OPEN = 600.0  # 夹爪张开范围 g
        self.V_Z1 = 212.         # 底部高度 z1 (绝对坐标，单位mm) - 请确保该高度不会撞击
        self.V_Z2 = 245.0       # 顶部高度 z2 (抓取后上升到的高度)
        self.V_SPEED_UP = 3.         # 上升速率 (mm/tick, 100Hz下约等于 50mm/s)
        self.V_SPEED_DOWN = 3. # 下降速率 (mm/tick)
        self.V_RG = 180.0         # 夹爪闭合速率 rg (单位/tick, 100Hz下 800单位/s)

        self.vgrab_state = 0    # 0=Idle, 1=Descending, 2=Ascending+Closing

        # [新增] 录制起始/结束位置 (从图片中读取的参数)
        self.RECORD_HOME_POSE = [
            116.22736202727263,      # x
            -562.0755453327145,     # y
            278.6781595848309,      # z
            -176.34093460159997,    # rx
            -0.3747282984917232,    # ry
            82.69540118216263       # rz
        ]
        self.RECORD_HOME_GRIPPER = 1000.0  # 夹爪初始位置（完全张开）
        self.moving_to_home = False  # 是否正在移动到起始位置
        self.home_move_callback = None  # 移动完成后的回调
        self.home_settling = False  # 是否正在等待到位后稳定
        self.home_settle_counter = 0  # 稳定等待计数器
        self.HOME_SETTLE_TICKS = 5  # 到位后等待的tick数 (100Hz下约1秒)

        # [新增] 复位平滑处理参数
        self.home_start_pose = None  # 复位开始时的位置
        self.home_total_distance = None  # 复位总距离（用于计算进度）
        self.home_progress = 0.0  # 当前进度 [0, 1]

        self.running = True
        self.control_enabled = True  # 新增：控制启用/禁用标志
        self.target_pose = [0.0] * 6
        self.keys_pressed = set()

        # 启动独立的夹爪控制线程
        self.gripper_worker = GripperManager(self.robot.node, self.mouse_state)
        self.gripper_worker.start()

        # [新增] 启动独立的夹爪状态反馈线程（需要在 control_enabled 定义之后）
        self.gripper_feedback = GripperStateFeedback(self.robot.node, self.gripper_worker.comm, self.gripper_worker, self)
        self.gripper_feedback.start()

        self._init_pose()
        
        # 输入监听（suppress改为False，允许其他程序接收输入）
        self.kb_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release, suppress=False)
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_scroll,
            suppress=False
        )
        self.kb_listener.start()
        self.mouse_listener.start()

        # 屏幕中心初始化
        root = tk.Tk()
        self.center_x = root.winfo_screenwidth() // 2
        self.center_y = root.winfo_screenheight() // 2
        root.destroy()
        
        self.mouse_controller = mouse.Controller()
        self.mouse_controller.position = (self.center_x, self.center_y)
        
        try: os.system("xdotool getactivewindow windowactivate")
        except: pass

        print("=== 系统就绪 ===")
        print(" [Main Loop]       : 负责机械臂运动 (ServoP)")
        print(" [Gripper Thread]  : 负责夹爪控制")
        print(" [Feedback Thread] : 定期发布夹爪状态 (当前+目标 @ 100Hz，基于时间的线性插补)")
        print("")
        print("=== VLA 数据采集话题 ===")
        print(" - 机械臂当前状态: /dobot_msgs_v3/msg/ToolVectorActual (100Hz)")
        print(" - 机械臂目标状态: /robot/target_pose (100Hz)")
        print(" - 夹爪当前状态:   /gripper/state_feedback (100Hz，10Hz采样+时间插补)")
        print(" - 夹爪目标状态:   /gripper/command_update (100Hz，10Hz采样+时间插补)")
        print(" - 相机帧率:       30Hz（数据采集时可降采样）")
        print("")
        print(" [O/P/L]           : 录制控制")
        print(" [Z]               : 触发垂直抓取序列")
        print(" [Tab]             : 暂停/恢复控制（可操作其他窗口)")

    def _init_pose(self):
        for _ in range(5):
            pose = self.robot.get_pose()
            if pose:
                self.target_pose = pose
                # 初始化垂直抓取的高度参数，防止硬编码导致撞击（可选，如果需要绝对值请注释掉）
                # self.V_Z2 = self.target_pose[2]
                # self.V_Z1 = max(0.0, self.target_pose[2] - 50.0)
                return
            time.sleep(0.5)
        print("错误：无法获取机械臂初始位置！")
        sys.exit(1)

    def move_to_home_pose(self, callback=None):
        """
        开始移动到录制起始位置
        callback: 移动完成后执行的回调函数
        """
        self.moving_to_home = True
        self.home_move_callback = callback
        self.home_settling = False
        self.home_settle_counter = 0

        # [新增] 记录起始位置，用于平滑插值
        self.home_start_pose = self.target_pose.copy()
        self.home_progress = 0.0

        # 计算各轴的总距离（用于进度计算）
        # 使用加权距离：位置轴权重1，旋转轴权重较小
        pos_weight = 1.0
        rot_weight = 0.1  # 旋转轴权重较小
        weights = [pos_weight, pos_weight, pos_weight, rot_weight, rot_weight, rot_weight]

        total_dist_sq = 0.0
        for i in range(6):
            diff = self.RECORD_HOME_POSE[i] - self.home_start_pose[i]
            total_dist_sq += (diff * weights[i]) ** 2
        self.home_total_distance = math.sqrt(total_dist_sq)

        # 启用夹爪自动模式，移动到初始位置
        self.gripper_worker.set_auto_mode(True, self.RECORD_HOME_GRIPPER)
        print(">>> 正在移动到录制起始位置...")

    def _check_home_reached(self):
        """检查是否到达起始位置（允许一定误差）"""
        threshold = 2.0  # 位置误差阈值 (mm/度)
        for i in range(6):
            if abs(self.target_pose[i] - self.RECORD_HOME_POSE[i]) > threshold:
                return False
        return True

    def _smoothstep(self, t):
        """
        S曲线缓动函数 (smoothstep)
        输入 t 在 [0, 1] 范围内，输出也在 [0, 1] 范围内
        实现平滑的加速和减速
        """
        # 使用 smootherstep (Ken Perlin 改进版)，更平滑
        t = max(0.0, min(1.0, t))  # 限制在 [0, 1]
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _update_move_to_home(self):
        """
        平滑移动到起始位置（在主循环中调用）
        使用S曲线实现平滑的加速和减速
        返回: True 如果仍在移动, False 如果已到达
        """
        if not self.moving_to_home:
            return False

        # 移动速度参数
        PROGRESS_SPEED = 0.25  # 每tick进度增量，控制整体移动速度
        GRIPPER_THRESHOLD = 50.0  # 夹爪位置误差阈值

        # 更新进度
        self.home_progress += PROGRESS_SPEED
        self.home_progress = min(1.0, self.home_progress)

        # 使用S曲线计算平滑后的进度
        smooth_progress = self._smoothstep(self.home_progress)

        # 根据平滑进度插值计算目标位置
        for i in range(6):
            start = self.home_start_pose[i]
            end = self.RECORD_HOME_POSE[i]
            self.target_pose[i] = start + (end - start) * smooth_progress

        # 检查是否到达目标
        all_reached = self.home_progress >= 1.0

        # 检查夹爪是否到位
        gripper_diff = abs(self.gripper_worker.current_pos - self.RECORD_HOME_GRIPPER)
        if gripper_diff > GRIPPER_THRESHOLD:
            all_reached = False

        if all_reached:
            if not self.home_settling:
                # 刚到达目标位置，进入稳定等待阶段
                self.home_settling = True
                self.home_settle_counter = 0
                print(">>> 已到达录制起始位置，等待稳定...")
                return True

            self.home_settle_counter += 1
            if self.home_settle_counter < self.HOME_SETTLE_TICKS:
                # 仍在稳定等待中，继续发送保持指令
                return True

            # 稳定等待完成
            self.moving_to_home = False
            self.home_settling = False
            # 关闭夹爪自动模式，交还手动控制
            self.gripper_worker.set_auto_mode(False)
            print(">>> 稳定完成，执行录制指令")
            if self.home_move_callback:
                self.home_move_callback()
                self.home_move_callback = None
            return False

        return True

    # --- 输入回调 (运行在 pynput 线程) ---
    def _on_mouse_move(self, x, y):
        if not self.control_enabled:
            return
        dx = x - self.center_x
        dy = y - self.center_y
        if dx != 0 or dy != 0:
            self.mouse_state.update_move(dx, dy)
            self.mouse_controller.position = (self.center_x, self.center_y)

    def _on_mouse_click(self, x, y, button, pressed):
        if not self.control_enabled:
            return
        # 如果正在自动抓取，忽略鼠标点击
        if self.vgrab_state != 0:
            return

        # 更新共享状态，供 GripperManager 读取
        if button == mouse.Button.left:
            self.mouse_state.left_pressed = pressed
            print(f"[DEBUG] 左键 {'按下' if pressed else '松开'}")
        elif button == mouse.Button.right:
            self.mouse_state.right_pressed = pressed
            print(f"[DEBUG] 右键 {'按下' if pressed else '松开'}")

    def _on_scroll(self, x, y, dx, dy):
        if not self.control_enabled:
            return
        self.mouse_state.update_scroll(dy)

    def _on_key_press(self, key):
        if key == keyboard.Key.esc: self.running = False

        # Tab键切换控制启用/禁用
        if key == keyboard.Key.tab:
            self.control_enabled = not self.control_enabled
            status = "启用" if self.control_enabled else "禁用"
            print(f"\n>>> 控制已{status}，现在可以操作其他窗口\n")
            return

        try:
            if hasattr(key, 'char'):
                char_key = key.char.lower()
                self.keys_pressed.add(char_key)

                # 录制指令
                msg = Int32()
                if char_key == 'o':
                    # 录制开始：先移动到起始位置，然后发送开始指令
                    def start_recording():
                        msg = Int32()
                        msg.data = 1
                        self.robot.node.pub_record_cmd.publish(msg)
                        print("Rec Start")
                    self.move_to_home_pose(callback=start_recording)
                elif char_key == 'p':
                    # 录制结束：先移动到起始位置，然后发送保存指令
                    def save_recording():
                        msg = Int32()
                        msg.data = 2
                        self.robot.node.pub_record_cmd.publish(msg)
                        print("Rec Save")
                    self.move_to_home_pose(callback=save_recording)
                elif char_key == 'l': msg.data = 0; self.robot.node.pub_record_cmd.publish(msg); print("Rec Discard")

                # [新增] Z键触发垂直抓取
                elif char_key == 'z':
                    print(">>> 开始垂直抓取序列")
                    self.vgrab_state = 1 # 进入下降阶段
                    # 开启夹爪自动模式，设置目标为张开 (g)
                    self.gripper_worker.set_auto_mode(True, self.V_G_OPEN)

            else:
                self.keys_pressed.add(key)
        except: pass

    def _on_key_release(self, key):
        try:
            if hasattr(key, 'char'): self.keys_pressed.discard(key.char.lower())
            else: self.keys_pressed.discard(key)
        except: pass

    # --- 主循环 (Main Process) ---
    def run(self):
        time_rate = 1.0

        while self.running:
            start_time = time.time()

            # 如果控制被禁用，target_pose 跟踪当前实际位置，但不发送控制指令
            if not self.control_enabled:
                # 读取当前实际位置并更新 target_pose
                current_pose = self.robot.get_pose()
                if current_pose:
                    self.target_pose = current_pose

                # 发布目标状态（与当前状态一致）
                self.robot.publish_target_pose(*self.target_pose)

                time.sleep(1.0 / self.LOOP_RATE)
                continue

            # [新增] 移动到录制起始位置的处理
            if self.moving_to_home:
                # 忽略鼠标输入
                self.mouse_state.get_and_clear_move()
                self.mouse_state.get_and_clear_scroll()
                # 更新移动
                self._update_move_to_home()
                # 发送机械臂指令
                self.robot.ServoP(*self.target_pose)
                self.robot.publish_target_pose(*self.target_pose)
                # 循环频率控制
                elapsed = time.time() - start_time
                sleep_time = (1.0 / self.LOOP_RATE) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # [新增] 垂直抓取状态机
            if self.vgrab_state != 0:
                if 'x' in self.keys_pressed:
                    print(">>> 手动退出抓去")
                    self.vgrab_state = 0
                    self.gripper_worker.set_auto_mode(False) # 交还夹爪控制权
                # 忽略鼠标输入，清除积压的移动量
                self.mouse_state.get_and_clear_move()
                self.mouse_state.get_and_clear_scroll()

                if self.vgrab_state == 1: # 阶段1: 张开夹爪并下降直到 Z1
                    # 确保夹爪目标是张开
                    self.gripper_worker.target_auto_pos = self.V_G_OPEN
                    
                    # 下降 Z
                    if self.target_pose[2] > self.V_Z1:
                        self.target_pose[2] -= self.V_SPEED_DOWN
                        self.target_pose[2] = max(self.target_pose[2], self.V_Z1)
                    else:
                        # 到达底部，切换状态
                        print(">>> 到达底部，开始闭合上升")
                        self.vgrab_state = 2
                
                elif self.vgrab_state == 2: # 阶段2: 闭合夹爪同时上升直到 Z2
                    # 上升 Z (速率 rz)
                    if abs(self.target_pose[2] - self.V_Z1) <= self.V_SPEED_UP * 2:
                        self.target_pose[2] += self.V_SPEED_UP / 2
                    else:
                        self.target_pose[2] += self.V_SPEED_UP
                    
                    # 闭合夹爪 (速率 rg)
                    new_grip = self.gripper_worker.target_auto_pos - self.V_RG
                    self.gripper_worker.target_auto_pos = max(0.0, new_grip)
                    
                    # 结束条件
                    if self.target_pose[2] > self.V_Z2:
                        print(">>> 抓取结束，恢复手动控制")
                        self.vgrab_state = 0
                        self.gripper_worker.set_auto_mode(False) # 交还夹爪控制权

            else:
                # === 原有的手动控制逻辑 ===
                
                # 1. 获取并清除累积的鼠标移动量 (非阻塞)
                dx, dy = self.mouse_state.get_and_clear_move()
                scroll_dy = self.mouse_state.get_and_clear_scroll()

                # 2. 计算机械臂运动 (XYZ)
                self.target_pose[0] += dx * self.MOUSE_SENSITIVITY
                self.target_pose[1] -= dy * self.MOUSE_SENSITIVITY

                if keyboard.Key.space in self.keys_pressed:
                    step = self.KEY_XYZ_STEP_FAST if (keyboard.Key.alt in self.keys_pressed) else self.KEY_XYZ_STEP
                    self.target_pose[2] += step * time_rate
                if keyboard.Key.ctrl in self.keys_pressed or keyboard.Key.ctrl_l in self.keys_pressed:
                    step = self.KEY_XYZ_STEP_FAST if (keyboard.Key.alt in self.keys_pressed) else self.KEY_XYZ_STEP
                    self.target_pose[2] -= step * time_rate

                # 3. 计算机械臂旋转 (RPY)
                if 'a' in self.keys_pressed: self.target_pose[3] += self.KEY_ROT_STEP * time_rate
                if 'd' in self.keys_pressed: self.target_pose[3] -= self.KEY_ROT_STEP * time_rate
                if 'w' in self.keys_pressed: self.target_pose[4] -= self.KEY_ROT_STEP * time_rate
                if 's' in self.keys_pressed: self.target_pose[4] += self.KEY_ROT_STEP * time_rate

                if 'x' in self.keys_pressed: 
                    self.gripper_worker.move(self.V_G_OPEN)

                if scroll_dy != 0:
                    self.target_pose[5] += scroll_dy * self.SCROLL_SENSITIVITY

            # 4. 发送机械臂指令 (夹爪指令由 GripperManager 处理)
            self.robot.ServoP(*self.target_pose)

            # [新增] 发布机械臂目标状态 (用于VLA数据采集)
            self.robot.publish_target_pose(*self.target_pose)

            # 5. 循环频率控制
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.LOOP_RATE) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                time_rate = 1.0
            else:
                time_rate = elapsed / (1.0 / self.LOOP_RATE) + 1

        self.gripper_worker.running = False
        self.gripper_worker.join()
        self.gripper_feedback.running = False
        self.gripper_feedback.join()
        self.robot.close()

if __name__ == "__main__":
    controller = TeleopController()
    controller.run()