# 顶部相机 精度1cm之内
# 两点， 包含抓取方向
# [修改版] 集成 VLA 数据转发功能

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import threading
import time
import os
import re
import sys

# ROS Messages & Services
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32
from dobot_msgs_v3.srv import *
# [新增] 引入用于数据转发的消息类型
from dobot_msgs_v3.msg import ToolVectorActual, ToolVectorTarget

# ================= 1. 视觉标定类 (保持不变) =================
class RobotVisionSystem:
    def __init__(self, 
                 cam_param_path, 
                 plane_eq_path, 
                 alignment_path, 
                 calib_resolution=(8160, 6120), 
                 current_resolution=None):
        
        if not all(os.path.exists(p) for p in [cam_param_path, plane_eq_path, alignment_path]):
            raise FileNotFoundError("部分标定文件未找到")

        with np.load(cam_param_path) as X:
            self.mtx_orig = X['mtx']
            self.dist = X['dist']
        
        self.plane_eq = np.load(plane_eq_path)
        
        with np.load(alignment_path) as X:
            self.R_cam2rob = X['R']
            self.T_cam2rob = X['T'] 

        self.calib_w, self.calib_h = calib_resolution
        if current_resolution is None:
            self.curr_w, self.curr_h = self.calib_w, self.calib_h
        else:
            self.curr_w, self.curr_h = current_resolution
        
        scale_x = self.curr_w / self.calib_w
        scale_y = self.curr_h / self.calib_h
        
        self.mtx_curr = self.mtx_orig.copy()
        self.mtx_curr[0, 0] *= scale_x 
        self.mtx_curr[0, 2] *= scale_x 
        self.mtx_curr[1, 1] *= scale_y 
        self.mtx_curr[1, 2] *= scale_y 
        
        self.R_rob2cam = self.R_cam2rob.T
        self.T_rob2cam = -self.R_rob2cam @ self.T_cam2rob

    def pixel_to_robot(self, u, v, z_offset_mm=0.0):
        pt_distorted = np.array([[[u, v]]], dtype=np.float64)
        pt_norm = cv2.undistortPoints(pt_distorted, self.mtx_curr, self.dist)
        x_n, y_n = pt_norm[0, 0]

        ray_cam = np.array([x_n, y_n, 1.0])
        origin_rob = self.T_cam2rob 
        dir_rob = self.R_cam2rob @ ray_cam
        
        z_target_m = z_offset_mm / 1000.0
        
        if abs(dir_rob[2]) < 1e-6:
            return None
            
        t = (z_target_m - origin_rob[2]) / dir_rob[2]
        pt_rob_m = origin_rob + t * dir_rob
        
        return pt_rob_m * 1000.0 

    def pixel_pair_to_robot_pose(self, u_start, v_start, u_end, v_end, z_offset_mm=0.0):
        p_start = self.pixel_to_robot(u_start, v_start, z_offset_mm)
        p_end = self.pixel_to_robot(u_end, v_end, z_offset_mm)
        
        if p_start is None or p_end is None:
            return None
            
        x1, y1 = p_start[0], p_start[1]
        x2, y2 = p_end[0], p_end[1]
        
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # 归一化到 [0, 360)
        angle_deg = (angle_deg + 360) % 360
        
        return (x1, y1), angle_deg

# ================= 2. 相机驱动 (保持不变) =================
class Camera8KDriver:
    def __init__(self, device_index=0):
        self.dev = f"/dev/video{device_index}"
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 8160)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 6120)
        self.cap.set(cv2.CAP_PROP_FPS, 5)
        
        self.current_exp = 440
        self.current_focus = 240
        self.setup_hardware()

    def setup_hardware(self):
        os.system(f"v4l2-ctl -d {self.dev} -c auto_exposure=1") 
        os.system(f"v4l2-ctl -d {self.dev} -c exposure_time_absolute={self.current_exp}")
        os.system(f"v4l2-ctl -d {self.dev} -c focus_automatic_continuous=0") 
        os.system(f"v4l2-ctl -d {self.dev} -c focus_absolute={self.current_focus}")

    def set_exposure(self, val):
        self.current_exp = max(1, min(10000, val))
        os.system(f"v4l2-ctl -d {self.dev} -c exposure_time_absolute={self.current_exp}")

    def set_focus(self, val):
        self.current_focus = max(0, min(1023, val))
        os.system(f"v4l2-ctl -d {self.dev} -c focus_absolute={self.current_focus}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

# ================= 3. Dobot ROS2 Wrapper (核心修改：数据桥接) =================
class DobotRosWrapper(Node):
    def __init__(self, node_name="click_pair_pick"):
        super().__init__(node_name)
        # 基础运动服务
        self.cli_enable = self.create_client(EnableRobot, '/dobot_bringup_v3/srv/EnableRobot')
        self.cli_clear_error = self.create_client(ClearError, '/dobot_bringup_v3/srv/ClearError')
        self.cli_speed_factor = self.create_client(SpeedFactor, '/dobot_bringup_v3/srv/SpeedFactor')
        self.cli_mov_j = self.create_client(MovJ, '/dobot_bringup_v3/srv/MovJ')
        self.cli_mov_l = self.create_client(MovL, '/dobot_bringup_v3/srv/MovL')
        self.cli_servo_p = self.create_client(ServoP, '/dobot_bringup_v3/srv/ServoP')
        self.cli_sync = self.create_client(Sync, '/dobot_bringup_v3/srv/Sync')
        
        # 逆解客户端
        self.cli_inverse = self.create_client(InverseSolution, '/dobot_bringup_v3/srv/InverseSolution')

        # Modbus
        self.cli_modbus_create = self.create_client(ModbusCreate, '/dobot_bringup_v3/srv/ModbusCreate')
        self.cli_modbus_close = self.create_client(ModbusClose, '/dobot_bringup_v3/srv/ModbusClose')
        self.cli_set_hold_regs = self.create_client(SetHoldRegs, '/dobot_bringup_v3/srv/SetHoldRegs')
        self.cli_get_hold_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs')

        # === 1. 录制器交互话题 ===
        # 夹爪相关
        self.pub_gripper_update = self.create_publisher(PointStamped, "/gripper/command_update", 5)
        self.pub_gripper_state = self.create_publisher(PointStamped, "/gripper/state_feedback", 5)
        # 录制控制
        self.pub_record_cmd = self.create_publisher(Int32, "/recorder/command", 10)

        # === 2. 机械臂 Target 数据桥接 (Bridge) ===
        # 订阅：来自底层反馈节点的硬件目标
        self.sub_hardware_target = self.create_subscription(
            ToolVectorTarget, 
            '/dobot_msgs_v3/msg/ToolVectorTarget', 
            self.relay_target_callback, 
            10
        )
        # 发布：给录制器使用的统一 Target 话题
        self.pub_recorder_target = self.create_publisher(
            ToolVectorActual, 
            '/robot/target_pose', 
            10
        )

        # 检查服务
        if not self.cli_inverse.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Warning: InverseSolution service not available!")

    def relay_target_callback(self, msg):
        """
        [关键] 将底层反馈的 ToolVectorTarget 转换为录制器需要的 ToolVectorActual 格式
        并发布到 /robot/target_pose。
        这样，无论视觉脚本如何通过服务控制机器人，录制器都能收到真实的硬件目标位姿。
        """
        relay_msg = ToolVectorActual()
        # 复制所有字段
        relay_msg.header = msg.header
        relay_msg.x = msg.x
        relay_msg.y = msg.y
        relay_msg.z = msg.z
        relay_msg.rx = msg.rx
        relay_msg.ry = msg.ry
        relay_msg.rz = msg.rz
        
        # 转发给录制器
        self.pub_recorder_target.publish(relay_msg)

    def call_service(self, client, request):
        if not client.service_is_ready(): return None
        future = client.call_async(request)
        while not future.done(): time.sleep(0.002)
        return future.result()

    def call_service_async_no_wait(self, client, request):
        if client.service_is_ready(): client.call_async(request)

    def check_accessibility(self, x, y, z, rx, ry, rz):
        if not self.cli_inverse.service_is_ready():
            print("Error: Inverse service not ready, skipping check (assuming risky true)")
            return True 
        req = InverseSolution.Request()
        req.offset1, req.offset2, req.offset3 = float(x), float(y), float(z)
        req.offset4, req.offset5, req.offset6 = float(rx), float(ry), float(rz)
        req.user = 0; req.tool = 0; req.is_jointnear = 0; req.joint_near = "" 
        res = self.call_service(self.cli_inverse, req)
        return True if (res is not None and res.res == 0) else False

class GripperComm:
    """
    底层 Modbus 通信类，只负责发协议，不含控制逻辑
    [新增] 后台异步读取线程，避免阻塞主循环
    """
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.id = 0
        self.init_connection()

        # [新增] 缓存的夹爪位置，由后台线程更新
        self.cached_position = None
        self.cached_position_lock = threading.Lock()

        # [新增] 启动后台读取线程
        self.reader_thread = threading.Thread(target=self._background_reader, daemon=True)
        self.reader_running = True
        self.reader_thread.start()

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
            print("[DEBUG] 寄存器257 (Force/Speed) = 60")

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
        """阻塞读取寄存器（仅用于初始化等场景）"""
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

    def _background_reader(self):
        """后台线程：持续读取夹爪位置并更新缓存"""
        print("[DEBUG] GripperComm 后台读取线程已启动")
        while self.reader_running:
            if self.id > 0:
                try:
                    # 阻塞读取（在后台线程中不影响主循环）
                    pos = self.read_reg(514)
                    if pos is not None:
                        with self.cached_position_lock:
                            self.cached_position = pos
                except Exception as e:
                    print(f"[ERROR] 后台读取夹爪位置失败: {e}")
            time.sleep(0.05)  # 20Hz 读取频率，避免过于频繁

    def get_cached_position(self):
        """非阻塞获取缓存的夹爪位置"""
        with self.cached_position_lock:
            return self.cached_position

    def stop_reader(self):
        """停止后台读取线程"""
        self.reader_running = False
        if self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)

class GripperManager(threading.Thread):
    """
    [新架构] 独立的夹爪控制进程（线程）
    完全负责：状态维护、指令发送
    """
    def __init__(self, wrapper):
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.comm = GripperComm(wrapper)
        self.running = True
        self.target_pos = 1000.0

    def set_target(self, pos):
        self.target_pos = float(pos)

    def run(self):
        while self.running:
            self.comm.write_reg(259, 1, str(int(self.target_pos)), wait=False)
            time.sleep(0.02)

class GripperStateFeedback(threading.Thread):
    """
    [新增] 独立的夹爪状态反馈线程
    专门负责定期读取夹爪实际位置并发布当前状态和目标状态
    使用高频发布(100Hz)+低频采样(10Hz)+基于时间的线性插补
    """
    def __init__(self, wrapper, comm, gripper_manager):
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.comm = comm
        self.gripper_manager = gripper_manager
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
        loop_count = 0
        start_time = time.time()

        # 性能分析变量
        time_get_clock = 0
        time_read_modbus = 0
        time_interpolate = 0
        time_publish = 0

        while self.running:
            try:
                now = time.time()

                t0 = time.time()
                timestamp = self.wrapper.get_clock().now().to_msg()
                time_get_clock += time.time() - t0

                # === 每0.1秒读取一次真实值 ===
                if now - self.last_read_time >= self.read_interval:
                    self.last_read_time = now

                    # [优化] 非阻塞读取缓存的夹爪位置（后台线程已在更新）
                    t0 = time.time()
                    real_pos = self.comm.get_cached_position()
                    time_read_modbus += time.time() - t0

                    if real_pos is not None:
                        self.last_real_pos = self.current_real_pos
                        self.last_real_read_time = self.current_real_read_time

                        self.current_real_pos = float(real_pos)
                        self.current_real_read_time = now

                    # 读取目标位置（直接使用控制指令）
                    target = self.gripper_manager.target_pos

                    self.last_target_pos = self.current_target_pos
                    self.last_target_read_time = self.current_target_read_time

                    self.current_target_pos = target
                    self.current_target_read_time = now

                # === 基于时间的线性插补计算 ===
                t0 = time.time()
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
                time_interpolate += time.time() - t0

                # === 发布插补后的状态（100Hz）===
                t0 = time.time()
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
                time_publish += time.time() - t0

            except Exception as e:
                # 避免异常导致线程崩溃
                print(f"[ERROR] GripperStateFeedback: {e}")

            # 控制循环频率 100Hz
            time.sleep(1.0 / self.feedback_rate)

            # [调试] 每秒打印一次实际频率和性能分析
            loop_count += 1
            if loop_count % 100 == 0:
                elapsed = time.time() - start_time
                actual_hz = loop_count / elapsed
                avg_clock = (time_get_clock / loop_count) * 1000
                avg_modbus = (time_read_modbus / max(1, loop_count//10)) * 1000  # 只每10次读一次
                avg_interp = (time_interpolate / loop_count) * 1000
                avg_pub = (time_publish / loop_count) * 1000

                print(f"[GripperFeedback] 频率: {actual_hz:.1f} Hz | "
                      f"时钟: {avg_clock:.1f}ms | Modbus: {avg_modbus:.1f}ms | "
                      f"插值: {avg_interp:.2f}ms | 发布: {avg_pub:.1f}ms")

                if loop_count >= 1000:  # 重置计数器避免溢出
                    loop_count = 0
                    start_time = time.time()
                    time_get_clock = 0
                    time_read_modbus = 0
                    time_interpolate = 0
                    time_publish = 0

# ================= 4. 主程序 (集成逆解验证与数据采集桥接) =================

V_Z1 = 212.0
V_Z2 = 250.0
GRIPPER_OPEN = 600.0
GRIPPER_CLOSE = 0.0
V_SPEED_UP = 0.1
V_RG = 80.0
DEFAULT_POSE = [383.0, -61.0, V_Z2, 180.0, 0.0, 90.0]

class AutoPickApp:
    def __init__(self):
        rclpy.init()
        self.node = DobotRosWrapper()
        # ROS 必须在后台 Spin 以接收硬件反馈并转发
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        self.gripper = GripperManager(self.node)
        self.gripper.start()

        # [新增] 启动独立的夹爪状态反馈线程（传入 comm 参数）
        self.gripper_feedback = GripperStateFeedback(self.node, self.gripper.comm, self.gripper)
        self.gripper_feedback.start()

        self.cam_driver = Camera8KDriver(device_index=0) 
        
        self.vision = RobotVisionSystem(
            cam_param_path='/home/hit/camera_high/camera_matrix/calibration_data.npz',
            plane_eq_path='/home/hit/camera_high/table_matrix/plane_equation.npy',
            alignment_path='/home/hit/camera_high/robot_matrix/robot_alignment.npz',
            calib_resolution=(8160, 6120),
            current_resolution=(8160, 6120)
        )

        self.running = True
        self.robot_busy = False
        self.stop_flag = False
        self.window_name = "8K Verified Pick (Recorder Bridge Mode)"
        
        self.click_state = 0 
        self.pt1_uv = None
        self.pt1_real = None

        self.init_robot()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def init_robot(self):
        try:
            self.node.call_service(self.node.cli_enable, EnableRobot.Request())
            self.node.call_service(self.node.cli_clear_error, ClearError.Request())
            self.node.call_service(self.node.cli_speed_factor, SpeedFactor.Request(ratio=50))
            self.gripper.set_target(GRIPPER_OPEN)
            self.move_to_default()
        except: pass

    def move_to_default(self):
        if self.robot_busy: return
        req = MovJ.Request()
        req.x, req.y, req.z = DEFAULT_POSE[0], DEFAULT_POSE[1], DEFAULT_POSE[2]
        req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
        self.node.call_service(self.node.cli_mov_j, req)
        self.node.call_service(self.node.cli_sync, Sync.Request())

    def stop_robot(self):
        self.stop_flag = True
        self.node.call_service(self.node.cli_clear_error, ClearError.Request())
        self.robot_busy = False
        self.click_state = 0

    def find_reachable_rz(self, x, y, z, original_rz):
        rx_def = DEFAULT_POSE[3]
        ry_def = DEFAULT_POSE[4]
        candidates = [
            (original_rz, "Original"),
            ((original_rz + 180) % 360, "+180"),
            ((original_rz - 180) % 360, "-180"),
            ((original_rz + 90) % 360, "+90"),
            ((original_rz - 90) % 360, "-90")
        ]
        print(f"--- 验证可达性 (Pos: {x:.1f}, {y:.1f}, {z:.1f}) ---")
        for rz_test, tag in candidates:
            if self.node.check_accessibility(x, y, z, rx_def, ry_def, rz_test):
                print(f"√ 可行 [{tag}]: Rz = {rz_test:.2f}")
                return rz_test
        return None

    def execution_thread(self, target_pos_mm, target_angle):
        if self.robot_busy: return
        self.robot_busy = True
        self.stop_flag = False
        tx, ty = target_pos_mm
        
        valid_rz = self.find_reachable_rz(tx, ty, V_Z1, target_angle)
        if valid_rz is None:
            print(">>> [错误] 目标点无法到达")
            self.robot_busy = False
            return
        
        print(f">>> 执行抓取: X={tx:.1f}, Y={ty:.1f}, Valid_Rz={valid_rz:.1f}°")
        try:
            # 1. 移动到上方 (V_Z2)
            if self.stop_flag: raise InterruptedError()
            self.gripper.set_target(GRIPPER_OPEN)
            req = MovJ.Request()
            req.x, req.y, req.z = float(tx), float(ty), float(V_Z2)
            req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], float(valid_rz)
            self.node.call_service(self.node.cli_mov_j, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # 2. 下降 (V_Z1)
            req = MovL.Request()
            req.x, req.y, req.z = float(tx), float(ty), float(V_Z2)
            req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], float(valid_rz)
            if self.stop_flag: raise InterruptedError()
            req.z = float(V_Z1)
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # 3. 动态抓取 (ServoP 模拟)
            print(">>> 闭合夹爪并上升...")
            current_z = V_Z1
            current_grip = GRIPPER_OPEN
            while current_z < V_Z2:
                if self.stop_flag: raise InterruptedError()
                current_z += V_SPEED_UP
                current_grip -= V_RG
                if current_z > V_Z2: current_z = V_Z2
                if current_grip < GRIPPER_CLOSE: current_grip = GRIPPER_CLOSE
                
                sp_req = ServoP.Request()
                sp_req.x, sp_req.y, sp_req.z = float(tx), float(ty), float(current_z)
                sp_req.rx, sp_req.ry, sp_req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], float(valid_rz)
                self.node.call_service_async_no_wait(self.node.cli_servo_p, sp_req)
                
                self.gripper.set_target(current_grip)
                time.sleep(0.02)

            time.sleep(0.5)
            self.gripper.set_target(GRIPPER_OPEN)
            time.sleep(0.5)

        except InterruptedError: print("Action Interrupted")
        except Exception as e:
            print(f"Action Error: {e}")
            self.stop_robot()
        finally:
            self.robot_busy = False
            print(">>> 回归原点")
            self.move_to_default()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.robot_busy: return
            
            scale_w = 8160 / 1280
            scale_h = 6120 / 960
            real_u = x * scale_w
            real_v = y * scale_h
            
            if self.click_state == 0:
                self.pt1_uv = (x, y)
                self.pt1_real = (real_u, real_v)
                self.click_state = 1
                print(f"[Click 1] 抓取点: ({x}, {y})")
            elif self.click_state == 1:
                print(f"[Click 2] 方向点: ({x}, {y})")
                result = self.vision.pixel_pair_to_robot_pose(
                    self.pt1_real[0], self.pt1_real[1], real_u, real_v, z_offset_mm=0
                )
                self.click_state = 0
                self.pt1_uv = None
                
                if result is not None:
                    (rx, ry), angle = result
                    t = threading.Thread(target=self.execution_thread, args=((rx, ry), angle))
                    t.start()
                else: print("解算失败")

    def run(self):
        print("\n=== 视觉操控 (Bridge Mode) ===")
        print("  此脚本已集成数据桥接，可与录制器完美配合")
        print("\n=== 录制控制 ===")
        print("  [O] - 开始录制")
        print("  [P] - 保存录制")
        print("  [L] - 丢弃录制")
        print("\n=== 数据流 ===")
        print("  1. 接收: /dobot_msgs_v3/msg/ToolVectorTarget (from feedback node)")
        print("  2. 转发: /robot/target_pose (to recorder)")
        print("  3. 夹爪: /gripper/command_update & feedback (100Hz)")
        print("")

        while self.running:
            ret, frame = self.cam_driver.read()
            if not ret: break
            
            preview = cv2.resize(frame, (1280, 960))
            if self.click_state == 1 and self.pt1_uv is not None:
                cv2.circle(preview, self.pt1_uv, 10, (0, 255, 255), 2)
            
            status_text = "BUSY" if self.robot_busy else ("WAITING" if self.click_state == 1 else "READY")
            cv2.putText(preview, f"Status: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(self.window_name, preview)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                self.running = False
                self.stop_robot()
            elif key == ord('x'): self.stop_robot()
            
            # 录制控制
            elif key == ord('o'):
                self.node.pub_record_cmd.publish(Int32(data=1))
                print("[录制] START")
            elif key == ord('p'):
                self.node.pub_record_cmd.publish(Int32(data=2))
                print("[录制] SAVE")
            elif key == ord('l'):
                self.node.pub_record_cmd.publish(Int32(data=0))
                print("[录制] DISCARD")
            
            # 相机调节
            elif key in [ord('+'), ord('=')]: self.cam_driver.set_exposure(self.cam_driver.current_exp + 20)
            elif key == ord('-'): self.cam_driver.set_exposure(self.cam_driver.current_exp - 20)
            elif key in [ord('*'), ord('8')]: self.cam_driver.set_focus(self.cam_driver.current_focus + 10)
            elif key in [ord('/'), ord('2')]: self.cam_driver.set_focus(self.cam_driver.current_focus - 10)

        self.gripper.running = False
        self.gripper_feedback.running = False
        self.gripper.comm.stop_reader()  # [新增] 停止后台 Modbus 读取线程
        self.cam_driver.release()
        self.node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = AutoPickApp()
        app.run()
    except KeyboardInterrupt: pass