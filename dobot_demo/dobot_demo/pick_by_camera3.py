# 顶部相机 精度1cm之内
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
from dobot_msgs_v3.srv import *

# ================= 1. 最新手眼标定类 (RobotVisionSystem) =================
class RobotVisionSystem:
    def __init__(self, 
                 cam_param_path, 
                 plane_eq_path, 
                 alignment_path, 
                 calib_resolution=(8160, 6120), 
                 current_resolution=None):
        """
        初始化视觉转换系统
        """
        # 1. 加载数据
        if not all(os.path.exists(p) for p in [cam_param_path, plane_eq_path, alignment_path]):
            print(f"检查路径: {cam_param_path}, {plane_eq_path}, {alignment_path}")
            raise FileNotFoundError("部分标定文件未找到，请检查当前目录下是否存在 .npz/.npy 文件")

        # 加载内参
        with np.load(cam_param_path) as X:
            self.mtx_orig = X['mtx']
            self.dist = X['dist']
        
        # 加载平面方程 (ax + by + cz + d = 0, 单位: m)
        self.plane_eq = np.load(plane_eq_path)
        
        # 加载手眼标定矩阵 (单位: m)
        with np.load(alignment_path) as X:
            self.R_cam2rob = X['R']
            self.T_cam2rob = X['T'] 

        # 2. 处理分辨率缩放
        self.calib_w, self.calib_h = calib_resolution
        if current_resolution is None:
            self.curr_w, self.curr_h = self.calib_w, self.calib_h
        else:
            self.curr_w, self.curr_h = current_resolution
        
        scale_x = self.curr_w / self.calib_w
        scale_y = self.curr_h / self.calib_h
        
        self.mtx_curr = self.mtx_orig.copy()
        self.mtx_curr[0, 0] *= scale_x # fx
        self.mtx_curr[0, 2] *= scale_x # cx
        self.mtx_curr[1, 1] *= scale_y # fy
        self.mtx_curr[1, 2] *= scale_y # cy
        
        self.R_rob2cam = self.R_cam2rob.T
        self.T_rob2cam = -self.R_rob2cam @ self.T_cam2rob

    def pixel_to_robot(self, u, v, z_offset_mm=0.0):
        """像素坐标 -> 机械臂坐标 (求交点)"""
        # 1. 去畸变
        pt_distorted = np.array([[[u, v]]], dtype=np.float64)
        pt_norm = cv2.undistortPoints(pt_distorted, self.mtx_curr, self.dist)
        x_n, y_n = pt_norm[0, 0]

        # 2. 射线与平面求交
        ray_cam = np.array([x_n, y_n, 1.0])
        origin_rob = self.T_cam2rob 
        dir_rob = self.R_cam2rob @ ray_cam
        
        z_target_m = z_offset_mm / 1000.0
        
        if abs(dir_rob[2]) < 1e-6:
            print("Warning: Ray is parallel to the plane!")
            return None
            
        t = (z_target_m - origin_rob[2]) / dir_rob[2]
        pt_rob_m = origin_rob + t * dir_rob
        
        return pt_rob_m * 1000.0 # m -> mm

# ================= 2. 8K 相机驱动 (含指定参数) =================
class Camera8KDriver:
    def __init__(self, device_index=6):
        self.dev = f"/dev/video{device_index}"
        # 初始化 8K
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 8160)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 6120)
        self.cap.set(cv2.CAP_PROP_FPS, 5)
        
        # 默认参数 (用户指定)
        self.current_exp = 440
        self.current_focus = 240
        
        # 预览缩放比例
        self.PREVIEW_W = 1280
        self.PREVIEW_H = 960
        
        self.setup_hardware()

    def setup_hardware(self):
        """配置初始曝光和对焦"""
        print(f"正在配置相机: {self.dev} ...")
        # 1. 设置手动曝光 440
        os.system(f"v4l2-ctl -d {self.dev} -c auto_exposure=1") # 1=Manual
        os.system(f"v4l2-ctl -d {self.dev} -c exposure_time_absolute={self.current_exp}")
        
        # 2. 设置手动对焦 240
        os.system(f"v4l2-ctl -d {self.dev} -c focus_automatic_continuous=0") # 关闭自动对焦
        os.system(f"v4l2-ctl -d {self.dev} -c focus_absolute={self.current_focus}")
        
        print(f"相机就绪: 8160x6120 | Exp={self.current_exp} | Focus={self.current_focus} (Manual)")

    def set_exposure(self, val):
        self.current_exp = max(1, min(10000, val))
        os.system(f"v4l2-ctl -d {self.dev} -c exposure_time_absolute={self.current_exp}")
        print(f"曝光设置: {self.current_exp}")

    def set_focus(self, val):
        self.current_focus = max(0, min(1023, val))
        os.system(f"v4l2-ctl -d {self.dev} -c focus_absolute={self.current_focus}")
        print(f"对焦设置: {self.current_focus}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

# ================= 3. Dobot ROS2 接口 & 夹爪控制 =================
class DobotRosWrapper(Node):
    def __init__(self, node_name="click_pick_8k"):
        super().__init__(node_name)
        # 服务客户端
        self.cli_enable = self.create_client(EnableRobot, '/dobot_bringup_v3/srv/EnableRobot')
        self.cli_clear_error = self.create_client(ClearError, '/dobot_bringup_v3/srv/ClearError')
        self.cli_speed_factor = self.create_client(SpeedFactor, '/dobot_bringup_v3/srv/SpeedFactor')
        self.cli_mov_j = self.create_client(MovJ, '/dobot_bringup_v3/srv/MovJ')
        self.cli_mov_l = self.create_client(MovL, '/dobot_bringup_v3/srv/MovL')
        self.cli_servo_p = self.create_client(ServoP, '/dobot_bringup_v3/srv/ServoP')
        self.cli_sync = self.create_client(Sync, '/dobot_bringup_v3/srv/Sync')
        # Modbus
        self.cli_modbus_create = self.create_client(ModbusCreate, '/dobot_bringup_v3/srv/ModbusCreate')
        self.cli_modbus_close = self.create_client(ModbusClose, '/dobot_bringup_v3/srv/ModbusClose')
        self.cli_set_hold_regs = self.create_client(SetHoldRegs, '/dobot_bringup_v3/srv/SetHoldRegs')
        self.cli_get_hold_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs')

        if not self.cli_mov_j.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("警告: 无法连接 Dobot 服务，请检查是否启动 dobot_bringup")

    def call_service(self, client, request):
        if not client.service_is_ready(): return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.002)
        return future.result()

    def call_service_async_no_wait(self, client, request):
        if client.service_is_ready():
            client.call_async(request)

class GripperManager(threading.Thread):
    def __init__(self, wrapper):
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.id = 0
        self.running = True
        self.target_pos = 1000.0
        self.init_connection()

    def init_connection(self):
        try:
            # 重置连接
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
                match = re.search(r'(\d+)', str(res.index))
                self.id = int(match.group(1)) if match else int(res.index)
                # Enable & Speed
                self.write_reg(256, 1, "1", wait=True)
                self.write_reg(257, 1, "60", wait=True)
                print(f"[Gripper] Connected ID: {self.id}")
            else:
                print("[Gripper] Connection Failed")
        except Exception as e:
            print(f"[Gripper] Init Error: {e}")

    def write_reg(self, addr, count, val_str, wait=False):
        if self.id <= 0: return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_str
        if wait:
            self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)
        else:
            self.wrapper.call_service_async_no_wait(self.wrapper.cli_set_hold_regs, req)

    def set_target(self, pos):
        self.target_pos = float(pos)

    def run(self):
        while self.running:
            # 持续发送位置指令
            self.write_reg(259, 1, str(int(self.target_pos)), wait=False)
            time.sleep(0.02)

# ================= 4. 主程序应用 =================

# 参数配置
V_Z1 = 212.0  # 抓取高度
V_Z2 = 250.0  # 安全高度
GRIPPER_OPEN = 600.0
GRIPPER_CLOSE = 0.0
V_SPEED_UP = 0.1
V_RG = 80.0
DEFAULT_POSE = [383.0, -61.0, V_Z2, 180.0, 0.0, 90.0]

class AutoPickApp:
    def __init__(self):
        # 1. 初始化 ROS
        rclpy.init()
        self.node = DobotRosWrapper()
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        # 2. 初始化硬件
        self.gripper = GripperManager(self.node)
        self.gripper.start()
        
        # 你的设备ID，如果不是6请修改
        self.cam_driver = Camera8KDriver(device_index=0) 
        
        # 3. 初始化视觉系统
        # 确保当前目录下有这些文件
        self.vision = RobotVisionSystem(
            cam_param_path='/home/hit/camera_high/camera_matrix/calibration_data.npz',
            plane_eq_path='/home/hit/camera_high/table_matrix/plane_equation.npy',
            alignment_path='/home/hit/camera_high/robot_matrix/robot_alignment.npz',
            calib_resolution=(8160, 6120),
            current_resolution=(8160, 6120) # 我们用全分辨率计算
        )

        # 4. 状态标志
        self.running = True
        self.robot_busy = False
        self.stop_flag = False
        self.window_name = "8K Pick System (Click to Pick)"

        # 5. 初始化机械臂
        self.init_robot()
        
        # 6. UI设置
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def init_robot(self):
        print("Initializing Robot Pose...")
        try:
            self.node.call_service(self.node.cli_enable, EnableRobot.Request())
            self.node.call_service(self.node.cli_clear_error, ClearError.Request())
            self.node.call_service(self.node.cli_speed_factor, SpeedFactor.Request(ratio=50))
            self.gripper.set_target(GRIPPER_OPEN)
            self.move_to_default()
        except Exception as e:
            print(f"Robot Init Warning: {e}")

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

    def execution_thread(self, target_pos_mm):
        """执行抓取动作"""
        if self.robot_busy: return
        self.robot_busy = True
        self.stop_flag = False
        
        tx, ty, _ = target_pos_mm # Z由高度参数控制
        print(f">>> 开始抓取: X={tx:.1f}, Y={ty:.1f}")

        try:
            # 1. 移动到上方 (V_Z2)
            if self.stop_flag: raise InterruptedError()
            self.gripper.set_target(GRIPPER_OPEN)
            
            req = MovL.Request()
            req.x, req.y, req.z = float(tx), float(ty), float(V_Z2)
            req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # 2. 下降 (V_Z1)
            if self.stop_flag: raise InterruptedError()
            req.z = float(V_Z1)
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # 3. 动态抓取 (ServoP + Gripper)
            print(">>> 闭合夹爪并上升...")
            current_z = V_Z1
            current_grip = GRIPPER_OPEN
            
            while current_z < V_Z2:
                if self.stop_flag: raise InterruptedError()
                
                # 运动更新
                current_z += V_SPEED_UP
                current_grip -= V_RG
                
                # 限制
                if current_z > V_Z2: current_z = V_Z2
                if current_grip < GRIPPER_CLOSE: current_grip = GRIPPER_CLOSE
                
                # 发送
                sp_req = ServoP.Request()
                sp_req.x, sp_req.y, sp_req.z = float(tx), float(ty), float(current_z)
                sp_req.rx, sp_req.ry, sp_req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
                self.node.call_service_async_no_wait(self.node.cli_servo_p, sp_req)
                
                self.gripper.set_target(current_grip)
                time.sleep(0.02) # 50Hz

            # 4. 放置逻辑 (此处简单演示为松开)
            time.sleep(0.5)
            print(">>> 释放")
            self.gripper.set_target(GRIPPER_OPEN)
            time.sleep(0.5)

        except InterruptedError:
            print("Action Interrupted")
        except Exception as e:
            print(f"Action Error: {e}")
        finally:
            self.robot_busy = False
            self.move_to_default()
            print(">>> 动作结束")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.robot_busy:
                print("Robot is Busy!")
                return
            
            # 1. 坐标映射: 预览(1280x960) -> 全图(8160x6120)
            # 比例: 8160/1280 = 6.375
            scale_w = 8160 / 1280
            scale_h = 6120 / 960
            
            real_u = x * scale_w
            real_v = y * scale_h
            
            print(f"Click Preview({x},{y}) -> Real({real_u:.1f}, {real_v:.1f})")
            
            # 2. 调用手眼标定转换
            # z_offset_mm=0 代表抓取桌面平面
            rob_pos = self.vision.pixel_to_robot(real_u, real_v, z_offset_mm=0)
            
            if rob_pos is not None:
                print(f"Calculated Robot Pos: {rob_pos}")
                # 3. 启动线程执行
                t = threading.Thread(target=self.execution_thread, args=(rob_pos,))
                t.start()
            else:
                print("坐标转换失败 (可能射线平行或超出范围)")

    def run(self):
        print("\n=== 系统运行中 ===")
        print("  鼠标左键: 点击物体进行抓取")
        print("  键盘 +/-: 调整曝光 (当前 440)")
        print("  键盘 * /: 调整对焦 (当前 240)")
        print("  Q / Esc : 退出")
        
        while self.running:
            # 读取 8K 帧
            ret, frame = self.cam_driver.read()
            if not ret:
                print("Error reading frame")
                break
            
            # 缩放用于显示
            preview = cv2.resize(frame, (1280, 960))
            
            # 绘制UI
            status_text = "BUSY" if self.robot_busy else "IDLE"
            color = (0, 0, 255) if self.robot_busy else (0, 255, 0)
            
            cv2.putText(preview, f"Status: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(preview, f"Exp: {self.cam_driver.current_exp} | Foc: {self.cam_driver.current_focus}", 
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow(self.window_name, preview)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]: # Q or Esc
                self.running = False
                self.stop_robot()
            elif key == ord('x'):
                self.stop_robot()
            
            # 曝光控制
            elif key == ord('+') or key == ord('='):
                self.cam_driver.set_exposure(self.cam_driver.current_exp + 20)
            elif key == ord('-'):
                self.cam_driver.set_exposure(self.cam_driver.current_exp - 20)
            
            # 对焦控制
            elif key == ord('*') or key == ord('8'):
                self.cam_driver.set_focus(self.cam_driver.current_focus + 10)
            elif key == ord('/') or key == ord('2'):
                self.cam_driver.set_focus(self.cam_driver.current_focus - 10)

        # 清理
        self.gripper.running = False
        self.cam_driver.release()
        self.node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = AutoPickApp()
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Main Loop Error: {e}")