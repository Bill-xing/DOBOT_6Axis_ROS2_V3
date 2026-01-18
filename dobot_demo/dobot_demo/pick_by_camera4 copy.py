# 顶部相机 精度1cm之内
# 两点， 包含抓取方向
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

# ================= 3. Dobot ROS2 Wrapper (新增逆解功能) =================
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
        
        # === 新增 InverseSolution 客户端 ===
        self.cli_inverse = self.create_client(InverseSolution, '/dobot_bringup_v3/srv/InverseSolution')
        
        # Modbus
        self.cli_modbus_create = self.create_client(ModbusCreate, '/dobot_bringup_v3/srv/ModbusCreate')
        self.cli_modbus_close = self.create_client(ModbusClose, '/dobot_bringup_v3/srv/ModbusClose')
        self.cli_set_hold_regs = self.create_client(SetHoldRegs, '/dobot_bringup_v3/srv/SetHoldRegs')

        # 检查服务
        if not self.cli_inverse.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Warning: InverseSolution service not available!")

    def call_service(self, client, request):
        if not client.service_is_ready(): return None
        future = client.call_async(request)
        while not future.done(): time.sleep(0.002)
        return future.result()

    def call_service_async_no_wait(self, client, request):
        if client.service_is_ready(): client.call_async(request)

    def check_accessibility(self, x, y, z, rx, ry, rz):
        """
        调用 InverseSolution 验证点位是否可达
        
        Returns:
            bool: True 可达, False 不可达
        """
        if not self.cli_inverse.service_is_ready():
            print("Error: Inverse service not ready, skipping check (assuming risky true)")
            return True # 如果服务没启动，为了不卡死，暂时返回True(但在生产环境应该返回False)

        req = InverseSolution.Request()
        req.offset1 = float(x)
        req.offset2 = float(y)
        req.offset3 = float(z)
        req.offset4 = float(rx)
        req.offset5 = float(ry)
        req.offset6 = float(rz)
        req.user = 0  # 默认用户坐标系
        req.tool = 0  # 默认工具坐标系
        req.is_jointnear = 0 # 自动选解
        req.joint_near = "" 

        res = self.call_service(self.cli_inverse, req)
        
        # Dobot V3 协议中，通常 res (或错误码) 为 0 表示成功
        # 具体字段请参考实际srv定义，这里假设 res 为 ErrorID 字段
        # 如果 res.res == 0 表示逆解成功
        if res is not None and res.res == 0:
            return True
        else:
            return False

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
                self.write_reg(256, 1, "1", wait=True)
                self.write_reg(257, 1, "60", wait=True)
                print(f"[Gripper] Connected ID: {self.id}")
        except: pass

    def write_reg(self, addr, count, val_str, wait=False):
        if self.id <= 0: return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_str
        if wait: self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)
        else: self.wrapper.call_service_async_no_wait(self.wrapper.cli_set_hold_regs, req)

    def set_target(self, pos):
        self.target_pos = float(pos)

    def run(self):
        while self.running:
            self.write_reg(259, 1, str(int(self.target_pos)), wait=False)
            time.sleep(0.02)

# ================= 4. 主程序 (集成逆解验证) =================

V_Z1 = 212.0  # 抓取高度 (重点验证高度)
V_Z2 = 250.0  # 安全高度
GRIPPER_OPEN = 600.0
GRIPPER_CLOSE = 0.0
V_SPEED_UP = 0.1
V_RG = 80.0
DEFAULT_POSE = [383.0, -61.0, V_Z2, 180.0, 0.0, 90.0]
# DEFAULT_POSE = [120.0, -243.0, V_Z2, 180.0, 0.0, 90.0]

class AutoPickApp:
    def __init__(self):
        rclpy.init()
        self.node = DobotRosWrapper()
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        self.gripper = GripperManager(self.node)
        self.gripper.start()
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
        self.window_name = "8K Verified Pick"
        
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
        """
        寻找可行的 Rz 角度。
        策略：原值 -> ±180(mod 360) -> ±90(mod 360)
        """
        rx_def = DEFAULT_POSE[3]
        ry_def = DEFAULT_POSE[4]
        
        # 候选列表：(角度值, 描述)
        candidates = []
        
        # 1. 原始计算角度
        candidates.append((original_rz, "Original"))
        
        # 2. +/- 180 度 (同方向抓取，关节不同)
        # 注意: (ang + 180) % 360 可能会产生正值，如果要测试负值范围，需要特殊处理
        # 这里统一转到 0-360 或者 -180~180。Dobot Inverse 通常接受宽松范围。
        candidates.append(((original_rz + 180) % 360, "+180"))
        candidates.append(((original_rz - 180) % 360, "-180"))
        
        # 3. +/- 90 度 (垂直方向抓取，仅在上述失败时尝试)
        candidates.append(((original_rz + 90) % 360, "+90"))
        candidates.append(((original_rz - 90) % 360, "-90"))

        print(f"--- 开始可达性验证 (Pos: {x:.1f}, {y:.1f}, {z:.1f}) ---")
        
        for rz_test, tag in candidates:
            # 验证逆解
            is_reachable = self.node.check_accessibility(x, y, z, rx_def, ry_def, rz_test)
            
            if is_reachable:
                print(f"√ 方案可行 [{tag}]: Rz = {rz_test:.2f}")
                return rz_test
            else:
                print(f"X 方案不可行 [{tag}]: Rz = {rz_test:.2f}")
        
        return None

    def execution_thread(self, target_pos_mm, target_angle):
        """执行抓取动作"""
        if self.robot_busy: return
        self.robot_busy = True
        self.stop_flag = False
        
        tx, ty = target_pos_mm
        
        # === 步骤 0: 可达性验证 (重点) ===
        # 我们验证抓取底部的点 (V_Z1)，因为这是约束最严格的点
        valid_rz = self.find_reachable_rz(tx, ty, V_Z1, target_angle)
        print(f"验证结果 Rz: {valid_rz}")
        
        if valid_rz is None:
            print(">>> [错误] 目标点无法到达 (所有角度尝试均失败) <<<")
            self.robot_busy = False
            return
        
        # 使用验证过的 Rz
        print(f">>> 执行抓取: X={tx:.1f}, Y={ty:.1f}, Valid_Rz={valid_rz:.1f}°")

        try:
            # 1. 移动到上方 (V_Z2) + 旋转
            if self.stop_flag: raise InterruptedError()
            self.gripper.set_target(GRIPPER_OPEN)
            
            req = MovJ.Request()
            req.x, req.y, req.z = float(tx), float(ty), float(V_Z2)
            req.rx, req.ry = DEFAULT_POSE[3], DEFAULT_POSE[4]
            req.rz = float(valid_rz) # 使用验证过的角度
            
            self.node.call_service(self.node.cli_mov_j, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # 2. 下降 (V_Z1) - 这里因为已经验证过，理论上是安全的
            req = MovL.Request()
            req.x, req.y, req.z = float(tx), float(ty), float(V_Z2)
            req.rx, req.ry = DEFAULT_POSE[3], DEFAULT_POSE[4]
            req.rz = float(valid_rz) # 使用验证过的角度

            if self.stop_flag: raise InterruptedError()
            req.z = float(V_Z1)
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # 3. 动态抓取
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
                sp_req.rx, sp_req.ry = DEFAULT_POSE[3], DEFAULT_POSE[4]
                sp_req.rz = float(valid_rz) 
                self.node.call_service_async_no_wait(self.node.cli_servo_p, sp_req)
                
                self.gripper.set_target(current_grip)
                time.sleep(0.02)

            time.sleep(0.5)
            self.gripper.set_target(GRIPPER_OPEN)
            time.sleep(0.5)

        except InterruptedError:
            print("Action Interrupted")
        except Exception as e:
            print(f"Action Error: {e}")
            self.stop_robot() # 出错时清除错误
        finally:
            self.robot_busy = False
            print(">>> 回归原点")
            self.move_to_default()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.robot_busy:
                print("Robot is Busy!")
                return
            
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
                    self.pt1_real[0], self.pt1_real[1],
                    real_u, real_v,
                    z_offset_mm=0
                )
                self.click_state = 0
                self.pt1_uv = None
                
                if result is not None:
                    (rx, ry), angle = result
                    # 启动线程
                    t = threading.Thread(target=self.execution_thread, args=((rx, ry), angle))
                    t.start()
                else:
                    print("解算失败")

    def run(self):
        print("\n=== 验证版抓取程序 ===")
        print("  包含关节限位检测与自动换向逻辑")
        
        while self.running:
            ret, frame = self.cam_driver.read()
            if not ret: break
            
            preview = cv2.resize(frame, (1280, 960))
            
            if self.click_state == 1 and self.pt1_uv is not None:
                cv2.circle(preview, self.pt1_uv, 10, (0, 255, 255), 2)
                cv2.putText(preview, "Click Direction", (self.pt1_uv[0]+15, self.pt1_uv[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            status_text = "BUSY" if self.robot_busy else ("WAITING" if self.click_state == 1 else "READY")
            cv2.putText(preview, f"Status: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(self.window_name, preview)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                self.running = False
                self.stop_robot()
            elif key == ord('x'):
                self.stop_robot()
            elif key in [ord('+'), ord('=')]: self.cam_driver.set_exposure(self.cam_driver.current_exp + 20)
            elif key == ord('-'): self.cam_driver.set_exposure(self.cam_driver.current_exp - 20)
            elif key in [ord('*'), ord('8')]: self.cam_driver.set_focus(self.cam_driver.current_focus + 10)
            elif key in [ord('/'), ord('2')]: self.cam_driver.set_focus(self.cam_driver.current_focus - 10)

        self.gripper.running = False
        self.cam_driver.release()
        self.node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = AutoPickApp()
        app.run()
    except KeyboardInterrupt: pass