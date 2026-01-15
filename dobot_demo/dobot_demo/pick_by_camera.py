# 问题 抓取逻辑错误：直接下坠，应该到位置之后下坠；回到位置之后松开夹爪，应该抬起后立刻松开
# pkl文件解算手眼未验证
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import threading
import time
import pickle
import sys
import glob
import os
from cv_bridge import CvBridge

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped
from dobot_msgs_v3.srv import *

# 配置参数
V_Z1 = 212.0  # 底部/抓取高度 (mm)
V_Z2 = 245.0  # 抬起/安全高度 (mm)
DEFAULT_POSE = [383.0, -61.0, V_Z2, 180.0, 0.0, 90.0]
GLOBAL_SPEED_RATIO = 30  # 全局速度 30%

# 修正参数 (对应Windows代码中的 delta_dict)
# 请根据实际 Device UID 修改或添加，这里使用默认逻辑
delta_dict = {
    "default": [0, 0, 0],
    "2e742906": [0, -2, 10],
    "26fef4210": [0, 0, -2],
}

class DobotRosWrapper(Node):
    def __init__(self, node_name="click_and_pick_node"):
        super().__init__(node_name)
        
        # 客户端初始化
        self.cli_enable = self.create_client(EnableRobot, '/dobot_bringup_v3/srv/EnableRobot')
        self.cli_clear_error = self.create_client(ClearError, '/dobot_bringup_v3/srv/ClearError')
        self.cli_speed_factor = self.create_client(SpeedFactor, '/dobot_bringup_v3/srv/SpeedFactor')
        self.cli_mov_j = self.create_client(MovJ, '/dobot_bringup_v3/srv/MovJ')
        self.cli_mov_l = self.create_client(MovL, '/dobot_bringup_v3/srv/MovL')
        self.cli_servo_p = self.create_client(ServoP, '/dobot_bringup_v3/srv/ServoP')
        self.cli_sync = self.create_client(Sync, '/dobot_bringup_v3/srv/Sync')
        self.cli_modbus_create = self.create_client(ModbusCreate, '/dobot_bringup_v3/srv/ModbusCreate')
        self.cli_modbus_close = self.create_client(ModbusClose, '/dobot_bringup_v3/srv/ModbusClose')
        self.cli_set_hold_regs = self.create_client(SetHoldRegs, '/dobot_bringup_v3/srv/SetHoldRegs')

        self.get_logger().info("Waiting for Dobot services...")
        if not self.cli_mov_j.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Dobot services not available!")
            sys.exit(1)

    def call_service(self, client, request):
        if not client.service_is_ready(): return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.005)
        return future.result()

    def call_service_async(self, client, request):
        if client.service_is_ready():
            client.call_async(request)

class GripperController:
    """夹爪控制器 (Modbus)"""
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.id = 0
        self.init_connection()
        self.current_val = 0 # 0=Close, 1000=Open (Simulated logic from prompt)
        # Windows代码逻辑: 0=Close, 1000=Open (通常夹爪0是闭合，1000是张开，或者反之，需根据实际硬件调整)
        # 参考Linux脚本: Open=600~1000, Close=0. 
        self.OPEN_VAL = 600
        self.CLOSE_VAL = 0
        
    def init_connection(self):
        # Close old connections
        for i in range(1, 5):
            req = ModbusClose.Request()
            req.index = i
            self.wrapper.call_service(self.wrapper.cli_modbus_close, req)
        
        # Create new connection
        req = ModbusCreate.Request()
        req.ip = "127.0.0.1"
        req.port = 60000
        req.slave_id = 1
        req.is_rtu = 1
        res = self.wrapper.call_service(self.wrapper.cli_modbus_create, req)
        
        if res and res.res == 0:
            import re
            match = re.search(r'(\d+)', str(res.index))
            self.id = int(match.group(1)) if match else int(res.index)
            # Init Enable and Speed
            self.write_reg(256, 1, "1", wait=True)
            self.write_reg(257, 1, "60", wait=True) # Speed
            print(f"Gripper Connected ID: {self.id}")
        else:
            print("Gripper Connection Failed")

    def write_reg(self, addr, count, val_str, wait=True):
        if self.id <= 0: return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_str
        if wait:
            self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)
        else:
            self.wrapper.call_service_async(self.wrapper.cli_set_hold_regs, req)

    def open(self):
        self.write_reg(259, 1, str(self.OPEN_VAL), wait=True)
        time.sleep(0.5)

    def close(self):
        self.write_reg(259, 1, str(self.CLOSE_VAL), wait=True)
        time.sleep(0.5)

class CameraManager:
    """相机管理：ROS订阅 + 坐标转换"""
    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()
        self.color_img = None
        self.depth_img = None
        self.camera_intrinsic = None # [fx, fy, cx, cy]
        self.camera_extrinsic = None
        self.inverse_K = None
        self.device_uid = "default" # 默认UID，如果有真实UID需获取

        # 订阅
        self.sub_info = self.node.create_subscription(CameraInfo, "/camera/color/camera_info", self.info_callback, 10)
        self.sub_color = self.node.create_subscription(Image, "/camera/color/image_raw", self.color_callback, 10)
        self.sub_depth = self.node.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)
        
        # 尝试加载外参
        self.load_extrinsics()

    def info_callback(self, msg):
        if self.camera_intrinsic is None:
            K = msg.k
            # K is [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.camera_intrinsic = [K[0], K[4], K[2], K[5]]
            print(f"Camera Intrinsic Loaded: {self.camera_intrinsic}")
            self.inverse_K = np.array([
                [1 / self.camera_intrinsic[0], 0, -self.camera_intrinsic[2] / self.camera_intrinsic[0]],
                [0, 1 / self.camera_intrinsic[1], -self.camera_intrinsic[3] / self.camera_intrinsic[1]],
                [0, 0, 1],
            ], dtype=float)

    def color_callback(self, msg):
        try:
            self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            print(e)

    def depth_callback(self, msg):
        try:
            # ROS depth usually is uint16 (mm) or float32 (m). Orbbec ROS wrapper typically gives uint16 mm.
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_img = img
        except Exception as e:
            print(e)

    def load_extrinsics(self):
        """模拟Windows代码中的 pickle 加载与 lstsq 计算"""
        try:
            # 寻找 .pkl 文件
            pkl_files = glob.glob("./calibrate_points/*world_points.pkl")
            if not pkl_files:
                print("Warning: No calibration files found in ./calibrate_points/")
                return

            # 假设取第一个找到的文件的UID
            filename = os.path.basename(pkl_files[0])
            self.device_uid = filename.split('_')[0]
            print(f"Loading calibration for device: {self.device_uid}")

            with open(f'./calibrate_points/camera_points.pkl', 'rb') as f:
                camera_points = pickle.load(f)
                print(len(camera_points))
            with open(f'./calibrate_points/world_points.pkl', 'rb') as f:
                world_points = pickle.load(f)
                print(len(world_points))
            
            camera_points = np.array(camera_points)
            world_points = np.array(world_points)
            
            # 过滤无效点
            valid_indices = ~np.isnan(camera_points).any(axis=1)
            camera_points = camera_points[valid_indices]
            world_points = world_points[valid_indices]
            
            # 计算外参
            self.camera_extrinsic = np.linalg.lstsq(camera_points, world_points, rcond=None)[0]
            print("Extrinsics loaded successfully.")
            
        except Exception as e:
            print(f"Error loading extrinsics: {e}")

    def pixel_to_world(self, u, v):
        if self.depth_img is None or self.inverse_K is None or self.camera_extrinsic is None:
            print(self.depth_img, self.inverse_K, self.camera_extrinsic)
            print("Camera not ready (Missing Depth, Intrinsic or Extrinsic)")
            return None

        # 获取深度
        try:
            d = self.depth_img[v, u]
        except IndexError:
            return None
            
        if d == 0: return None
        
        # ROS uint16 is usually mm. The calculation relies on consistent units.
        # Assuming camera_points used for calibration were in the same unit as depth (mm).
        depth_value = float(d) 

        # Pixel to Camera
        uv_point = np.array([[u, v]])
        z_axis = np.ones((1, 1))
        xyz_point = np.concatenate((uv_point, z_axis), axis=1) # [u, v, 1]
        
        camera_point = np.dot(self.inverse_K, xyz_point.T).T
        camera_point *= depth_value # [x_c, y_c, z_c]
        
        # Camera to World
        # Add homogeneous coord for dot product [x, y, z, 1]
        camera_point_homo = np.concatenate((camera_point, np.ones((camera_point.shape[0], 1))), axis=1)
        
        world_coords = np.round(np.dot(camera_point_homo, self.camera_extrinsic), 3)
        
        # Apply Delta
        delta = delta_dict.get(self.device_uid, delta_dict["default"])
        world_point = world_coords[0, :3] + np.array(delta)
        
        return world_point

class PickApp:
    def __init__(self):
        rclpy.init()
        self.node = DobotRosWrapper()
        
        # 启动ROS Spin线程
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        # 初始化模块
        self.gripper = GripperController(self.node)
        self.camera = CameraManager(self.node)
        
        # 状态控制
        self.running = True
        self.stop_flag = False # 按下X停止
        self.reset_flag = False # 按下X复位
        self.robot_busy = False
        
        # 机械臂初始化
        self.init_robot()
        
        # UI
        self.window_name = "Robot View (Click to Pick, 'x' to Stop/Reset)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def init_robot(self):
        print("Initializing Robot...")
        self.node.call_service(self.node.cli_enable, EnableRobot.Request())
        self.node.call_service(self.node.cli_clear_error, ClearError.Request())
        self.node.call_service(self.node.cli_speed_factor, SpeedFactor.Request(ratio=GLOBAL_SPEED_RATIO))
        self.gripper.open() # 初始张开
        self.move_to_default()

    def move_to_default(self):
        print("Moving to Default Position...")
        # 使用 MovJ 安全归位
        req = MovJ.Request()
        req.x, req.y, req.z = DEFAULT_POSE[0], DEFAULT_POSE[1], DEFAULT_POSE[2]
        req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
        self.node.call_service(self.node.cli_mov_j, req)
        self.node.call_service(self.node.cli_sync, Sync.Request())

    def stop_robot(self):
        """紧急停止/清除错误"""
        # Dobot ROS通常没有直接的StopService，用ClearError或发送当前位置的指令来覆盖
        # 这里的Stop是逻辑层面的，不发新指令，并尝试清除队列
        print(">>> EMERGENCY STOP <<<")
        self.node.call_service(self.node.cli_clear_error, ClearError.Request())
        self.robot_busy = False

    def check_stop(self):
        if self.stop_flag:
            raise InterruptedError("Stopped by user")

    def execution_thread(self, target_point):
        """执行抓取逻辑的线程"""
        if self.robot_busy: return
        self.robot_busy = True
        self.stop_flag = False
        
        tx, ty, tz = target_point
        print(f"Starting Pick Sequence -> Target: {tx:.1f}, {ty:.1f}, {tz:.1f}")

        try:
            # 1. 移动到目标XY，高度保持 V_Z1 (MovL直线运动)
            self.check_stop()
            print(f"Moving to XY: {tx}, {ty} at Height {V_Z1}")
            req = MovL.Request()
            req.x, req.y, req.z = float(tx), float(ty), float(V_Z1)
            req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # 2. 执行抓取动作 (等同于控制脚本按下Z)
            # 逻辑: 当前已经在V_Z1 -> 闭合夹爪 -> 抬升到V_Z2
            self.check_stop()
            print("Closing Gripper...")
            self.gripper.close()
            
            self.check_stop()
            print(f"Lifting to {V_Z2}...")
            req.z = float(V_Z2)
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # 3. 回到默认位置
            self.check_stop()
            print("Returning to Default...")
            self.move_to_default()

            # 4. 松开夹爪
            self.check_stop()
            print("Releasing Gripper...")
            self.gripper.open()
            
            print("Sequence Complete.")

        except InterruptedError:
            print("Sequence Interrupted by User!")
            self.stop_robot()
        except Exception as e:
            print(f"Execution Error: {e}")
            self.stop_robot()
        finally:
            self.robot_busy = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.robot_busy:
                print("Robot is busy, please wait or press 'x' to stop.")
                return
            
            # 计算世界坐标
            world_pt = self.camera.pixel_to_world(x, y)
            if world_pt is not None:
                print(f"Click Detected: {x}, {y} -> World: {world_pt}")
                # 启动执行线程
                t = threading.Thread(target=self.execution_thread, args=(world_pt,))
                t.start()
            else:
                print("Could not calculate world coordinates (Depth invalid or uncalibrated).")

    def run(self):
        print("App Running. Press 'q' to quit, 'x' to Stop/Reset.")
        
        while self.running:
            if self.camera.color_img is not None:
                display_img = self.camera.color_img.copy()
                
                # 简单的UI提示
                status_text = "Status: " + ("BUSY" if self.robot_busy else "IDLE")
                cv2.putText(display_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_img, f"Default Z: {V_Z1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                cv2.imshow(self.window_name, display_img)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                self.running = False
                self.stop_flag = True
                
            elif key == ord('x'):
                if self.robot_busy:
                    print(">>> X Pressed: STOPPING <<<")
                    self.stop_flag = True
                else:
                    print(">>> X Pressed: RESETTING TO DEFAULT <<<")
                    # 在非忙碌状态下按下X，强制复位
                    t = threading.Thread(target=self.move_to_default)
                    t.start()

        self.node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app = PickApp()
        app.run()
    except KeyboardInterrupt:
        pass