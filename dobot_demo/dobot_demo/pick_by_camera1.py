# 问题 抓取逻辑错误：直接下坠，应该到位置之后下坠；回到位置之后松开夹爪，应该抬起后立刻松开
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
import threading
import time
import sys
from cv_bridge import CvBridge

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from dobot_msgs_v3.srv import *

# TF2 Imports (核心新增)
import tf2_ros
import tf2_geometry_msgs  # 必须导入这个，否则无法转换 PointStamped

# 配置参数
V_Z1 = 212.0  # 底部/抓取高度 (mm)
V_Z2 = 245.0  # 抬起/安全高度 (mm)
DEFAULT_POSE = [383.0, -61.0, V_Z2, 180.0, 0.0, 90.0]
GLOBAL_SPEED_RATIO = 30  # 全局速度 30%

# 修正参数 (仍然保留，用于微调 TF 转换后的误差)
delta_dict = {
    "default": [0, 0, 0],
    "2e742906": [0, -2, 10], 
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
        # 减少等待时间防止阻塞调试
        if not self.cli_mov_j.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Dobot services not available! (Running in simulation/vision only mode)")
        
    def call_service(self, client, request):
        if not client.service_is_ready(): return None
        future = client.call_async(request)
        # 简单的同步等待实现
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
        self.OPEN_VAL = 600
        self.CLOSE_VAL = 0
        
    def init_connection(self):
        # 尝试关闭旧连接并建立新连接
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
                import re
                match = re.search(r'(\d+)', str(res.index))
                self.id = int(match.group(1)) if match else int(res.index)
                self.write_reg(256, 1, "1", wait=True)
                self.write_reg(257, 1, "60", wait=True) 
                print(f"Gripper Connected ID: {self.id}")
            else:
                print("Gripper Connection Failed (Service might be offline)")
        except Exception as e:
            print(f"Gripper Init Error: {e}")

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
    """相机管理：ROS订阅 + TF坐标转换"""
    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()
        self.color_img = None
        self.depth_img = None
        self.camera_intrinsic = None # [fx, fy, cx, cy]
        self.inverse_K = None
        
        # TF2 初始化
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.device_uid = "default" 

        # 订阅
        # 注意：这里订阅的是修正后的 camera_info (如果你的 Fixer 节点在运行)
        self.sub_info = self.node.create_subscription(CameraInfo, "/camera/color/camera_info", self.info_callback, 10)
        self.sub_color = self.node.create_subscription(Image, "/camera/color/image_raw", self.color_callback, 10)
        self.sub_depth = self.node.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)
        
        self.node.get_logger().info("CameraManager Initialized. Waiting for TF and Images...")

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
            # 这里的编码通常是 16UC1 (mm)
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_img = img
        except Exception as e:
            print(e)

    def pixel_to_world(self, u, v):
        """
        核心函数：像素 -> 相机坐标 -> 查找TF -> 机器人基座坐标
        """
        if self.depth_img is None or self.inverse_K is None:
            print("Camera not ready (Missing Depth or Intrinsic)")
            return None

        # 1. 获取深度值 (mm)
        try:
            d = self.depth_img[v, u]
        except IndexError:
            return None
            
        if d == 0: 
            print("Zero depth at this pixel.")
            return None
        
        # 将深度转为浮点数
        depth_value = float(d) 
        # 注意：如果 TF 中的平移单位是米(m)，这里可能需要根据情况 /1000.0
        # 但 Dobot ROS 消息通常使用 mm 作为单位，且你的 static_transform_publisher 输入看起来像米 (0.455)，
        # 而 MovL 指令通常用 mm。
        # 这里存在一个常见的单位坑。假设深度图是 mm。
        
        # 2. 像素坐标 -> 相机坐标系 (Camera Frame)
        uv_point = np.array([[u, v]])
        z_axis = np.ones((1, 1))
        xyz_point = np.concatenate((uv_point, z_axis), axis=1) # [u, v, 1]
        
        # 计算相机坐标系下的点 (x_c, y_c, z_c) 单位: mm
        camera_point = np.dot(self.inverse_K, xyz_point.T).T
        camera_point *= depth_value 
        
        x_c, y_c, z_c = camera_point[0]

        # 3. 构建 PointStamped 消息 (准备进行 TF 变换)
        # 关键：我们需要把这个毫米单位的点转换成 TF 树使用的单位。
        # 通常 ROS 的 TF 树标准单位是 米 (m)。
        # 你的 publisher 参数：x=-0.777 (看起来是米)。
        # 因此，我们需要先将相机点转为米，变换后，再转回毫米给 Dobot 控制。
        
        point_cam = PointStamped()
        point_cam.header.stamp = self.node.get_clock().now().to_msg()
        # 必须与 static_transform_publisher 中的 child-frame-id 一致
        point_cam.header.frame_id = "camera_color_optical_frame" 
        
        point_cam.point.x = x_c / 1000.0
        point_cam.point.y = y_c / 1000.0
        point_cam.point.z = z_c / 1000.0

        # 4. TF 变换: Camera Frame -> Base Link
        try:
            # 查找最新的变换关系，超时时间 1.0秒
            # target_frame: "base_link" (机器人基座)
            # source_frame: "camera_color_optical_frame" (点所在的坐标系)
            transform = self.tf_buffer.lookup_transform(
                "base_link", 
                "camera_color_optical_frame",
                rclpy.time.Time(), # 获取最新时间
                timeout=Duration(seconds=1.0)
            )
            
            # 执行点的坐标变换
            point_base = tf2_geometry_msgs.do_transform_point(point_cam, transform)
            
            # 5. 提取结果并转回 mm (Dobot 控制单位)
            final_x = point_base.point.x * 1000.0
            final_y = point_base.point.y * 1000.0
            final_z = point_base.point.z * 1000.0
            
            # 6. 应用微调 Delta
            delta = delta_dict.get(self.device_uid, delta_dict["default"])
            
            world_point = np.array([final_x, final_y, final_z]) + np.array(delta)
            
            print(f"Cam(mm): [{x_c:.1f}, {y_c:.1f}, {z_c:.1f}] -> Base(mm): [{world_point[0]:.1f}, {world_point[1]:.1f}, {world_point[2]:.1f}]")
            return world_point

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(f"TF Transform Error: {e}")
            return None

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
        self.stop_flag = False 
        self.robot_busy = False
        
        # 机械臂初始化
        self.init_robot()
        
        # UI
        self.window_name = "Robot View (Click to Pick, 'x' to Stop)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def init_robot(self):
        print("Initializing Robot...")
        # 尝试调用服务，若失败则只作为视觉调试器运行
        try:
            self.node.call_service(self.node.cli_enable, EnableRobot.Request())
            self.node.call_service(self.node.cli_clear_error, ClearError.Request())
            self.node.call_service(self.node.cli_speed_factor, SpeedFactor.Request(ratio=GLOBAL_SPEED_RATIO))
            self.gripper.open() 
            self.move_to_default()
        except Exception:
            print("Robot init skipped (Service unavailable)")

    def move_to_default(self):
        print("Moving to Default Position...")
        if not self.node.cli_mov_j.service_is_ready(): return
        req = MovJ.Request()
        req.x, req.y, req.z = DEFAULT_POSE[0], DEFAULT_POSE[1], DEFAULT_POSE[2]
        req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
        self.node.call_service(self.node.cli_mov_j, req)
        self.node.call_service(self.node.cli_sync, Sync.Request())

    def stop_robot(self):
        print(">>> EMERGENCY STOP <<<")
        if self.node.cli_clear_error.service_is_ready():
            self.node.call_service(self.node.cli_clear_error, ClearError.Request())
        self.robot_busy = False

    def execution_thread(self, target_point):
        """执行抓取逻辑的线程"""
        if self.robot_busy: return
        self.robot_busy = True
        self.stop_flag = False
        
        tx, ty, tz = target_point
        
        # 安全检查: 如果目标点是 None，或者坐标极其异常，不执行
        if np.isnan(tx) or np.isnan(ty):
            print("Invalid coordinates")
            self.robot_busy = False
            return

        print(f"Starting Pick Sequence -> Target: {tx:.1f}, {ty:.1f}, {tz:.1f}")

        try:
            if self.stop_flag: raise InterruptedError("Stopped")

            # 1. 移动到目标XY，高度保持 V_Z1
            print(f"Moving to XY: {tx:.1f}, {ty:.1f}")
            req = MovL.Request()
            req.x, req.y, req.z = float(tx), float(ty), float(V_Z1)
            # 保持默认姿态
            req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            if self.stop_flag: raise InterruptedError("Stopped")

            # 2. 闭合夹爪 (抓取)
            print("Closing Gripper...")
            self.gripper.close()
            
            # 3. 抬升到 V_Z2
            print(f"Lifting to {V_Z2}...")
            req.z = float(V_Z2)
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # 4. 回到默认位置
            print("Returning to Default...")
            self.move_to_default()

            # 5. 松开
            print("Releasing Gripper...")
            self.gripper.open()
            
            print("Sequence Complete.")

        except Exception as e:
            print(f"Execution Error: {e}")
            self.stop_robot()
        finally:
            self.robot_busy = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.robot_busy:
                print("Robot is busy, please wait or press 'x'.")
                return
            
            # 计算世界坐标 (通过TF)
            world_pt = self.camera.pixel_to_world(x, y)
            
            if world_pt is not None:
                # 启动执行线程
                t = threading.Thread(target=self.execution_thread, args=(world_pt,))
                t.start()
            else:
                print("Could not calculate world coordinates (Check Depth/TF).")

    def run(self):
        print("App Running. Press 'q' to quit, 'x' to Stop/Reset.")
        
        while self.running:
            if self.camera.color_img is not None:
                display_img = self.camera.color_img.copy()
                
                status_text = "Status: " + ("BUSY" if self.robot_busy else "IDLE")
                cv2.putText(display_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(self.window_name, display_img)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                self.running = False
                self.stop_flag = True
                
            elif key == ord('x'):
                print(">>> X Pressed <<<")
                self.stop_flag = True
                if not self.robot_busy:
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