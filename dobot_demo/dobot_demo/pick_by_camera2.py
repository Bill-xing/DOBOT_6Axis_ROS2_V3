import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
import threading
import time
import re
import sys
from cv_bridge import CvBridge

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped 
from std_msgs.msg import Int32
from dobot_msgs_v3.srv import *

# TF2 Imports
import tf2_ros
import tf2_geometry_msgs

# ================= 配置参数 =================
# 高度定义 (mm)
V_Z1 = 212.0  # 底部/抓取高度 (下潜最低点)
V_Z2 = 245.0  # 抬起/安全高度 (移动平面)

# 抓取参数 (参考第二份代码)
GRIPPER_OPEN = 600.0   # 张开值
GRIPPER_CLOSE = 0.0    # 闭合值 (或根据物体调整)
V_SPEED_UP = 0.1       # 上升速度 (mm per tick)
V_RG = 80.0            # 夹爪闭合速度 (value per tick)
LOOP_RATE = 0.02       # 动态控制周期 (s) -> 50Hz

# 默认姿态
DEFAULT_POSE = [383.0, -61.0, V_Z2, 180.0, 0.0, 90.0]
GLOBAL_SPEED_RATIO = 50 

# 坐标微调
delta_dict = {
    "default": [0, 0, 0],
}

# ================= ROS 包装类 =================
class DobotRosWrapper(Node):
    def __init__(self, node_name="click_and_pick_node"):
        super().__init__(node_name)
        
        # 服务客户端
        self.cli_enable = self.create_client(EnableRobot, '/dobot_bringup_v3/srv/EnableRobot')
        self.cli_clear_error = self.create_client(ClearError, '/dobot_bringup_v3/srv/ClearError')
        self.cli_speed_factor = self.create_client(SpeedFactor, '/dobot_bringup_v3/srv/SpeedFactor')
        self.cli_mov_j = self.create_client(MovJ, '/dobot_bringup_v3/srv/MovJ')
        self.cli_mov_l = self.create_client(MovL, '/dobot_bringup_v3/srv/MovL')
        self.cli_servo_p = self.create_client(ServoP, '/dobot_bringup_v3/srv/ServoP') # 用于动态抓取
        self.cli_sync = self.create_client(Sync, '/dobot_bringup_v3/srv/Sync')
        
        # Modbus 相关
        self.cli_modbus_create = self.create_client(ModbusCreate, '/dobot_bringup_v3/srv/ModbusCreate')
        self.cli_modbus_close = self.create_client(ModbusClose, '/dobot_bringup_v3/srv/ModbusClose')
        self.cli_set_hold_regs = self.create_client(SetHoldRegs, '/dobot_bringup_v3/srv/SetHoldRegs')
        self.cli_get_hold_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs')

        # 发布夹爪状态 (用于调试/记录)
        self.pub_gripper_update = self.create_publisher(PointStamped, "/gripper/command_update", 5)

        self.get_logger().info("Waiting for Dobot services...")
        if not self.cli_mov_j.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Dobot services not available! (Running in simulation/vision only mode)")
        
    def call_service(self, client, request):
        if not client.service_is_ready(): return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.002)
        return future.result()

    def call_service_async_no_wait(self, client, request):
        if client.service_is_ready():
            client.call_async(request)

# ================= 夹爪通信与控制 (移植自新版代码) =================
class GripperComm:
    """底层 Modbus 通信"""
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.id = 0
        self.init_connection()

    def init_connection(self):
        try:
            # 关闭旧连接
            for i in range(1, 5):
                req = ModbusClose.Request()
                req.index = i
                self.wrapper.call_service(self.wrapper.cli_modbus_close, req)

            # 创建新连接
            req = ModbusCreate.Request()
            req.ip = "127.0.0.1"
            req.port = 60000
            req.slave_id = 1
            req.is_rtu = 1
            res = self.wrapper.call_service(self.wrapper.cli_modbus_create, req)

            if res and res.res == 0:
                match = re.search(r'(\d+)', str(res.index))
                self.id = int(match.group(1)) if match else int(res.index)
                # 初始化
                self.write_reg(256, 1, "1", wait=True)  # Enable
                self.write_reg(257, 1, "60", wait=True) # Speed
                print(f"[Gripper] Connected ID: {self.id}")
            else:
                print(f"[Gripper] Connection Failed")
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
    独立夹爪线程：负责执行自动指令
    """
    def __init__(self, wrapper):
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.comm = GripperComm(wrapper)
        self.running = True
        
        self.current_pos = 1000.0
        
        # 自动控制接口
        self.target_pos = 1000.0  # 目标位置

        # 初始化同步
        init_val = self.comm.read_reg(514)
        if init_val is not None:
            self.current_pos = float(init_val)
            self.target_pos = self.current_pos

    def publish_state(self, pos):
        msg = PointStamped()
        msg.header.stamp = self.wrapper.get_clock().now().to_msg()
        msg.header.frame_id = "gripper"
        msg.point.x = float(pos)
        self.wrapper.pub_gripper_update.publish(msg)

    def set_target(self, pos):
        """设置期望的夹爪位置"""
        self.target_pos = float(pos)

    def run(self):
        while self.running:
            # 简单的 P 控制或直接赋值，这里直接发指令
            # 只有当目标值改变较大时才频繁发送，或者维持低频刷新
            # 在新版逻辑中，为了配合 ServoP，我们需要高频响应
            
            # 发送指令
            self.comm.write_reg(259, 1, str(int(self.target_pos)), wait=False)
            self.publish_state(self.target_pos)
            
            # 记录当前值 (模拟)
            self.current_pos = self.target_pos
            
            # 50Hz 刷新率
            time.sleep(0.02)

# ================= 视觉与 TF 管理 (保留原第一版逻辑) =================
class CameraManager:
    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()
        self.color_img = None
        self.depth_img = None
        self.camera_intrinsic = None
        self.inverse_K = None
        
        # TF2 初始化
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.device_uid = "default" 

        # 订阅
        self.sub_info = self.node.create_subscription(CameraInfo, "/camera/color/camera_info", self.info_callback, 10)
        self.sub_color = self.node.create_subscription(Image, "/camera/color/image_raw", self.color_callback, 10)
        self.sub_depth = self.node.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)
        
        self.node.get_logger().info("CameraManager Initialized. Waiting for TF and Images...")

    def info_callback(self, msg):
        if self.camera_intrinsic is None:
            K = msg.k
            self.camera_intrinsic = [K[0], K[4], K[2], K[5]]
            self.inverse_K = np.array([
                [1 / self.camera_intrinsic[0], 0, -self.camera_intrinsic[2] / self.camera_intrinsic[0]],
                [0, 1 / self.camera_intrinsic[1], -self.camera_intrinsic[3] / self.camera_intrinsic[1]],
                [0, 0, 1],
            ], dtype=float)

    def color_callback(self, msg):
        try:
            self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception: pass

    def depth_callback(self, msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception: pass

    def pixel_to_world(self, u, v):
        """像素 -> 相机坐标 -> TF变换 -> 基座坐标"""
        if self.depth_img is None or self.inverse_K is None:
            print("Camera not ready.")
            return None

        try:
            d = self.depth_img[v, u]
        except IndexError:
            return None
            
        if d == 0: return None
        depth_value = float(d) 

        # 1. 像素 -> 相机坐标 (mm)
        uv_point = np.array([[u, v]])
        z_axis = np.ones((1, 1))
        xyz_point = np.concatenate((uv_point, z_axis), axis=1)
        camera_point = np.dot(self.inverse_K, xyz_point.T).T
        camera_point *= depth_value 
        x_c, y_c, z_c = camera_point[0]

        # 2. 构建消息 (转为米进行TF变换)
        point_cam = PointStamped()
        point_cam.header.stamp = self.node.get_clock().now().to_msg()
        point_cam.header.frame_id = "camera_color_optical_frame"
        point_cam.point.x = x_c / 1000.0
        point_cam.point.y = y_c / 1000.0
        point_cam.point.z = z_c / 1000.0

        # 3. TF 变换
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link", "camera_color_optical_frame",
                rclpy.time.Time(), timeout=Duration(seconds=1.0)
            )
            point_base = tf2_geometry_msgs.do_transform_point(point_cam, transform)
            
            # 4. 转回 mm
            world_point = np.array([
                point_base.point.x * 1000.0,
                point_base.point.y * 1000.0,
                point_base.point.z * 1000.0
            ])
            
            # 应用偏移
            delta = delta_dict.get("default", [0,0,0])
            world_point += np.array(delta)
            
            print(f"World Coord: {world_point}")
            return world_point

        except Exception as e:
            print(f"TF Error: {e}")
            return None

# ================= 主控制应用 =================
class PickApp:
    def __init__(self):
        rclpy.init()
        self.node = DobotRosWrapper()
        
        # ROS Spin
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        # 模块初始化
        self.gripper_manager = GripperManager(self.node)
        self.gripper_manager.start() # 启动夹爪线程
        
        self.camera = CameraManager(self.node)
        
        # 状态
        self.running = True
        self.stop_flag = False 
        self.robot_busy = False
        
        self.init_robot()
        
        # UI
        self.window_name = "Auto Pick (Click to Pick, X to Stop)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def init_robot(self):
        print("Initializing Robot...")
        try:
            self.node.call_service(self.node.cli_enable, EnableRobot.Request())
            self.node.call_service(self.node.cli_clear_error, ClearError.Request())
            self.node.call_service(self.node.cli_speed_factor, SpeedFactor.Request(ratio=GLOBAL_SPEED_RATIO))
            self.gripper_manager.set_target(GRIPPER_OPEN) # 初始张开
            self.move_to_default()
        except Exception as e:
            print(f"Robot Init Failed: {e}")

    def move_to_default(self):
        if self.robot_busy: return
        req = MovJ.Request()
        req.x, req.y, req.z = DEFAULT_POSE[0], DEFAULT_POSE[1], DEFAULT_POSE[2]
        req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
        self.node.call_service(self.node.cli_mov_j, req)
        self.node.call_service(self.node.cli_sync, Sync.Request())

    def stop_robot(self):
        print(">>> STOP <<<")
        self.node.call_service(self.node.cli_clear_error, ClearError.Request())
        self.robot_busy = False

    def execution_thread(self, target_point):
        """
        [核心修改] 
        新的抓取逻辑：
        1. MovL 平移到 (TargetX, TargetY, V_Z2)
        2. MovL 下降到 (TargetX, TargetY, V_Z1) [夹爪张开]
        3. ServoP 循环: 边上升(-> V_Z2) 边闭合夹爪
        4. Wait 0.5s -> Open -> Return
        """
        if self.robot_busy: return
        self.robot_busy = True
        self.stop_flag = False
        
        tx, ty, tz = target_point
        # 忽略计算出的 Z，使用预设的作业高度 V_Z1/V_Z2
        
        if np.isnan(tx) or np.isnan(ty):
            print("Invalid coordinates")
            self.robot_busy = False
            return

        print(f"Action -> Target XY: {tx:.1f}, {ty:.1f}")

        try:
            # -------------------------------------------------
            # 1. 移动到 XY 平面安全高度 (V_Z2)
            # -------------------------------------------------
            if self.stop_flag: raise InterruptedError()
            
            # 确保夹爪是张开的
            self.gripper_manager.set_target(GRIPPER_OPEN)
            
            req = MovL.Request()
            req.x, req.y, req.z = float(tx), float(ty), float(V_Z2)
            req.rx, req.ry, req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
            
            print(f"Step 1: Moving to Approach Position ({V_Z2})")
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # -------------------------------------------------
            # 2. 垂直下降到 V_Z1
            # -------------------------------------------------
            if self.stop_flag: raise InterruptedError()
            
            print(f"Step 2: Descending to Bottom ({V_Z1})")
            req.z = float(V_Z1)
            self.node.call_service(self.node.cli_mov_l, req)
            self.node.call_service(self.node.cli_sync, Sync.Request())

            # -------------------------------------------------
            # 3. 动态上升 + 闭合夹爪 (ServoP Loop)
            # -------------------------------------------------
            print("Step 3: Lifting & Closing Gripper...")
            
            current_z = V_Z1
            current_grip = GRIPPER_OPEN
            
            # 循环直到高度回到 V_Z2
            while current_z < V_Z2:
                if self.stop_flag: raise InterruptedError()

                # 上升 Z (速率 rz)
                if abs(current_z - V_Z1) <= V_SPEED_UP * 2:
                    current_z += V_SPEED_UP / 2
                else:
                    current_z += V_SPEED_UP
                
                # 计算下一帧状态
                current_z += V_SPEED_UP
                current_grip -= V_RG
                
                # 限制范围
                if current_z > V_Z2: current_z = V_Z2
                if current_grip < GRIPPER_CLOSE: current_grip = GRIPPER_CLOSE
                
                # 1. 发送机械臂运动指令 (ServoP)
                sp_req = ServoP.Request()
                sp_req.x, sp_req.y, sp_req.z = float(tx), float(ty), float(current_z)
                sp_req.rx, sp_req.ry, sp_req.rz = DEFAULT_POSE[3], DEFAULT_POSE[4], DEFAULT_POSE[5]
                self.node.call_service_async_no_wait(self.node.cli_servo_p, sp_req)
                
                # 2. 发送夹爪指令 (通过 Manager)
                self.gripper_manager.set_target(current_grip)
                
                # 控制循环频率
                time.sleep(LOOP_RATE)

            # -------------------------------------------------
            # 4. 完成抓取，等待松手
            # -------------------------------------------------
            print("Step 4: Hold (0.5s)")
            time.sleep(0.5)
            
            print("Step 5: Releasing")
            self.gripper_manager.set_target(GRIPPER_OPEN)
            time.sleep(0.5) # 等待张开
            
            print("Sequence Complete.")

        except InterruptedError:
            print("Sequence Interrupted!")
            self.stop_robot()
        except Exception as e:
            print(f"Execution Error: {e}")
            self.stop_robot()
        finally:
            self.robot_busy = False
            print("Step 6: Return Home")
            self.move_to_default()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.robot_busy:
                print("Busy...")
                return
            
            world_pt = self.camera.pixel_to_world(x, y)
            if world_pt is not None:
                t = threading.Thread(target=self.execution_thread, args=(world_pt,))
                t.start()
            else:
                print("Invalid Depth/TF.")

    def run(self):
        print("App Running...")
        while self.running:
            if self.camera.color_img is not None:
                display_img = self.camera.color_img.copy()
                status = "BUSY" if self.robot_busy else "IDLE"
                cv2.putText(display_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(self.window_name, display_img)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self.running = False
                self.stop_flag = True
            elif key == ord('x'):
                self.stop_flag = True
                if not self.robot_busy:
                    t = threading.Thread(target=self.move_to_default)
                    t.start()

        self.gripper_manager.running = False
        self.node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app = PickApp()
        app.run()
    except KeyboardInterrupt:
        pass