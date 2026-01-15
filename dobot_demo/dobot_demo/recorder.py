
import rclpy
from rclpy.node import Node
import h5py
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
import threading

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32
from dobot_msgs_v3.msg import ToolVectorActual

# Message Filters for time synchronization
from message_filters import ApproximateTimeSynchronizer, Subscriber

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder_node')

        # 参数配置
        self.data_dir = "./data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.bridge = CvBridge()

        # 状态控制
        # 0: Idle/Discard, 1: Recording, 2: Save
        self.recording_state = 0
        self.episode_buffer = [] # 暂存当前episode数据
        self.mutex = threading.Lock()
        self.last_camera_info = None

        # === 订阅器 ===
        # 1. 指令订阅器 (独立订阅,不参与时间同步)
        self.sub_cmd = self.create_subscription(Int32, "/recorder/command", self.cmd_callback, 10)

        # 2. 相机内参订阅器 (独立订阅,不参与时间同步)
        self.sub_info = self.create_subscription(CameraInfo, "/camera/color/camera_info", self.info_callback, 10)

        # 3. 使用 message_filters 进行时间同步的订阅器
        # 创建 message_filters Subscriber (而不是普通的 create_subscription)
        self.sub_color = Subscriber(self, Image, "/camera/color/image_raw")
        self.sub_depth = Subscriber(self, Image, "/camera/depth/image_raw")
        self.sub_robot_current = Subscriber(self, ToolVectorActual, "/dobot_msgs_v3/msg/ToolVectorActual")
        self.sub_robot_target = Subscriber(self, ToolVectorActual, "/robot/target_pose")
        self.sub_gripper_current = Subscriber(self, PointStamped, "/gripper/state_feedback")
        self.sub_gripper_target = Subscriber(self, PointStamped, "/gripper/command_update")

        # 4. 创建 ApproximateTimeSynchronizer
        # slop: 允许的最大时间差 (秒)
        # 相机是30Hz (33.3ms间隔), 机械臂/夹爪是100Hz (10ms间隔)
        # 设置slop为50ms (0.05秒) 来容忍网络延迟和不同频率
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_robot_current, self.sub_robot_target,
             self.sub_gripper_current, self.sub_gripper_target],
            queue_size=10,
            slop=0.05  # 50ms tolerance
        )
        self.sync.registerCallback(self.synchronized_callback)

        self.get_logger().info("=== VLA Data Recorder Initialized (ApproximateTimeSynchronizer) ===")
        self.get_logger().info("Time synchronization: ApproximateTimeSynchronizer with 50ms tolerance")
        self.get_logger().info("Subscribed topics (time-synchronized):")
        self.get_logger().info("  - Camera (30Hz): /camera/color/image_raw, /camera/depth/image_raw")
        self.get_logger().info("  - Robot Current (100Hz): /dobot_msgs_v3/msg/ToolVectorActual")
        self.get_logger().info("  - Robot Target (100Hz): /robot/target_pose")
        self.get_logger().info("  - Gripper Current (100Hz): /gripper/state_feedback")
        self.get_logger().info("  - Gripper Target (100Hz): /gripper/command_update")
        self.get_logger().info("Press 'O' in controller to start recording.")

    # === 回调函数 ===

    def cmd_callback(self, msg):
        """处理录制指令"""
        new_state = msg.data
        with self.mutex:
            if new_state == 1 and self.recording_state != 1:
                self.get_logger().info(">>> START RECORDING")
                self.episode_buffer = [] # 清空缓存
                self.recording_state = 1

            elif new_state == 2 and self.recording_state == 1:
                self.get_logger().info(">>> STOP & SAVING...")
                self.recording_state = 2 # 标记为保存中
                self.save_dataset()
                self.recording_state = 0 # 回到空闲

            elif new_state == 0:
                if self.recording_state == 1:
                    self.get_logger().info(">>> STOP & DISCARD")
                self.recording_state = 0
                self.episode_buffer = []

    def info_callback(self, msg):
        """缓存相机内参"""
        self.last_camera_info = msg

    def synchronized_callback(self, color_msg, depth_msg, robot_current_msg, robot_target_msg,
                            gripper_current_msg, gripper_target_msg):
        """
        时间同步回调函数
        当所有传感器数据的时间戳在50ms容差内对齐时调用此函数

        参数说明：
        - color_msg: Image - RGB图像
        - depth_msg: Image - 深度图像
        - robot_current_msg: ToolVectorActual - 机械臂当前位姿
        - robot_target_msg: ToolVectorActual - 机械臂目标位姿
        - gripper_current_msg: PointStamped - 夹爪当前状态
        - gripper_target_msg: PointStamped - 夹爪目标状态

        时间对齐方案：
        - 使用 ApproximateTimeSynchronizer 进行基于时间戳的软同步
        - slop=50ms: 允许50ms的时间戳差异
        - 算法会选择时间戳最接近的消息组合进行打包
        """
        if self.recording_state != 1:
            return

        with self.mutex:
            try:
                # 转换图像
                cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                cv_depth = None
                if depth_msg:
                    cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

                # 提取时间戳 (使用color图像的时间戳作为参考)
                timestamp = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9

                # 打包数据
                frame_data = {
                    'timestamp': timestamp,
                    'image': cv_image,
                    'depth': cv_depth,

                    # 机械臂当前状态 (x, y, z, rx, ry, rz)
                    'robot_current': np.array([
                        robot_current_msg.x,
                        robot_current_msg.y,
                        robot_current_msg.z,
                        robot_current_msg.rx,
                        robot_current_msg.ry,
                        robot_current_msg.rz
                    ], dtype=np.float32),

                    # 机械臂目标状态 (x, y, z, rx, ry, rz)
                    'robot_target': np.array([
                        robot_target_msg.x,
                        robot_target_msg.y,
                        robot_target_msg.z,
                        robot_target_msg.rx,
                        robot_target_msg.ry,
                        robot_target_msg.rz
                    ], dtype=np.float32),

                    # 夹爪当前状态 (position: 0-1000)
                    'gripper_current': np.array([gripper_current_msg.point.x], dtype=np.float32),

                    # 夹爪目标状态 (position: 0-1000)
                    'gripper_target': np.array([gripper_target_msg.point.x], dtype=np.float32),

                    # [调试信息] 记录各传感器的时间戳，便于验证时间对齐
                    'timestamps_debug': {
                        'color': timestamp,
                        'depth': depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9,
                        'robot_current': robot_current_msg.header.stamp.sec + robot_current_msg.header.stamp.nanosec * 1e-9,
                        'robot_target': robot_target_msg.header.stamp.sec + robot_target_msg.header.stamp.nanosec * 1e-9,
                        'gripper_current': gripper_current_msg.header.stamp.sec + gripper_current_msg.header.stamp.nanosec * 1e-9,
                        'gripper_target': gripper_target_msg.header.stamp.sec + gripper_target_msg.header.stamp.nanosec * 1e-9,
                    }
                }

                # 如果有相机内参，也记录
                if self.last_camera_info:
                    frame_data['camera_intrinsics'] = np.array(self.last_camera_info.k, dtype=np.float32).reshape(3, 3)

                self.episode_buffer.append(frame_data)

                # 打印进度和时间戳差异 (用于验证同步质量)
                if len(self.episode_buffer) % 30 == 0:
                    ts_debug = frame_data['timestamps_debug']
                    max_diff = max(ts_debug.values()) - min(ts_debug.values())
                    print(f"Recording... {len(self.episode_buffer)} frames, max timestamp diff: {max_diff*1000:.2f}ms")

            except Exception as e:
                self.get_logger().error(f"Error in synchronized_callback: {e}")

    # === 保存逻辑 ===

    def get_next_episode_index(self):
        """遍历data目录，找到下一个可用的编号"""
        existing_files = [f for f in os.listdir(self.data_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
        if not existing_files:
            return 0

        indices = []
        for f in existing_files:
            try:
                idx = int(f.split('_')[1].split('.')[0])
                indices.append(idx)
            except: pass

        return max(indices) + 1 if indices else 0

    def save_dataset(self):
        if not self.episode_buffer:
            self.get_logger().warn("Buffer empty, nothing to save.")
            return

        idx = self.get_next_episode_index()
        file_path = os.path.join(self.data_dir, f"episode_{idx}.hdf5")

        self.get_logger().info(f"Saving {len(self.episode_buffer)} frames to {file_path}...")

        try:
            with h5py.File(file_path, 'w') as f:
                # 1. 提取各个数据流
                data_len = len(self.episode_buffer)

                # 预分配数据集 (Chunked storage for images)
                # 图像：(N, H, W, 3)
                img_sample = self.episode_buffer[0]['image']
                dset_img = f.create_dataset('observations/images/color',
                                          (data_len, img_sample.shape[0], img_sample.shape[1], 3),
                                          dtype='uint8', chunks=(1, img_sample.shape[0], img_sample.shape[1], 3))

                if self.episode_buffer[0]['depth'] is not None:
                    depth_sample = self.episode_buffer[0]['depth']
                    dset_depth = f.create_dataset('observations/images/depth',
                                                (data_len, depth_sample.shape[0], depth_sample.shape[1]),
                                                dtype='uint16', chunks=(1, depth_sample.shape[0], depth_sample.shape[1]))

                # 机械臂状态：当前 + 目标 (x, y, z, rx, ry, rz)
                dset_robot_current = f.create_dataset('observations/robot_current', (data_len, 6), dtype='float32')
                dset_robot_target = f.create_dataset('actions/robot_target', (data_len, 6), dtype='float32')

                # 夹爪状态：当前 + 目标 (position: 0-1000)
                dset_gripper_current = f.create_dataset('observations/gripper_current', (data_len, 1), dtype='float32')
                dset_gripper_target = f.create_dataset('actions/gripper_target', (data_len, 1), dtype='float32')

                # 时间戳
                dset_time = f.create_dataset('timestamp', (data_len,), dtype='float64')

                # 2. 写入数据
                for i, frame in enumerate(self.episode_buffer):
                    dset_img[i] = frame['image']
                    if frame['depth'] is not None:
                        dset_depth[i] = frame['depth']

                    dset_robot_current[i] = frame['robot_current']
                    dset_robot_target[i] = frame['robot_target']
                    dset_gripper_current[i] = frame['gripper_current']
                    dset_gripper_target[i] = frame['gripper_target']
                    dset_time[i] = frame['timestamp']

                # 保存相机内参 (取第一帧即可)
                if 'camera_intrinsics' in self.episode_buffer[0]:
                    f.create_dataset('camera/intrinsics', data=self.episode_buffer[0]['camera_intrinsics'])

                # 保存元数据
                f.attrs['sim'] = False
                f.attrs['total_frames'] = data_len
                f.attrs['sync_method'] = 'ApproximateTimeSynchronizer'
                f.attrs['sync_tolerance_ms'] = 50.0
                f.attrs['camera_frequency_hz'] = 30
                f.attrs['robot_frequency_hz'] = 100

            self.get_logger().info(f"Save successfully! Saved to {file_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to save dataset: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
