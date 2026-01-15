#!/usr/bin/env python3
"""
简化版 recorder - 用于测试消息接收
不使用 message_filters，而是手动进行时间对齐
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import h5py
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
import threading
from collections import deque
import time

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32
from dobot_msgs_v3.msg import ToolVectorActual

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder_node')

        # 参数配置
        self.data_dir = "./data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.bridge = CvBridge()

        # 状态控制
        self.recording_state = 0
        self.episode_buffer = []
        self.mutex = threading.Lock()
        self.last_camera_info = None

        # 使用带时间戳的缓存队列
        self.msg_buffer = {
            'color': deque(maxlen=60),  # 2秒@30Hz
            'depth': deque(maxlen=60),
            'robot_current': deque(maxlen=200),  # 2秒@100Hz
            'robot_target': deque(maxlen=200),
            'gripper_current': deque(maxlen=200),
            'gripper_target': deque(maxlen=200),
        }

        # 消息接收计数
        self.msg_count = {
            'color': 0, 'depth': 0, 'robot_current': 0,
            'robot_target': 0, 'gripper_current': 0, 'gripper_target': 0
        }

        # QoS 配置 - 传感器数据通常使用 BEST_EFFORT
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # === 订阅器 ===
        self.sub_cmd = self.create_subscription(Int32, "/recorder/command", self.cmd_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, "/camera/color/camera_info", self.info_callback, 10)

        # 使用普通订阅器 + 手动时间对齐
        self.sub_color = self.create_subscription(Image, "/camera/color/image_raw",
                                                   self.color_callback, sensor_qos)
        self.sub_depth = self.create_subscription(Image, "/camera/depth/image_raw",
                                                   self.depth_callback, sensor_qos)
        self.sub_robot_current = self.create_subscription(ToolVectorActual, "/dobot_msgs_v3/msg/ToolVectorActual",
                                                          self.robot_current_callback, sensor_qos)
        self.sub_robot_target = self.create_subscription(ToolVectorActual, "/robot/target_pose",
                                                         self.robot_target_callback, sensor_qos)
        self.sub_gripper_current = self.create_subscription(PointStamped, "/gripper/state_feedback",
                                                            self.gripper_current_callback, sensor_qos)
        self.sub_gripper_target = self.create_subscription(PointStamped, "/gripper/command_update",
                                                           self.gripper_target_callback, sensor_qos)

        self.get_logger().info("=== VLA Data Recorder Initialized (Manual Time Alignment) ===")
        self.get_logger().info("Time synchronization: Manual matching with 50ms tolerance")
        self.get_logger().info("QoS: BEST_EFFORT + VOLATILE")
        self.get_logger().info("Press 'O' to start recording.")

        # 调试定时器
        self.create_timer(2.0, self.debug_callback)

    def get_timestamp(self, msg):
        """提取消息的时间戳（秒）"""
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def cmd_callback(self, msg):
        """处理录制指令"""
        new_state = msg.data
        with self.mutex:
            if new_state == 1 and self.recording_state != 1:
                self.get_logger().info(">>> START RECORDING")
                self.episode_buffer = []
                self.recording_state = 1

            elif new_state == 2 and self.recording_state == 1:
                self.get_logger().info(">>> STOP & SAVING...")
                self.recording_state = 2
                self.save_dataset()
                self.recording_state = 0

            elif new_state == 0:
                if self.recording_state == 1:
                    self.get_logger().info(">>> STOP & DISCARD")
                self.recording_state = 0
                self.episode_buffer = []

    def info_callback(self, msg):
        self.last_camera_info = msg

    def color_callback(self, msg):
        """RGB图像回调 - 触发数据对齐和录制"""
        self.msg_count['color'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['color'].append((timestamp, msg))

        # 以相机帧为基准触发数据对齐
        if self.recording_state == 1:
            self.try_align_and_record(timestamp)

    def depth_callback(self, msg):
        self.msg_count['depth'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['depth'].append((timestamp, msg))

    def robot_current_callback(self, msg):
        self.msg_count['robot_current'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['robot_current'].append((timestamp, msg))

    def robot_target_callback(self, msg):
        self.msg_count['robot_target'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['robot_target'].append((timestamp, msg))

    def gripper_current_callback(self, msg):
        self.msg_count['gripper_current'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['gripper_current'].append((timestamp, msg))

    def gripper_target_callback(self, msg):
        self.msg_count['gripper_target'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['gripper_target'].append((timestamp, msg))

    def debug_callback(self):
        """定期打印各话题的接收状态"""
        self.get_logger().info(f"[DEBUG] Message counts (last 2s): "
                              f"color={self.msg_count['color']}, "
                              f"depth={self.msg_count['depth']}, "
                              f"robot_curr={self.msg_count['robot_current']}, "
                              f"robot_targ={self.msg_count['robot_target']}, "
                              f"grip_curr={self.msg_count['gripper_current']}, "
                              f"grip_targ={self.msg_count['gripper_target']}")

        self.get_logger().info(f"[DEBUG] Buffer sizes: "
                              f"color={len(self.msg_buffer['color'])}, "
                              f"robot_curr={len(self.msg_buffer['robot_current'])}, "
                              f"robot_targ={len(self.msg_buffer['robot_target'])}")

        # 重置计数
        for key in self.msg_count:
            self.msg_count[key] = 0

    def find_closest_msg(self, buffer, target_time, tolerance=0.05):
        """在缓冲区中查找时间戳最接近目标时间的消息"""
        if not buffer:
            return None

        best_match = None
        min_diff = float('inf')

        for timestamp, msg in buffer:
            diff = abs(timestamp - target_time)
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                best_match = (timestamp, msg)

        return best_match

    def try_align_and_record(self, reference_time):
        """尝试对齐所有传感器数据并录制"""
        with self.mutex:
            tolerance = 0.05  # 50ms

            # 查找与参考时间最接近的消息
            color_match = self.find_closest_msg(self.msg_buffer['color'], reference_time, tolerance)
            depth_match = self.find_closest_msg(self.msg_buffer['depth'], reference_time, tolerance)
            robot_current_match = self.find_closest_msg(self.msg_buffer['robot_current'], reference_time, tolerance)
            robot_target_match = self.find_closest_msg(self.msg_buffer['robot_target'], reference_time, tolerance)
            gripper_current_match = self.find_closest_msg(self.msg_buffer['gripper_current'], reference_time, tolerance)
            gripper_target_match = self.find_closest_msg(self.msg_buffer['gripper_target'], reference_time, tolerance)

            # 检查必要数据是否存在
            if not all([color_match, robot_current_match, robot_target_match,
                       gripper_current_match, gripper_target_match]):
                return

            try:
                # 提取消息
                _, color_msg = color_match
                _, robot_current_msg = robot_current_match
                _, robot_target_msg = robot_target_match
                _, gripper_current_msg = gripper_current_match
                _, gripper_target_msg = gripper_target_match

                # 转换图像
                cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                cv_depth = None
                if depth_match:
                    _, depth_msg = depth_match
                    cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

                # 打包数据
                frame_data = {
                    'timestamp': reference_time,
                    'image': cv_image,
                    'depth': cv_depth,

                    'robot_current': np.array([
                        robot_current_msg.x, robot_current_msg.y, robot_current_msg.z,
                        robot_current_msg.rx, robot_current_msg.ry, robot_current_msg.rz
                    ], dtype=np.float32),

                    'robot_target': np.array([
                        robot_target_msg.x, robot_target_msg.y, robot_target_msg.z,
                        robot_target_msg.rx, robot_target_msg.ry, robot_target_msg.rz
                    ], dtype=np.float32),

                    'gripper_current': np.array([gripper_current_msg.point.x], dtype=np.float32),
                    'gripper_target': np.array([gripper_target_msg.point.x], dtype=np.float32),
                }

                if self.last_camera_info:
                    frame_data['camera_intrinsics'] = np.array(self.last_camera_info.k, dtype=np.float32).reshape(3, 3)

                self.episode_buffer.append(frame_data)

                # 打印进度
                if len(self.episode_buffer) % 30 == 0:
                    print(f"Recording... {len(self.episode_buffer)} frames")

            except Exception as e:
                self.get_logger().error(f"Error in try_align_and_record: {e}")

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
                data_len = len(self.episode_buffer)

                # 图像
                img_sample = self.episode_buffer[0]['image']
                dset_img = f.create_dataset('observations/images/color',
                                          (data_len, img_sample.shape[0], img_sample.shape[1], 3),
                                          dtype='uint8', chunks=(1, img_sample.shape[0], img_sample.shape[1], 3))

                if self.episode_buffer[0]['depth'] is not None:
                    depth_sample = self.episode_buffer[0]['depth']
                    dset_depth = f.create_dataset('observations/images/depth',
                                                (data_len, depth_sample.shape[0], depth_sample.shape[1]),
                                                dtype='uint16', chunks=(1, depth_sample.shape[0], depth_sample.shape[1]))

                # 机械臂和夹爪状态
                dset_robot_current = f.create_dataset('observations/robot_current', (data_len, 6), dtype='float32')
                dset_robot_target = f.create_dataset('actions/robot_target', (data_len, 6), dtype='float32')
                dset_gripper_current = f.create_dataset('observations/gripper_current', (data_len, 1), dtype='float32')
                dset_gripper_target = f.create_dataset('actions/gripper_target', (data_len, 1), dtype='float32')
                dset_time = f.create_dataset('timestamp', (data_len,), dtype='float64')

                # 写入数据
                for i, frame in enumerate(self.episode_buffer):
                    dset_img[i] = frame['image']
                    if frame['depth'] is not None:
                        dset_depth[i] = frame['depth']

                    dset_robot_current[i] = frame['robot_current']
                    dset_robot_target[i] = frame['robot_target']
                    dset_gripper_current[i] = frame['gripper_current']
                    dset_gripper_target[i] = frame['gripper_target']
                    dset_time[i] = frame['timestamp']

                if 'camera_intrinsics' in self.episode_buffer[0]:
                    f.create_dataset('camera/intrinsics', data=self.episode_buffer[0]['camera_intrinsics'])

                # 元数据
                f.attrs['sim'] = False
                f.attrs['total_frames'] = data_len
                f.attrs['sync_method'] = 'ManualTimeAlignment'
                f.attrs['sync_tolerance_ms'] = 50.0
                f.attrs['camera_frequency_hz'] = 30
                f.attrs['robot_frequency_hz'] = 100

            self.get_logger().info(f"✓ Save successfully! Saved to {file_path}")

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
