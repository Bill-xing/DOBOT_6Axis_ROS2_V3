#!/usr/bin/env python3
"""
播放评估录制器 - 评估数据集播放质量

功能：
在播放数据集时，同步录制机械臂和夹爪的目标状态与实际状态，
用于后续分析跟踪误差、延迟等性能指标。

录制数据：
1. 机械臂目标状态 - 来自播放器发布的 /robot/target_pose
2. 机械臂实际状态 - 来自机械臂反馈的 /dobot_msgs_v3/msg/ToolVectorActual
3. 夹爪目标状态 - 来自播放器发布的 /gripper/command_update
4. 夹爪实际状态 - 来自播放器发布的 /gripper/state_feedback

使用方法：
1. 启动评估录制器（在播放前启动）:
   python3 playback_evaluator.py

2. 在另一个终端播放数据集:
   python3 dataset_player.py data/episode_0.hdf5

3. 播放结束后，按 Ctrl+C 停止录制并保存

输出：
- playback_eval/evaluation_X.hdf5: 包含目标和实际状态的对比数据
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import h5py
import numpy as np
import os
import time
from collections import deque

# ROS Messages
from geometry_msgs.msg import PointStamped
from dobot_msgs_v3.msg import ToolVectorActual
from std_msgs.msg import Int32

class PlaybackEvaluator(Node):
    """
    播放评估录制器

    订阅目标和实际状态话题，同步录制用于评估播放质量
    """

    def __init__(self):
        super().__init__('playback_evaluator')

        # 配置
        self.data_dir = "./playback_eval"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # 录制状态
        self.is_recording = False
        self.episode_buffer = []

        # 消息缓冲区（用于时间对齐）
        self.msg_buffer = {
            'robot_target': deque(maxlen=100),
            'robot_actual': deque(maxlen=100),
            'gripper_target': deque(maxlen=100),
            'gripper_actual': deque(maxlen=100),
        }

        # 消息计数
        self.msg_count = {
            'robot_target': 0,
            'robot_actual': 0,
            'gripper_target': 0,
            'gripper_actual': 0,
        }

        # QoS配置
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 订阅器
        # 机械臂目标状态（来自播放器）
        self.sub_robot_target = self.create_subscription(
            ToolVectorActual,
            "/robot/target_pose",
            self.robot_target_callback,
            10
        )

        # 机械臂实际状态（来自机械臂驱动）
        self.sub_robot_actual = self.create_subscription(
            ToolVectorActual,
            "/dobot_msgs_v3/msg/ToolVectorActual",
            self.robot_actual_callback,
            sensor_qos
        )

        # 夹爪目标状态（来自播放器）
        self.sub_gripper_target = self.create_subscription(
            PointStamped,
            "/gripper/command_update",
            self.gripper_target_callback,
            10
        )

        # 夹爪实际状态（来自播放器的状态反馈）
        self.sub_gripper_actual = self.create_subscription(
            PointStamped,
            "/gripper/state_feedback",
            self.gripper_actual_callback,
            10
        )

        # 定时器：定期打印状态
        self.create_timer(2.0, self.status_callback)

        # 自动录制控制
        self.auto_start_triggered = False

        self.get_logger().info("=" * 70)
        self.get_logger().info("播放评估录制器已启动")
        self.get_logger().info("=" * 70)
        self.get_logger().info("等待播放器发布话题...")
        self.get_logger().info("提示：启动 dataset_player.py 开始播放，本节点将自动开始录制")
        self.get_logger().info("=" * 70)

    def get_timestamp(self, msg):
        """提取消息的时间戳（秒）"""
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    # ========== 回调函数 ==========

    def robot_target_callback(self, msg):
        """机械臂目标状态回调"""
        self.msg_count['robot_target'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['robot_target'].append((timestamp, msg))

        # 自动开始录制
        if not self.is_recording and not self.auto_start_triggered:
            self.get_logger().info("检测到播放开始，自动启动录制...")
            self.start_recording()
            self.auto_start_triggered = True

    def robot_actual_callback(self, msg):
        """机械臂实际状态回调"""
        self.msg_count['robot_actual'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['robot_actual'].append((timestamp, msg))

        # 如果正在录制，处理数据
        if self.is_recording:
            self.process_frame(timestamp)

    def gripper_target_callback(self, msg):
        """夹爪目标状态回调"""
        self.msg_count['gripper_target'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['gripper_target'].append((timestamp, msg))

    def gripper_actual_callback(self, msg):
        """夹爪实际状态回调"""
        self.msg_count['gripper_actual'] += 1
        timestamp = self.get_timestamp(msg)
        self.msg_buffer['gripper_actual'].append((timestamp, msg))

    # ========== 数据处理 ==========

    def find_closest_msg(self, buffer, target_time, tolerance=0.05):
        """
        查找最接近目标时间的消息

        Args:
            buffer: 消息缓冲区
            target_time: 目标时间戳
            tolerance: 时间容差（秒），默认50ms

        Returns:
            (timestamp, msg) 或 None
        """
        if not buffer:
            return None

        best_match = None
        best_diff = float('inf')

        for timestamp, msg in buffer:
            diff = abs(timestamp - target_time)
            if diff < best_diff:
                best_diff = diff
                best_match = (timestamp, msg)

        if best_diff <= tolerance:
            return best_match
        return None

    def process_frame(self, reference_time):
        """
        处理单帧数据：查找时间对齐的所有消息

        以机械臂实际状态的时间戳为基准，查找最接近的其他消息
        """
        try:
            # 查找对齐的消息（50ms容差）
            robot_target_match = self.find_closest_msg(
                self.msg_buffer['robot_target'], reference_time, tolerance=0.05
            )
            robot_actual_match = self.find_closest_msg(
                self.msg_buffer['robot_actual'], reference_time, tolerance=0.05
            )
            gripper_target_match = self.find_closest_msg(
                self.msg_buffer['gripper_target'], reference_time, tolerance=0.05
            )
            gripper_actual_match = self.find_closest_msg(
                self.msg_buffer['gripper_actual'], reference_time, tolerance=0.05
            )

            # 检查数据完整性
            if not all([robot_target_match, robot_actual_match,
                       gripper_target_match, gripper_actual_match]):
                return

            # 提取消息
            _, robot_target_msg = robot_target_match
            _, robot_actual_msg = robot_actual_match
            _, gripper_target_msg = gripper_target_match
            _, gripper_actual_msg = gripper_actual_match

            # 打包数据
            frame_data = {
                'timestamp': reference_time,

                # 机械臂目标状态
                'robot_target': np.array([
                    robot_target_msg.x, robot_target_msg.y, robot_target_msg.z,
                    robot_target_msg.rx, robot_target_msg.ry, robot_target_msg.rz
                ], dtype=np.float32),

                # 机械臂实际状态
                'robot_actual': np.array([
                    robot_actual_msg.x, robot_actual_msg.y, robot_actual_msg.z,
                    robot_actual_msg.rx, robot_actual_msg.ry, robot_actual_msg.rz
                ], dtype=np.float32),

                # 夹爪目标位置
                'gripper_target': np.array([gripper_target_msg.point.x], dtype=np.float32),

                # 夹爪实际位置
                'gripper_actual': np.array([gripper_actual_msg.point.x], dtype=np.float32),
            }

            # 添加到缓冲区
            self.episode_buffer.append(frame_data)

            # 每50帧打印一次进度
            if len(self.episode_buffer) % 50 == 0:
                self.get_logger().info(f"录制进度: {len(self.episode_buffer)} 帧")

        except Exception as e:
            self.get_logger().error(f"处理帧时出错: {e}")

    # ========== 录制控制 ==========

    def start_recording(self):
        """开始录制"""
        if not self.is_recording:
            self.is_recording = True
            self.episode_buffer = []
            self.get_logger().info(">>> 开始录制评估数据")

    def stop_recording(self):
        """停止录制并保存"""
        if self.is_recording:
            self.is_recording = False
            self.get_logger().info(">>> 停止录制")

            if len(self.episode_buffer) > 0:
                self.save_dataset()
            else:
                self.get_logger().warn("缓冲区为空，没有数据可保存")

    # ========== 状态监控 ==========

    def status_callback(self):
        """定期打印状态"""
        self.get_logger().info(
            f"[状态] 消息计数(过去2s): "
            f"robot_target={self.msg_count['robot_target']}, "
            f"robot_actual={self.msg_count['robot_actual']}, "
            f"gripper_target={self.msg_count['gripper_target']}, "
            f"gripper_actual={self.msg_count['gripper_actual']}"
        )

        if self.is_recording:
            self.get_logger().info(f"[录制中] 已录制帧数: {len(self.episode_buffer)}")

        # 重置计数
        for key in self.msg_count:
            self.msg_count[key] = 0

    # ========== 保存数据 ==========

    def get_next_episode_index(self):
        """获取下一个评估文件编号"""
        existing_files = [
            f for f in os.listdir(self.data_dir)
            if f.startswith('evaluation_') and f.endswith('.hdf5')
        ]

        if not existing_files:
            return 0

        indices = []
        for f in existing_files:
            try:
                idx = int(f.split('_')[1].split('.')[0])
                indices.append(idx)
            except:
                pass

        return max(indices) + 1 if indices else 0

    def save_dataset(self):
        """保存评估数据集"""
        if not self.episode_buffer:
            self.get_logger().warn("缓冲区为空，无法保存")
            return

        idx = self.get_next_episode_index()
        file_path = os.path.join(self.data_dir, f"evaluation_{idx}.hdf5")

        self.get_logger().info(f"正在保存 {len(self.episode_buffer)} 帧到 {file_path}...")

        try:
            with h5py.File(file_path, 'w') as f:
                data_len = len(self.episode_buffer)

                # 创建数据集
                dset_time = f.create_dataset('timestamp', (data_len,), dtype='float64')

                # 机械臂数据
                dset_robot_target = f.create_dataset('robot/target', (data_len, 6), dtype='float32')
                dset_robot_actual = f.create_dataset('robot/actual', (data_len, 6), dtype='float32')

                # 夹爪数据
                dset_gripper_target = f.create_dataset('gripper/target', (data_len, 1), dtype='float32')
                dset_gripper_actual = f.create_dataset('gripper/actual', (data_len, 1), dtype='float32')

                # 写入数据
                for i, frame in enumerate(self.episode_buffer):
                    dset_time[i] = frame['timestamp']
                    dset_robot_target[i] = frame['robot_target']
                    dset_robot_actual[i] = frame['robot_actual']
                    dset_gripper_target[i] = frame['gripper_target']
                    dset_gripper_actual[i] = frame['gripper_actual']

                # 元数据
                f.attrs['total_frames'] = data_len
                f.attrs['evaluation_type'] = 'playback'
                f.attrs['recording_frequency_hz'] = 100  # 基于机械臂反馈频率
                f.attrs['time_tolerance_ms'] = 50  # 时间对齐容差

            self.get_logger().info(f"✓ 成功保存评估数据到 {file_path}")
            self.get_logger().info(f"  总帧数: {data_len}")
            self.get_logger().info(f"  总时长: {self.episode_buffer[-1]['timestamp'] - self.episode_buffer[0]['timestamp']:.2f} 秒")

        except Exception as e:
            self.get_logger().error(f"保存失败: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PlaybackEvaluator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n检测到中断信号，正在停止录制...")
        node.stop_recording()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
