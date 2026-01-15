#!/usr/bin/env python3
"""
数据集丢帧检查工具

专门用于分析录制数据集的丢帧情况，评估数据质量是否适合VLA训练。

主要功能：
1. 检测丢帧位置和频率
2. 分析丢帧分布（是否集中）
3. 评估对VLA训练的影响
4. 生成详细报告和可视化
5. 提供数据清洗建议
"""

import h5py
import numpy as np
import argparse
import sys
import os
from collections import defaultdict

class FrameDropAnalyzer:
    """丢帧分析器"""

    def __init__(self, hdf5_path, expected_fps=30.0):
        """
        初始化分析器

        Args:
            hdf5_path: HDF5数据集路径
            expected_fps: 期望的帧率（Hz）
        """
        self.hdf5_path = hdf5_path
        self.expected_fps = expected_fps
        self.expected_interval = 1.0 / expected_fps  # 期望时间间隔（秒）

        # 容差设置
        self.tolerance = 0.5  # 允许50%的时间误差
        self.drop_threshold = self.expected_interval * (1 + self.tolerance)

        # 分析结果
        self.total_frames = 0
        self.total_duration = 0.0
        self.expected_frames = 0
        self.dropped_frames_count = 0
        self.drop_locations = []
        self.timestamps = None
        self.time_diffs = None

    def load_dataset(self):
        """加载数据集"""
        print(f"正在加载数据集: {self.hdf5_path}\n")

        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                # 读取时间戳
                if 'timestamp' not in f:
                    print("❌ 错误：数据集中没有timestamp字段")
                    sys.exit(1)

                self.timestamps = np.array(f['timestamp'])
                self.total_frames = len(self.timestamps)

                # 读取元数据
                if 'camera_frequency_hz' in f.attrs:
                    recorded_fps = f.attrs['camera_frequency_hz']
                    print(f"数据集记录的帧率: {recorded_fps} Hz")

                # 检查是否有机械臂数据
                has_robot = 'actions/robot_target' in f or 'observations/robot_current' in f
                has_gripper = 'actions/gripper_target' in f or 'observations/gripper_current' in f
                has_images = 'observations/images/color' in f

                print(f"数据完整性: 机械臂={'✓' if has_robot else '✗'}, "
                      f"夹爪={'✓' if has_gripper else '✗'}, "
                      f"图像={'✓' if has_images else '✗'}\n")

        except Exception as e:
            print(f"❌ 加载数据集失败: {e}")
            sys.exit(1)

    def analyze_frame_drops(self):
        """分析丢帧情况"""
        print("="*70)
        print("丢帧分析")
        print("="*70)

        # 基本统计
        self.total_duration = self.timestamps[-1] - self.timestamps[0]
        self.expected_frames = int(self.total_duration * self.expected_fps)
        self.dropped_frames_count = self.expected_frames - self.total_frames

        print(f"\n[基本信息]")
        print(f"  实际帧数: {self.total_frames}")
        print(f"  录制时长: {self.total_duration:.2f} 秒")
        print(f"  期望帧数: {self.expected_frames} (按 {self.expected_fps} Hz)")
        print(f"  丢帧数量: {self.dropped_frames_count}")
        print(f"  丢帧率: {self.dropped_frames_count / self.expected_frames * 100:.2f}%")

        # 计算时间间隔
        self.time_diffs = np.diff(self.timestamps)

        print(f"\n[时间间隔统计]")
        print(f"  期望间隔: {self.expected_interval*1000:.2f} ms")
        print(f"  平均间隔: {self.time_diffs.mean()*1000:.2f} ms")
        print(f"  最小间隔: {self.time_diffs.min()*1000:.2f} ms")
        print(f"  最大间隔: {self.time_diffs.max()*1000:.2f} ms")
        print(f"  标准差: {self.time_diffs.std()*1000:.2f} ms")

        # 检测丢帧位置
        drop_mask = self.time_diffs > self.drop_threshold
        self.drop_locations = np.where(drop_mask)[0]

        print(f"\n[丢帧位置检测]")
        print(f"  检测到 {len(self.drop_locations)} 个丢帧位置")
        print(f"  判定阈值: >{self.drop_threshold*1000:.2f} ms")

        if len(self.drop_locations) > 0:
            # 统计每个位置丢了多少帧
            total_detected_drops = 0
            for idx in self.drop_locations:
                interval = self.time_diffs[idx]
                estimated_drops = int(interval / self.expected_interval) - 1
                total_detected_drops += estimated_drops

            print(f"  估算丢帧总数: {total_detected_drops}")

            # 显示前10个丢帧位置
            print(f"\n  前10个丢帧位置:")
            for i, idx in enumerate(self.drop_locations[:10]):
                interval = self.time_diffs[idx]
                estimated_drops = int(interval / self.expected_interval) - 1
                print(f"    #{i+1} 帧{idx}→{idx+1}: "
                      f"间隔={interval*1000:.1f}ms, "
                      f"估算丢{estimated_drops}帧, "
                      f"时间戳={self.timestamps[idx]:.2f}s")

    def analyze_drop_distribution(self):
        """分析丢帧分布"""
        print(f"\n[丢帧分布分析]")

        if len(self.drop_locations) == 0:
            print("  ✅ 无丢帧")
            return

        # 计算丢帧之间的间隔
        if len(self.drop_locations) > 1:
            drop_intervals = np.diff(self.drop_locations)

            print(f"  丢帧间隔统计:")
            print(f"    平均间隔: {drop_intervals.mean():.1f} 帧")
            print(f"    最小间隔: {drop_intervals.min()} 帧")
            print(f"    最大间隔: {drop_intervals.max()} 帧")

            # 检测连续丢帧
            consecutive_drops = []
            current_streak = 1
            for i in range(1, len(self.drop_locations)):
                if self.drop_locations[i] - self.drop_locations[i-1] <= 5:
                    current_streak += 1
                else:
                    if current_streak >= 3:
                        consecutive_drops.append(current_streak)
                    current_streak = 1

            if consecutive_drops:
                print(f"\n  ⚠️  检测到 {len(consecutive_drops)} 个连续丢帧区域:")
                for i, streak in enumerate(consecutive_drops[:5]):
                    print(f"    区域{i+1}: 连续{streak}次丢帧")
            else:
                print(f"\n  ✓ 丢帧分散，无严重连续丢帧")

        # 按时间段分析
        segment_duration = self.total_duration / 10  # 分成10段
        segment_drops = defaultdict(int)

        for idx in self.drop_locations:
            timestamp = self.timestamps[idx]
            segment = int(timestamp / segment_duration)
            segment_drops[segment] += 1

        print(f"\n  时间段分布 (每段{segment_duration:.1f}秒):")
        for seg in range(10):
            drops = segment_drops.get(seg, 0)
            bar = '█' * (drops // 2) if drops > 0 else ''
            print(f"    段{seg+1}: {drops:3d} 次 {bar}")

    def evaluate_training_suitability(self):
        """评估数据集是否适合VLA训练"""
        print(f"\n" + "="*70)
        print("VLA训练适用性评估")
        print("="*70)

        drop_rate = self.dropped_frames_count / self.expected_frames if self.expected_frames > 0 else 0

        # 评估标准
        if drop_rate < 0.01:
            quality = "优秀"
            emoji = "🟢"
            recommendation = "✅ 可直接用于VLA训练"
            score = 95
        elif drop_rate < 0.05:
            quality = "良好"
            emoji = "🟡"
            recommendation = "⚠️  建议检查丢帧分布，可谨慎使用"
            score = 75
        elif drop_rate < 0.10:
            quality = "一般"
            emoji = "🟠"
            recommendation = "⚠️  建议过滤严重片段或重新录制"
            score = 50
        else:
            quality = "差"
            emoji = "🔴"
            recommendation = "❌ 不建议使用，强烈建议重新录制"
            score = 20

        print(f"\n数据质量等级: {emoji} {quality} (得分: {score}/100)")
        print(f"丢帧率: {drop_rate*100:.2f}%")
        print(f"\n建议: {recommendation}")

        # 详细分析
        print(f"\n[影响分析]")

        # 1. 时序一致性
        if self.time_diffs.std() < self.expected_interval * 0.2:
            print(f"  ✓ 时序一致性: 良好 (标准差 {self.time_diffs.std()*1000:.2f}ms)")
        else:
            print(f"  ✗ 时序一致性: 差 (标准差 {self.time_diffs.std()*1000:.2f}ms)")

        # 2. 最大间隔
        max_gap_frames = int(self.time_diffs.max() / self.expected_interval)
        if max_gap_frames <= 2:
            print(f"  ✓ 最大间隔: {max_gap_frames}帧 (可接受)")
        elif max_gap_frames <= 5:
            print(f"  ⚠️  最大间隔: {max_gap_frames}帧 (可能影响流畅性)")
        else:
            print(f"  ✗ 最大间隔: {max_gap_frames}帧 (严重影响动作连续性)")

        # 3. 丢帧分布
        if len(self.drop_locations) == 0:
            print(f"  ✓ 丢帧分布: 无丢帧")
        elif len(self.drop_locations) < self.total_frames * 0.05:
            print(f"  ✓ 丢帧分布: 零星分散 (影响较小)")
        else:
            print(f"  ✗ 丢帧分布: 频繁 (可能影响学习效果)")

        return score >= 50

    def generate_report(self, output_path=None):
        """生成详细报告"""
        if output_path is None:
            output_path = self.hdf5_path.replace('.hdf5', '_frame_drop_report.txt')

        print(f"\n正在生成详细报告: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"数据集丢帧分析报告\n")
            f.write(f"数据集: {self.hdf5_path}\n")
            f.write("="*70 + "\n\n")

            f.write("[摘要]\n")
            f.write(f"实际帧数: {self.total_frames}\n")
            f.write(f"期望帧数: {self.expected_frames}\n")
            f.write(f"丢帧数量: {self.dropped_frames_count}\n")
            f.write(f"丢帧率: {self.dropped_frames_count / self.expected_frames * 100:.2f}%\n")
            f.write(f"录制时长: {self.total_duration:.2f}秒\n\n")

            f.write("[所有丢帧位置]\n")
            for i, idx in enumerate(self.drop_locations):
                interval = self.time_diffs[idx]
                estimated_drops = int(interval / self.expected_interval) - 1
                f.write(f"#{i+1:3d} 帧{idx:4d}→{idx+1:4d}: "
                       f"间隔={interval*1000:6.1f}ms, "
                       f"丢{estimated_drops}帧, "
                       f"时间={self.timestamps[idx]:7.2f}s\n")

        print(f"✓ 报告已保存")

    def run_analysis(self):
        """运行完整分析"""
        self.load_dataset()
        self.analyze_frame_drops()
        self.analyze_drop_distribution()
        is_suitable = self.evaluate_training_suitability()

        return is_suitable

def main():
    parser = argparse.ArgumentParser(description='分析数据集丢帧情况')
    parser.add_argument('dataset', type=str, help='HDF5数据集路径')
    parser.add_argument('--fps', type=float, default=30.0, help='期望帧率 (默认: 30Hz)')
    parser.add_argument('--report', action='store_true', help='生成详细报告文件')

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"❌ 错误：文件不存在 {args.dataset}")
        sys.exit(1)

    # 创建分析器
    analyzer = FrameDropAnalyzer(args.dataset, expected_fps=args.fps)

    # 运行分析
    is_suitable = analyzer.run_analysis()

    # 生成报告
    if args.report:
        analyzer.generate_report()

    print("\n" + "="*70)
    print("分析完成")
    print("="*70)

    # 返回退出码
    sys.exit(0 if is_suitable else 1)

if __name__ == "__main__":
    main()
