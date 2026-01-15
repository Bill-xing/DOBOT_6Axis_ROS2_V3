#!/usr/bin/env python3
"""
数据集清洗工具 - 过滤丢帧片段

功能：
1. 识别丢帧严重的片段
2. 切分数据集为多个高质量的子片段
3. 可选择删除或插值修复丢帧
4. 保存清洗后的数据集

使用场景：
- 数据集整体可用，但部分片段丢帧严重
- 想保留高质量部分用于训练
"""

import h5py
import numpy as np
import argparse
import os
import sys

class DatasetCleaner:
    """数据集清洗器"""

    def __init__(self, hdf5_path, expected_fps=30.0, max_gap_frames=3):
        """
        初始化清洗器

        Args:
            hdf5_path: 数据集路径
            expected_fps: 期望帧率
            max_gap_frames: 允许的最大帧间隔（超过则视为严重丢帧）
        """
        self.hdf5_path = hdf5_path
        self.expected_fps = expected_fps
        self.expected_interval = 1.0 / expected_fps
        self.max_gap_frames = max_gap_frames
        self.max_gap_time = self.expected_interval * max_gap_frames

        self.data = {}
        self.timestamps = None
        self.good_segments = []

    def load_data(self):
        """加载数据集"""
        print(f"正在加载数据集: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, 'r') as f:
            # 加载所有数据
            self.timestamps = np.array(f['timestamp'])

            # 加载图像
            if 'observations/images/color' in f:
                print("  加载RGB图像...")
                self.data['color'] = np.array(f['observations/images/color'])

            if 'observations/images/depth' in f:
                print("  加载深度图像...")
                self.data['depth'] = np.array(f['observations/images/depth'])

            # 加载机械臂数据
            if 'observations/robot_current' in f:
                self.data['robot_current'] = np.array(f['observations/robot_current'])

            if 'actions/robot_target' in f:
                self.data['robot_target'] = np.array(f['actions/robot_target'])

            # 加载夹爪数据
            if 'observations/gripper_current' in f:
                self.data['gripper_current'] = np.array(f['observations/gripper_current'])

            if 'actions/gripper_target' in f:
                self.data['gripper_target'] = np.array(f['actions/gripper_target'])

            # 加载相机内参
            if 'camera/intrinsics' in f:
                self.data['camera_intrinsics'] = np.array(f['camera/intrinsics'])

            # 加载元数据
            self.attrs = dict(f.attrs)

        print(f"✓ 加载完成，总帧数: {len(self.timestamps)}")

    def find_good_segments(self, min_segment_length=30):
        """
        查找高质量片段（无严重丢帧）

        Args:
            min_segment_length: 最小片段长度（帧数）

        Returns:
            good_segments: [(start_idx, end_idx), ...]
        """
        print(f"\n正在查找高质量片段...")
        print(f"  判定标准: 帧间隔 < {self.max_gap_time*1000:.1f}ms ({self.max_gap_frames}帧)")
        print(f"  最小片段长度: {min_segment_length}帧")

        time_diffs = np.diff(self.timestamps)
        bad_gaps = time_diffs > self.max_gap_time

        # 查找连续的好片段
        segments = []
        start_idx = 0

        for i, is_bad in enumerate(bad_gaps):
            if is_bad:
                # 遇到坏间隔，保存之前的片段
                if i - start_idx >= min_segment_length:
                    segments.append((start_idx, i))
                    print(f"  找到片段: 帧{start_idx}-{i}, 长度={i-start_idx}, "
                          f"时长={self.timestamps[i]-self.timestamps[start_idx]:.1f}s")
                start_idx = i + 1

        # 处理最后一个片段
        if len(self.timestamps) - start_idx >= min_segment_length:
            segments.append((start_idx, len(self.timestamps) - 1))
            print(f"  找到片段: 帧{start_idx}-{len(self.timestamps)-1}, "
                  f"长度={len(self.timestamps)-start_idx}, "
                  f"时长={self.timestamps[-1]-self.timestamps[start_idx]:.1f}s")

        self.good_segments = segments
        print(f"\n✓ 共找到 {len(segments)} 个高质量片段")

        return segments

    def save_segments(self, output_dir=None, save_merged=True):
        """
        保存清洗后的片段

        Args:
            output_dir: 输出目录
            save_merged: 是否保存合并的完整数据集
        """
        if output_dir is None:
            output_dir = os.path.dirname(self.hdf5_path)
            output_dir = os.path.join(output_dir, "cleaned")

        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.hdf5_path))[0]

        # 保存各个片段
        print(f"\n正在保存片段到: {output_dir}")

        for seg_idx, (start, end) in enumerate(self.good_segments):
            output_path = os.path.join(output_dir, f"{base_name}_seg{seg_idx}.hdf5")

            self._save_segment(output_path, start, end, seg_idx)

        # 保存合并的数据集
        if save_merged and len(self.good_segments) > 0:
            merged_path = os.path.join(output_dir, f"{base_name}_cleaned_merged.hdf5")
            self._save_merged(merged_path)

        print(f"\n✓ 清洗完成！")

    def _save_segment(self, output_path, start_idx, end_idx, seg_idx):
        """保存单个片段"""
        segment_length = end_idx - start_idx + 1

        print(f"  保存片段{seg_idx}: {output_path} ({segment_length}帧)")

        with h5py.File(output_path, 'w') as f:
            # 保存时间戳
            f.create_dataset('timestamp', data=self.timestamps[start_idx:end_idx+1])

            # 保存各类数据
            if 'color' in self.data:
                f.create_dataset('observations/images/color',
                               data=self.data['color'][start_idx:end_idx+1],
                               compression='gzip', compression_opts=4)

            if 'depth' in self.data:
                f.create_dataset('observations/images/depth',
                               data=self.data['depth'][start_idx:end_idx+1],
                               compression='gzip', compression_opts=4)

            if 'robot_current' in self.data:
                f.create_dataset('observations/robot_current',
                               data=self.data['robot_current'][start_idx:end_idx+1])

            if 'robot_target' in self.data:
                f.create_dataset('actions/robot_target',
                               data=self.data['robot_target'][start_idx:end_idx+1])

            if 'gripper_current' in self.data:
                f.create_dataset('observations/gripper_current',
                               data=self.data['gripper_current'][start_idx:end_idx+1])

            if 'gripper_target' in self.data:
                f.create_dataset('actions/gripper_target',
                               data=self.data['gripper_target'][start_idx:end_idx+1])

            if 'camera_intrinsics' in self.data:
                f.create_dataset('camera/intrinsics', data=self.data['camera_intrinsics'])

            # 保存元数据
            for key, value in self.attrs.items():
                f.attrs[key] = value

            f.attrs['segment_index'] = seg_idx
            f.attrs['original_start_frame'] = start_idx
            f.attrs['original_end_frame'] = end_idx
            f.attrs['cleaned'] = True
            f.attrs['total_frames'] = segment_length

    def _save_merged(self, output_path):
        """保存合并的完整数据集（去除丢帧片段）"""
        print(f"\n  保存合并数据集: {output_path}")

        # 计算总长度
        total_length = sum(end - start + 1 for start, end in self.good_segments)

        with h5py.File(output_path, 'w') as f:
            # 预分配数据集
            if 'color' in self.data:
                img_shape = self.data['color'].shape[1:]
                dset_color = f.create_dataset('observations/images/color',
                                             (total_length, *img_shape),
                                             dtype='uint8',
                                             compression='gzip', compression_opts=4)

            if 'depth' in self.data:
                depth_shape = self.data['depth'].shape[1:]
                dset_depth = f.create_dataset('observations/images/depth',
                                             (total_length, *depth_shape),
                                             dtype='uint16',
                                             compression='gzip', compression_opts=4)

            dset_time = f.create_dataset('timestamp', (total_length,), dtype='float64')

            if 'robot_current' in self.data:
                dset_robot_curr = f.create_dataset('observations/robot_current',
                                                  (total_length, 6), dtype='float32')

            if 'robot_target' in self.data:
                dset_robot_targ = f.create_dataset('actions/robot_target',
                                                  (total_length, 6), dtype='float32')

            if 'gripper_current' in self.data:
                dset_grip_curr = f.create_dataset('observations/gripper_current',
                                                  (total_length, 1), dtype='float32')

            if 'gripper_target' in self.data:
                dset_grip_targ = f.create_dataset('actions/gripper_target',
                                                  (total_length, 1), dtype='float32')

            # 复制数据
            write_idx = 0
            for start, end in self.good_segments:
                seg_len = end - start + 1

                if 'color' in self.data:
                    dset_color[write_idx:write_idx+seg_len] = self.data['color'][start:end+1]

                if 'depth' in self.data:
                    dset_depth[write_idx:write_idx+seg_len] = self.data['depth'][start:end+1]

                dset_time[write_idx:write_idx+seg_len] = self.timestamps[start:end+1]

                if 'robot_current' in self.data:
                    dset_robot_curr[write_idx:write_idx+seg_len] = self.data['robot_current'][start:end+1]

                if 'robot_target' in self.data:
                    dset_robot_targ[write_idx:write_idx+seg_len] = self.data['robot_target'][start:end+1]

                if 'gripper_current' in self.data:
                    dset_grip_curr[write_idx:write_idx+seg_len] = self.data['gripper_current'][start:end+1]

                if 'gripper_target' in self.data:
                    dset_grip_targ[write_idx:write_idx+seg_len] = self.data['gripper_target'][start:end+1]

                write_idx += seg_len

            # 保存相机内参
            if 'camera_intrinsics' in self.data:
                f.create_dataset('camera/intrinsics', data=self.data['camera_intrinsics'])

            # 元数据
            for key, value in self.attrs.items():
                f.attrs[key] = value

            f.attrs['cleaned'] = True
            f.attrs['total_frames'] = total_length
            f.attrs['num_segments'] = len(self.good_segments)
            f.attrs['removed_frames'] = len(self.timestamps) - total_length

        print(f"    原始帧数: {len(self.timestamps)}")
        print(f"    清洗后: {total_length}")
        print(f"    移除帧数: {len(self.timestamps) - total_length}")

    def run(self, min_segment_length=30, output_dir=None, save_merged=True):
        """运行完整清洗流程"""
        print("="*70)
        print("数据集清洗")
        print("="*70 + "\n")

        self.load_data()
        self.find_good_segments(min_segment_length)

        if len(self.good_segments) == 0:
            print("❌ 没有找到符合标准的片段，数据质量太差")
            return False

        self.save_segments(output_dir, save_merged)

        return True

def main():
    parser = argparse.ArgumentParser(description='清洗数据集，过滤丢帧片段')
    parser.add_argument('dataset', type=str, help='HDF5数据集路径')
    parser.add_argument('--fps', type=float, default=30.0, help='期望帧率 (默认: 30Hz)')
    parser.add_argument('--max-gap', type=int, default=3,
                       help='允许的最大帧间隔 (默认: 3帧)')
    parser.add_argument('--min-length', type=int, default=30,
                       help='最小片段长度 (默认: 30帧)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录 (默认: ./cleaned/)')
    parser.add_argument('--no-merge', action='store_true',
                       help='不保存合并的数据集')

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"❌ 错误：文件不存在 {args.dataset}")
        sys.exit(1)

    # 创建清洗器
    cleaner = DatasetCleaner(args.dataset,
                            expected_fps=args.fps,
                            max_gap_frames=args.max_gap)

    # 运行清洗
    success = cleaner.run(min_segment_length=args.min_length,
                         output_dir=args.output,
                         save_merged=not args.no_merge)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
