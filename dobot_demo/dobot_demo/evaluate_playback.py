#!/usr/bin/env python3
"""
数据集对比评估工具
用于对比原始录制数据和播放重现数据，评估播放精度
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_hdf5_data(filepath):
    """加载HDF5数据集"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        # 读取机械臂数据
        if 'actions/robot_target' in f:
            data['robot_target'] = np.array(f['actions/robot_target'])
        if 'observations/robot_current' in f:
            data['robot_current'] = np.array(f['observations/robot_current'])

        # 读取夹爪数据
        if 'actions/gripper_target' in f:
            data['gripper_target'] = np.array(f['actions/gripper_target'])
        if 'observations/gripper_current' in f:
            data['gripper_current'] = np.array(f['observations/gripper_current'])

        # 读取时间戳
        if 'timestamp' in f:
            data['timestamp'] = np.array(f['timestamp'])

        # 读取元数据
        data['attrs'] = dict(f.attrs)

    return data

def align_sequences(seq1, seq2):
    """对齐两个序列长度"""
    min_len = min(len(seq1), len(seq2))
    return seq1[:min_len], seq2[:min_len]

def compute_errors(original, replay):
    """计算误差统计"""
    errors = np.abs(original - replay)

    stats = {
        'mean': np.mean(errors, axis=0),
        'std': np.std(errors, axis=0),
        'max': np.max(errors, axis=0),
        'rmse': np.sqrt(np.mean(errors**2, axis=0)),
        'median': np.median(errors, axis=0)
    }

    return errors, stats

def print_error_summary(stats, data_name, dim_names=None):
    """打印误差摘要"""
    print(f"\n{'='*60}")
    print(f"{data_name} 误差统计")
    print(f"{'='*60}")

    n_dims = len(stats['mean'])

    if dim_names is None:
        dim_names = [f"Dim{i}" for i in range(n_dims)]

    # 表头
    print(f"{'维度':<10} {'平均':<10} {'标准差':<10} {'最大':<10} {'RMSE':<10} {'中位数':<10}")
    print("-" * 60)

    # 各维度
    for i, name in enumerate(dim_names[:n_dims]):
        print(f"{name:<10} {stats['mean'][i]:<10.3f} {stats['std'][i]:<10.3f} "
              f"{stats['max'][i]:<10.3f} {stats['rmse'][i]:<10.3f} {stats['median'][i]:<10.3f}")

    # 总体
    print("-" * 60)
    print(f"{'总体':<10} {np.mean(stats['mean']):<10.3f} {np.mean(stats['std']):<10.3f} "
          f"{np.max(stats['max']):<10.3f} {np.mean(stats['rmse']):<10.3f} {np.mean(stats['median']):<10.3f}")

def plot_comparison(original_data, replay_data, save_dir=None):
    """绘制对比图"""

    # 对齐数据
    orig_robot = original_data.get('robot_target')
    replay_robot = replay_data.get('robot_current')

    if orig_robot is not None and replay_robot is not None:
        orig_robot, replay_robot = align_sequences(orig_robot, replay_robot)

        # 机械臂位置对比
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('机械臂轨迹对比 (原始目标 vs 播放实际)', fontsize=14)

        dim_names = ['X (mm)', 'Y (mm)', 'Z (mm)', 'RX (deg)', 'RY (deg)', 'RZ (deg)']

        for i in range(6):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            ax.plot(orig_robot[:, i], label='原始目标', linewidth=1.5, alpha=0.8)
            ax.plot(replay_robot[:, i], label='播放实际', linewidth=1.5, alpha=0.8)
            ax.set_title(dim_names[i])
            ax.set_xlabel('帧')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, 'robot_trajectory_comparison.png'), dpi=150)
            print(f"保存图表: {os.path.join(save_dir, 'robot_trajectory_comparison.png')}")
        else:
            plt.show()

    # 对齐夹爪数据
    orig_gripper = original_data.get('gripper_target')
    replay_gripper = replay_data.get('gripper_current')

    if orig_gripper is not None and replay_gripper is not None:
        orig_gripper, replay_gripper = align_sequences(orig_gripper, replay_gripper)

        # 夹爪位置对比
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(orig_gripper, label='原始目标', linewidth=1.5, alpha=0.8)
        ax.plot(replay_gripper, label='播放实际', linewidth=1.5, alpha=0.8)
        ax.set_title('夹爪位置对比')
        ax.set_xlabel('帧')
        ax.set_ylabel('位置 (0-1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, 'gripper_trajectory_comparison.png'), dpi=150)
            print(f"保存图表: {os.path.join(save_dir, 'gripper_trajectory_comparison.png')}")
        else:
            plt.show()

    # 误差图
    if orig_robot is not None and replay_robot is not None:
        errors = np.abs(orig_robot - replay_robot)

        fig, ax = plt.subplots(figsize=(12, 4))
        for i, name in enumerate(dim_names):
            ax.plot(errors[:, i], label=name, alpha=0.7)
        ax.set_title('机械臂各维度误差')
        ax.set_xlabel('帧')
        ax.set_ylabel('绝对误差')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, 'robot_errors.png'), dpi=150)
            print(f"保存图表: {os.path.join(save_dir, 'robot_errors.png')}")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='对比原始数据集与播放重现数据集')
    parser.add_argument('original', type=str, help='原始数据集路径 (例如: ./data/episode_0.hdf5)')
    parser.add_argument('replay', type=str, help='播放录制数据集路径 (例如: ./data/episode_1.hdf5)')
    parser.add_argument('--plot', action='store_true', help='绘制对比图表')
    parser.add_argument('--save-dir', type=str, default=None, help='保存图表的目录')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.original):
        print(f"错误: 原始数据集不存在 - {args.original}")
        return

    if not os.path.exists(args.replay):
        print(f"错误: 播放数据集不存在 - {args.replay}")
        return

    # 加载数据
    print("加载数据集...")
    original_data = load_hdf5_data(args.original)
    replay_data = load_hdf5_data(args.replay)

    print(f"\n原始数据集: {args.original}")
    print(f"  帧数: {len(original_data.get('robot_target', []))}")
    print(f"  元数据: {original_data.get('attrs', {})}")

    print(f"\n播放数据集: {args.replay}")
    print(f"  帧数: {len(replay_data.get('robot_current', []))}")
    print(f"  元数据: {replay_data.get('attrs', {})}")

    # 对齐并计算误差
    # 机械臂误差
    if 'robot_target' in original_data and 'robot_current' in replay_data:
        orig_robot, replay_robot = align_sequences(
            original_data['robot_target'],
            replay_data['robot_current']
        )

        robot_errors, robot_stats = compute_errors(orig_robot, replay_robot)
        dim_names = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
        print_error_summary(robot_stats, "机械臂位置", dim_names)

    # 夹爪误差
    if 'gripper_target' in original_data and 'gripper_current' in replay_data:
        orig_gripper, replay_gripper = align_sequences(
            original_data['gripper_target'],
            replay_data['gripper_current']
        )

        gripper_errors, gripper_stats = compute_errors(orig_gripper, replay_gripper)
        print_error_summary(gripper_stats, "夹爪位置", ['Position'])

    # 绘图
    if args.plot:
        if args.save_dir and not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        plot_comparison(original_data, replay_data, args.save_dir)

    print("\n评估完成！")

if __name__ == "__main__":
    main()
