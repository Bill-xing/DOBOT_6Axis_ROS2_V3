#!/usr/bin/env python3
"""
最优时间偏移分析工具

通过尝试不同的时间偏移值，找到使跟踪误差最小的最优偏移。
这个工具帮助确定 ServoP 命令的响应延迟，用于评估器的时间补偿。

原理：
- 目标位置在时刻 T 发送
- 机械臂在时刻 T + delay 到达目标位置
- 评估器记录的是同一时刻的目标和实际位置
- 通过时间偏移补偿，可以正确对齐命令和响应

使用方法：
    python3 find_optimal_offset.py playback_eval/evaluation_X.hdf5
"""

import h5py
import numpy as np
import argparse
import os

def load_evaluation_data(hdf5_path):
    """加载评估数据"""
    if not os.path.exists(hdf5_path):
        print(f"错误：文件不存在 {hdf5_path}")
        return None

    with h5py.File(hdf5_path, 'r') as f:
        data = {
            'timestamps': np.array(f['timestamp']),
            'robot_target': np.array(f['robot/target']),
            'robot_actual': np.array(f['robot/actual']),
            'gripper_target': np.array(f['gripper/target']),
            'gripper_actual': np.array(f['gripper/actual']),
        }
    return data

def compute_error_with_offset(data, frame_offset):
    """
    计算给定帧偏移下的误差

    frame_offset > 0: 目标位置向后偏移（即用更早的目标与当前实际对比）
    这模拟了机械臂的响应延迟

    Args:
        data: 评估数据字典
        frame_offset: 帧偏移量（正数表示目标滞后于实际）

    Returns:
        平均位置误差, X/Y/Z分量误差
    """
    n_frames = len(data['timestamps'])

    if frame_offset >= n_frames:
        return float('inf'), float('inf'), float('inf'), float('inf')

    # 有效帧范围
    if frame_offset >= 0:
        # 目标向后偏移：用 target[i] 与 actual[i + offset] 对比
        # 即：当前目标 vs 未来实际位置
        # 但我们的数据是"当前目标 vs 当前实际"
        # 所以应该用 target[i - offset] 与 actual[i] 对比
        # 即：过去的目标 vs 当前实际位置
        start_idx = frame_offset
        end_idx = n_frames

        target_pos = data['robot_target'][0:end_idx-frame_offset, :3]
        actual_pos = data['robot_actual'][start_idx:end_idx, :3]
    else:
        # 负偏移（通常不需要）
        offset = -frame_offset
        start_idx = offset
        end_idx = n_frames

        target_pos = data['robot_target'][start_idx:end_idx, :3]
        actual_pos = data['robot_actual'][0:end_idx-offset, :3]

    # 计算误差
    pos_diff = target_pos - actual_pos
    pos_errors = np.linalg.norm(pos_diff, axis=1)

    x_errors = np.abs(pos_diff[:, 0])
    y_errors = np.abs(pos_diff[:, 1])
    z_errors = np.abs(pos_diff[:, 2])

    return np.mean(pos_errors), np.mean(x_errors), np.mean(y_errors), np.mean(z_errors)

def find_optimal_offset(data, max_offset_frames=50):
    """
    搜索最优时间偏移

    Args:
        data: 评估数据
        max_offset_frames: 最大搜索帧数

    Returns:
        最优帧偏移, 最优时间偏移(秒), 最小误差
    """
    # 计算平均帧间隔
    timestamps = data['timestamps']
    avg_dt = np.mean(np.diff(timestamps))

    print(f"数据统计:")
    print(f"  总帧数: {len(timestamps)}")
    print(f"  总时长: {timestamps[-1] - timestamps[0]:.2f} 秒")
    print(f"  平均帧间隔: {avg_dt*1000:.2f} ms")
    print(f"  平均频率: {1.0/avg_dt:.1f} Hz")
    print()

    # 搜索最优偏移
    results = []

    print("搜索最优时间偏移...")
    print("-" * 80)
    print(f"{'帧偏移':>8} | {'时间偏移':>10} | {'位置误差':>10} | {'X误差':>10} | {'Y误差':>10} | {'Z误差':>10}")
    print("-" * 80)

    for offset in range(0, min(max_offset_frames, len(timestamps)//2)):
        pos_err, x_err, y_err, z_err = compute_error_with_offset(data, offset)
        time_offset = offset * avg_dt
        results.append((offset, time_offset, pos_err, x_err, y_err, z_err))

        # 每5帧打印一次
        if offset % 5 == 0 or offset < 10:
            print(f"{offset:>8} | {time_offset*1000:>8.1f}ms | {pos_err:>8.2f}mm | {x_err:>8.2f}mm | {y_err:>8.2f}mm | {z_err:>8.2f}mm")

    print("-" * 80)

    # 找到最小误差
    min_result = min(results, key=lambda x: x[2])
    optimal_frame_offset = min_result[0]
    optimal_time_offset = min_result[1]
    min_error = min_result[2]

    return optimal_frame_offset, optimal_time_offset, min_error, results

def analyze_velocity(data):
    """分析运动速度，帮助理解误差来源"""
    timestamps = data['timestamps']
    positions = data['robot_target'][:, :3]

    # 计算速度
    dt = np.diff(timestamps)
    dp = np.diff(positions, axis=0)

    velocities = np.linalg.norm(dp, axis=1) / dt
    vx = np.abs(dp[:, 0]) / dt
    vy = np.abs(dp[:, 1]) / dt
    vz = np.abs(dp[:, 2]) / dt

    print("\n运动速度分析:")
    print(f"  总体速度: 平均={np.mean(velocities):.1f}mm/s, 最大={np.max(velocities):.1f}mm/s")
    print(f"  X轴速度: 平均={np.mean(vx):.1f}mm/s, 最大={np.max(vx):.1f}mm/s")
    print(f"  Y轴速度: 平均={np.mean(vy):.1f}mm/s, 最大={np.max(vy):.1f}mm/s")
    print(f"  Z轴速度: 平均={np.mean(vz):.1f}mm/s, 最大={np.max(vz):.1f}mm/s")

    return np.mean(velocities), np.max(velocities)

def main():
    parser = argparse.ArgumentParser(description='分析最优时间偏移')
    parser.add_argument('evaluation_file', type=str, help='评估数据文件路径')
    parser.add_argument('--max-offset', type=int, default=50, help='最大搜索帧数 (默认50)')
    parser.add_argument('--plot', action='store_true', help='生成图表')
    args = parser.parse_args()

    print("=" * 80)
    print("最优时间偏移分析")
    print("=" * 80)
    print()

    # 加载数据
    print(f"加载数据: {args.evaluation_file}")
    data = load_evaluation_data(args.evaluation_file)
    if data is None:
        return

    # 分析速度
    avg_vel, max_vel = analyze_velocity(data)

    # 原始误差（无偏移）
    print("\n原始误差（无时间偏移）:")
    pos_err, x_err, y_err, z_err = compute_error_with_offset(data, 0)
    print(f"  位置误差: {pos_err:.2f} mm")
    print(f"  X轴误差: {x_err:.2f} mm")
    print(f"  Y轴误差: {y_err:.2f} mm")
    print(f"  Z轴误差: {z_err:.2f} mm")

    # 搜索最优偏移
    print()
    optimal_frame, optimal_time, min_error, results = find_optimal_offset(data, args.max_offset)

    # 结果汇总
    print("\n" + "=" * 80)
    print("分析结果")
    print("=" * 80)
    print(f"\n最优帧偏移: {optimal_frame} 帧")
    print(f"最优时间偏移: {optimal_time*1000:.1f} ms")
    print(f"补偿后位置误差: {min_error:.2f} mm")
    print(f"误差改善: {pos_err - min_error:.2f} mm ({(pos_err - min_error)/pos_err*100:.1f}%)")

    # 理论验证
    if avg_vel > 0:
        theoretical_delay = pos_err / avg_vel
        print(f"\n理论延迟估算:")
        print(f"  基于误差/速度: {theoretical_delay*1000:.1f} ms")
        print(f"  实际最优偏移: {optimal_time*1000:.1f} ms")

    # 建议
    print(f"\n" + "=" * 80)
    print("使用建议")
    print("=" * 80)
    print(f"\n在 playback_evaluator.py 中使用以下参数:")
    print(f"  --time-offset {optimal_time*1000:.0f}")
    print(f"\n或者在代码中设置:")
    print(f"  self.time_offset = {optimal_time:.4f}  # 秒")

    # 生成图表
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            offsets = [r[0] for r in results]
            time_offsets = [r[1] * 1000 for r in results]  # 转换为ms
            errors = [r[2] for r in results]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # 帧偏移 vs 误差
            ax1.plot(offsets, errors, 'b-', linewidth=2)
            ax1.axvline(x=optimal_frame, color='r', linestyle='--',
                       label=f'最优: {optimal_frame} 帧')
            ax1.axhline(y=min_error, color='g', linestyle='--',
                       label=f'最小误差: {min_error:.2f}mm')
            ax1.set_xlabel('帧偏移')
            ax1.set_ylabel('平均位置误差 (mm)')
            ax1.set_title('帧偏移 vs 位置误差')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 时间偏移 vs 误差
            ax2.plot(time_offsets, errors, 'b-', linewidth=2)
            ax2.axvline(x=optimal_time*1000, color='r', linestyle='--',
                       label=f'最优: {optimal_time*1000:.1f}ms')
            ax2.axhline(y=min_error, color='g', linestyle='--',
                       label=f'最小误差: {min_error:.2f}mm')
            ax2.set_xlabel('时间偏移 (ms)')
            ax2.set_ylabel('平均位置误差 (mm)')
            ax2.set_title('时间偏移 vs 位置误差')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            output_path = args.evaluation_file.replace('.hdf5', '_offset_analysis.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n图表已保存: {output_path}")

        except ImportError:
            print("\n警告: 需要 matplotlib 来生成图表")

if __name__ == "__main__":
    main()
