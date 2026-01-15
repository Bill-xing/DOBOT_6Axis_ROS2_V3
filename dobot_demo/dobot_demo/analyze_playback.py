#!/usr/bin/env python3
"""
播放质量分析工具

分析播放评估数据，计算跟踪误差、延迟等性能指标，
生成详细的评估报告和可视化图表。

功能：
1. 加载评估数据（evaluation_X.hdf5）
2. 计算跟踪误差：
   - 位置误差（笛卡尔空间）
   - 旋转误差（姿态空间）
   - 夹爪误差
3. 统计分析：平均值、最大值、标准差、百分位数
4. 时序分析：误差随时间变化
5. 生成评估报告

使用方法：
python3 analyze_playback.py playback_eval/evaluation_0.hdf5
python3 analyze_playback.py playback_eval/evaluation_0.hdf5 --plot  # 生成图表
"""

import h5py
import numpy as np
import argparse
import sys
import os

class PlaybackAnalyzer:
    """播放质量分析器"""

    def __init__(self, hdf5_path):
        """
        初始化分析器

        Args:
            hdf5_path: 评估数据文件路径
        """
        self.hdf5_path = hdf5_path
        self.data = {}
        self.timestamps = None
        self.results = {}

    def load_data(self):
        """加载评估数据"""
        print(f"正在加载评估数据: {self.hdf5_path}\n")

        if not os.path.exists(self.hdf5_path):
            print(f"错误：文件不存在 {self.hdf5_path}")
            return False

        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                # 加载时间戳
                self.timestamps = np.array(f['timestamp'])

                # 加载机械臂数据
                self.data['robot_target'] = np.array(f['robot/target'])
                self.data['robot_actual'] = np.array(f['robot/actual'])

                # 加载夹爪数据
                self.data['gripper_target'] = np.array(f['gripper/target'])
                self.data['gripper_actual'] = np.array(f['gripper/actual'])

                # 元数据
                self.attrs = dict(f.attrs)

            print(f"✓ 加载完成")
            print(f"  总帧数: {len(self.timestamps)}")
            print(f"  总时长: {self.timestamps[-1] - self.timestamps[0]:.2f} 秒")
            print(f"  平均频率: {len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0]):.1f} Hz")
            print()

            return True

        except Exception as e:
            print(f"加载失败: {e}")
            return False

    def compute_errors(self):
        """计算跟踪误差"""
        print("=" * 70)
        print("计算跟踪误差")
        print("=" * 70 + "\n")

        # 机械臂位置误差（笛卡尔空间）
        pos_target = self.data['robot_target'][:, :3]  # [x, y, z]
        pos_actual = self.data['robot_actual'][:, :3]
        pos_errors = np.linalg.norm(pos_target - pos_actual, axis=1)  # 欧氏距离

        # 机械臂旋转误差（姿态空间）
        rot_target = self.data['robot_target'][:, 3:]  # [rx, ry, rz]
        rot_actual = self.data['robot_actual'][:, 3:]
        rot_errors = np.linalg.norm(rot_target - rot_actual, axis=1)

        # 夹爪误差
        gripper_target = self.data['gripper_target'].flatten()
        gripper_actual = self.data['gripper_actual'].flatten()
        gripper_errors = np.abs(gripper_target - gripper_actual)

        # 分量误差（X, Y, Z）
        pos_diff = pos_target - pos_actual
        x_errors = np.abs(pos_diff[:, 0])
        y_errors = np.abs(pos_diff[:, 1])
        z_errors = np.abs(pos_diff[:, 2])

        # 保存结果
        self.results = {
            'pos_errors': pos_errors,
            'rot_errors': rot_errors,
            'gripper_errors': gripper_errors,
            'x_errors': x_errors,
            'y_errors': y_errors,
            'z_errors': z_errors,
        }

        print("✓ 误差计算完成\n")

    def print_statistics(self):
        """打印统计信息"""
        print("=" * 70)
        print("跟踪误差统计")
        print("=" * 70 + "\n")

        # 位置误差统计
        pos_errors = self.results['pos_errors']
        print("[机械臂位置误差] (mm)")
        print(f"  平均值: {np.mean(pos_errors):.3f} mm")
        print(f"  中位数: {np.median(pos_errors):.3f} mm")
        print(f"  标准差: {np.std(pos_errors):.3f} mm")
        print(f"  最小值: {np.min(pos_errors):.3f} mm")
        print(f"  最大值: {np.max(pos_errors):.3f} mm")
        print(f"  95%分位: {np.percentile(pos_errors, 95):.3f} mm")
        print(f"  99%分位: {np.percentile(pos_errors, 99):.3f} mm")

        # 分量误差
        print(f"\n[位置分量误差] (mm)")
        print(f"  X轴: 平均={np.mean(self.results['x_errors']):.3f}, 最大={np.max(self.results['x_errors']):.3f}")
        print(f"  Y轴: 平均={np.mean(self.results['y_errors']):.3f}, 最大={np.max(self.results['y_errors']):.3f}")
        print(f"  Z轴: 平均={np.mean(self.results['z_errors']):.3f}, 最大={np.max(self.results['z_errors']):.3f}")

        # 旋转误差统计
        rot_errors = self.results['rot_errors']
        print(f"\n[机械臂旋转误差] (deg)")
        print(f"  平均值: {np.mean(rot_errors):.3f} deg")
        print(f"  中位数: {np.median(rot_errors):.3f} deg")
        print(f"  标准差: {np.std(rot_errors):.3f} deg")
        print(f"  最小值: {np.min(rot_errors):.3f} deg")
        print(f"  最大值: {np.max(rot_errors):.3f} deg")
        print(f"  95%分位: {np.percentile(rot_errors, 95):.3f} deg")

        # 夹爪误差统计
        gripper_errors = self.results['gripper_errors']
        print(f"\n[夹爪位置误差] (0-1000)")
        print(f"  平均值: {np.mean(gripper_errors):.3f}")
        print(f"  中位数: {np.median(gripper_errors):.3f}")
        print(f"  标准差: {np.std(gripper_errors):.3f}")
        print(f"  最小值: {np.min(gripper_errors):.3f}")
        print(f"  最大值: {np.max(gripper_errors):.3f}")
        print(f"  95%分位: {np.percentile(gripper_errors, 95):.3f}")

        print()

    def analyze_tracking_quality(self):
        """分析跟踪质量"""
        print("=" * 70)
        print("跟踪质量评估")
        print("=" * 70 + "\n")

        pos_errors = self.results['pos_errors']
        rot_errors = self.results['rot_errors']

        # 位置跟踪质量
        pos_good = np.sum(pos_errors < 1.0) / len(pos_errors) * 100  # <1mm
        pos_acceptable = np.sum(pos_errors < 5.0) / len(pos_errors) * 100  # <5mm
        pos_poor = np.sum(pos_errors >= 5.0) / len(pos_errors) * 100  # >=5mm

        print("[位置跟踪质量]")
        print(f"  优秀 (<1mm):   {pos_good:.1f}% ({int(pos_good * len(pos_errors) / 100)} 帧)")
        print(f"  良好 (<5mm):   {pos_acceptable:.1f}% ({int(pos_acceptable * len(pos_errors) / 100)} 帧)")
        print(f"  较差 (>=5mm):  {pos_poor:.1f}% ({int(pos_poor * len(pos_errors) / 100)} 帧)")

        # 旋转跟踪质量
        rot_good = np.sum(rot_errors < 0.5) / len(rot_errors) * 100  # <0.5deg
        rot_acceptable = np.sum(rot_errors < 2.0) / len(rot_errors) * 100  # <2deg
        rot_poor = np.sum(rot_errors >= 2.0) / len(rot_errors) * 100  # >=2deg

        print(f"\n[旋转跟踪质量]")
        print(f"  优秀 (<0.5°):  {rot_good:.1f}% ({int(rot_good * len(rot_errors) / 100)} 帧)")
        print(f"  良好 (<2°):    {rot_acceptable:.1f}% ({int(rot_acceptable * len(rot_errors) / 100)} 帧)")
        print(f"  较差 (>=2°):   {rot_poor:.1f}% ({int(rot_poor * len(rot_errors) / 100)} 帧)")

        # 总体评级
        print(f"\n[总体评级]")
        if np.mean(pos_errors) < 1.0 and np.mean(rot_errors) < 0.5:
            rating = "优秀"
            comment = "跟踪精度非常高，适合高精度任务"
        elif np.mean(pos_errors) < 3.0 and np.mean(rot_errors) < 1.5:
            rating = "良好"
            comment = "跟踪精度良好，适合大部分任务"
        elif np.mean(pos_errors) < 5.0 and np.mean(rot_errors) < 2.0:
            rating = "中等"
            comment = "跟踪精度一般，建议优化控制参数"
        else:
            rating = "较差"
            comment = "跟踪精度不足，需要检查系统配置"

        print(f"  评级: {rating}")
        print(f"  建议: {comment}")
        print()

    def find_large_errors(self, pos_threshold=5.0, rot_threshold=2.0):
        """查找大误差位置"""
        print("=" * 70)
        print("大误差分析")
        print("=" * 70 + "\n")

        pos_errors = self.results['pos_errors']
        rot_errors = self.results['rot_errors']

        # 查找位置大误差
        large_pos_idx = np.where(pos_errors > pos_threshold)[0]
        print(f"[位置误差 >{pos_threshold}mm 的帧]")
        if len(large_pos_idx) > 0:
            print(f"  发现 {len(large_pos_idx)} 个大误差点")
            print(f"  前5个位置:")
            for i, idx in enumerate(large_pos_idx[:5]):
                print(f"    #{i+1} 帧{idx}: 误差={pos_errors[idx]:.2f}mm, "
                      f"时间={self.timestamps[idx]-self.timestamps[0]:.2f}s")
        else:
            print(f"  ✓ 未发现位置大误差")

        # 查找旋转大误差
        large_rot_idx = np.where(rot_errors > rot_threshold)[0]
        print(f"\n[旋转误差 >{rot_threshold}° 的帧]")
        if len(large_rot_idx) > 0:
            print(f"  发现 {len(large_rot_idx)} 个大误差点")
            print(f"  前5个位置:")
            for i, idx in enumerate(large_rot_idx[:5]):
                print(f"    #{i+1} 帧{idx}: 误差={rot_errors[idx]:.2f}°, "
                      f"时间={self.timestamps[idx]-self.timestamps[0]:.2f}s")
        else:
            print(f"  ✓ 未发现旋转大误差")

        print()

    def plot_results(self, output_dir=None):
        """
        生成可视化图表

        Args:
            output_dir: 输出目录，默认为评估数据所在目录
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("错误：需要安装 matplotlib 来生成图表")
            print("安装命令: pip install matplotlib")
            return

        if output_dir is None:
            output_dir = os.path.dirname(self.hdf5_path)

        print("=" * 70)
        print("生成可视化图表")
        print("=" * 70 + "\n")

        # 时间序列（相对时间）
        time_series = self.timestamps - self.timestamps[0]

        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('播放质量评估', fontsize=16, fontweight='bold')

        # 1. 位置误差时序图
        ax = axes[0, 0]
        ax.plot(time_series, self.results['pos_errors'], 'b-', linewidth=0.5, alpha=0.7)
        ax.axhline(y=np.mean(self.results['pos_errors']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(self.results["pos_errors"]):.2f}mm')
        ax.axhline(y=np.percentile(self.results['pos_errors'], 95), color='orange', linestyle='--',
                   label=f'95%: {np.percentile(self.results["pos_errors"], 95):.2f}mm')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (mm)')
        ax.set_title('Position Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 位置误差直方图
        ax = axes[0, 1]
        ax.hist(self.results['pos_errors'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(self.results['pos_errors']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(self.results["pos_errors"]):.2f}mm')
        ax.set_xlabel('Position Error (mm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Position Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 旋转误差时序图
        ax = axes[1, 0]
        ax.plot(time_series, self.results['rot_errors'], 'g-', linewidth=0.5, alpha=0.7)
        ax.axhline(y=np.mean(self.results['rot_errors']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(self.results["rot_errors"]):.2f}°')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rotation Error (deg)')
        ax.set_title('Rotation Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 夹爪误差时序图
        ax = axes[1, 1]
        ax.plot(time_series, self.results['gripper_errors'], 'm-', linewidth=0.5, alpha=0.7)
        ax.axhline(y=np.mean(self.results['gripper_errors']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(self.results["gripper_errors"]):.2f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gripper Error')
        ax.set_title('Gripper Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. XYZ分量误差对比
        ax = axes[2, 0]
        ax.plot(time_series, self.results['x_errors'], 'r-', linewidth=0.5, alpha=0.6, label='X')
        ax.plot(time_series, self.results['y_errors'], 'g-', linewidth=0.5, alpha=0.6, label='Y')
        ax.plot(time_series, self.results['z_errors'], 'b-', linewidth=0.5, alpha=0.6, label='Z')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Component Error (mm)')
        ax.set_title('Position Error Components (X, Y, Z)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. 误差累积分布函数 (CDF)
        ax = axes[2, 1]
        sorted_pos_errors = np.sort(self.results['pos_errors'])
        cdf = np.arange(1, len(sorted_pos_errors) + 1) / len(sorted_pos_errors) * 100
        ax.plot(sorted_pos_errors, cdf, 'b-', linewidth=2)
        ax.axvline(x=np.percentile(self.results['pos_errors'], 95), color='orange',
                   linestyle='--', label='95% percentile')
        ax.axhline(y=95, color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('Position Error (mm)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Position Error CDF')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        base_name = os.path.splitext(os.path.basename(self.hdf5_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存到: {output_path}")

        # 显示图表（可选）
        # plt.show()

    def run(self, generate_plot=False):
        """运行完整分析流程"""
        print("=" * 70)
        print("播放质量分析")
        print("=" * 70 + "\n")

        # 加载数据
        if not self.load_data():
            return False

        # 计算误差
        self.compute_errors()

        # 打印统计信息
        self.print_statistics()

        # 分析跟踪质量
        self.analyze_tracking_quality()

        # 查找大误差
        self.find_large_errors()

        # 生成图表
        if generate_plot:
            self.plot_results()

        print("=" * 70)
        print("分析完成")
        print("=" * 70)

        return True

def main():
    parser = argparse.ArgumentParser(description='分析播放评估数据')
    parser.add_argument('evaluation_file', type=str,
                       help='评估数据文件路径 (例如: playback_eval/evaluation_0.hdf5)')
    parser.add_argument('--plot', action='store_true',
                       help='生成可视化图表')
    parser.add_argument('--pos-threshold', type=float, default=5.0,
                       help='位置大误差阈值(mm)，默认5.0')
    parser.add_argument('--rot-threshold', type=float, default=2.0,
                       help='旋转大误差阈值(度)，默认2.0')

    args = parser.parse_args()

    # 创建分析器
    analyzer = PlaybackAnalyzer(args.evaluation_file)

    # 运行分析
    success = analyzer.run(generate_plot=args.plot)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
