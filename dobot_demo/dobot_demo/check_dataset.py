#!/usr/bin/env python3
"""
数据集综合检查工具

整合了丢帧检查和安全检查，提供全面的数据集质量评估。

功能：
1. 丢帧检测 - 检测时序问题，评估VLA训练适用性
2. 安全检查 - 检测轨迹异常，防止机械臂爆冲
3. 统计分析 - 数据集基本信息统计
4. 详细报告 - 生成完整的检查报告

用法：
    python check_dataset.py dataset.hdf5              # 运行所有检查
    python check_dataset.py dataset.hdf5 --frames     # 仅检查丢帧
    python check_dataset.py dataset.hdf5 --safety     # 仅检查安全性
    python check_dataset.py dataset.hdf5 --report     # 生成详细报告
"""

import h5py
import numpy as np
import argparse
import sys
import os
from collections import defaultdict
from typing import Optional, List, Tuple


class DatasetChecker:
    """数据集综合检查器"""

    # 安全阈值
    MAX_POSITION_JUMP = 50.0     # mm
    MAX_ROTATION_JUMP = 10.0     # deg
    MAX_GRIPPER_JUMP = 200.0
    MAX_VELOCITY = 100.0         # mm/s

    # 工作空间限制
    WORKSPACE_X_MIN = -600.0
    WORKSPACE_X_MAX = 600.0
    WORKSPACE_Y_MIN = -600.0
    WORKSPACE_Y_MAX = 600.0
    WORKSPACE_Z_MIN = -100.0
    WORKSPACE_Z_MAX = 600.0

    def __init__(self, hdf5_path: str, expected_fps: float = 30.0, strict: bool = False):
        """
        初始化检查器

        Args:
            hdf5_path: HDF5数据集路径
            expected_fps: 期望帧率
            strict: 是否使用严格模式
        """
        self.hdf5_path = hdf5_path
        self.expected_fps = expected_fps
        self.expected_interval = 1.0 / expected_fps
        self.strict = strict

        # 丢帧检测参数
        self.tolerance = 0.5
        self.drop_threshold = self.expected_interval * (1 + self.tolerance)

        # 严格模式
        if strict:
            self.MAX_POSITION_JUMP = 20.0
            self.MAX_ROTATION_JUMP = 5.0
            self.MAX_GRIPPER_JUMP = 100.0

        # 数据容器
        self.timestamps: Optional[np.ndarray] = None
        self.robot_trajectory: Optional[np.ndarray] = None
        self.gripper_trajectory: Optional[np.ndarray] = None
        self.total_frames = 0
        self.total_duration = 0.0

        # 检查结果
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.info: List[str] = []

    def load_dataset(self) -> bool:
        """加载数据集"""
        print(f"正在加载数据集: {self.hdf5_path}\n")

        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                # 读取时间戳
                if 'timestamp' not in f:
                    self.errors.append("❌ 数据集中没有timestamp字段")
                    return False

                self.timestamps = np.array(f['timestamp'])
                self.total_frames = len(self.timestamps)
                self.total_duration = self.timestamps[-1] - self.timestamps[0]

                # 读取机械臂轨迹
                if 'actions/robot_target' in f:
                    self.robot_trajectory = np.array(f['actions/robot_target'])
                elif 'observations/robot_current' in f:
                    self.robot_trajectory = np.array(f['observations/robot_current'])
                else:
                    self.errors.append("❌ 未找到机械臂轨迹数据")
                    return False

                # 读取夹爪轨迹
                if 'actions/gripper_target' in f:
                    self.gripper_trajectory = np.array(f['actions/gripper_target']).flatten()
                elif 'observations/gripper_current' in f:
                    self.gripper_trajectory = np.array(f['observations/gripper_current']).flatten()
                else:
                    self.gripper_trajectory = None

                # 读取元数据
                if 'camera_frequency_hz' in f.attrs:
                    recorded_fps = f.attrs['camera_frequency_hz']
                    self.info.append(f"数据集记录的帧率: {recorded_fps} Hz")

                # 检查数据完整性
                has_robot = self.robot_trajectory is not None
                has_gripper = self.gripper_trajectory is not None
                has_images = 'observations/images/color' in f

                print(f"数据完整性: 机械臂={'✓' if has_robot else '✗'}, "
                      f"夹爪={'✓' if has_gripper else '✗'}, "
                      f"图像={'✓' if has_images else '✗'}\n")

                return True

        except Exception as e:
            self.errors.append(f"❌ 加载数据集失败: {e}")
            return False

    def check_statistics(self):
        """统计信息"""
        print("=" * 70)
        print("数据集基本信息")
        print("=" * 70)

        print(f"\n[文件信息]")
        print(f"  数据集路径: {self.hdf5_path}")
        print(f"  文件大小: {os.path.getsize(self.hdf5_path) / 1024 / 1024:.1f} MB")

        print(f"\n[时序信息]")
        print(f"  总帧数: {self.total_frames}")
        print(f"  总时长: {self.total_duration:.2f} 秒")
        print(f"  平均帧率: {self.total_frames / self.total_duration:.1f} Hz")
        print(f"  期望帧率: {self.expected_fps} Hz")

        print(f"\n[位置范围]")
        print(f"  X: [{self.robot_trajectory[:, 0].min():.1f}, {self.robot_trajectory[:, 0].max():.1f}] mm")
        print(f"  Y: [{self.robot_trajectory[:, 1].min():.1f}, {self.robot_trajectory[:, 1].max():.1f}] mm")
        print(f"  Z: [{self.robot_trajectory[:, 2].min():.1f}, {self.robot_trajectory[:, 2].max():.1f}] mm")

        print(f"\n[旋转范围]")
        print(f"  RX: [{self.robot_trajectory[:, 3].min():.1f}, {self.robot_trajectory[:, 3].max():.1f}] deg")
        print(f"  RY: [{self.robot_trajectory[:, 4].min():.1f}, {self.robot_trajectory[:, 4].max():.1f}] deg")
        print(f"  RZ: [{self.robot_trajectory[:, 5].min():.1f}, {self.robot_trajectory[:, 5].max():.1f}] deg")

        if self.gripper_trajectory is not None:
            print(f"\n[夹爪范围]")
            print(f"  范围: [{self.gripper_trajectory.min():.1f}, {self.gripper_trajectory.max():.1f}]")

    def check_frame_drops(self):
        """检查丢帧"""
        print("\n" + "=" * 70)
        print("丢帧检查")
        print("=" * 70)

        # 计算期望帧数和丢帧数
        expected_frames = int(self.total_duration * self.expected_fps)
        dropped_frames = expected_frames - self.total_frames

        print(f"\n[丢帧统计]")
        print(f"  实际帧数: {self.total_frames}")
        print(f"  期望帧数: {expected_frames}")
        print(f"  丢帧数量: {dropped_frames}")
        print(f"  丢帧率: {dropped_frames / expected_frames * 100:.2f}%")

        # 计算时间间隔
        time_diffs = np.diff(self.timestamps)

        print(f"\n[时间间隔]")
        print(f"  期望间隔: {self.expected_interval*1000:.2f} ms")
        print(f"  平均间隔: {time_diffs.mean()*1000:.2f} ms")
        print(f"  最小间隔: {time_diffs.min()*1000:.2f} ms")
        print(f"  最大间隔: {time_diffs.max()*1000:.2f} ms")
        print(f"  标准差: {time_diffs.std()*1000:.2f} ms")

        # 检测丢帧位置
        drop_mask = time_diffs > self.drop_threshold
        drop_locations = np.where(drop_mask)[0]

        print(f"\n[丢帧位置]")
        print(f"  检测到 {len(drop_locations)} 个丢帧位置")
        print(f"  判定阈值: >{self.drop_threshold*1000:.2f} ms")

        if len(drop_locations) > 0:
            # 显示前5个丢帧位置
            print(f"\n  前5个丢帧位置:")
            for i, idx in enumerate(drop_locations[:5]):
                interval = time_diffs[idx]
                estimated_drops = int(interval / self.expected_interval) - 1
                print(f"    #{i+1} 帧{idx}→{idx+1}: "
                      f"间隔={interval*1000:.1f}ms, "
                      f"估算丢{estimated_drops}帧")

        # 评估VLA训练适用性
        drop_rate = dropped_frames / expected_frames if expected_frames > 0 else 0

        print(f"\n[VLA训练适用性]")
        if drop_rate < 0.01:
            print(f"  🟢 优秀 (丢帧率 {drop_rate*100:.2f}%)")
            print(f"  ✅ 可直接用于VLA训练")
        elif drop_rate < 0.05:
            print(f"  🟡 良好 (丢帧率 {drop_rate*100:.2f}%)")
            print(f"  ⚠️  建议检查丢帧分布，可谨慎使用")
        elif drop_rate < 0.10:
            print(f"  🟠 一般 (丢帧率 {drop_rate*100:.2f}%)")
            print(f"  ⚠️  建议过滤严重片段或重新录制")
            self.warnings.append(f"丢帧率较高: {drop_rate*100:.2f}%")
        else:
            print(f"  🔴 差 (丢帧率 {drop_rate*100:.2f}%)")
            print(f"  ❌ 不建议使用，强烈建议重新录制")
            self.errors.append(f"丢帧率过高: {drop_rate*100:.2f}%")

    def check_workspace_limits(self):
        """检查工作空间"""
        print("\n" + "=" * 70)
        print("工作空间检查")
        print("=" * 70)

        violations = []

        for i in range(self.total_frames):
            x, y, z = self.robot_trajectory[i, :3]

            if not (self.WORKSPACE_X_MIN <= x <= self.WORKSPACE_X_MAX):
                violations.append(f"帧{i}: X={x:.1f} 超出范围")

            if not (self.WORKSPACE_Y_MIN <= y <= self.WORKSPACE_Y_MAX):
                violations.append(f"帧{i}: Y={y:.1f} 超出范围")

            if not (self.WORKSPACE_Z_MIN <= z <= self.WORKSPACE_Z_MAX):
                violations.append(f"帧{i}: Z={z:.1f} 超出范围")

        if violations:
            self.errors.extend(violations[:10])
            print(f"\n  ✗ 发现 {len(violations)} 个超出工作空间的点")
            for v in violations[:5]:
                print(f"    {v}")
        else:
            print(f"\n  ✓ 所有点在工作空间内")

    def check_sudden_jumps(self):
        """检查突变"""
        print("\n" + "=" * 70)
        print("突变检查")
        print("=" * 70)

        # 位置突变
        position_diffs = np.diff(self.robot_trajectory[:, :3], axis=0)
        position_jumps = np.linalg.norm(position_diffs, axis=1)
        position_violations = np.where(position_jumps > self.MAX_POSITION_JUMP)[0]

        print(f"\n[位置突变]")
        if len(position_violations) > 0:
            print(f"  ✗ 发现 {len(position_violations)} 个位置突变点 (> {self.MAX_POSITION_JUMP} mm)")
            for idx in position_violations[:5]:
                jump_size = position_jumps[idx]
                print(f"    帧{idx}→{idx+1}: 跳变 {jump_size:.2f} mm")
                print(f"      前: [{self.robot_trajectory[idx, 0]:.1f}, "
                      f"{self.robot_trajectory[idx, 1]:.1f}, "
                      f"{self.robot_trajectory[idx, 2]:.1f}]")
                print(f"      后: [{self.robot_trajectory[idx+1, 0]:.1f}, "
                      f"{self.robot_trajectory[idx+1, 1]:.1f}, "
                      f"{self.robot_trajectory[idx+1, 2]:.1f}]")
            self.errors.append(f"发现 {len(position_violations)} 个位置突变点")
        else:
            print(f"  ✓ 位置变化正常 (最大: {position_jumps.max():.2f} mm)")

        # 旋转突变
        rotation_diffs = np.diff(self.robot_trajectory[:, 3:6], axis=0)
        rotation_jumps = np.linalg.norm(rotation_diffs, axis=1)
        rotation_violations = np.where(rotation_jumps > self.MAX_ROTATION_JUMP)[0]

        print(f"\n[旋转突变]")
        if len(rotation_violations) > 0:
            print(f"  ! 发现 {len(rotation_violations)} 个旋转突变点 (> {self.MAX_ROTATION_JUMP} deg)")
            for idx in rotation_violations[:3]:
                jump_size = rotation_jumps[idx]
                print(f"    帧{idx}→{idx+1}: 旋转跳变 {jump_size:.2f} deg")
            self.warnings.append(f"发现 {len(rotation_violations)} 个旋转突变点")
        else:
            print(f"  ✓ 旋转变化正常 (最大: {rotation_jumps.max():.2f} deg)")

        # 夹爪突变
        if self.gripper_trajectory is not None:
            gripper_diffs = np.abs(np.diff(self.gripper_trajectory))
            gripper_violations = np.where(gripper_diffs > self.MAX_GRIPPER_JUMP)[0]

            print(f"\n[夹爪突变]")
            if len(gripper_violations) > 0:
                print(f"  ! 发现 {len(gripper_violations)} 个夹爪突变点 (> {self.MAX_GRIPPER_JUMP})")
                for idx in gripper_violations[:3]:
                    jump_size = gripper_diffs[idx]
                    print(f"    帧{idx}→{idx+1}: 夹爪跳变 {jump_size:.1f}")
                self.warnings.append(f"发现 {len(gripper_violations)} 个夹爪突变点")
            else:
                print(f"  ✓ 夹爪变化正常 (最大: {gripper_diffs.max():.1f})")

    def check_velocity(self):
        """检查速度"""
        print("\n" + "=" * 70)
        print("速度检查")
        print("=" * 70)

        time_diffs = np.diff(self.timestamps)
        position_diffs = np.diff(self.robot_trajectory[:, :3], axis=0)
        velocities = np.linalg.norm(position_diffs, axis=1) / time_diffs

        velocity_violations = np.where(velocities > self.MAX_VELOCITY)[0]

        print(f"\n[速度统计]")
        print(f"  平均速度: {velocities.mean():.1f} mm/s")
        print(f"  最大速度: {velocities.max():.1f} mm/s")
        print(f"  速度阈值: {self.MAX_VELOCITY} mm/s")

        if len(velocity_violations) > 0:
            print(f"\n  ! 发现 {len(velocity_violations)} 个高速运动点")
            for idx in velocity_violations[:3]:
                vel = velocities[idx]
                print(f"    帧{idx}→{idx+1}: 速度 {vel:.1f} mm/s")
            self.warnings.append(f"发现 {len(velocity_violations)} 个高速运动点")
        else:
            print(f"\n  ✓ 速度正常")

    def generate_report(self, output_path: Optional[str] = None):
        """生成详细报告"""
        if output_path is None:
            output_path = self.hdf5_path.replace('.hdf5', '_check_report.txt')

        print(f"\n正在生成详细报告: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("数据集综合检查报告\n")
            f.write(f"数据集: {self.hdf5_path}\n")
            f.write("=" * 70 + "\n\n")

            f.write("[基本信息]\n")
            f.write(f"总帧数: {self.total_frames}\n")
            f.write(f"总时长: {self.total_duration:.2f} 秒\n")
            f.write(f"平均帧率: {self.total_frames / self.total_duration:.1f} Hz\n\n")

            if self.errors:
                f.write("[错误]\n")
                for err in self.errors:
                    f.write(f"{err}\n")
                f.write("\n")

            if self.warnings:
                f.write("[警告]\n")
                for warn in self.warnings:
                    f.write(f"{warn}\n")
                f.write("\n")

            if self.info:
                f.write("[附加信息]\n")
                for info in self.info:
                    f.write(f"{info}\n")

        print(f"✓ 报告已保存")

    def run_all_checks(self) -> bool:
        """运行所有检查"""
        if not self.load_dataset():
            return False

        self.check_statistics()
        self.check_frame_drops()
        self.check_workspace_limits()
        self.check_sudden_jumps()
        self.check_velocity()

        return True

    def print_summary(self) -> bool:
        """打印检查总结"""
        print("\n" + "=" * 70)
        print("检查总结")
        print("=" * 70)

        has_errors = len(self.errors) > 0
        has_warnings = len(self.warnings) > 0

        if has_errors:
            print(f"\n❌ 发现 {len(self.errors)} 个错误:")
            for err in self.errors[:10]:
                print(f"  • {err}")
            if len(self.errors) > 10:
                print(f"  ... 还有 {len(self.errors) - 10} 个错误")

        if has_warnings:
            print(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
            for warn in self.warnings[:10]:
                print(f"  • {warn}")
            if len(self.warnings) > 10:
                print(f"  ... 还有 {len(self.warnings) - 10} 个警告")

        if not has_errors and not has_warnings:
            print("\n✅ 所有检查通过！")
            print("  • 数据集质量优秀")
            print("  • 可安全播放")
            print("  • 适合用于VLA训练")
            return True
        elif not has_errors:
            print("\n⚠️  存在警告但无严重错误")
            print("  • 数据集基本可用")
            print("  • 建议谨慎播放")
            return True
        else:
            print("\n❌ 存在严重错误")
            print("  • 不建议直接使用")
            print("  • 建议修复后再使用")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='数据集综合检查工具 - 整合丢帧检查和安全检查',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s dataset.hdf5                    # 运行所有检查
  %(prog)s dataset.hdf5 --frames           # 仅检查丢帧
  %(prog)s dataset.hdf5 --safety           # 仅检查安全性
  %(prog)s dataset.hdf5 --report           # 生成详细报告
  %(prog)s dataset.hdf5 --strict           # 使用严格模式
  %(prog)s dataset.hdf5 --fps 20           # 指定期望帧率
        """
    )

    parser.add_argument('dataset', type=str, help='HDF5数据集路径')
    parser.add_argument('--fps', type=float, default=30.0, help='期望帧率 (默认: 30Hz)')
    parser.add_argument('--strict', action='store_true', help='使用严格模式（更低的阈值）')
    parser.add_argument('--report', action='store_true', help='生成详细报告文件')

    # 选择性检查
    parser.add_argument('--frames', action='store_true', help='仅运行丢帧检查')
    parser.add_argument('--safety', action='store_true', help='仅运行安全检查')

    # 自定义阈值
    parser.add_argument('--max-position-jump', type=float, help='自定义位置跳变阈值 (mm)')
    parser.add_argument('--max-rotation-jump', type=float, help='自定义旋转跳变阈值 (deg)')
    parser.add_argument('--max-velocity', type=float, help='自定义最大速度阈值 (mm/s)')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.dataset):
        print(f"❌ 错误：文件不存在 {args.dataset}")
        sys.exit(1)

    # 创建检查器
    checker = DatasetChecker(args.dataset, expected_fps=args.fps, strict=args.strict)

    # 自定义阈值
    if args.max_position_jump is not None:
        checker.MAX_POSITION_JUMP = args.max_position_jump
    if args.max_rotation_jump is not None:
        checker.MAX_ROTATION_JUMP = args.max_rotation_jump
    if args.max_velocity is not None:
        checker.MAX_VELOCITY = args.max_velocity

    # 加载数据集
    if not checker.load_dataset():
        print("\n❌ 无法加载数据集")
        sys.exit(1)

    # 运行检查
    checker.check_statistics()

    # 根据参数选择要运行的检查
    if args.frames and not args.safety:
        # 仅丢帧检查
        checker.check_frame_drops()
    elif args.safety and not args.frames:
        # 仅安全检查
        checker.check_workspace_limits()
        checker.check_sudden_jumps()
        checker.check_velocity()
    else:
        # 运行所有检查
        checker.check_frame_drops()
        checker.check_workspace_limits()
        checker.check_sudden_jumps()
        checker.check_velocity()

    # 打印总结
    is_ok = checker.print_summary()

    # 生成报告
    if args.report:
        checker.generate_report()

    print("\n" + "=" * 70)
    print("检查完成")
    print("=" * 70 + "\n")

    # 返回退出码
    sys.exit(0 if is_ok else 1)


if __name__ == "__main__":
    main()
