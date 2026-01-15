#!/usr/bin/env python3
"""
数据集安全检查工具
在播放前检查轨迹中是否存在剧烈变化，防止机械臂爆冲
"""

import h5py
import numpy as np
import argparse
import sys

class SafetyChecker:
    """数据集安全检查器"""

    # 安全阈值（根据实际情况调整）
    MAX_POSITION_JUMP = 50.0     # mm - 单帧最大位置跳变
    MAX_ROTATION_JUMP = 10.0     # deg - 单帧最大旋转跳变
    MAX_GRIPPER_JUMP = 200.0     # 夹爪最大跳变
    MAX_VELOCITY = 100.0         # mm/s - 最大速度（假设30Hz）
    MAX_ACCELERATION = 500.0     # mm/s^2 - 最大加速度

    # 工作空间限制（根据机械臂实际工作空间调整）
    WORKSPACE_X_MIN = -600.0
    WORKSPACE_X_MAX = 600.0
    WORKSPACE_Y_MIN = -600.0
    WORKSPACE_Y_MAX = 600.0
    WORKSPACE_Z_MIN = -100.0
    WORKSPACE_Z_MAX = 600.0

    def __init__(self, hdf5_path, strict=False):
        self.hdf5_path = hdf5_path
        self.strict = strict

        # 如果是严格模式，降低阈值
        if strict:
            self.MAX_POSITION_JUMP = 20.0
            self.MAX_ROTATION_JUMP = 5.0
            self.MAX_GRIPPER_JUMP = 100.0

        self.warnings = []
        self.errors = []

    def load_data(self):
        """加载HDF5数据"""
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                # 读取轨迹数据
                if 'actions/robot_target' in f:
                    self.robot_trajectory = np.array(f['actions/robot_target'])
                elif 'observations/robot_current' in f:
                    self.robot_trajectory = np.array(f['observations/robot_current'])
                else:
                    raise ValueError("未找到机械臂轨迹数据")

                if 'actions/gripper_target' in f:
                    self.gripper_trajectory = np.array(f['actions/gripper_target']).flatten()
                elif 'observations/gripper_current' in f:
                    self.gripper_trajectory = np.array(f['observations/gripper_current']).flatten()
                else:
                    self.gripper_trajectory = None

                # 读取时间戳
                if 'timestamp' in f:
                    self.timestamps = np.array(f['timestamp'])
                else:
                    # 假设30Hz
                    self.timestamps = np.arange(len(self.robot_trajectory)) / 30.0

                self.total_frames = len(self.robot_trajectory)

        except Exception as e:
            print(f"[错误] 无法加载数据集: {e}")
            sys.exit(1)

    def check_workspace_limits(self):
        """检查是否超出工作空间"""
        print("\n[检查1] 工作空间边界检查...")

        violations = []

        # 检查XYZ是否在工作空间内
        for i in range(self.total_frames):
            x, y, z = self.robot_trajectory[i, :3]

            if not (self.WORKSPACE_X_MIN <= x <= self.WORKSPACE_X_MAX):
                violations.append(f"  帧{i}: X={x:.1f} 超出范围 [{self.WORKSPACE_X_MIN}, {self.WORKSPACE_X_MAX}]")

            if not (self.WORKSPACE_Y_MIN <= y <= self.WORKSPACE_Y_MAX):
                violations.append(f"  帧{i}: Y={y:.1f} 超出范围 [{self.WORKSPACE_Y_MIN}, {self.WORKSPACE_Y_MAX}]")

            if not (self.WORKSPACE_Z_MIN <= z <= self.WORKSPACE_Z_MAX):
                violations.append(f"  帧{i}: Z={z:.1f} 超出范围 [{self.WORKSPACE_Z_MIN}, {self.WORKSPACE_Z_MAX}]")

        if violations:
            self.errors.extend(violations[:10])  # 只显示前10个
            print(f"  ✗ 发现 {len(violations)} 个超出工作空间的点")
            if len(violations) > 10:
                print(f"  (仅显示前10个)")
        else:
            print("  ✓ 所有点在工作空间内")

    def check_sudden_jumps(self):
        """检查位置突变"""
        print("\n[检查2] 位置突变检查...")

        # 计算位置差分
        position_diffs = np.diff(self.robot_trajectory[:, :3], axis=0)
        rotation_diffs = np.diff(self.robot_trajectory[:, 3:6], axis=0)

        # 计算跳变幅度
        position_jumps = np.linalg.norm(position_diffs, axis=1)
        rotation_jumps = np.linalg.norm(rotation_diffs, axis=1)

        # 检查位置跳变
        position_violations = np.where(position_jumps > self.MAX_POSITION_JUMP)[0]
        rotation_violations = np.where(rotation_jumps > self.MAX_ROTATION_JUMP)[0]

        if len(position_violations) > 0:
            self.errors.append(f"  ✗ 发现 {len(position_violations)} 个位置突变点 (>  {self.MAX_POSITION_JUMP} mm)")
            for idx in position_violations[:5]:  # 显示前5个
                jump_size = position_jumps[idx]
                self.errors.append(f"    帧{idx}->{idx+1}: 跳变 {jump_size:.2f} mm")
                self.errors.append(f"      前: [{self.robot_trajectory[idx, 0]:.1f}, {self.robot_trajectory[idx, 1]:.1f}, {self.robot_trajectory[idx, 2]:.1f}]")
                self.errors.append(f"      后: [{self.robot_trajectory[idx+1, 0]:.1f}, {self.robot_trajectory[idx+1, 1]:.1f}, {self.robot_trajectory[idx+1, 2]:.1f}]")
        else:
            print(f"  ✓ 位置变化正常 (最大: {position_jumps.max():.2f} mm < {self.MAX_POSITION_JUMP} mm)")

        if len(rotation_violations) > 0:
            self.warnings.append(f"  ! 发现 {len(rotation_violations)} 个旋转突变点 (> {self.MAX_ROTATION_JUMP} deg)")
            for idx in rotation_violations[:3]:
                jump_size = rotation_jumps[idx]
                self.warnings.append(f"    帧{idx}->{idx+1}: 旋转跳变 {jump_size:.2f} deg")
        else:
            print(f"  ✓ 旋转变化正常 (最大: {rotation_jumps.max():.2f} deg < {self.MAX_ROTATION_JUMP} deg)")

    def check_gripper_jumps(self):
        """检查夹爪突变"""
        if self.gripper_trajectory is None:
            return

        print("\n[检查3] 夹爪突变检查...")

        gripper_diffs = np.abs(np.diff(self.gripper_trajectory))
        gripper_violations = np.where(gripper_diffs > self.MAX_GRIPPER_JUMP)[0]

        if len(gripper_violations) > 0:
            self.warnings.append(f"  ! 发现 {len(gripper_violations)} 个夹爪突变点 (> {self.MAX_GRIPPER_JUMP})")
            for idx in gripper_violations[:3]:
                jump_size = gripper_diffs[idx]
                self.warnings.append(f"    帧{idx}->{idx+1}: 夹爪跳变 {jump_size:.1f}")
        else:
            print(f"  ✓ 夹爪变化正常 (最大: {gripper_diffs.max():.1f} < {self.MAX_GRIPPER_JUMP})")

    def check_velocity(self):
        """检查速度是否合理"""
        print("\n[检查4] 速度检查...")

        # 计算时间间隔
        time_diffs = np.diff(self.timestamps)

        # 计算位置差分
        position_diffs = np.diff(self.robot_trajectory[:, :3], axis=0)

        # 计算速度 (mm/s)
        velocities = np.linalg.norm(position_diffs, axis=1) / time_diffs

        velocity_violations = np.where(velocities > self.MAX_VELOCITY)[0]

        if len(velocity_violations) > 0:
            self.warnings.append(f"  ! 发现 {len(velocity_violations)} 个高速运动点 (> {self.MAX_VELOCITY} mm/s)")
            for idx in velocity_violations[:3]:
                vel = velocities[idx]
                self.warnings.append(f"    帧{idx}->{idx+1}: 速度 {vel:.1f} mm/s")
        else:
            print(f"  ✓ 速度正常 (最大: {velocities.max():.1f} mm/s < {self.MAX_VELOCITY} mm/s)")

        print(f"  平均速度: {velocities.mean():.1f} mm/s")
        print(f"  最大速度: {velocities.max():.1f} mm/s")

    def check_statistics(self):
        """打印统计信息"""
        print("\n[统计信息]")
        print(f"  总帧数: {self.total_frames}")
        print(f"  总时长: {self.timestamps[-1] - self.timestamps[0]:.2f} 秒")
        print(f"  平均帧率: {self.total_frames / (self.timestamps[-1] - self.timestamps[0]):.1f} Hz")

        # 位置范围
        print(f"\n  位置范围:")
        print(f"    X: [{self.robot_trajectory[:, 0].min():.1f}, {self.robot_trajectory[:, 0].max():.1f}] mm")
        print(f"    Y: [{self.robot_trajectory[:, 1].min():.1f}, {self.robot_trajectory[:, 1].max():.1f}] mm")
        print(f"    Z: [{self.robot_trajectory[:, 2].min():.1f}, {self.robot_trajectory[:, 2].max():.1f}] mm")

        # 旋转范围
        print(f"\n  旋转范围:")
        print(f"    RX: [{self.robot_trajectory[:, 3].min():.1f}, {self.robot_trajectory[:, 3].max():.1f}] deg")
        print(f"    RY: [{self.robot_trajectory[:, 4].min():.1f}, {self.robot_trajectory[:, 4].max():.1f}] deg")
        print(f"    RZ: [{self.robot_trajectory[:, 5].min():.1f}, {self.robot_trajectory[:, 5].max():.1f}] deg")

        if self.gripper_trajectory is not None:
            print(f"\n  夹爪范围: [{self.gripper_trajectory.min():.1f}, {self.gripper_trajectory.max():.1f}]")

    def run_all_checks(self):
        """运行所有检查"""
        print("="*70)
        print(f"数据集安全检查: {self.hdf5_path}")
        print("="*70)

        # 加载数据
        self.load_data()

        # 运行检查
        self.check_statistics()
        self.check_workspace_limits()
        self.check_sudden_jumps()
        self.check_gripper_jumps()
        self.check_velocity()

        # 打印结果
        print("\n" + "="*70)
        print("检查结果")
        print("="*70)

        if self.errors:
            print(f"\n❌ 发现 {len(self.errors)} 个错误:")
            for err in self.errors:
                print(err)

        if self.warnings:
            print(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
            for warn in self.warnings:
                print(warn)

        if not self.errors and not self.warnings:
            print("\n✅ 所有检查通过，数据集安全可播放！")
            return True
        elif not self.errors:
            print("\n⚠️  存在警告，建议谨慎播放")
            return True
        else:
            print("\n❌ 存在错误，建议修复后再播放！")
            return False

def main():
    parser = argparse.ArgumentParser(description='检查数据集安全性，防止机械臂爆冲')
    parser.add_argument('dataset', type=str, help='HDF5数据集路径')
    parser.add_argument('--strict', action='store_true', help='使用严格模式（更低的阈值）')
    parser.add_argument('--max-position-jump', type=float, default=None, help='自定义位置跳变阈值 (mm)')
    parser.add_argument('--max-rotation-jump', type=float, default=None, help='自定义旋转跳变阈值 (deg)')
    parser.add_argument('--max-velocity', type=float, default=None, help='自定义最大速度阈值 (mm/s)')

    args = parser.parse_args()

    # 创建检查器
    checker = SafetyChecker(args.dataset, strict=args.strict)

    # 自定义阈值
    if args.max_position_jump is not None:
        checker.MAX_POSITION_JUMP = args.max_position_jump
    if args.max_rotation_jump is not None:
        checker.MAX_ROTATION_JUMP = args.max_rotation_jump
    if args.max_velocity is not None:
        checker.MAX_VELOCITY = args.max_velocity

    # 运行检查
    is_safe = checker.run_all_checks()

    # 返回退出码
    sys.exit(0 if is_safe else 1)

if __name__ == "__main__":
    main()
