#!/usr/bin/env python3
"""
LeRobot v2.0 数据集验证脚本

验证 LeRobot v2.0 格式数据集的完整性和正确性

使用方法:
    python validate_lerobot_v2.py --root ./lerobot_dataset
"""

import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq


def validate_lerobot_v2_dataset(dataset_root):
    """验证 LeRobot v2.0 数据集"""

    dataset_path = Path(dataset_root)

    print("=" * 70)
    print("🔍 LeRobot v2.0 数据集验证")
    print("=" * 70)
    print(f"数据集路径: {dataset_path.absolute()}\n")

    errors = []
    warnings = []

    # ========== 1. 检查目录结构 ==========
    print("[1/6] 检查目录结构...")
    required_dirs = {
        "data": dataset_path / "data",
        "videos": dataset_path / "videos",
        "meta": dataset_path / "meta"
    }

    for name, dir_path in required_dirs.items():
        if dir_path.exists():
            print(f"  ✓ {name}/ 存在")
        else:
            errors.append(f"缺少目录: {name}/")
            print(f"  ✗ {name}/ 缺失")

    # ========== 2. 检查元数据文件 ==========
    print("\n[2/6] 检查元数据文件...")
    meta_files = {
        "info.json": dataset_path / "meta" / "info.json",
        "episodes.jsonl": dataset_path / "meta" / "episodes.jsonl"
    }

    info_data = None
    episodes_data = []

    for name, file_path in meta_files.items():
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"  ✓ meta/{name} 存在 ({size_kb:.2f} KB)")

            if name == "info.json":
                try:
                    with open(file_path) as f:
                        info_data = json.load(f)
                except Exception as e:
                    errors.append(f"无法解析 {name}: {e}")
                    print(f"    ✗ 解析失败: {e}")

            elif name == "episodes.jsonl":
                try:
                    with open(file_path) as f:
                        for line in f:
                            episodes_data.append(json.loads(line))
                except Exception as e:
                    errors.append(f"无法解析 {name}: {e}")
                    print(f"    ✗ 解析失败: {e}")
        else:
            errors.append(f"缺少文件: meta/{name}")
            print(f"  ✗ meta/{name} 缺失")

    # ========== 3. 验证 info.json ==========
    print("\n[3/6] 验证 info.json 内容...")
    if info_data:
        required_fields = {
            'codebase_version': 'str',
            'robot_type': 'str',
            'total_episodes': 'int',
            'total_frames': 'int',
            'fps': 'int'
        }

        all_fields_ok = True
        for field, expected_type in required_fields.items():
            if field in info_data:
                value = info_data[field]

                # 类型检查
                if expected_type == 'int' and not isinstance(value, int):
                    warnings.append(f"{field} 应该是 int 类型")
                    print(f"  ⚠  {field}: {value} (类型: {type(value).__name__}, 期望: {expected_type})")
                elif expected_type == 'str' and not isinstance(value, str):
                    warnings.append(f"{field} 应该是 str 类型")
                    print(f"  ⚠  {field}: {value} (类型: {type(value).__name__}, 期望: {expected_type})")
                else:
                    print(f"  ✓ {field}: {value}")
            else:
                errors.append(f"info.json 缺少字段: {field}")
                print(f"  ✗ 缺少字段: {field}")
                all_fields_ok = False

        # 检查可选但推荐的字段
        optional_fields = ['cameras', 'observation_shapes', 'action_shape', 'stats']
        print(f"\n  可选字段:")
        for field in optional_fields:
            if field in info_data:
                print(f"  ✓ {field} 存在")
            else:
                warnings.append(f"info.json 缺少推荐字段: {field}")
                print(f"  ⚠  {field} 不存在（推荐添加）")
    else:
        errors.append("info.json 无法读取或解析")

    # ========== 4. 验证 episodes.jsonl ==========
    print("\n[4/6] 验证 episodes.jsonl...")
    if episodes_data:
        print(f"  ✓ 总共 {len(episodes_data)} 个 episodes")

        # 检查索引连续性
        index_errors = []
        for i, ep in enumerate(episodes_data):
            expected_idx = i
            actual_idx = ep.get('episode_index')

            if actual_idx != expected_idx:
                index_errors.append(
                    f"Episode 索引不连续: 位置 {i}, 期望 {expected_idx}, 实际 {actual_idx}"
                )

        if index_errors:
            for err in index_errors[:5]:  # 只显示前5个错误
                errors.append(err)
                print(f"  ✗ {err}")
            if len(index_errors) > 5:
                print(f"  ... 还有 {len(index_errors)-5} 个索引错误")
        else:
            print(f"  ✓ Episode 索引从 0 到 {len(episodes_data)-1} 连续")

        # 显示每个 episode 的信息
        print(f"\n  Episode 详情:")
        for ep in episodes_data[:5]:  # 只显示前5个
            ep_idx = ep.get('episode_index', '?')
            ep_len = ep.get('length', '?')
            print(f"    Episode {ep_idx}: {ep_len} 帧")
        if len(episodes_data) > 5:
            print(f"    ... 还有 {len(episodes_data)-5} 个 episodes")

        # 与 info.json 对比
        if info_data and 'total_episodes' in info_data:
            if len(episodes_data) != info_data['total_episodes']:
                errors.append(
                    f"episodes.jsonl 行数 ({len(episodes_data)}) 与 "
                    f"info.json 中的 total_episodes ({info_data['total_episodes']}) 不一致"
                )
                print(f"  ✗ Episodes 数量不匹配")
            else:
                print(f"  ✓ Episodes 数量与 info.json 一致")
    else:
        errors.append("episodes.jsonl 为空或无法读取")

    # ========== 5. 检查 Parquet 数据文件 ==========
    print("\n[5/6] 检查 Parquet 数据文件...")

    data_dir = dataset_path / "data" / "chunk-000"
    if data_dir.exists():
        parquet_files = sorted(data_dir.glob("episode_*.parquet"))
        print(f"  ✓ 找到 {len(parquet_files)} 个 parquet 文件")

        if len(parquet_files) != len(episodes_data):
            warnings.append(
                f"Parquet 文件数量 ({len(parquet_files)}) 与 "
                f"episodes 数量 ({len(episodes_data)}) 不一致"
            )
            print(f"  ⚠  文件数量不匹配")

        # 抽样检查前3个文件
        print(f"\n  数据文件抽样检查:")
        total_frames = 0
        for pf in parquet_files[:3]:
            try:
                table = pq.read_table(pf)
                num_rows = table.num_rows
                total_frames += num_rows

                # 检查必需的列
                required_columns = [
                    'episode_index', 'frame_index', 'timestamp',
                    'observation.state', 'action', 'next.done'
                ]

                missing_cols = [col for col in required_columns
                               if col not in table.column_names]

                if missing_cols:
                    errors.append(f"{pf.name} 缺少列: {missing_cols}")
                    print(f"  ✗ {pf.name}: {num_rows} 行, 缺少列: {missing_cols}")
                else:
                    print(f"  ✓ {pf.name}: {num_rows} 行, schema 有效")

            except Exception as e:
                errors.append(f"无法读取 {pf.name}: {e}")
                print(f"  ✗ 无法读取 {pf.name}: {e}")

        if len(parquet_files) > 3:
            print(f"    ... 还有 {len(parquet_files)-3} 个文件未检查")

        # 验证总帧数（如果检查了所有文件）
        if len(parquet_files) <= 3 and info_data and 'total_frames' in info_data:
            if total_frames != info_data['total_frames']:
                warnings.append(
                    f"实际帧数 ({total_frames}) 与 "
                    f"info.json 中的 total_frames ({info_data['total_frames']}) 不一致"
                )
    else:
        errors.append("data/chunk-000/ 目录不存在")
        print(f"  ✗ data/chunk-000/ 不存在")

    # ========== 6. 检查视频文件 ==========
    print("\n[6/6] 检查视频文件...")

    videos_dir = dataset_path / "videos" / "chunk-000"
    if videos_dir.exists():
        camera_dirs = [d for d in videos_dir.iterdir() if d.is_dir()]
        print(f"  ✓ 找到 {len(camera_dirs)} 个相机目录")

        for cam_dir in camera_dirs:
            video_files = sorted(cam_dir.glob("episode_*.mp4"))
            print(f"    ✓ {cam_dir.name}: {len(video_files)} 个视频")

            if len(video_files) != len(episodes_data):
                warnings.append(
                    f"{cam_dir.name} 视频数量 ({len(video_files)}) 与 "
                    f"episodes 数量 ({len(episodes_data)}) 不一致"
                )
                print(f"      ⚠  视频数量不匹配")
    else:
        warnings.append("videos/chunk-000/ 目录不存在")
        print(f"  ⚠  videos/chunk-000/ 不存在")

    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("📊 验证总结")
    print("=" * 70)

    if len(errors) == 0:
        print("✅ 数据集验证通过！没有发现错误。")
    else:
        print(f"❌ 发现 {len(errors)} 个错误:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    if len(warnings) > 0:
        print(f"\n⚠️  发现 {len(warnings)} 个警告:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    if len(errors) == 0 and len(warnings) == 0:
        print("\n🎉 数据集完全符合 LeRobot v2.0 规范！")

    print("=" * 70)

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="验证 LeRobot v2.0 格式数据集"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="数据集根目录（包含 data/, videos/, meta/ 的目录）"
    )

    args = parser.parse_args()

    success = validate_lerobot_v2_dataset(args.root)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
