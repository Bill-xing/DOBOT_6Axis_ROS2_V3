#!/usr/bin/env python3
"""
LeRobot数据集验证脚本

用于验证转换后的LeRobot v2.0数据集的完整性和正确性

使用方法:
    python verify_lerobot_dataset.py --dataset ./lerobot_dataset/data

功能:
    - 检查数据集结构
    - 验证数据完整性
    - 显示数据统计
    - 可视化样本数据
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from datasets import load_from_disk
except ImportError:
    print("Error: Please install datasets package:")
    print("pip install datasets")
    exit(1)


def verify_dataset(dataset_path, visualize=True, num_samples=3):
    """
    验证LeRobot数据集

    参数:
        dataset_path (str): 数据集路径
        visualize (bool): 是否可视化样本
        num_samples (int): 可视化样本数量
    """
    print("=== LeRobot Dataset Verification ===\n")

    dataset_path = Path(dataset_path)

    # 检查数据集是否存在
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return False

    # 加载数据集
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(str(dataset_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    print(f"✓ Dataset loaded successfully\n")

    # 检查元数据
    meta_dir = dataset_path.parent / "meta"
    if meta_dir.exists():
        info_path = meta_dir / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            print("Dataset Info:")
            print(f"  - Total Episodes: {info.get('total_episodes', 'N/A')}")
            print(f"  - Total Frames: {info.get('total_frames', 'N/A')}")
            print(f"  - FPS: {info.get('fps', 'N/A')}")
            print(f"  - Robot Type: {info.get('robot_type', 'N/A')}")
            print()

    # 数据集基本信息
    print("Dataset Structure:")
    print(f"  - Total Samples: {len(dataset)}")
    print(f"  - Features: {list(dataset.features.keys())}")
    print()

    # 检查episode分布
    episode_indices = np.array(dataset['episode_index'])
    unique_episodes = np.unique(episode_indices)
    print(f"Episode Distribution:")
    print(f"  - Number of Episodes: {len(unique_episodes)}")
    print(f"  - Episode Indices: {unique_episodes.tolist()}")

    # 每个episode的帧数
    print(f"\n  Frames per Episode:")
    for ep_idx in unique_episodes[:10]:  # 只显示前10个
        num_frames = np.sum(episode_indices == ep_idx)
        print(f"    Episode {ep_idx}: {num_frames} frames")
    if len(unique_episodes) > 10:
        print(f"    ... and {len(unique_episodes) - 10} more episodes")
    print()

    # 数据范围统计
    print("Data Statistics:")

    # Observation state
    obs_states = np.array([sample for sample in dataset['observation.state']])
    print(f"  observation.state: shape={obs_states.shape}")
    print(f"    - Min: {obs_states.min(axis=0)}")
    print(f"    - Max: {obs_states.max(axis=0)}")
    print(f"    - Mean: {obs_states.mean(axis=0)}")

    # Gripper
    grippers = np.array([sample for sample in dataset['observation.gripper']])
    print(f"\n  observation.gripper: shape={grippers.shape}")
    print(f"    - Min: {grippers.min():.3f}")
    print(f"    - Max: {grippers.max():.3f}")
    print(f"    - Mean: {grippers.mean():.3f}")

    # Actions
    actions = np.array([sample for sample in dataset['action']])
    print(f"\n  action: shape={actions.shape}")
    print(f"    - Min: {actions.min(axis=0)}")
    print(f"    - Max: {actions.max(axis=0)}")
    print(f"    - Mean: {actions.mean(axis=0)}")
    print()

    # 检查图像
    print("Image Statistics:")
    sample_image = dataset[0]['observation.images.top']
    if isinstance(sample_image, str):
        # 如果是路径，加载图像
        img = Image.open(sample_image)
        img_array = np.array(img)
    else:
        # 如果是PIL Image对象
        img_array = np.array(sample_image)

    print(f"  - Image Shape: {img_array.shape}")
    print(f"  - Image Dtype: {img_array.dtype}")
    print(f"  - Value Range: [{img_array.min()}, {img_array.max()}]")
    print()

    # 数据完整性检查
    print("Data Integrity Check:")
    issues = []

    # 检查是否有NaN
    if np.any(np.isnan(obs_states)):
        issues.append("✗ Found NaN in observation.state")
    else:
        print("  ✓ No NaN in observation.state")

    if np.any(np.isnan(actions)):
        issues.append("✗ Found NaN in actions")
    else:
        print("  ✓ No NaN in actions")

    # 检查时间戳单调性
    for ep_idx in unique_episodes:
        ep_mask = episode_indices == ep_idx
        ep_timestamps = np.array(dataset['timestamp'])[ep_mask]
        if not np.all(ep_timestamps[1:] >= ep_timestamps[:-1]):
            issues.append(f"✗ Non-monotonic timestamps in episode {ep_idx}")

    if len(issues) == 0:
        print("  ✓ Timestamps are monotonic in all episodes")
    else:
        for issue in issues:
            print(f"  {issue}")
    print()

    # 可视化样本
    if visualize and len(dataset) > 0:
        print(f"Visualizing {num_samples} random samples...")
        visualize_samples(dataset, num_samples)

    # 总结
    print("\n=== Verification Summary ===")
    if len(issues) == 0:
        print("✓ All checks passed! Dataset is valid.")
        return True
    else:
        print(f"✗ Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  {issue}")
        return False


def visualize_samples(dataset, num_samples=3):
    """
    可视化数据集样本

    参数:
        dataset: Hugging Face Dataset对象
        num_samples: 要可视化的样本数量
    """
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]

        # 显示图像
        img = sample['observation.images.top']
        if isinstance(img, str):
            img = Image.open(img)
        img_array = np.array(img)

        axes[i, 0].imshow(img_array)
        axes[i, 0].set_title(f"Frame {idx} (Episode {sample['episode_index']})")
        axes[i, 0].axis('off')

        # 显示状态数据
        obs_state = np.array(sample['observation.state'])
        action = np.array(sample['action'])

        axes[i, 1].bar(range(6), obs_state, alpha=0.7, label='Observation')
        axes[i, 1].bar(range(6), action[:6], alpha=0.7, label='Action')
        axes[i, 1].set_xlabel('Dimension')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].set_title('Robot Pose (x,y,z,rx,ry,rz)')
        axes[i, 1].legend()
        axes[i, 1].set_xticks(range(6))
        axes[i, 1].set_xticklabels(['x', 'y', 'z', 'rx', 'ry', 'rz'])

        # 显示夹爪状态
        gripper_obs = sample['observation.gripper'][0]
        gripper_action = action[6]

        axes[i, 2].bar(['Observation', 'Action'], [gripper_obs, gripper_action], alpha=0.7)
        axes[i, 2].set_ylabel('Gripper Opening')
        axes[i, 2].set_title('Gripper State')
        axes[i, 2].set_ylim([0, max(1000, gripper_obs * 1.1, gripper_action * 1.1)])

    plt.tight_layout()

    # 保存图像
    output_path = Path("dataset_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")

    # 显示图像（如果在图形环境中）
    try:
        plt.show(block=False)
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="Verify LeRobot v2.0 dataset")

    parser.add_argument(
        "--dataset",
        type=str,
        default="./lerobot_dataset/data",
        help="Path to LeRobot dataset directory (default: ./lerobot_dataset/data)"
    )

    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization (default: False)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to visualize (default: 3)"
    )

    args = parser.parse_args()

    # 执行验证
    success = verify_dataset(
        dataset_path=args.dataset,
        visualize=not args.no_visualize,
        num_samples=args.num_samples
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
