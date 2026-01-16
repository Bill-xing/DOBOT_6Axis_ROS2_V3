#!/usr/bin/env python3
"""测试并可视化 LeRobot 数据集"""

import sys
from pathlib import Path
import cv2
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print("✓ lerobot 库导入成功")
except ImportError as e:
    print(f"❌ 无法导入 lerobot: {e}")
    sys.exit(1)

# 配置
REPO_ID = "leng234/cr51"
ROOT_DIR = "./lerobot_dataset"
OUTPUT_DIR = Path("./dataset_preview")

def main():
    print(f"\n{'='*60}")
    print(f"加载数据集: {REPO_ID}")
    print(f"根目录: {ROOT_DIR}")
    print(f"{'='*60}\n")

    try:
        # 加载数据集（只使用本地文件）
        dataset = LeRobotDataset(
            repo_id=REPO_ID,
            root=ROOT_DIR,
        )
        print("✓ 数据集加载成功！\n")

    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 打印数据集信息
    print(f"📊 数据集统计:")
    print(f"  - Episodes 数量: {dataset.num_episodes}")
    print(f"  - 总帧数: {len(dataset)}")
    print(f"  - FPS: {dataset.fps}")
    print(f"  - 特征列表: {list(dataset.features.keys())}")

    # 打印每个 episode 的信息
    print(f"\n📁 Episodes 详情:")
    for ep_idx in range(dataset.num_episodes):
        ep_length = len(dataset.episode_data_index['to'][ep_idx]) - len(dataset.episode_data_index['from'][ep_idx])
        print(f"  Episode {ep_idx}: {ep_length} 帧")

    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n💾 保存示例图像到: {OUTPUT_DIR}")

    # 保存每个 episode 的第一帧
    for ep_idx in range(min(3, dataset.num_episodes)):
        try:
            # 获取该 episode 的第一帧索引
            first_frame_idx = dataset.episode_data_index['from'][ep_idx][0]
            sample = dataset[first_frame_idx]

            # 提取图像
            img_tensor = sample['observation.images.top']

            # 转换为 numpy 数组 (C, H, W) -> (H, W, C)
            img = img_tensor.numpy().transpose(1, 2, 0)

            # 转换为 0-255 范围
            img = (img * 255).astype(np.uint8)

            # RGB -> BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 添加文本信息
            state = sample['observation.state'].numpy()
            action = sample['action'].numpy()

            text_lines = [
                f"Episode {ep_idx} - Frame 0",
                f"State: [{state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f}]",
                f"Action: [{action[0]:.1f}, {action[1]:.1f}, {action[2]:.1f}]",
            ]

            y_offset = 30
            for line in text_lines:
                cv2.putText(img_bgr, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

            # 保存图像
            output_path = OUTPUT_DIR / f"episode_{ep_idx:03d}_frame_000.jpg"
            cv2.imwrite(str(output_path), img_bgr)
            print(f"  ✓ {output_path}")

            # 打印该帧的详细信息
            print(f"    观测状态: {state}")
            print(f"    动作: {action}")

        except Exception as e:
            print(f"  ❌ Episode {ep_idx} 处理失败: {e}")

    # 生成简单的统计报告
    print(f"\n📈 数据统计:")
    if hasattr(dataset.meta, 'stats') and dataset.meta.stats:
        stats = dataset.meta.stats

        if 'observation.state' in stats:
            print(f"\n  观测状态 (observation.state):")
            print(f"    - 最小值: {stats['observation.state']['min']}")
            print(f"    - 最大值: {stats['observation.state']['max']}")
            print(f"    - 均值: {stats['observation.state']['mean']}")

        if 'action' in stats:
            print(f"\n  动作 (action):")
            print(f"    - 最小值: {stats['action']['min']}")
            print(f"    - 最大值: {stats['action']['max']}")
            print(f"    - 均值: {stats['action']['mean']}")

    print(f"\n{'='*60}")
    print(f"✅ 数据集验证完成！")
    print(f"{'='*60}\n")
    print(f"📂 查看生成的图像: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
