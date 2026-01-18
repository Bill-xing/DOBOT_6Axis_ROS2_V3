#!/usr/bin/env python3
"""
HDF5数据集转LeRobot v2.0格式转换器

参考 any4lerobot (https://github.com/Tavish9/any4lerobot) 的标准格式
将data_collector4.py和recorder_optimized.py录制的HDF5数据转换为LeRobot v2.0标准格式

LeRobot v2.0 标准目录结构:
    dataset/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    ├── videos/
    │   └── chunk-000/
    │       ├── observation.images.top/
    │       │   ├── episode_000000.mp4
    │       │   ├── episode_000001.mp4
    │       │   └── ...
    │       └── ...
    └── meta/
        ├── info.json          # 数据集schema和配置
        ├── stats.json         # 统计信息（min/max/mean/std）
        ├── episodes.jsonl     # 每个episode的元数据
        └── tasks.jsonl        # 任务描述

使用方法:
    python convert_to_lerobot.py --input ./data --output ./lerobot_dataset --repo-id "dobot/teleop"

依赖安装:
    pip install h5py numpy tqdm pyarrow opencv-python pillow
"""

import argparse
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: Please install pyarrow:")
    print("pip install pyarrow")
    exit(1)


class HDF5ToLeRobotConverter:
    """
    HDF5到LeRobot v2.0格式转换器

    遵循 LeRobot v2.0 标准:
    - 使用 Parquet 格式存储每个 episode 的数据
    - 使用 MP4 格式存储视频（而非单独图片）
    - 使用 episodes.jsonl（而非 episodes.json）
    - 标准目录结构: data/chunk-000/, videos/chunk-000/, meta/
    """

    def __init__(self, input_dir, output_dir, repo_id="dobot/teleop_dataset",
                 fps=30, video_codec='libx264', robot_type='dobot_cr3'):
        """
        初始化转换器

        参数:
            input_dir (str): HDF5文件所在目录
            output_dir (str): LeRobot数据集输出目录
            repo_id (str): Hugging Face数据集ID
            fps (int): 视频帧率
            video_codec (str): 视频编码器 (libx264, h264, etc.)
            robot_type (str): 机器人类型
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.fps = fps
        self.video_codec = video_codec
        self.robot_type = robot_type

        # 创建标准目录结构
        self.data_dir = self.output_dir / "data" / "chunk-000"
        self.videos_dir = self.output_dir / "videos" / "observation.images.top" / "chunk-000"
        self.meta_dir = self.output_dir / "meta"

        # 统计信息
        self.stats = {
            'observation.state': {'min': [], 'max': [], 'mean': [], 'std': []},
            'action': {'min': [], 'max': [], 'mean': [], 'std': []},
        }

        # 相机内参
        self.camera_intrinsics = None

        # Episode信息列表（用于生成episodes.jsonl）
        self.episodes_info = []

        # 全局帧索引计数器（跨所有episode）
        self.global_index = 0

    def scan_episodes(self):
        """扫描所有 episode_*.hdf5 文件"""
        episode_files = sorted(self.input_dir.glob("episode_*.hdf5"))
        print(f"Found {len(episode_files)} episodes in {self.input_dir}")
        return episode_files

    def load_hdf5_episode(self, hdf5_path):
        """
        加载单个HDF5 episode

        返回:
            dict: episode数据
        """
        data = {
            'images': None,
            'depth_images': None,
            'robot_current': None,
            'robot_target': None,
            'gripper_current': None,
            'gripper_target': None,
            'timestamps': None,
            'episode_index': None,
        }

        with h5py.File(hdf5_path, 'r') as f:
            # 读取图像
            if 'observations/images/color' in f:
                data['images'] = f['observations/images/color'][:]

            # 读取深度图（可选）
            if 'observations/images/depth' in f:
                data['depth_images'] = f['observations/images/depth'][:]

            # 读取状态数据
            data['robot_current'] = f['observations/robot_current'][:]
            data['robot_target'] = f['actions/robot_target'][:]
            data['gripper_current'] = f['observations/gripper_current'][:]
            data['gripper_target'] = f['actions/gripper_target'][:]

            # 生成均匀时间戳（强制使用，确保 LeRobot 兼容性）
            n_frames = len(data['robot_current'])

            # 检查原始时间戳是否存在（仅用于诊断）
            if 'timestamp' in f:
                original_timestamps = f['timestamp'][:]
                dt = np.diff(original_timestamps)

                # 检测时间戳问题
                has_issues = False
                if len(original_timestamps) != n_frames:
                    print(f"  ⚠ Timestamp count mismatch: {len(original_timestamps)} vs {n_frames} frames")
                    has_issues = True
                elif np.all(original_timestamps == original_timestamps[0]):
                    print(f"  ⚠ All timestamps identical: {original_timestamps[0]}")
                    has_issues = True
                elif np.any(dt < 0):
                    print(f"  ⚠ Non-monotonic timestamps detected")
                    has_issues = True
                elif np.any(dt > 1.0 / self.fps * 1.5):
                    max_gap = np.max(dt)
                    print(f"  ⚠ Large time gaps detected: max {max_gap:.3f}s (expected ~{1.0/self.fps:.3f}s)")
                    has_issues = True
                elif np.any(dt == 0):
                    print(f"  ⚠ Zero time intervals detected: {np.sum(dt == 0)} frames")
                    has_issues = True

                if has_issues:
                    print(f"  → Using generated uniform timestamps for LeRobot compatibility")
            else:
                print(f"  ℹ No timestamps found in HDF5, generating uniform timestamps")

            # 始终生成均匀时间戳（确保 LeRobot 兼容）
            data['timestamps'] = np.arange(n_frames, dtype=np.float64) / self.fps

            # 读取相机内参
            if self.camera_intrinsics is None and 'camera/intrinsics' in f:
                self.camera_intrinsics = f['camera/intrinsics'][:]

        # 提取episode索引
        filename = hdf5_path.stem  # episode_0
        data['episode_index'] = int(filename.split('_')[1])

        return data

    def create_video_from_images(self, images, output_path, fps=30):
        """
        将图像序列编码为MP4视频

        参数:
            images (np.ndarray): 图像数组 (N, H, W, 3) BGR格式
            output_path (Path): 输出视频路径
            fps (int): 帧率
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_frames, height, width, channels = images.shape

        # 配置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        # 写入每一帧
        for i in range(n_frames):
            frame = images[i]  # 已经是BGR格式
            writer.write(frame)

        writer.release()

        return str(output_path.relative_to(self.output_dir))

    def convert_episode_to_parquet(self, episode_data):
        """
        将单个episode转换为Parquet格式

        参数:
            episode_data (dict): episode数据

        返回:
            tuple: (parquet_path, video_path, episode_info)
        """
        ep_idx = episode_data['episode_index']
        n_frames = len(episode_data['robot_current'])

        # 1. 保存视频
        video_filename = f"episode_{ep_idx:06d}.mp4"
        video_rel_path = self.create_video_from_images(
            episode_data['images'],
            self.videos_dir / video_filename,
            fps=self.fps
        )

        # 2. 构建 observation.state (7D: 6D pose + 1D gripper)
        obs_state = np.concatenate([
            episode_data['robot_current'],     # (N, 6)
            episode_data['gripper_current']    # (N, 1)
        ], axis=1)  # (N, 7)

        # 3. 构建 action (7D: 6D pose + 1D gripper)
        action = np.concatenate([
            episode_data['robot_target'],      # (N, 6)
            episode_data['gripper_target']     # (N, 1)
        ], axis=1)  # (N, 7)

        timestamps = episode_data['timestamps']

        # 4. 构建 next.done (标记episode结束)
        next_done = np.zeros(n_frames, dtype=bool)
        next_done[-1] = True  # 最后一帧标记为done

        # 6. 构建全局索引（跨所有episode连续编号）
        global_indices = list(range(self.global_index, self.global_index + n_frames))

        # 7. 创建 PyArrow Table
        # 注意: 不在Parquet中存储视频字段,LeRobot会根据info.json自动加载视频
        table_data = {
            'index': pa.array(global_indices, type=pa.int64()),  # 全局索引
            'episode_index': pa.array([ep_idx] * n_frames, type=pa.int64()),
            'frame_index': pa.array(list(range(n_frames)), type=pa.int64()),
            'timestamp': pa.array(timestamps, type=pa.float64()),
            'task_index': pa.array([0] * n_frames, type=pa.int64()),  # 所有数据属于任务0

            # Observations (只存储状态,不存储图像)
            'observation.state': pa.array(obs_state.tolist(), type=pa.list_(pa.float32(), 7)),

            # Actions
            'action': pa.array(action.tolist(), type=pa.list_(pa.float32(), 7)),

            # Episode终止标志
            'next.done': pa.array(next_done, type=pa.bool_()),
        }

        table = pa.Table.from_pydict(table_data)

        # 8. 保存为Parquet文件
        parquet_filename = f"episode_{ep_idx:06d}.parquet"
        parquet_path = self.data_dir / parquet_filename
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(table, parquet_path)

        # 9. 更新全局索引计数器
        self.global_index += n_frames

        # 10. 记录episode信息（用于episodes.jsonl）
        episode_info = {
            'episode_index': ep_idx,
            'length': n_frames,
            'tasks': ['robot_teleoperation'],  # 任务描述
        }

        return parquet_path, video_rel_path, episode_info

    def compute_statistics(self, all_episodes_data):
        """计算数据集统计信息"""
        print("\nComputing dataset statistics...")

        all_obs_states = []
        all_actions = []

        for episode_data in all_episodes_data:
            # 构建7D observation state (6D pose + 1D gripper)
            obs_state = np.concatenate([
                episode_data['robot_current'],
                episode_data['gripper_current']
            ], axis=1)
            all_obs_states.append(obs_state)

            # 构建7D action (6D pose + 1D gripper)
            action = np.concatenate([
                episode_data['robot_target'],
                episode_data['gripper_target']
            ], axis=1)
            all_actions.append(action)

        # 合并所有数据
        all_obs_states = np.concatenate(all_obs_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        # 计算统计量
        for key, data in [('observation.state', all_obs_states), ('action', all_actions)]:
            self.stats[key]['min'] = np.min(data, axis=0).tolist()
            self.stats[key]['max'] = np.max(data, axis=0).tolist()
            self.stats[key]['mean'] = np.mean(data, axis=0).tolist()
            self.stats[key]['std'] = np.std(data, axis=0).tolist()

            print(f"{key}:")
            print(f"  min: {self.stats[key]['min']}")
            print(f"  max: {self.stats[key]['max']}")

    def save_metadata(self, total_episodes, total_frames):
        """
        保存 LeRobot v2.0 标准元数据

        包括:
        - info.json: 数据集总体信息（不包含stats）
        - stats.json: 数据集统计信息（独立文件）
        - episodes.jsonl: 每个episode的信息（每行一个JSON）
        - tasks.jsonl: 任务信息
        """
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存 info.json（使用标准占位符格式）
        info = {
            'codebase_version': 'v2.0',
            'robot_type': self.robot_type,
            'total_episodes': total_episodes,
            'total_frames': total_frames,
            'total_tasks': 1,  # HuggingFace 需要：任务总数
            'total_videos': total_episodes,  # HuggingFace 需要：视频总数
            'total_chunks': 1,  # HuggingFace 需要：chunk总数
            'chunks_size': 1000,  # LeRobot v2.0 必需字段：每个chunk的episode数量
            'fps': self.fps,

            # HuggingFace 需要：数据集划分信息
            'splits': {
                'train': f'0:{total_episodes}'
            },

            # 数据路径（使用标准占位符格式）
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4',

            # LeRobot 必需的 features 字段
            'features': {
                'index': {
                    'dtype': 'int64',
                    'shape': [1],
                    'names': None
                },
                'observation.state': {
                    'dtype': 'float32',
                    'shape': [7],
                    'names': ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
                },
                'observation.images.top': {
                    'dtype': 'video',
                    'shape': [3, 480, 640],
                    'names': ['channel', 'height', 'width'],
                    'video_info': {
                        'video.fps': self.fps,
                        'video.codec': self.video_codec,
                        'video.pix_fmt': 'yuv420p',
                        'video.is_depth_map': False,
                        'has_audio': False
                    }
                },
                'action': {
                    'dtype': 'float32',
                    'shape': [7],
                    'names': ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
                },
                'episode_index': {
                    'dtype': 'int64',
                    'shape': [1],  # 标量字段使用 [1] 而不是 []
                    'names': None
                },
                'frame_index': {
                    'dtype': 'int64',
                    'shape': [1],
                    'names': None
                },
                'timestamp': {
                    'dtype': 'float64',
                    'shape': [1],
                    'names': None
                },
                'task_index': {
                    'dtype': 'int64',
                    'shape': [1],
                    'names': None
                },
                'next.done': {
                    'dtype': 'bool',
                    'shape': [1],
                    'names': None
                }
            },
        }

        info_path = self.meta_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Saved info.json to {info_path}")

        # 2. 保存 stats.json（独立文件）
        stats_data = {
            'observation.state': self.stats['observation.state'],
            'action': self.stats['action']
        }

        stats_path = self.meta_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"✓ Saved stats.json to {stats_path}")

        # 3. 保存 episodes.jsonl（每行一个JSON对象）
        episodes_path = self.meta_dir / "episodes.jsonl"
        with open(episodes_path, 'w') as f:
            for ep_info in self.episodes_info:
                f.write(json.dumps(ep_info) + '\n')
        print(f"✓ Saved episodes.jsonl to {episodes_path}")

        # 4. 保存 tasks.jsonl（任务描述）
        tasks = [
            {
                'task_index': 0,
                'task': 'robot_teleoperation',
                'task_description': 'Teleoperation of Dobot CR3 robotic arm with gripper'
            }
        ]
        tasks_path = self.meta_dir / "tasks.jsonl"
        with open(tasks_path, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')
        print(f"✓ Saved tasks.jsonl to {tasks_path}")

    def generate_readme(self, total_episodes, total_frames):
        """生成README文件"""
        readme_content = f"""---
license: mit
task_categories:
- robotics
tags:
- LeRobot
- Dobot
- teleoperation
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# {self.repo_id}

## Dataset Description

This dataset contains robot teleoperation demonstrations recorded from a Dobot CR3 robotic arm.
Converted to LeRobot v2.0 format using the standard any4lerobot converter.

## Dataset Statistics

- **Robot**: {self.robot_type}
- **Total Episodes**: {total_episodes}
- **Total Frames**: {total_frames}
- **FPS**: {self.fps}
- **Format Version**: LeRobot v2.0

## Data Structure

```
dataset/
├── data/
│   └── chunk-000/              # Parquet files (one per episode)
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── observation.images.top/    # Camera view
│       └── chunk-000/              # MP4 videos (one per episode)
│           ├── episode_000000.mp4
│           ├── episode_000001.mp4
│           └── ...
└── meta/                       # Metadata files
    ├── info.json               # Dataset information and schema
    ├── stats.json              # Statistics (min/max/mean/std)
    ├── episodes.jsonl          # Per-episode metadata
    └── tasks.jsonl             # Task descriptions
```

## Observation Space

- **observation.state**: Robot end-effector 6D pose (x, y, z, rx, ry, rz) in mm and degrees
- **observation.images.top**: RGB camera view (VideoFrame format)

## Action Space

- **action**: 7D vector (x, y, z, rx, ry, rz, gripper_opening)
  - First 6 dimensions: target 6D pose
  - Last dimension: gripper opening (0-1000)

## Usage with LeRobot

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset
dataset = LeRobotDataset("{self.repo_id}")

# Access a sample
sample = dataset[0]
print(sample.keys())
# dict_keys(['observation.state', 'observation.images.top', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.done'])

# Get observation
obs_state = sample['observation.state']  # torch.Tensor [6]
obs_image = sample['observation.images.top']  # torch.Tensor [C, H, W]

# Get action
action = sample['action']  # torch.Tensor [7]
```

## Usage with Hugging Face Datasets

```python
import pyarrow.parquet as pq

# Load a single episode
episode_0 = pq.read_table("{self.output_dir}/data/chunk-000/episode_000000.parquet")
print(episode_0.schema)

# Convert to pandas
df = episode_0.to_pandas()
print(df.head())
```

## Citation

```bibtex
@dataset{{dobot_teleop_{total_episodes}_episodes,
  title = {{Dobot Teleoperation Dataset}},
  author = {{Your Name}},
  year = {{2026}},
  howpublished = {{\\url{{{self.repo_id}}}}},
}}
```

## References

- [LeRobot](https://github.com/huggingface/lerobot)
- [any4lerobot](https://github.com/Tavish9/any4lerobot)

---

Generated with HDF5 to LeRobot v2.0 converter (any4lerobot compatible)
"""
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"✓ Generated README at {readme_path}")

    def convert(self):
        """执行完整的转换流程"""
        print(f"=== HDF5 to LeRobot v2.0 Converter ===")
        print(f"Based on any4lerobot format specification")
        print(f"Input:  {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Repo:   {self.repo_id}")
        print()

        # 1. 扫描episode文件
        episode_files = self.scan_episodes()
        if len(episode_files) == 0:
            print("Error: No episode files found!")
            return

        # 2. 加载所有episodes
        print("\nLoading episodes...")
        all_episodes_data = []
        for hdf5_path in tqdm(episode_files, desc="Loading HDF5"):
            episode_data = self.load_hdf5_episode(hdf5_path)
            all_episodes_data.append(episode_data)

        # 3. 计算统计信息
        self.compute_statistics(all_episodes_data)

        # 4. 转换每个episode为Parquet + MP4
        print("\nConverting episodes to LeRobot format...")
        total_frames = 0

        for episode_data in tqdm(all_episodes_data, desc="Converting to Parquet"):
            _, _, episode_info = self.convert_episode_to_parquet(episode_data)
            self.episodes_info.append(episode_info)
            total_frames += episode_info['length']

        # 5. 保存元数据
        print("\nSaving metadata...")
        self.save_metadata(len(all_episodes_data), total_frames)

        # 6. 生成README
        self.generate_readme(len(all_episodes_data), total_frames)

        print("\n=== Conversion Complete ===")
        print(f"✓ Total episodes: {len(all_episodes_data)}")
        print(f"✓ Total frames: {total_frames}")
        print(f"✓ Output directory: {self.output_dir}")
        print()
        print("Dataset structure:")
        print(f"  - Parquet files: {self.data_dir}")
        print(f"  - Video files: {self.videos_dir}")
        print(f"  - Metadata: {self.meta_dir}")
        print(f"    ├── info.json    (dataset schema)")
        print(f"    ├── stats.json   (statistics)")
        print(f"    ├── episodes.jsonl")
        print(f"    └── tasks.jsonl")
        print()
        print("To use with LeRobot:")
        print(f"  from lerobot.common.datasets.lerobot_dataset import LeRobotDataset")
        print(f"  dataset = LeRobotDataset('{self.repo_id}')")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 robot dataset to LeRobot v2.0 format (any4lerobot compatible)"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./data",
        help="Input directory containing episode_*.hdf5 files (default: ./data)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./lerobot_dataset",
        help="Output directory for LeRobot dataset (default: ./lerobot_dataset)"
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="dobot/teleop_dataset",
        help="Dataset repository ID (default: dobot/teleop_dataset)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)"
    )

    parser.add_argument(
        "--video-codec",
        type=str,
        default="libx264",
        help="Video codec (default: libx264)"
    )

    parser.add_argument(
        "--robot-type",
        type=str,
        default="dobot_cr3",
        help="Robot type identifier (default: dobot_cr3)"
    )

    args = parser.parse_args()

    # 创建转换器并执行
    converter = HDF5ToLeRobotConverter(
        input_dir=args.input,
        output_dir=args.output,
        repo_id=args.repo_id,
        fps=args.fps,
        video_codec=args.video_codec,
        robot_type=args.robot_type
    )

    converter.convert()


if __name__ == "__main__":
    main()
