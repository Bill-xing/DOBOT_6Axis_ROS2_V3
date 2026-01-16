# HDF5 到 LeRobot v2.0 数据集转换工具

## 概述

这套工具基于 [any4lerobot](https://github.com/Tavish9/any4lerobot) 标准，将 `data_collector4.py` 和 `recorder_optimized.py` 录制的 HDF5 格式数据转换为 LeRobot v2.0 标准格式，用于机器人学习和模仿学习任务。

## 主要特性

✅ **完全兼容 LeRobot v2.0 标准**
- Parquet 格式存储 episode 数据
- MP4 视频编码（而非单独图片）
- episodes.jsonl 元数据（每行一个JSON）
- 标准目录结构：`data/chunk-000/`, `videos/chunk-000/`, `meta/`

✅ **基于 any4lerobot 格式规范**
- 遵循 [any4lerobot](https://github.com/Tavish9/any4lerobot) 的转换标准
- 可直接用于 LeRobot 训练框架
- 支持上传到 Hugging Face Hub

## 文件说明

- **convert_to_lerobot.py** - 主转换脚本（any4lerobot 兼容）
- **verify_lerobot_dataset.py** - 数据集验证脚本
- **USAGE.md** - 本文档

## 安装依赖

```bash
# 安装必要的Python包
pip install h5py numpy tqdm pyarrow opencv-python pillow
```

## LeRobot v2.0 数据集结构

转换后的数据集遵循以下标准结构（any4lerobot 格式）：

```
lerobot_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.top/
│           ├── episode_000000.mp4
│           ├── episode_000001.mp4
│           └── ...
├── meta/
│   ├── info.json           # 数据集总体信息
│   ├── episodes.jsonl      # 每个episode的信息（JSONL格式）
│   └── tasks.jsonl         # 任务描述
└── README.md               # 数据集说明
```

## 使用方法

### 1. 基本转换

```bash
python convert_to_lerobot.py --input ./data --output ./lerobot_dataset
```

### 2. 完整参数示例

```bash
python convert_to_lerobot.py \
  --input ./data \
  --output ./lerobot_dataset \
  --repo-id "your_username/dobot_teleop" \
  --fps 30 \
  --video-codec libx264 \
  --robot-type dobot_cr3
```

**参数说明:**

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `./data` | HDF5 文件所在目录 |
| `--output` | `-o` | `./lerobot_dataset` | LeRobot 数据集输出目录 |
| `--repo-id` | - | `dobot/teleop_dataset` | Hugging Face 数据集 ID |
| `--fps` | - | `30` | 视频帧率 |
| `--video-codec` | - | `libx264` | 视频编码器 |
| `--robot-type` | - | `dobot_cr3` | 机器人类型标识 |

## 在 Python 中使用数据集

### 使用 LeRobot 框架

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("your_username/dobot_teleop")

# 访问单个样本
sample = dataset[0]
print(sample.keys())
# dict_keys(['observation.state', 'observation.images.top', 'action',
#            'episode_index', 'frame_index', 'timestamp', 'next.done'])

# 获取数据
obs_state = sample['observation.state']  # torch.Tensor [6]
obs_image = sample['observation.images.top']  # torch.Tensor [C, H, W]
action = sample['action']  # torch.Tensor [7]
```

### 使用 PyArrow 读取 Parquet

```python
import pyarrow.parquet as pq

# 读取单个 episode
table = pq.read_table("./lerobot_dataset/data/chunk-000/episode_000000.parquet")
print(table.schema)

# 转换为 pandas
df = table.to_pandas()
print(df.head())
```

### 训练策略

```python
from lerobot.scripts.train import train

train(
    dataset_repo_id="your_username/dobot_teleop",
    policy_name="act",  # 或 "diffusion"
    env_name="dobot_real",
    training_steps=100000,
    batch_size=32,
    output_dir="./outputs/act_dobot",
)
```

## 数据格式说明

### Parquet 字段

| 字段名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `episode_index` | int64 | - | Episode 编号 |
| `frame_index` | int64 | - | 帧编号 |
| `timestamp` | float64 | - | ROS 时间戳 |
| `observation.state` | list[float32] | (6,) | 机械臂当前 6D 位姿 |
| `observation.images.top` | struct | - | VideoFrame（path + timestamp） |
| `action` | list[float32] | (7,) | 目标位姿 + 夹爪 |
| `next.done` | bool | - | Episode 结束标志 |

### episodes.jsonl 示例

```jsonl
{"episode_index": 0, "length": 416, "tasks": ["robot_teleoperation"]}
{"episode_index": 1, "length": 470, "tasks": ["robot_teleoperation"]}
```

## 上传到 Hugging Face Hub

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your_username/dobot_teleop"

# 创建数据集仓库
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# 上传整个数据集目录
api.upload_folder(
    folder_path="./lerobot_dataset",
    repo_id=repo_id,
    repo_type="dataset",
)
```

## 常见问题

### Q1: 为什么使用 MP4 而不是单独的图片？

**A:** MP4 格式的优势：
- 文件数量大幅减少
- 存储空间更小（视频压缩更高效）
- 读取速度更快
- 符合 LeRobot v2.0 标准

### Q2: 转换很慢怎么办？

**A:** 视频编码是主要瓶颈，优化建议：
- 使用更快的编码器（如 GPU 加速的 h264_nvenc）
- 降低视频分辨率
- 使用 SSD 硬盘

### Q3: 如何验证数据集？

**A:** 使用验证脚本：

```bash
python verify_lerobot_dataset.py --dataset ./lerobot_dataset/data
```

或手动检查：

```python
import pyarrow.parquet as pq
import cv2

# 检查 parquet
table = pq.read_table("./lerobot_dataset/data/chunk-000/episode_000000.parquet")
print(table.schema)

# 检查视频
cap = cv2.VideoCapture("./lerobot_dataset/videos/chunk-000/observation.images.top/episode_000000.mp4")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
```

## 参考资料

- [GitHub - Tavish9/any4lerobot](https://github.com/Tavish9/any4lerobot/)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot Dataset Format](https://docs.phospho.ai/learn/lerobot-dataset)
- [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
