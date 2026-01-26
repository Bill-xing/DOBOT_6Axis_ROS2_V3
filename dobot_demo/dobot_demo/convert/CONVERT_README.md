# HDF5 到 LeRobot v2.0 转换器使用说明

## 功能概述

`convert_to_lerobot.py` 是一个将机器人录制的HDF5数据转换为 LeRobot v2.0 标准格式的工具。

### 主要功能

1. **读取HDF5数据**：自动扫描并加载所有 `episode_*.hdf5` 文件
2. **视频编码**：将图像序列转换为标准MP4视频（H.264编码）
3. **Parquet格式转换**：每个episode保存为一个Parquet文件
4. **元数据生成**：自动生成所有必需的元数据文件
5. **统计信息计算**：生成数据集的统计信息（min/max/mean/std）

### 输出格式

符合 LeRobot v2.0 标准和 [any4lerobot](https://github.com/Tavish9/any4lerobot) 规范：

```
lerobot_dataset/
├── data/
│   └── chunk-000/              # Parquet 数据文件
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/              # MP4 视频文件
│       └── observation.images.top/
│           ├── episode_000000.mp4
│           ├── episode_000001.mp4
│           └── ...
└── meta/                       # 元数据文件
    ├── info.json               # 数据集schema和配置
    ├── stats.json              # 统计信息（独立文件）
    ├── episodes.jsonl          # 每个episode的元数据
    └── tasks.jsonl             # 任务描述
```

---

## 安装依赖

### 必需依赖

```bash
pip install h5py numpy tqdm pyarrow opencv-python pillow
```

### 依赖说明

| 库 | 用途 |
|----|------|
| `h5py` | 读取HDF5文件 |
| `numpy` | 数值计算和数据处理 |
| `tqdm` | 进度条显示 |
| `pyarrow` | Parquet格式读写 |
| `opencv-python` | 视频编码 |
| `pillow` | 图像处理（可选） |

---

## 使用方法

### 基本用法

```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3/dobot_demo/dobot_demo/convert

# 转换数据集
python convert_to_lerobot.py --input ../data --output ./lerobot_dataset
```

### 完整参数示例

```bash
python convert_to_lerobot.py \
    --input ../data \
    --output ./lerobot_dataset \
    --repo-id "dobot/teleop_cr3" \
    --fps 30 \
    --video-codec libx264 \
    --robot-type dobot_cr3
```

---

## 命令行参数详解

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input` | `-i` | string | `./data` | HDF5文件所在目录 |
| `--output` | `-o` | string | `./lerobot_dataset` | 输出目录 |
| `--repo-id` | - | string | `dobot/teleop_dataset` | 数据集ID（用于Hugging Face） |
| `--fps` | - | int | `30` | 视频帧率 |
| `--video-codec` | - | string | `libx264` | 视频编码器 |
| `--robot-type` | - | string | `dobot_cr3` | 机器人类型标识 |

---

## 输入数据要求

### HDF5 文件格式

工具要求输入目录包含 `episode_*.hdf5` 文件，每个文件应包含以下数据集：

```
episode_X.hdf5
├── observations/
│   ├── images/
│   │   └── color              # (N, H, W, 3) uint8 - RGB图像
│   ├── robot_current          # (N, 6) float32 - 当前位姿
│   └── gripper_current        # (N, 1) float32 - 当前夹爪开度
├── actions/
│   ├── robot_target           # (N, 6) float32 - 目标位姿
│   └── gripper_target         # (N, 1) float32 - 目标夹爪开度
├── timestamp                  # (N,) float64 - 时间戳
└── camera/
    └── intrinsics             # (3, 3) float32 - 相机内参（可选）
```

### 数据格式说明

- **robot pose**: `[x, y, z, rx, ry, rz]` (6D位姿)
  - `x, y, z`: 位置坐标（mm）
  - `rx, ry, rz`: 欧拉角旋转（度）

- **gripper**: `[opening]` (夹爪开度)
  - 范围: 0-1000

- **action**: `[x, y, z, rx, ry, rz, gripper]` (7D动作)
  - 前6维：目标位姿
  - 第7维：夹爪开度

---

## 输出文件详解

### 1. info.json - 数据集配置

包含数据集的schema定义和配置信息：

```json
{
  "codebase_version": "v2.0",
  "robot_type": "dobot_cr3",
  "total_episodes": 10,
  "total_frames": 3000,
  "chunks_size": 1000,
  "fps": 30,

  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",

  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [6],
      "names": ["x", "y", "z", "rx", "ry", "rz"]
    },
    "observation.images.top": {
      "dtype": "video",
      "shape": [3, 480, 640],
      "names": ["channel", "height", "width"]
    },
    "action": {
      "dtype": "float32",
      "shape": [7],
      "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    }
  }
}
```

**关键字段说明**：
- `data_path`, `video_path`: 使用占位符格式（`{episode_chunk:03d}`, `{episode_index:06d}`, `{video_key}`）
- `features`: 完整的数据schema定义
- `chunks_size`: 每个chunk包含的episode数量（默认1000）

### 2. stats.json - 统计信息（新增）

**独立文件**，包含数据集的统计信息：

```json
{
  "observation.state": {
    "min": [-400.59, -509.78, 222.05, -180.0, -0.11, 89.89],
    "max": [397.70, -413.00, 391.51, 180.0, 0.11, 90.24],
    "mean": [58.19, -461.82, 306.90, -142.85, -0.00, 90.00],
    "std": [220.82, 16.13, 52.12, 104.75, 0.01, 0.01]
  },
  "action": {
    "min": [-401.96, -509.90, 222.08, -180.0, -0.00, 90.0, 320.0],
    "max": [397.70, -412.93, 391.59, -180.0, -0.00, 90.0, 1000.0],
    "mean": [58.88, -462.01, 306.89, -180.0, -0.00, 90.0, 700.74],
    "std": [221.55, 16.24, 52.11, 0.00, 0.00, 0.00, 303.01]
  }
}
```

**用途**：
- 数据归一化
- 分析数据分布
- 检查数据质量

### 3. episodes.jsonl - Episode元数据

每行一个JSON对象，记录每个episode的信息：

```jsonl
{"episode_index": 0, "length": 300, "tasks": ["robot_teleoperation"]}
{"episode_index": 1, "length": 450, "tasks": ["robot_teleoperation"]}
{"episode_index": 2, "length": 250, "tasks": ["robot_teleoperation"]}
```

### 4. tasks.jsonl - 任务描述

任务定义和描述：

```jsonl
{"task_index": 0, "task": "robot_teleoperation", "task_description": "Teleoperation of Dobot CR3 robotic arm with gripper"}
```

### 5. Parquet 文件 - Episode数据

每个episode一个Parquet文件，包含所有帧的数据：

**Schema**:
```
episode_index: int64
frame_index: int64
timestamp: float64
observation.state: list<float32>[6]
observation.images.top: struct<path: string, timestamp: float64>
action: list<float32>[7]
next.done: bool
```

**示例数据**:
```python
import pyarrow.parquet as pq

table = pq.read_table('data/chunk-000/episode_000000.parquet')
df = table.to_pandas()

# 输出:
#    episode_index  frame_index  timestamp  observation.state  action  next.done
# 0              0            0  12345.678  [x,y,z,rx,ry,rz]  [...]   False
# 1              0            1  12345.711  [x,y,z,rx,ry,rz]  [...]   False
# ...
# 299            0          299  12355.678  [x,y,z,rx,ry,rz]  [...]   True
```

### 6. MP4 视频文件

标准H.264编码的MP4视频：
- 编码: libx264
- 像素格式: yuv420p
- 帧率: 30 fps（可配置）
- 命名: `episode_{episode_index:06d}.mp4`

---

## 转换流程

```
1️⃣ 扫描 episode_*.hdf5 文件
    └─ 按文件名排序

2️⃣ 加载所有episodes
    └─ 读取图像、状态、动作数据
    └─ 读取时间戳和相机内参

3️⃣ 计算统计信息
    └─ observation.state: min/max/mean/std
    └─ action: min/max/mean/std

4️⃣ 转换每个episode
    └─ 编码MP4视频
    └─ 生成Parquet文件
    └─ 记录episode信息

5️⃣ 保存元数据
    ├─ info.json (schema)
    ├─ stats.json (statistics)
    ├─ episodes.jsonl
    └─ tasks.jsonl

6️⃣ 生成 README.md
```

---

## 使用转换后的数据集

### 方法 1: 使用 LeRobot

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("dobot/teleop_cr3")

# 访问样本
sample = dataset[0]
print(sample.keys())
# dict_keys(['observation.state', 'observation.images.top', 'action',
#            'episode_index', 'frame_index', 'timestamp', 'next.done'])

# 获取数据
obs_state = sample['observation.state']      # torch.Tensor [6]
obs_image = sample['observation.images.top'] # torch.Tensor [C, H, W]
action = sample['action']                     # torch.Tensor [7]
```

### 方法 2: 直接读取 Parquet

```python
import pyarrow.parquet as pq
import pandas as pd

# 读取单个episode
episode_0 = pq.read_table('lerobot_dataset/data/chunk-000/episode_000000.parquet')

# 转换为pandas
df = episode_0.to_pandas()

# 查看数据
print(df.head())
print(df.columns)

# 访问特定列
robot_states = df['observation.state'].tolist()
actions = df['action'].tolist()
```

### 方法 3: 使用 Hugging Face Datasets

```python
from datasets import load_dataset

# 从本地加载
dataset = load_dataset('parquet', data_dir='lerobot_dataset/data/chunk-000')

# 或推送到Hugging Face Hub后加载
# dataset = load_dataset('dobot/teleop_cr3')

# 访问数据
print(dataset['train'][0])
```

---

## 示例：完整转换流程

### 步骤 1: 准备数据

```bash
# 确保有HDF5数据文件
ls ../data/
# episode_0.hdf5
# episode_1.hdf5
# episode_2.hdf5
```

### 步骤 2: 执行转换

```bash
python convert_to_lerobot.py \
    --input ../data \
    --output ./my_dataset \
    --repo-id "my_org/dobot_demo"
```

### 步骤 3: 查看输出

```
=== HDF5 to LeRobot v2.0 Converter ===
Based on any4lerobot format specification
Input:  ../data
Output: ./my_dataset
Repo:   my_org/dobot_demo

Found 3 episodes in ../data

Loading episodes...
Loading HDF5: 100%|████████████████| 3/3 [00:01<00:00, 2.10it/s]

Computing dataset statistics...
observation.state:
  min: [-400.59, -509.78, 222.05, -180.0, -0.11, 89.89]
  max: [397.70, -413.00, 391.51, 180.0, 0.11, 90.24]
action:
  min: [-401.96, -509.90, 222.08, -180.0, -0.00, 90.0, 320.0]
  max: [397.70, -412.93, 391.59, -180.0, -0.00, 90.0, 1000.0]

Converting episodes to LeRobot format...
Converting to Parquet: 100%|████████| 3/3 [00:05<00:00, 1.67s/it]

Saving metadata...
✓ Saved info.json to my_dataset/meta/info.json
✓ Saved stats.json to my_dataset/meta/stats.json
✓ Saved episodes.jsonl to my_dataset/meta/episodes.jsonl
✓ Saved tasks.jsonl to my_dataset/meta/tasks.jsonl

✓ Generated README at my_dataset/README.md

=== Conversion Complete ===
✓ Total episodes: 3
✓ Total frames: 900
✓ Output directory: ./my_dataset

Dataset structure:
  - Parquet files: my_dataset/data/chunk-000
  - Video files: my_dataset/videos/chunk-000
  - Metadata: my_dataset/meta
    ├── info.json    (dataset schema)
    ├── stats.json   (statistics)
    ├── episodes.jsonl
    └── tasks.jsonl

To use with LeRobot:
  from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
  dataset = LeRobotDataset('my_org/dobot_demo')
```

### 步骤 4: 验证数据

```python
import pyarrow.parquet as pq
import json

# 1. 检查info.json
with open('my_dataset/meta/info.json') as f:
    info = json.load(f)
    print(f"Total episodes: {info['total_episodes']}")
    print(f"Total frames: {info['total_frames']}")

# 2. 检查stats.json
with open('my_dataset/meta/stats.json') as f:
    stats = json.load(f)
    print(f"Observation state range:")
    print(f"  Min: {stats['observation.state']['min']}")
    print(f"  Max: {stats['observation.state']['max']}")

# 3. 读取一个episode
table = pq.read_table('my_dataset/data/chunk-000/episode_000000.parquet')
print(f"Episode 0 shape: {table.num_rows} frames")
print(f"Columns: {table.column_names}")
```

---

## 故障排查

### 问题 1: 找不到 episode 文件

**错误信息**:
```
Found 0 episodes in ../data
Error: No episode files found!
```

**解决方法**:
```bash
# 检查文件命名是否正确
ls ../data/
# 应该看到: episode_0.hdf5, episode_1.hdf5, ...

# 检查文件路径是否正确
python convert_to_lerobot.py --input /absolute/path/to/data
```

### 问题 2: HDF5 数据集缺失

**错误信息**:
```
KeyError: 'observations/images/color'
```

**解决方法**:
- 确认HDF5文件是用 `recorder_optimized.py` 录制的
- 检查数据集结构：
```python
import h5py
with h5py.File('episode_0.hdf5', 'r') as f:
    print(list(f.keys()))
    print(list(f['observations'].keys()))
```

### 问题 3: 视频编码失败

**错误信息**:
```
cv2.error: OpenCV(4.x.x) ...
```

**解决方法**:
```bash
# 重新安装 opencv-python
pip uninstall opencv-python opencv-python-headless
pip install opencv-python

# 或者尝试使用 headless 版本
pip install opencv-python-headless
```

### 问题 4: 内存不足

**错误信息**:
```
MemoryError: Unable to allocate array...
```

**解决方法**:
- 分批处理episode（修改代码，每次处理部分文件）
- 增加系统交换空间
- 使用更高内存的机器

### 问题 5: Parquet 写入失败

**错误信息**:
```
pyarrow.lib.ArrowInvalid: ...
```

**解决方法**:
```bash
# 更新 pyarrow
pip install --upgrade pyarrow

# 检查数据类型是否正确
# 确保 numpy array 是 contiguous 的
```

---

## 性能指标

### 典型转换时间

| Episode数量 | 每个Episode帧数 | 图像分辨率 | 总时间 | 输出大小 |
|------------|----------------|-----------|--------|----------|
| 10 | 300 | 640x480 | ~2分钟 | ~500 MB |
| 50 | 300 | 640x480 | ~8分钟 | ~2.5 GB |
| 100 | 300 | 640x480 | ~15分钟 | ~5 GB |

*测试环境：Intel i7-9700K, 16GB RAM, SSD*

### 优化建议

1. **使用SSD存储**：显著提升I/O性能
2. **增加内存**：减少磁盘交换
3. **并行处理**：修改代码支持多进程（高级）

---

## 与 any4lerobot 的兼容性

本转换器完全兼容 [any4lerobot](https://github.com/Tavish9/any4lerobot) 规范：

✅ 支持特性：
- Parquet + MP4 格式
- 标准路径占位符格式
- VideoFrame 结构
- episodes.jsonl 格式
- tasks.jsonl 格式
- 独立的 stats.json 文件
- 完整的 features schema

---

## 相关文件

- **数据录制器**: `recorder_optimized.py` - 用于录制HDF5数据
- **数据转换器**: `convert_to_lerobot.py` - 当前工具
- **可视化工具**: `show_vla.py` - 生成可视化视频

---

## 常见问题 (FAQ)

### Q1: 如何修改视频分辨率？

修改代码中的 `features` 定义：
```python
'observation.images.top': {
    'dtype': 'video',
    'shape': [3, 720, 1280],  # 修改为目标分辨率
    ...
}
```

### Q2: 如何添加多个相机视图？

在转换函数中添加额外的视频处理：
```python
# 保存第二个相机视图
video_filename_wrist = f"episode_{ep_idx:06d}.mp4"
self.create_video_from_images(
    episode_data['wrist_images'],
    self.videos_dir / "observation.images.wrist" / video_filename_wrist,
    fps=self.fps
)
```

### Q3: 如何导出到 Hugging Face Hub？

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./lerobot_dataset",
    repo_id="your-username/your-dataset-name",
    repo_type="dataset"
)
```

### Q4: 转换后的数据可以直接训练模型吗？

是的！转换后的数据符合 LeRobot 标准，可以直接用于：
- LeRobot 官方训练脚本
- 自定义训练pipeline
- 其他支持 LeRobot 格式的工具

---

## 更新日志

### v2.0 (2024-01-18)
- ✨ 生成独立的 stats.json 文件
- ✨ 使用标准路径占位符格式
- ✨ 修正 features schema（标量字段使用 [1]）
- ✅ 完全兼容 any4lerobot 规范
- 📝 完善文档和注释

### v1.0 (2024)
- ✨ 初始版本
- ✅ 支持 HDF5 到 LeRobot v2.0 转换
- ✅ Parquet + MP4 格式
- ✅ 自动生成元数据

---

## 开发者信息

- **开发团队**: DOBOT Team
- **参考规范**: [any4lerobot](https://github.com/Tavish9/any4lerobot)
- **LeRobot**: [huggingface/lerobot](https://github.com/huggingface/lerobot)
- **更新日期**: 2024-01-18
