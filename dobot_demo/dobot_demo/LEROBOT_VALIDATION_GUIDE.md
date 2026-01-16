# LeRobot 官方数据集验证指南

LeRobot 提供了多个官方工具来验证和检查数据集的合理性。

## 🛠️ 官方验证工具

### 1. **visualize_dataset_html** - 可视化验证（推荐）

LeRobot 提供了在线和本地的数据集可视化工具。

#### 在线可视化（最简单）

访问官方可视化工具：
- **网址**: https://huggingface.co/spaces/lerobot/visualize_dataset
- **功能**:
  - 查看所有 episode
  - 播放视频
  - 查看机器人状态曲线
  - 检查数据完整性

#### 本地可视化（命令行）

```bash
# 可视化已上传到 Hub 的数据集
python -m lerobot.scripts.visualize_dataset_html \
  --repo-id your_username/your_dataset

# 可视化本地数据集
python -m lerobot.scripts.visualize_dataset_html \
  --repo-id your_username/your_dataset \
  --local-files-only 1 \
  --root ./lerobot_dataset
```

**输出**: 生成 HTML 可视化报告，可在浏览器中查看

---

### 2. **lerobot-dataset-viz** - 交互式可视化

使用 rerun.io 进行实时可视化：

```bash
# 安装 LeRobot（如果还没安装）
pip install lerobot

# 可视化数据集
lerobot-dataset-viz \
  --repo-id your_username/your_dataset \
  --episode-index 0
```

**功能**:
- 3D 可视化机器人轨迹
- 多相机视角同步显示
- 时间轴回放
- 状态和动作曲线

---

### 3. **validate_dataset** - 数据完整性验证（核心工具）

**最重要的验证工具**，检查数据集结构和数据完整性：

```bash
python -m lerobot.scripts.validate_dataset \
  --root ./lerobot_dataset
```

**检查项目**:
- ✅ 目录结构是否正确
- ✅ Parquet 文件格式
- ✅ 视频文件完整性
- ✅ episodes.jsonl 格式
- ✅ info.json 元数据
- ✅ 数据维度一致性
- ✅ 时间戳单调性
- ✅ 统计信息正确性

---

### 4. **LeRobotDataset 类加载验证**

通过加载数据集来验证其可用性：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集（会自动验证）
try:
    dataset = LeRobotDataset(
        repo_id="your_username/your_dataset",
        root="./lerobot_dataset"  # 本地路径
    )

    print(f"✅ Dataset loaded successfully!")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Total frames: {len(dataset)}")
    print(f"Features: {dataset.features}")

    # 测试访问第一帧
    sample = dataset[0]
    print(f"✅ Data access successful!")
    print(f"Keys: {sample.keys()}")

except Exception as e:
    print(f"❌ Dataset validation failed: {e}")
```

---

### 5. **统计信息验证**

验证数据集的统计信息（均值、方差等）：

```python
from lerobot.common.datasets.v2_1.convert_stats import check_aggregate_stats
from pathlib import Path

# 检查统计信息
dataset_path = Path("./lerobot_dataset")
stats_file = dataset_path / "meta" / "stats.json"

if stats_file.exists():
    import json
    with open(stats_file) as f:
        stats = json.load(f)

    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
else:
    print("⚠️ Warning: No stats.json found")
```

---

## 📊 完整验证流程

### 步骤 1: 结构验证

```bash
# 检查目录结构
tree lerobot_dataset -L 3

# 预期结构:
# lerobot_dataset/
# ├── data/
# │   └── chunk-000/
# │       └── episode_*.parquet
# ├── videos/
# │   └── chunk-000/
# │       └── observation.images.*/
# │           └── episode_*.mp4
# └── meta/
#     ├── info.json
#     ├── episodes.jsonl
#     └── tasks.jsonl
```

### 步骤 2: 运行官方验证

```bash
# 使用官方验证脚本
python -m lerobot.scripts.validate_dataset --root ./lerobot_dataset
```

### 步骤 3: 可视化检查

```bash
# 在线可视化（推荐）
# 访问 https://huggingface.co/spaces/lerobot/visualize_dataset

# 或本地可视化
python -m lerobot.scripts.visualize_dataset_html \
  --repo-id local/dataset \
  --local-files-only 1 \
  --root ./lerobot_dataset
```

### 步骤 4: 加载测试

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 测试加载
dataset = LeRobotDataset(repo_id="local/dataset", root="./lerobot_dataset")

# 检查数据
print(f"Episodes: {dataset.num_episodes}")
print(f"Frames: {len(dataset)}")

# 随机采样验证
import random
indices = random.sample(range(len(dataset)), min(10, len(dataset)))
for idx in indices:
    sample = dataset[idx]
    assert 'observation.state' in sample
    assert 'action' in sample
    assert 'observation.images.top' in sample
```

---

## 🔍 常见验证检查项

### 元数据验证

```python
import json
from pathlib import Path

dataset_path = Path("./lerobot_dataset")

# 1. 检查 info.json
info_file = dataset_path / "meta" / "info.json"
with open(info_file) as f:
    info = json.load(f)

required_fields = [
    'codebase_version',
    'robot_type',
    'total_episodes',
    'total_frames',
    'fps'
]

for field in required_fields:
    assert field in info, f"Missing field: {field}"
    print(f"✓ {field}: {info[field]}")

# 2. 检查 episodes.jsonl
episodes_file = dataset_path / "meta" / "episodes.jsonl"
episodes = []
with open(episodes_file) as f:
    for line in f:
        episodes.append(json.loads(line))

print(f"✓ Total episodes in metadata: {len(episodes)}")

# 验证 episode 索引连续性
for i, ep in enumerate(episodes):
    assert ep['episode_index'] == i, f"Episode index mismatch at {i}"

print("✓ Episode indices are sequential")
```

### Parquet 文件验证

```python
import pyarrow.parquet as pq
from pathlib import Path

dataset_path = Path("./lerobot_dataset")
data_dir = dataset_path / "data" / "chunk-000"

# 检查所有 parquet 文件
parquet_files = sorted(data_dir.glob("episode_*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

for pf in parquet_files:
    table = pq.read_table(pf)

    # 验证必需字段
    required_columns = [
        'episode_index',
        'frame_index',
        'timestamp',
        'observation.state',
        'action',
        'next.done'
    ]

    for col in required_columns:
        assert col in table.column_names, f"Missing column {col} in {pf.name}"

    # 验证数据类型
    assert table.column('episode_index').type == pq.int64()
    assert table.column('frame_index').type == pq.int64()
    assert table.column('timestamp').type == pq.float64()

    print(f"✓ {pf.name}: {table.num_rows} rows, schema valid")
```

### 视频文件验证

```python
import cv2
from pathlib import Path

dataset_path = Path("./lerobot_dataset")
videos_dir = dataset_path / "videos" / "chunk-000" / "observation.images.top"

video_files = sorted(videos_dir.glob("episode_*.mp4"))
print(f"Found {len(video_files)} video files")

for vf in video_files:
    cap = cv2.VideoCapture(str(vf))

    if not cap.isOpened():
        print(f"✗ Failed to open {vf.name}")
        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    print(f"✓ {vf.name}: {frame_count} frames, {fps} fps, {width}x{height}")
```

---

## 🚨 常见问题排查

### 问题 1: "Missing required field in info.json"

**解决方案**: 确保 `info.json` 包含所有必需字段：
```json
{
  "codebase_version": "v2.0",
  "robot_type": "dobot_cr3",
  "total_episodes": 10,
  "total_frames": 5000,
  "fps": 30,
  "repo_id": "username/dataset_name"
}
```

### 问题 2: "Episode index mismatch"

**解决方案**: episodes.jsonl 中的 episode_index 必须从 0 开始连续递增。

### 问题 3: "Video frame count mismatch"

**解决方案**: 确保视频帧数与 Parquet 文件中的行数一致。

---

## 📚 参考资料

- [LeRobot 官方文档](https://github.com/huggingface/lerobot)
- [数据集工具使用指南](https://huggingface.co/docs/lerobot/en/using_dataset_tools)
- [LeRobot 数据集可视化工具](https://huggingface.co/spaces/lerobot/visualize_dataset)
- [LeRobotDataset v3.0 文档](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [数据集验证问题讨论](https://github.com/huggingface/lerobot/issues/1546)

---

## 📝 总结

LeRobot 提供了完善的数据集验证工具：

1. ✅ **visualize_dataset_html** - 可视化检查（推荐首选）
2. ✅ **validate_dataset** - 完整性验证（核心工具）
3. ✅ **lerobot-dataset-viz** - 交互式 3D 可视化
4. ✅ **LeRobotDataset 类** - 加载验证
5. ✅ **自定义验证脚本** - 深度检查

建议使用 **visualize_dataset_html** + **validate_dataset** 组合进行全面验证！
