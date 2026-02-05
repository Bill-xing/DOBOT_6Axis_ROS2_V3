# Dobot机械臂数据采集与播放完整流程

## 概览

本文档总结从机械臂数据录制到播放评估的完整工作流程，提供所有需要的命令行操作。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        完整工作流程                                   │
└─────────────────────────────────────────────────────────────────────┘

步骤1: 数据录制
    recorder_optimized.py
         ↓
    data/episode_X.hdf5

步骤2: 数据质量检查
    check_dataset.py
         ↓
    检查报告（丢帧、突变、速度）

步骤3: 数据清洗（可选）
    clean_dataset.py
         ↓
    cleaned/episode_X_cleaned_merged.hdf5

步骤4: 数据播放（带安全检查）
    dataset_player.py
         ↓
    机械臂重放轨迹

步骤5: 播放评估录制
    playback_evaluator.py (后台运行)
    + dataset_player.py
         ↓
    playback_eval/evaluation_X.hdf5

步骤6: 播放质量分析
    analyze_playback.py
         ↓
    统计报告 + 可视化图表
```

---

## 涉及的核心文件

| 文件名 | 功能 | 用途 |
|--------|------|------|
| `recorder_optimized.py` | 优化版数据录制器 | 录制机械臂轨迹、图像、夹爪状态 |
| `check_dataset.py` | 数据集质量检查 | 检查丢帧、突变、速度异常 |
| `clean_dataset.py` | 数据集清洗工具 | 过滤丢帧片段，切分高质量片段 |
| `dataset_player.py` | 数据集播放器 | 重放轨迹（带安全检查） |
| `playback_evaluator.py` | 播放评估录制器 | 录制目标vs实际状态 |
| `analyze_playback.py` | 播放质量分析 | 计算跟踪误差，生成报告 |

---

## 完整操作流程

### 阶段一：数据录制

#### 1.1 启动录制系统

**终端1：启动ROS2机械臂驱动**
```bash
cd /home/hit/dobot_ws_xing
source install/setup.bash

# 启动机械臂控制器（启动dobot_bringup和feedback两个节点）
ros2 launch dobot_bringup_v3 dobot_bringup_ros2.launch.py
```

**终端2：启动相机**
```bash

注意是ros_ws_xing目录下面
# # 切换到相机工作空间
# source ~/ros2_ws/install/setup.bash

# # 启动Orbbec Astra2相机
# ros2 launch orbbec_camera astra2.launch.py
# ```
使用共享内存
./start_camera_shm.sh

**终端3：启动优化版录制器**
```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3

# 启动录制器（不录制深度图，性能更好）
python3 dobot_demo/dobot_demo/recorder_optimized.py
```

输出：
```
=== Optimized Data Recorder Initialized ===
Depth recording: DISABLED
Async processing with efficient interpolation
Press 'O' to start recording.
```

**终端4：启动遥操作控制器**
```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3

# 启动键盘鼠标遥操作
python3 dobot_demo/dobot_demo/data_collector4.py
```

输出：        
```
=== 系统就绪 ===
 [Main Loop]       : 负责机械臂运动 (ServoP @ 100Hz)
 [Gripper Thread]  : 负责夹爪控制
 [Feedback Thread] : 定期发布夹爪状态 (100Hz)

=== VLA 数据采集话题 ===
 - 机械臂当前状态: /dobot_msgs_v3/msg/ToolVectorActual (100Hz)
 - 机械臂目标状态: /robot/target_pose (100Hz)
 - 夹爪当前状态:   /gripper/state_feedback (100Hz)
 - 夹爪目标状态:   /gripper/command_update (100Hz)

 [O/P/L]           : 录制控制
 [Z]               : 触发垂直抓取序列
 [Tab]             : 暂停/恢复控制
```

#### 1.2 遥操作与录制

**遥操作控制方式：**

| 操作 | 控制方式 | 说明 |
|------|---------|------|
| XY平面移动 | 鼠标移动 | 灵敏度可调 |
| Z轴上升 | 空格键 | 按住上升，Alt+空格快速上升 |
| Z轴下降 | Ctrl键 | 按住下降，Alt+Ctrl快速下降 |
| RX旋转 | A/D键 | A=逆时针，D=顺时针 |
| RY旋转 | W/S键 | W=俯仰向下，S=俯仰向上 |
| RZ旋转 | 鼠标滚轮 | 滚轮控制末端旋转 |
| 夹爪闭合 | 左键按住 | 按住持续闭合 |
| 夹爪张开 | 右键按住 | 按住持续张开 |
| 夹爪快速张开 | X键 | 一键张开到预设位置 |
| 垂直抓取 | Z键 | 自动执行：张开→下降→闭合→上升 |
| 暂停控制 | Tab键 | 切换控制启用/禁用，可操作其他窗口 |
| 退出 | Esc键 | 退出程序 |

**录制控制按键：**
- **O键** = 开始录制
- **P键** = 停止并保存
- **L键** = 丢弃当前录制

**录制流程：**
1. 使用遥操作移动机械臂到起始位置
2. 按 **O** 键开始录制（录制器会输出 "START RECORDING"）
3. 执行机械臂动作示教（抓取、搬运等）
4. 按 **P** 键停止并保存（录制器会保存为 `data/episode_X.hdf5`）
5. 或按 **L** 键丢弃本次录制

**输出：**
```
>>> START RECORDING
Recording... 30 frames (queue: 0)
Recording... 60 frames (queue: 0)
...
>>> STOP & SAVING...
✓ Saved successfully to ./data/episode_0.hdf5
```

---

### 阶段二：数据质量检查

#### 2.1 检查数据集完整性和质量

```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3

# 基本检查（使用默认参数：30Hz，最大3帧间隔）
python3 dobot_demo/dobot_demo/check_dataset.py data/episode_0.hdf5

# 自定义检查参数
python3 dobot_demo/dobot_demo/check_dataset.py data/episode_0.hdf5 \
    --fps 30.0 \
    --max-gap 3 \
    --max-speed 150.0
```

**输出报告包含：**
- 基本信息（帧数、时长、帧率）
- 丢帧检查（丢帧率、丢帧位置）
- 工作空间检查
- 突变检查（位置、旋转、夹爪）
- 速度检查（高速运动点）
- VLA训练适用性评估

**判断标准：**
- ✅ 优秀：丢帧率<1%
- 🟡 良好：丢帧率1-5%
- ⚠️ 中等：丢帧率5-10%
- ❌ 较差：丢帧率>10%

---

### 阶段三：数据清洗（如果需要）

#### 3.1 清洗数据集

如果检查发现丢帧率>5%，建议清洗：

```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3

# 基本清洗（使用默认参数）
python3 dobot_demo/dobot_demo/clean_dataset.py data/episode_0.hdf5

# 自定义参数
python3 dobot_demo/dobot_demo/clean_dataset.py data/episode_0.hdf5 \
    --fps 30.0 \
    --max-gap 3 \
    --min-length 30 \
    --output ./cleaned

# 不保存合并文件（只保存各个片段）
python3 dobot_demo/dobot_demo/clean_dataset.py data/episode_0.hdf5 --no-merge
```

**参数说明：**
- `--fps`: 期望帧率（默认30Hz）
- `--max-gap`: 允许的最大帧间隔（默认3帧，即100ms）
- `--min-length`: 最小片段长度（默认30帧，即1秒）
- `--output`: 输出目录（默认`./cleaned/`）
- `--no-merge`: 不保存合并的完整数据集

**输出：**
```
找到片段: 帧0-280, 长度=280, 时长=9.3s
找到片段: 帧291-845, 长度=554, 时长=18.5s
...
✓ 共找到 3 个高质量片段

保存片段0: ./cleaned/episode_0_seg0.hdf5 (280帧)
保存片段1: ./cleaned/episode_0_seg1.hdf5 (554帧)
保存片段2: ./cleaned/episode_0_seg2.hdf5 (291帧)

保存合并数据集: ./cleaned/episode_0_cleaned_merged.hdf5
  原始帧数: 1125
  清洗后: 1125
  移除帧数: 0
```

---

### 阶段四：数据播放（带安全检查）

#### 4.1 检查数据集兼容性（可选）

```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3

# 使用简单的Python检查
python3 << 'EOF'
import h5py
with h5py.File('data/episode_0.hdf5', 'r') as f:
    print("必需字段检查:")
    for field in ['actions/robot_target', 'actions/gripper_target', 'timestamp']:
        print(f"  {field}: {'✓' if field in f else '✗'}")
EOF
```

#### 4.2 播放数据集

**标准播放（推荐）：**
```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3

# 原速播放（带安全检查）
python3 dobot_demo/dobot_demo/dataset_player.py /home/hit/dobot_ws_xing/data/episode_0.hdf5

# 播放清洗后的数据
python3 dobot_demo/dobot_demo/dataset_player.py cleaned/episode_0_cleaned_merged.hdf5
```

**安全检查交互：**
```
======================================================================
安全检查：当前位置 vs 数据集起始位置
======================================================================

当前位姿:  X= 100.0 Y=-300.0 Z= 400.0 | RX= 180.0 RY=   0.0 RZ=  90.0
起始位姿:  X=-170.7 Y=-498.8 Z= 270.2 | RX= 179.1 RY=   2.7 RZ=  20.7

位置差距: 280.5 mm
旋转差距: 69.3 deg

⚠️  警告：差距超过阈值！

选项1: 自动移动到起始位置（使用MovJ平滑运动）  ← 推荐
选项2: 取消播放，手动调整位置
选项3: 忽略警告，强制播放（危险！）

请选择 [1/2/3]:
```

**其他播放选项：**
```bash
# 0.5倍速慢放（首次测试推荐）
python3 dobot_demo/dobot_demo/dataset_player.py data/episode_0.hdf5 --rate 0.5

# 2倍速快放
python3 dobot_demo/dobot_demo/dataset_player.py data/episode_0.hdf5 --rate 2.0

# 跳过安全检查（仅当确定位置接近时使用）
python3 dobot_demo/dobot_demo/dataset_player.py data/episode_0.hdf5 --no-safety-check

# 禁用自动移动功能
python3 dobot_demo/dobot_demo/dataset_player.py data/episode_0.hdf5 --no-auto-move

# 查看所有选项
python3 dobot_demo/dobot_demo/dataset_player.py --help
```

---

### 阶段五：播放质量评估

#### 5.1 启动评估录制器

**终端1：启动评估录制器**
```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3

python3 dobot_demo/dobot_demo/playback_evaluator.py
```

输出：
```
======================================================================
播放评估录制器已启动
======================================================================
等待播放器发布话题...
提示：启动 dataset_player.py 开始播放，本节点将自动开始录制
======================================================================
```

#### 5.2 播放数据集（在评估录制器运行时）

**终端2：播放数据集**
```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3

# 原速播放
python3 dobot_demo/dobot_demo/dataset_player.py data/episode_0.hdf5

# 或慢速播放
python3 dobot_demo/dobot_demo/dataset_player.py data/episode_0.hdf5 --rate 0.5
```

评估录制器会自动开始录制：
```
[INFO] 检测到播放开始，自动启动录制...
[INFO] >>> 开始录制评估数据
[INFO] 录制进度: 50 帧
[INFO] 录制进度: 100 帧
...
```

#### 5.3 停止评估录制

播放完成后，在**终端1**按 `Ctrl+C`：

```
^C
检测到中断信号，正在停止录制...
[INFO] >>> 停止录制
[INFO] 正在保存 1125 帧到 ./playback_eval/evaluation_0.hdf5...
[INFO] ✓ 成功保存评估数据到 ./playback_eval/evaluation_0.hdf5
[INFO]   总帧数: 1125
[INFO]   总时长: 39.39 秒
```

---

### 阶段六：播放质量分析

#### 6.1 查看统计报告


减少因为servoP延迟增加的误差
  使用流程

  1. 先用无补偿模式录制一次评估数据
     python3 playback_evaluator.py

  2. 分析最优偏移值
     python3 find_optimal_offset.py playback_eval/evaluation_X.hdf5

  3. 用推荐的偏移值重新录制
     python3 playback_evaluator.py --time-offset 737

  4. 分析补偿后的跟踪精度
     python3 analyze_playback.py playback_eval/evaluation_Y.hdf5
```bash
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3

# 基本分析（只显示统计信息）
python3 dobot_demo/dobot_demo/analyze_playback.py playback_eval/evaluation_0.hdf5
```

**输出示例：**
```
======================================================================
跟踪误差统计
======================================================================

[机械臂位置误差] (mm)
  平均值: 2.456 mm
  中位数: 2.123 mm
  标准差: 1.234 mm
  最小值: 0.123 mm
  最大值: 8.567 mm
  95%分位: 5.234 mm
  99%分位: 7.123 mm

[位置分量误差] (mm)
  X轴: 平均=1.234, 最大=5.678
  Y轴: 平均=1.456, 最大=6.789
  Z轴: 平均=0.987, 最大=4.321

[机械臂旋转误差] (deg)
  平均值: 0.456 deg
  ...

[夹爪位置误差] (0-1000)
  平均值: 12.345
  ...

======================================================================
跟踪质量评估
======================================================================

[位置跟踪质量]
  优秀 (<1mm):   45.2% (508 帧)
  良好 (<5mm):   89.3% (1004 帧)
  较差 (>=5mm):  10.7% (121 帧)

[总体评级]
  评级: 良好
  建议: 跟踪精度良好，适合大部分任务
```

#### 6.2 生成可视化图表

```bash
# 生成图表（包含6个子图）
python3 dobot_demo/dobot_demo/analyze_playback.py playback_eval/evaluation_0.hdf5 --plot
```

输出：
```
✓ 图表已保存到: playback_eval/evaluation_0_analysis.png
```

**图表内容：**
1. 位置误差时序图
2. 位置误差分布直方图
3. 旋转误差时序图
4. 夹爪误差时序图
5. XYZ分量误差对比
6. 误差累积分布函数(CDF)

#### 6.3 自定义分析参数

```bash
# 使用更严格的阈值
python3 dobot_demo/dobot_demo/analyze_playback.py playback_eval/evaluation_0.hdf5 \
    --plot \
    --pos-threshold 2.0 \
    --rot-threshold 1.0
```

---
source /home/hit/dobot_ws_xing/install/setup.bash
## 快速参考：常用命令

### 系统启动
```bash
# 终端1：启动机械臂
cd /home/hit/dobot_ws_xing
source install/setup.bash
ros2 launch dobot_bringup_v3 dobot_bringup_ros2.launch.py

# 终端2：启动相机
source ~/ros2_ws/install/setup.bash
ros2 launch orbbec_camera astra2.launch.py

# 终端3：启动录制器
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3
python3 dobot_demo/dobot_demo/recorder_optimized.py

# 终端4：启动遥操作
cd /home/hit/dobot_ws_xing/src/DOBOT_6Axis_ROS2_V3
python3 dobot_demo/dobot_demo/data_collector4.py
```

### 数据录制
```bash
# 使用遥操作控制机械臂：
# - 鼠标移动 = XY平面
# - 空格/Ctrl = Z轴升降
# - WASD = 旋转
# - 左右键 = 夹爪开合

# 录制控制：
# - O键 = 开始录制
# - P键 = 停止并保存
# - L键 = 丢弃录制
```

### 数据检查
```bash
# 检查数据质量
python3 dobot_demo/dobot_demo/check_dataset.py data/episode_0.hdf5
```

### 数据清洗
```bash
# 清洗数据（如果丢帧率>5%）
python3 dobot_demo/dobot_demo/clean_dataset.py data/episode_0.hdf5
```

### 数据播放
```bash
# 标准播放（带安全检查）
python3 dobot_demo/dobot_demo/dataset_player.py data/episode_0.hdf5

# 慢速播放（首次测试推荐）
python3 dobot_demo/dobot_demo/dataset_player.py data/episode_0.hdf5 --rate 0.5
```

### 播放评估
```bash
# 终端1：启动评估录制器
python3 dobot_demo/dobot_demo/playback_evaluator.py

# 终端2：播放数据集
python3 dobot_demo/dobot_demo/dataset_player.py data/episode_0.hdf5

# 终端1：播放完成后按 Ctrl+C 保存

# 分析评估结果
python3 dobot_demo/dobot_demo/analyze_playback.py playback_eval/evaluation_0.hdf5 --plot
```

---

## 质量标准与建议

### 数据录制质量标准

| 指标 | 优秀 | 良好 | 中等 | 较差 |
|------|------|------|------|------|
| 丢帧率 | <1% | 1-5% | 5-10% | >10% |
| 平均帧率 | >28Hz | 25-28Hz | 20-25Hz | <20Hz |
| 最大帧间隔 | <50ms | 50-100ms | 100-150ms | >150ms |

**建议：**
- 优秀/良好：可直接用于训练
- 中等：建议清洗后使用
- 较差：需重新录制

### 播放跟踪质量标准

| 指标 | 优秀 | 良好 | 中等 | 较差 |
|------|------|------|------|------|
| 位置误差平均值 | <1mm | 1-3mm | 3-5mm | >5mm |
| 位置误差95%分位 | <2mm | 2-5mm | 5-8mm | >8mm |
| 旋转误差平均值 | <0.5° | 0.5-1.5° | 1.5-2° | >2° |

**建议：**
- 优秀：可用于高精度任务
- 良好：适合大部分任务
- 中等：需优化控制参数
- 较差：需检查系统配置

---

## 故障排查

### 录制问题

**问题：丢帧率很高（>10%）**
- 检查相机是否稳定在30Hz
- 使用 `diagnose_camera.sh` 诊断相机
- 考虑禁用深度图录制

**问题：录制时卡顿**
- 降低图像分辨率
- 检查磁盘写入速度
- 增加消息缓冲区大小

### 播放问题

**问题：跟踪误差大（>5mm）**
- 降低播放速度（使用 `--rate 0.5`）
- 检查网络延迟
- 确认ServoP频率为100Hz

**问题：播放时机械臂突然停止**
- 检查安全限位
- 确认工作空间设置
- 查看机械臂驱动日志

### 评估问题

**问题：评估录制器没有收到消息**
- 确认播放器已启动
- 检查话题名称：`ros2 topic list`
- 确认时间同步

**问题：评估数据帧数少**
- 增大时间对齐容差
- 检查所有话题发布频率

---

## 文件组织建议

```
DOBOT_6Axis_ROS2_V3/
├── data/                          # 原始录制数据
│   ├── episode_0.hdf5
│   ├── episode_1.hdf5
│   └── ...
├── cleaned/                       # 清洗后数据
│   ├── episode_0_cleaned_merged.hdf5
│   └── episode_0_seg0.hdf5
├── playback_eval/                 # 播放评估数据
│   ├── evaluation_0.hdf5
│   ├── evaluation_0_analysis.png
│   └── ...
└── dobot_demo/dobot_demo/         # 工具脚本
    ├── recorder_optimized.py
    ├── check_dataset.py
    ├── clean_dataset.py
    ├── dataset_player.py
    ├── playback_evaluator.py
    └── analyze_playback.py
```

---

## 总结

完整流程：

1. **录制** → `recorder_optimized.py` → `data/episode_X.hdf5`
2. **检查** → `check_dataset.py` → 质量报告
3. **清洗**（如需） → `clean_dataset.py` → `cleaned/episode_X_cleaned.hdf5`
4. **播放** → `dataset_player.py` → 机械臂重放
5. **评估** → `playback_evaluator.py` + `dataset_player.py` → `evaluation_X.hdf5`
6. **分析** → `analyze_playback.py` → 统计报告 + 图表

**关键命令记忆：**
- 录制：`recorder_optimized.py` + `ros2 topic pub /recorder/command`
- 检查：`check_dataset.py data/episode_X.hdf5`
- 清洗：`clean_dataset.py data/episode_X.hdf5`
- 播放：`dataset_player.py data/episode_X.hdf5 --rate 0.5`
- 评估：`playback_evaluator.py` + `dataset_player.py` + `Ctrl+C`
- 分析：`analyze_playback.py playback_eval/evaluation_X.hdf5 --plot`
