# 数据集播放器使用说明

## 功能概述

`dataset_player.py` 用于播放录制的HDF5数据集，重现机械臂和夹爪的运动轨迹，并发布相应的ROS话题供评估。

## 工作流程

```
录制阶段:
  data_collector4 (遥操作) → recorder_simple (记录) → episode_X.hdf5

评估阶段:
  dataset_player (播放HDF5) + recorder_simple (记录播放) → 对比评估
```

## 编译安装

```bash
cd /home/hit/dobot_ws_xing
colcon build --packages-select dobot_demo
source install/setup.bash
```

## 使用方法

### 0. 安全检查（重要！）

**在播放数据集之前，务必先进行安全检查，防止机械臂爆冲！**

```bash
# 基本检查
ros2 run dobot_demo check_dataset_safety ./data/episode_0.hdf5

# 严格模式（更低的阈值）
ros2 run dobot_demo check_dataset_safety ./data/episode_0.hdf5 --strict

# 自定义阈值
ros2 run dobot_demo check_dataset_safety ./data/episode_0.hdf5 \
    --max-position-jump 30.0 \
    --max-rotation-jump 5.0 \
    --max-velocity 80.0
```

安全检查会验证：
- ✓ 是否超出工作空间边界
- ✓ 是否存在位置突变（>50mm）
- ✓ 是否存在旋转突变（>10deg）
- ✓ 夹爪变化是否剧烈
- ✓ 速度是否在合理范围内

**示例输出：**
```
======================================================================
数据集安全检查: ./data/episode_0.hdf5
======================================================================

[统计信息]
  总帧数: 150
  总时长: 5.00 秒
  平均帧率: 30.0 Hz

  位置范围:
    X: [250.5, 450.2] mm
    Y: [-150.3, 120.7] mm
    Z: [180.0, 280.5] mm

[检查1] 工作空间边界检查...
  ✓ 所有点在工作空间内

[检查2] 位置突变检查...
  ✓ 位置变化正常 (最大: 12.34 mm < 50.0 mm)
  ✓ 旋转变化正常 (最大: 3.21 deg < 10.0 deg)

[检查3] 夹爪突变检查...
  ✓ 夹爪变化正常 (最大: 45.0 < 200.0)

[检查4] 速度检查...
  ✓ 速度正常 (最大: 78.5 mm/s < 100.0 mm/s)
  平均速度: 25.3 mm/s
  最大速度: 78.5 mm/s

======================================================================
检查结果
======================================================================

✅ 所有检查通过，数据集安全可播放！
```

### 1. 播放单个数据集

```bash
# 基本用法
ros2 run dobot_demo dataset_player ./data/episode_0.hdf5

# 指定播放速率
ros2 run dobot_demo dataset_player ./data/episode_0.hdf5 --rate 0.5  # 0.5倍速
ros2 run dobot_demo dataset_player ./data/episode_0.hdf5 --rate 2.0  # 2倍速
```

### 2. 播放+录制评估

**终端1：启动录制器**
```bash
ros2 run dobot_demo recorder_simple
```

**终端2：播放数据集**
```bash
ros2 run dobot_demo dataset_player ./data/episode_0.hdf5
```

这样会生成新的episode文件（如`episode_1.hdf5`），其中包含播放过程中的实际机械臂状态。

### 3. 对比原始数据与播放数据

使用提供的评估工具对比两个HDF5文件：

```bash
# 基本评估（仅显示统计信息）
ros2 run dobot_demo evaluate_playback ./data/episode_0.hdf5 ./data/episode_1.hdf5

# 生成可视化图表
ros2 run dobot_demo evaluate_playback ./data/episode_0.hdf5 ./data/episode_1.hdf5 --plot

# 保存图表到指定目录
ros2 run dobot_demo evaluate_playback ./data/episode_0.hdf5 ./data/episode_1.hdf5 --plot --save-dir ./evaluation_results
```

输出示例：
```
============================================================
机械臂位置 误差统计
============================================================
维度       平均       标准差     最大       RMSE       中位数
------------------------------------------------------------
X          0.523      0.412      2.135      0.665      0.431
Y          0.489      0.391      1.987      0.625      0.398
Z          0.612      0.521      2.543      0.801      0.501
RX         0.234      0.187      0.921      0.299      0.192
RY         0.198      0.156      0.782      0.251      0.163
RZ         0.176      0.134      0.689      0.221      0.144
------------------------------------------------------------
总体       0.372      0.300      2.543      0.477      0.305

============================================================
夹爪位置 误差统计
============================================================
维度       平均       标准差     最大       RMSE       中位数
------------------------------------------------------------
Position   15.234     12.567     52.341     19.872     12.431
------------------------------------------------------------
总体       15.234     12.567     52.341     19.872     12.431
```

### 4. 完整评估流程示例（推荐）

```bash
# ===== 步骤1: 录制原始数据集 =====
# 终端1
ros2 run dobot_demo recorder_simple

# 终端2
ros2 run dobot_demo data_collector4
# 按O开始录制，操作机械臂，按P保存 -> 生成 episode_0.hdf5

# ===== 步骤2: 安全检查（重要！）=====
ros2 run dobot_demo check_dataset_safety ./data/episode_0.hdf5

# 如果检查通过，继续下一步
# 如果有错误，请检查数据集或调整阈值

# ===== 步骤3: 播放数据集并录制播放过程 =====
# 终端1
ros2 run dobot_demo recorder_simple
# 按O开始录制

# 终端2（在确认安全后再执行）
ros2 run dobot_demo dataset_player ./data/episode_0.hdf5
# 播放完成后，在终端1按P保存 -> 生成 episode_1.hdf5

# ===== 步骤4: 评估播放精度 =====
ros2 run dobot_demo evaluate_playback ./data/episode_0.hdf5 ./data/episode_1.hdf5 --plot --save-dir ./eval
```

**安全提示：**
1. 播放前必须运行安全检查
2. 确保机械臂周围无障碍物
3. 准备好急停按钮
4. 建议首次以0.5倍速播放：`--rate 0.5`

## 发布的话题

播放器会发布以下话题（与data_collector4保持一致）：

| 话题名称 | 消息类型 | 频率 | 说明 |
|---------|---------|------|------|
| `/robot/target_pose` | ToolVectorActual | 100Hz | 机械臂目标姿态（来自数据集） |
| `/gripper/state_feedback` | PointStamped | 100Hz | 夹爪当前状态（实际读取+插补） |
| `/gripper/command_update` | PointStamped | 100Hz | 夹爪目标状态（来自数据集） |

## 关键特性

1. **时间对齐**: 使用数据集中记录的时间戳，保持原始播放速率
2. **插补发布**: 夹爪状态以100Hz发布，使用10Hz采样+线性插补（与录制时一致）
3. **ServoP控制**: 使用ServoP模式进行机械臂控制，保证平滑运动
4. **播放速率可调**: 支持通过`--rate`参数调整播放速度

## 注意事项

1. **安全检查**: 播放前确保机械臂周围无障碍物，轨迹安全
2. **初始位置**: 播放器会直接跳转到数据集的第一帧位置，确保机械臂可以安全到达
3. **急停准备**: 随时准备按下机械臂的急停按钮
4. **频率匹配**: 播放频率默认100Hz，与录制时的控制频率一致

## 故障排除

**问题1: 机械臂不移动**
- 检查Dobot服务是否运行
- 确认机械臂已使能（EnableRobot）
- 查看终端是否有错误信息

**问题2: 夹爪不动作**
- 检查Modbus连接是否成功
- 确认夹爪初始化日志显示"连接成功"

**问题3: 播放延迟**
- 降低播放速率 `--rate 0.5`
- 检查系统CPU负载

**问题4: 轨迹不匹配**
- 确保录制和播放使用相同的坐标系
- 检查数据集中的sync_method是否为"LinearInterpolation"
