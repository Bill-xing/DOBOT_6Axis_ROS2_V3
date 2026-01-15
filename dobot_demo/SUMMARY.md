# VLA数据采集与播放系统

## 系统概述

完整的机械臂数据采集、播放和评估系统，用于VLA模型训练。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    数据采集与播放系统                          │
└─────────────────────────────────────────────────────────────┘

录制阶段:
  ┌──────────────┐    话题发布    ┌──────────────┐
  │data_collector4│──────────────>│recorder_simple│
  │(遥操作控制)   │               │(数据录制)     │
  └──────────────┘               └──────────────┘
         │                              │
         v                              v
   控制机械臂                       episode_X.hdf5
   + 夹爪                           (HDF5数据集)

播放阶段:
  ┌──────────────┐    话题发布    ┌──────────────┐
  │dataset_player │──────────────>│recorder_simple│
  │(播放HDF5)    │               │(记录播放)     │
  └──────────────┘               └──────────────┘
         │                              │
         v                              v
   控制机械臂                       episode_Y.hdf5
   + 夹爪                           (播放记录)

评估阶段:
  ┌──────────────┐
  │evaluate_      │
  │playback       │ 对比原始与播放数据
  │(精度评估)     │ 生成误差统计和图表
  └──────────────┘

安全检查:
  ┌──────────────┐
  │check_dataset_ │
  │safety         │ 播放前安全检查
  │(防爆冲检查)   │ 检测位置突变
  └──────────────┘
```

## 工具列表

### 1. data_collector4（遥操作控制）
- **功能**: 鼠标+键盘控制机械臂和夹爪
- **发布话题**:
  - `/robot/target_pose` (ToolVectorActual, 100Hz) - 机械臂目标状态
  - `/gripper/state_feedback` (PointStamped, 100Hz) - 夹爪当前状态
  - `/gripper/command_update` (PointStamped, 100Hz) - 夹爪目标状态
- **控制方式**:
  - 鼠标移动: XY平移
  - 空格/Ctrl: Z轴上下
  - A/D/W/S: 旋转
  - 左键/右键: 夹爪闭合/张开
  - O/P/L: 录制开始/保存/丢弃

### 2. recorder_simple（数据录制）
- **功能**: 订阅话题并保存为HDF5格式
- **订阅话题**:
  - `/camera/color/image_raw` (30Hz)
  - `/camera/depth/image_raw` (30Hz)
  - `/dobot_msgs_v3/msg/ToolVectorActual` (100Hz) - 机械臂当前状态
  - `/robot/target_pose` (100Hz) - 机械臂目标状态
  - `/gripper/state_feedback` (100Hz) - 夹爪当前状态
  - `/gripper/command_update` (100Hz) - 夹爪目标状态
- **同步方法**: 线性插补（保留所有图像，插补机械臂/夹爪状态）
- **输出格式**: HDF5文件包含:
  - `observations/images/color` (N, H, W, 3)
  - `observations/images/depth` (N, H, W)
  - `observations/robot_current` (N, 6)
  - `observations/gripper_current` (N, 1)
  - `actions/robot_target` (N, 6)
  - `actions/gripper_target` (N, 1)
  - `timestamp` (N,)

### 3. dataset_player（数据集播放）
- **功能**: 读取HDF5文件，控制机械臂重放轨迹
- **输入**: HDF5文件路径
- **发布话题**: 与data_collector4相同
- **特性**:
  - 时间对齐播放
  - 可调播放速率（--rate参数）
  - ServoP控制模式
  - 100Hz夹爪状态发布（10Hz采样+时间插补）

### 4. check_dataset_safety（安全检查）
- **功能**: 播放前安全检查，防止机械臂爆冲
- **检查项目**:
  - ✓ 工作空间边界
  - ✓ 位置突变（默认阈值：50mm）
  - ✓ 旋转突变（默认阈值：10deg）
  - ✓ 夹爪突变（默认阈值：200）
  - ✓ 速度限制（默认阈值：100mm/s）
- **使用模式**:
  - 普通模式: `check_dataset_safety episode_0.hdf5`
  - 严格模式: `--strict` (更低阈值)
  - 自定义阈值: `--max-position-jump 30.0`

### 5. evaluate_playback（精度评估）
- **功能**: 对比原始数据与播放数据
- **输入**: 两个HDF5文件
- **输出**:
  - 统计信息（平均、标准差、最大、RMSE、中位数）
  - 可视化图表（--plot参数）
  - 误差分析（各维度独立分析）

## 快速开始

### 编译安装

```bash
cd /home/hit/dobot_ws_xing
colcon build --packages-select dobot_demo
source install/setup.bash
```

### 完整工作流程

#### 阶段1: 数据录制
```bash
# 终端1: 启动录制器
ros2 run dobot_demo recorder_simple

# 终端2: 启动遥操作
ros2 run dobot_demo data_collector4

# 操作流程:
# 1. 按 O 开始录制
# 2. 用鼠标键盘操作机械臂完成任务
# 3. 按 P 保存数据集 (生成 episode_0.hdf5)
# 4. 按 L 丢弃数据集（如果录制失败）
```

#### 阶段2: 安全检查
```bash
# 检查数据集是否安全
ros2 run dobot_demo check_dataset_safety ./data/episode_0.hdf5

# 如果显示 "✅ 所有检查通过"，可以继续
# 如果有 "❌ 存在错误"，需要检查或重新录制
```

#### 阶段3: 播放验证
```bash
# 终端1: 启动录制器
ros2 run dobot_demo recorder_simple
# 按 O 开始录制播放过程

# 终端2: 播放数据集（建议首次使用慢速）
ros2 run dobot_demo dataset_player ./data/episode_0.hdf5 --rate 0.5

# 播放完成后，在终端1按 P 保存 (生成 episode_1.hdf5)
```

#### 阶段4: 精度评估
```bash
# 评估播放精度
ros2 run dobot_demo evaluate_playback ./data/episode_0.hdf5 ./data/episode_1.hdf5 --plot --save-dir ./eval

# 查看评估结果
# - 终端输出: 误差统计
# - ./eval/robot_trajectory_comparison.png: 机械臂轨迹对比
# - ./eval/gripper_trajectory_comparison.png: 夹爪轨迹对比
# - ./eval/robot_errors.png: 误差曲线
```

## 数据格式规范

### HDF5文件结构
```
episode_X.hdf5
├── observations/
│   ├── images/
│   │   ├── color (N, H, W, 3) uint8
│   │   └── depth (N, H, W) uint16
│   ├── robot_current (N, 6) float32  # [x, y, z, rx, ry, rz]
│   └── gripper_current (N, 1) float32  # [position]
├── actions/
│   ├── robot_target (N, 6) float32
│   └── gripper_target (N, 1) float32
├── timestamp (N,) float64
└── camera/
    └── intrinsics (3, 3) float32

属性 (attrs):
  - sim: False
  - total_frames: N
  - sync_method: "LinearInterpolation"
  - camera_frequency_hz: 30
  - robot_frequency_hz: 100
```

## 话题规范

| 话题名称 | 消息类型 | 频率 | 说明 |
|---------|---------|------|------|
| `/camera/color/image_raw` | sensor_msgs/Image | 30Hz | RGB图像 |
| `/camera/depth/image_raw` | sensor_msgs/Image | 30Hz | 深度图像 |
| `/dobot_msgs_v3/msg/ToolVectorActual` | ToolVectorActual | 100Hz | 机械臂当前状态 |
| `/robot/target_pose` | ToolVectorActual | 100Hz | 机械臂目标状态 |
| `/gripper/state_feedback` | PointStamped | 100Hz | 夹爪当前状态 |
| `/gripper/command_update` | PointStamped | 100Hz | 夹爪目标状态 |
| `/recorder/command` | std_msgs/Int32 | - | 录制控制指令 |

## 安全注意事项

### 播放前必查
1. **运行安全检查**: 使用 `check_dataset_safety` 检测位置突变
2. **清空周围空间**: 确保机械臂工作范围内无障碍物
3. **准备急停**: 随时准备按下急停按钮
4. **慢速首播**: 首次播放建议使用 `--rate 0.5` (0.5倍速)
5. **监控状态**: 观察机械臂是否按预期运动

### 危险信号
- ❌ 机械臂突然快速移动
- ❌ 出现异常噪音
- ❌ 轨迹与预期不符
- ❌ 终端显示错误消息

**遇到危险信号立即按下急停！**

## 故障排除

### 问题1: 编译失败
```bash
# 检查依赖
pip3 install h5py numpy matplotlib

# 清理后重新编译
cd /home/hit/dobot_ws_xing
rm -rf build install log
colcon build --packages-select dobot_demo
```

### 问题2: 播放器无法连接机械臂
```bash
# 检查服务是否运行
ros2 service list | grep dobot

# 重启机械臂服务
# (根据实际情况执行)
```

### 问题3: 录制时相机帧率低
- 检查相机连接
- 检查系统资源占用
- 降低录制频率或图像分辨率

### 问题4: 播放精度差
- 检查机械臂ServoP模式是否正确
- 检查播放速率是否合适
- 检查原始数据质量

## 性能指标

### 推荐配置
- 录制频率: 30Hz (相机) + 100Hz (机械臂/夹爪)
- 播放频率: 100Hz
- 时间同步: 线性插补
- 数据格式: HDF5 (压缩)

### 预期精度
- 机械臂位置误差: < 5mm (平均)
- 机械臂旋转误差: < 2deg (平均)
- 夹爪位置误差: < 20 (平均)
- 时间对齐误差: < 50ms

## 文件清单

```
dobot_demo/
├── dobot_demo/
│   ├── data_collector4.py          # 遥操作控制器
│   ├── recorder_simple.py          # 数据录制器
│   ├── dataset_player.py           # 数据集播放器
│   ├── check_dataset_safety.py     # 安全检查工具
│   └── evaluate_playback.py        # 播放评估工具
├── setup.py                        # ROS2包配置
├── README_dataset_player.md        # 详细使用文档
└── SUMMARY.md                      # 本文档

输出数据:
./data/
├── episode_0.hdf5                  # 原始录制数据
├── episode_1.hdf5                  # 播放记录数据
└── ...

评估结果:
./eval/
├── robot_trajectory_comparison.png
├── gripper_trajectory_comparison.png
└── robot_errors.png
```

## 未来改进

1. **可视化界面**: 添加GUI控制和监控
2. **实时反馈**: 播放时显示当前帧和误差
3. **自动对齐**: 自动对齐播放起始位置
4. **批量处理**: 支持批量播放和评估
5. **增强安全**: 添加碰撞检测和力控制

## 联系支持

如遇到问题，请检查：
1. ROS2日志: `ros2 topic echo /rosout`
2. 数据集完整性: `h5py文件查看工具`
3. 系统资源: `htop`

---

**版本**: v1.0
**最后更新**: 2026-01-15
**适用系统**: ROS2 + Dobot机械臂
