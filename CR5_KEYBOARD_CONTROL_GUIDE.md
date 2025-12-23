# CR5 机械臂 MoveIt 键盘控制指南

## 🎯 概述
这是一个为越疆 CR5 机械臂设计的安全键盘控制系统，基于 MoveIt 运动规划框架。
**核心特性：极小运动增量、防碰撞检测、Z 轴安全限制**

---

## ⚠️ 安全参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 关节运动步长 | ±1° | 每次按键增加/减少 1° |
| 笛卡尔运动步长 | ±3mm | 每次按键移动 3 毫米 |
| 速度限制 | 30% | 降低到标准速度的 30% |
| Z 轴最小高度 | 0.15m | 防止碰撞桌面的绝对限制 |
| Z 轴警告阈值 | 0.20m | 接近桌面时发出警告 |

---

## 🚀 启动步骤

### 第1步：配置环境

```bash
# 在 ~/.bashrc 中确保设置了以下环境变量
export IP_address=192.168.5.1    # 有线连接 IP（根据实际修改）
export DOBOT_TYPE=cr5
source ~/.bashrc
```

### 第2步：打开终端1 - 启动机械臂服务

```bash
cd ~/dobot_ws
source install/local_setup.sh
ros2 launch dobot_bringup_v3 dobot_bringup_ros2.launch.py
```
等待输出显示：`connection succeeded`

### 第3步：打开终端2 - 启动 MoveIt

```bash
cd ~/dobot_ws
source install/local_setup.sh
ros2 launch dobot_moveit dobot_moveit.launch.py
```
等待 RViz 窗口打开并显示机械臂模型

### 第4步：打开终端3 - 启动 Action Server

```bash
cd ~/dobot_ws
source install/local_setup.sh
ros2 run dobot_moveit action_move_server
```

### 第5步：打开终端4 - 启动键盘控制

```bash
cd ~/dobot_ws
source install/local_setup.sh
ros2 run dobot_demo keyboard_control
```

---

## 🎮 控制命令

### 关节控制
```
1-6    选择要控制的关节（1=joint1, 2=joint2, ..., 6=joint6）
+      所选关节增加 1°
-      所选关节减少 1°
```

**例子**：
```
输入命令: 1          ← 选择关节1
输入命令: +          ← 关节1 增加 1°
输入命令: +          ← 关节1 再增加 1°
输入命令: -          ← 关节1 减少 1°
```

### 笛卡尔空间控制
```
W      末端执行器向前移动 3mm (X+ 方向)
S      末端执行器向后移动 3mm (X- 方向)
A      末端执行器向左移动 3mm (Y+ 方向)
D      末端执行器向右移动 3mm (Y- 方向)
Q      末端执行器向上移动 3mm (Z+ 方向)
E      末端执行器向下移动 3mm (Z- 方向)
```

### 规划和执行
```
P      规划当前位置的运动轨迹
X      执行已规划的轨迹
```

**工作流程**：
1. 使用关节或笛卡尔命令设置目标位置
2. 按 `P` 规划
3. 在 RViz 中确认轨迹无碰撞
4. 按 `X` 执行

### 安全命令
```
R      重置到安全位置（打开构型）
T      移动到高位置 (TableTop)
G      显示当前位置和安全状态
L      显示关节限制信息
H      显示帮助
QUIT   退出程序
```

---

## 📊 安全监控

程序会自动监控以下安全指标：

### 1. **Z 轴高度监控**
```
Z > 0.20m     ✓ 安全区域
0.15m < Z ≤ 0.20m  ⚠️ 警告区域（接近表面）
Z ≤ 0.15m     🚨 危险区域（禁止）
```

### 2. **碰撞检测**
- MoveIt 会自动检测规划的轨迹是否与桌面碰撞
- 如果规划失败，会显示警告信息

### 3. **关节限制**
- 按 `L` 可查看所有关节的角度范围
- 超出范围的运动会被自动拒绝

---

## 💡 使用示例

### 示例1：Z 轴安全上升（避免碰撞）
```
输入命令: Q              ← 向上移动 3mm
输入命令: Q              ← 再向上移动 3mm
输入命令: Q              ← 持续上升至安全高度
输入命令: G              ← 检查当前位置
```

### 示例2：关节巡航
```
输入命令: 2              ← 选择关节2
输入命令: +              ← 关节2 +1°
输入命令: +              ← 关节2 +1°
输入命令: P              ← 规划
输入命令: X              ← 执行
```

### 示例3：返回安全位置
```
输入命令: R              ← 一键回到安全位置
```

---

## 🔧 调整安全参数

如需修改安全参数，编辑文件：
```
/home/hit/dobot_ws/src/DOBOT_6Axis_ROS2_V3/dobot_demo/dobot_demo/keyboard_control.py
```

在 `__init__` 方法中修改：
```python
self.JOINT_STEP = 1.0          # 关节步长（度数）
self.CARTESIAN_STEP = 0.003    # 笛卡尔步长（米）
self.MIN_Z = 0.15              # Z 轴最小高度（米）
self.Z_WARNING_THRESHOLD = 0.20 # Z 轴警告阈值（米）
```

修改后需重新编译：
```bash
cd ~/dobot_ws
colcon build --packages-select dobot_demo
source install/local_setup.sh
```

---

## ⚡ 常见问题

### Q1: 启动时报错 "service not available"
**原因**：终端1的机械臂服务未启动
**解决**：确保终端1正在运行 `dobot_bringup_ros2.launch.py`

### Q2: RViz 中看不到机械臂
**原因**：robot state publisher 未启动
**解决**：检查终端2 输出是否有 "robot_state_publisher" 信息

### Q3: 按键无反应
**原因**：可能没有获得规划
**解决**：确保按 `P` 进行规划后再按 `X` 执行

### Q4: "Z 高度过低" 警告频繁出现
**原因**：机械臂初始位置可能就接近桌面
**解决**：按 `T` 移动到高位置，或按 `Q` 多次上升

### Q5: 想要更大的运动步长
**原因**：默认是安全的小步长
**解决**：在 keyboard_control.py 中修改 `JOINT_STEP` 和 `CARTESIAN_STEP`

---

## 📝 日志查看

所有操作都会打印到终端，关键信息示例：
```
✓ 关节 2 移动 +1.0°
✓ 笛卡尔移动: X +0.0030m → 目标位置 Z=0.320m
⚠️  警告：接近表面！Z 高度: 0.195m
🚨 危险：Z 高度过低
✓ 规划成功，输入 X 执行
```

---

## 🛡️ 操作安全建议

1. **首次使用前**：
   - 确保周围环境清空，机械臂有足够活动空间
   - 穿戴安全防护装备

2. **开始控制时**：
   - 从安全位置开始（按 `R` 或 `T`）
   - 使用小步长逐步靠近目标
   - 观察 RViz 中的轨迹规划

3. **运动中**：
   - 随时准备按 Ctrl+C 停止程序
   - 监控终端输出的安全警告
   - 避免快速连续输入命令

4. **紧急情况**：
   - 按 Ctrl+C 立即停止
   - 机械臂会停止规划，但不会刹停
   - 如需硬停，使用示教盒或控制器的紧急按钮

---

## 📚 更多资源

- MoveIt 官方文档: http://docs.ros.org/en/humble/p/moveit2/
- ROS2 文档: https://docs.ros.org/en/humble/
- CR5 规格: [参考越疆官方文档]

---

**最后更新**: 2025年12月17日
**作者**: GitHub Copilot
