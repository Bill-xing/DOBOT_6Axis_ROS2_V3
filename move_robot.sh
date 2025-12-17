#!/bin/bash

# DOBOT机械臂姿态控制脚本
# 用法: ./move_robot.sh [姿态名称或自定义角度]

# 定义关节顺序: [joint2, joint3, joint1, joint4, joint5, joint6]

# 预定义姿态
declare -A poses=(
    ["home"]="0.0,0.0,0.0,0.0,0.0,0.0"
    ["ready"]="0.0,0.5,0.0,0.0,0.0,0.0"
    ["pickup"]="0.3,0.8,0.0,0.5,0.0,0.0"
    ["place"]="0.5,0.3,0.0,0.2,0.0,0.0"
    ["fold"]="1.0,1.2,0.0,1.0,0.0,0.0"
)

# 使用方法说明
function show_usage() {
    echo "用法: $0 [姿态名称或自定义角度]"
    echo ""
    echo "预定义姿态:"
    echo "  home    - 初始姿态 (所有关节归零)"
    echo "  ready   - 准备姿态"
    echo "  pickup  - 抓取姿态"
    echo "  place   - 放置姿态"
    echo "  fold    - 收起姿态"
    echo ""
    echo "自定义姿态示例:"
    echo "  $0 0.5,0.3,0.0,0.2,0.0,0.0"
    echo ""
    echo "显示关节顺序:"
    echo "  $0 --show-joints"
}

# 显示关节顺序
function show_joints() {
    echo "关节顺序: [joint2, joint3, joint1, joint4, joint5, joint6]"
    echo "对应角度: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
}

# 发送轨迹命令
function send_trajectory() {
    local positions="$1"
    local duration="${2:-3.0}"

    echo "发送轨迹指令..."
    echo "目标角度: [$positions]"
    echo "运动时间: ${duration}秒"

    ros2 topic pub /cr5_group_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
      header: {
        stamp: {sec: 0, nanosec: 0},
        frame_id: \"base_link\"
      },
      joint_names: [\"joint2\", \"joint3\", \"joint1\", \"joint4\", \"joint5\", \"joint6\"],
      points: [{
        positions: [$positions],
        velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        time_from_start: {sec: $duration, nanosec: 0}
      }]
    }" --once

    if [ $? -eq 0 ]; then
        echo "✓ 指令发送成功"
    else
        echo "✗ 指令发送失败"
    fi
}

# 主程序
case "$1" in
    ""|"--help"|"-h")
        show_usage
        ;;
    "--show-joints")
        show_joints
        ;;
    *)
        # 检查是否为预定义姿态
        if [[ -n "${poses[$1]}" ]]; then
            positions="${poses[$1]}"
            send_trajectory "$positions"
        elif [[ "$1" =~ ^-?[0-9]*\.?[0-9]+,-?[0-9]*\.?[0-9]+,-?[0-9]*\.?[0-9]+,-?[0-9]*\.?[0-9]+,-?[0-9]*\.?[0-9]+,-?[0-9]*\.?[0-9]+$ ]]; then
            # 验证自定义角度格式
            send_trajectory "$1" "$2"
        else
            echo "错误: 无效的姿态名称或角度格式"
            echo ""
            show_usage
            exit 1
        fi
        ;;
esac