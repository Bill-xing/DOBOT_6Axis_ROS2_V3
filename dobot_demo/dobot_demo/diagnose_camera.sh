#!/bin/bash
# 相机问题诊断脚本

echo "=========================================="
echo "相机帧率问题诊断"
echo "=========================================="
echo ""

echo "[1] USB连接检查"
echo "----------------------------------------"
USB_INFO=$(lsusb -t | grep -B 2 -A 2 -i "camera\|realsense")
echo "$USB_INFO"

if echo "$USB_INFO" | grep -q "5000M"; then
    echo "✅ USB 3.0连接"
elif echo "$USB_INFO" | grep -q "480M"; then
    echo "❌ USB 2.0连接 - 这是问题！需要换USB 3.0接口"
else
    echo "⚠️  无法判断USB版本"
fi
echo ""

echo "[2] 相机节点检查"
echo "----------------------------------------"
CAMERA_NODE=$(ros2 node list 2>/dev/null | grep -i camera | head -1)
if [ -z "$CAMERA_NODE" ]; then
    echo "❌ 相机节点未运行"
else
    echo "✅ 相机节点: $CAMERA_NODE"

    echo ""
    echo "[3] 相机参数"
    echo "----------------------------------------"
    ros2 param get $CAMERA_NODE color_width 2>/dev/null || echo "无法获取width参数"
    ros2 param get $CAMERA_NODE color_height 2>/dev/null || echo "无法获取height参数"
    ros2 param get $CAMERA_NODE color_fps 2>/dev/null || echo "无法获取fps参数"
    ros2 param get $CAMERA_NODE enable_depth 2>/dev/null || echo "无法获取depth参数"
fi
echo ""

echo "[4] 话题检查"
echo "----------------------------------------"
COLOR_TOPIC=$(ros2 topic list 2>/dev/null | grep "color/image_raw" | head -1)
if [ -z "$COLOR_TOPIC" ]; then
    echo "❌ 未找到RGB图像话题"
else
    echo "✅ RGB话题: $COLOR_TOPIC"

    echo ""
    echo "订阅者数量："
    ros2 topic info $COLOR_TOPIC 2>/dev/null | grep "Subscription count"

    echo ""
    echo "实际图像尺寸："
    timeout 3 ros2 topic echo --once $COLOR_TOPIC 2>/dev/null | grep -E "height:|width:" | head -2
fi

DEPTH_TOPIC=$(ros2 topic list 2>/dev/null | grep "depth/image_raw" | head -1)
if [ -n "$DEPTH_TOPIC" ]; then
    echo ""
    echo "⚠️  检测到深度话题: $DEPTH_TOPIC"
    DEPTH_SUBS=$(ros2 topic info $DEPTH_TOPIC 2>/dev/null | grep "Subscription count")
    echo "$DEPTH_SUBS"
    if echo "$DEPTH_SUBS" | grep -q "count: [1-9]"; then
        echo "❌ 深度流有订阅者，会占用带宽！"
    fi
fi
echo ""

echo "[5] 系统资源"
echo "----------------------------------------"
echo "CPU占用前5："
ps aux --sort=-%cpu | head -6
echo ""
echo "内存占用前5："
ps aux --sort=-%mem | head -6
echo ""

echo "[6] 实时帧率测试（10秒）"
echo "----------------------------------------"
if [ -n "$COLOR_TOPIC" ]; then
    echo "正在测量 $COLOR_TOPIC 的频率..."
    timeout 10 ros2 topic hz $COLOR_TOPIC 2>&1 | tail -5
else
    echo "跳过（话题不存在）"
fi
echo ""

echo "=========================================="
echo "诊断完成"
echo "=========================================="
echo ""
echo "建议："
echo "1. 如果USB是480M → 换USB 3.0接口/线缆"
echo "2. 如果深度流开启且有订阅 → 关闭深度流"
echo "3. 如果订阅者>1 → 减少订阅者或使用rosbag录制"
echo "4. 如果帧率<25Hz → 检查是否有其他高负载进程"
