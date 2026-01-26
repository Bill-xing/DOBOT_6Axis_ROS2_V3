#!/bin/bash
# 可视化所有 episodes 的脚本

# 设置参数
REPO_ID="dobot/teleop_dataset"
ROOT_PATH="./lerobot_dataset/dobot/teleop_dataset"
MODE="local"

# 从 info.json 读取 episode 总数
INFO_FILE="${ROOT_PATH}/meta/info.json"

if [ ! -f "$INFO_FILE" ]; then
    echo "Error: Cannot find info.json at $INFO_FILE"
    exit 1
fi

# 使用 Python 读取 total_episodes
TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$INFO_FILE'))['total_episodes'])")

echo "=== Visualizing All Episodes ==="
echo "Total episodes: $TOTAL_EPISODES"
echo "Dataset root: $ROOT_PATH"
echo "Repository ID: $REPO_ID"
echo ""

# 循环显示所有 episodes
for i in $(seq 0 $((TOTAL_EPISODES - 1))); do
    echo "----------------------------------------"
    echo "Visualizing Episode $i / $((TOTAL_EPISODES - 1))"
    echo "----------------------------------------"

    python -m lerobot.scripts.visualize_dataset \
        --repo-id "$REPO_ID" \
        --root "$ROOT_PATH" \
        --mode "$MODE" \
        --episode-index $i

    # 检查用户是否想要退出
    echo ""
    read -p "Press Enter to continue to next episode, or Ctrl+C to exit..."
done

echo ""
echo "=== Finished visualizing all episodes ==="
