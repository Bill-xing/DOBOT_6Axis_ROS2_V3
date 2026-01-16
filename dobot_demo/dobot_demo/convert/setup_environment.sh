#!/bin/bash
# LeRobot 转换环境快速设置脚本

set -e  # 遇到错误立即退出

echo "=================================="
echo "LeRobot Convert Environment Setup"
echo "=================================="
echo ""

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found!"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ conda found: $(conda --version)"
echo ""

# 设置环境名称
ENV_NAME="lerobot_convert"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠ Warning: Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Aborted."
        exit 0
    fi
fi

# 创建环境
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "=================================="
echo "✅ Installation Complete!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test the environment, run:"
echo "  python test_environment.py"
echo ""
echo "To start converting data, run:"
echo "  python convert_to_lerobot.py --input ./data --output ./lerobot_dataset"
echo ""
