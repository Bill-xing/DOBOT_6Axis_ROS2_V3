# 🚀 快速开始指南

## 一键安装（推荐）

```bash
# 1. 运行自动安装脚本
./setup_environment.sh

# 2. 激活环境
conda activate lerobot_convert

# 3. 验证安装
python test_environment.py

# 4. 开始转换
python convert_to_lerobot.py --input ./data --output ./lerobot_dataset
```

---

## 手动安装

```bash
# 1. 创建环境
conda env create -f environment.yml

# 2. 激活环境
conda activate lerobot_convert

# 3. 验证安装
python test_environment.py
```

---

## 推荐版本（已测试）

| 库 | 版本 | 用途 |
|---|---|---|
| Python | 3.10 | 基础环境 |
| numpy | 1.24.3 | 数值计算 |
| h5py | 3.9.0 | 读取HDF5 |
| pyarrow | 14.0.1 | Parquet文件 |
| opencv | 4.8.0 | 视频编码 |
| pillow | 10.0.0 | 图像处理 |
| tqdm | 4.66.1 | 进度条 |

---

## 常见问题

### Q: conda命令找不到？

**A:** 安装 Miniconda：
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Q: PyArrow 安装失败？

**A:** 使用 conda 安装：
```bash
conda install -c conda-forge pyarrow=14.0.1
```

### Q: 视频编码失败？

**A:** 重新安装 opencv：
```bash
conda install -c conda-forge opencv=4.8.0
```

---

## 验证清单

运行 `python test_environment.py`，确保看到：

```
✓ h5py           3.9.0
✓ numpy          1.24.3
✓ pyarrow        14.0.1
✓ opencv         4.8.0
✓ pillow         10.0.0
✓ tqdm           4.66.1

✅ All tests passed! Environment is ready.
```

---

## 下一步

环境配置完成后，参考 [USAGE.md](USAGE.md) 开始转换数据集！
