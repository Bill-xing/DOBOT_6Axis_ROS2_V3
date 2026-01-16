#!/usr/bin/env python3
"""
数据集对比评估工具 - 评估机械臂轨迹重放质量

功能概述：
本工具用于对比原始录制的HDF5数据集和播放后重新录制的数据集，评估播放重现的精度。
通过计算机械臂位置、姿态和夹爪位置的误差统计，判断播放是否满足要求。

主要功能：
1. load_hdf5_data: 从HDF5文件加载机械臂轨迹、夹爪轨迹、时间戳等数据
2. align_sequences: 对齐两个不同长度的数据序列（处理丢帧的情况）
3. compute_errors: 计算误差统计（平均、标准差、最大、RMSE、中位数）
4. print_error_summary: 以表格形式打印误差统计结果
5. plot_comparison: 生成对比图表，可视化原始轨迹与播放轨迹的差异
6. main: 命令行入口，协调整个评估流程

数据格式：
HDF5文件结构应包含以下字段：
  原始录制数据：
    - actions/robot_target: (N, 6) 机械臂目标位姿 [x,y,z,rx,ry,rz]
    - actions/gripper_target: (N, 1) 夹爪目标位置 [0-1000]
    - timestamp: (N,) 时间戳序列
  
  播放重新录制的数据：
    - observations/robot_current: (M, 6) 机械臂实际位姿（播放过程中采集）
    - observations/gripper_current: (M, 1) 夹爪实际位置（播放过程中采集）
    - 注：M与N可能不同（丢帧）

使用示例：
  # 基本用法：只查看误差统计
  python3 evaluate_playback.py original.hdf5 replay.hdf5
  
  # 生成并显示图表
  python3 evaluate_playback.py original.hdf5 replay.hdf5 --plot
  
  # 生成并保存图表
  python3 evaluate_playback.py original.hdf5 replay.hdf5 --plot --save-dir ./results/

误差指标说明：
  - 平均误差: 所有帧的误差均值，越小越好，反映总体准确度
  - 标准差: 误差的波动程度，越小说明重复性越好，越稳定
  - 最大误差: 最坏情况下的误差，用于安全评估（不能超过安全裕度）
  - RMSE: 均方根误差，综合考虑所有误差的大小
  - 中位数: 中间值，对极端值不敏感，反映典型情况

评估标准（参考）：
  机械臂位置：
    - 平均误差 < 5mm: 优秀
    - 平均误差 < 20mm: 良好
    - 平均误差 < 50mm: 可接受
  
  机械臂旋转：
    - 平均误差 < 1°: 优秀
    - 平均误差 < 5°: 良好
    - 平均误差 < 10°: 可接受
  
  夹爪位置：
    - 平均误差 < 20: 优秀
    - 平均误差 < 50: 良好
    - 平均误差 < 100: 可接受
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_hdf5_data(filepath):
    """
    从HDF5文件加载机械臂和夹爪数据
    
    该函数读取HDF5数据集中的以下信息：
    - robot_target: 原始录制的机械臂目标位姿 (N, 6) [x,y,z,rx,ry,rz]
    - robot_current: 播放过程中实时采集的机械臂实际位姿 (N, 6)
    - gripper_target: 原始录制的夹爪目标位置 (N, 1) [0-1000]
    - gripper_current: 播放过程中实时采集的夹爪实际位置 (N, 1)
    - timestamp: 时间戳序列，用于计算采样频率
    - attrs: 元数据（如相机频率、采样方法等）
    
    Args:
        filepath (str): HDF5文件的完整路径，例如 '/path/to/data.hdf5'
    
    Returns:
        dict: 包含读取数据的字典（字段可能不完整，取决于HDF5文件内容）
    """
    data = {}  # 初始化空字典存储读取的数据
    with h5py.File(filepath, 'r') as f:
        # 读取机械臂数据
        # robot_target: 原始录制时的目标位姿，包含所有控制指令
        if 'actions/robot_target' in f:
            data['robot_target'] = np.array(f['actions/robot_target'])
        # robot_current: 播放过程中机械臂的真实位姿，受控制延迟、精度等影响
        if 'observations/robot_current' in f:
            data['robot_current'] = np.array(f['observations/robot_current'])

        # 读取夹爪数据
        # gripper_target: 原始录制的夹爪位置指令 (0=全开，1000=全闭)
        if 'actions/gripper_target' in f:
            data['gripper_target'] = np.array(f['actions/gripper_target'])
        # gripper_current: 播放时夹爪的实际位置，反映夹爪响应情况
        if 'observations/gripper_current' in f:
            data['gripper_current'] = np.array(f['observations/gripper_current'])

        # 读取时间戳 (N,) shape，每一帧对应一个时间戳
        # 用途：计算帧率、检查数据同步、分析丢帧等
        if 'timestamp' in f:
            data['timestamp'] = np.array(f['timestamp'])

        # 读取元数据：包含录制时的配置信息
        # 例如：camera_frequency_hz (相机帧率), sync_method (同步方法)
        data['attrs'] = dict(f.attrs)

    return data  # 返回包含所有读取数据的字典

def align_sequences(seq1, seq2):
    """
    对齐两个序列长度，截断为最短序列的长度
    
    播放过程中可能出现丢帧现象（网络延迟、CPU过载等），导致某些数据采集不完整。
    该函数确保所有用于对比的序列长度一致，为统计分析做准备。
    
    参数说明：
    seq1: 第一个数据序列，通常是原始录制的目标轨迹
    seq2: 第二个数据序列，通常是播放过程中实时采集的实际轨迹
    
    返回值：
    tuple: 返回两个经过对齐的序列，均截断到较短序列的长度
           例如：seq1是100帧，seq2是95帧，则返回两个95帧的序列
    
    处理逻辑：
    1. 计算两个序列的长度，找出最小值
    2. 使用切片操作将两个序列都截断到最小长度
    3. 返回对齐后的序列对
    
    示例应用场景：
    >>> original = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])  # 4帧，完整录制
    >>> replayed = np.array([[1.1,2,3], [4,5,6], [7,8,9]])           # 3帧，由于丢帧
    >>> o_aligned, r_aligned = align_sequences(original, replayed)
    >>> print(o_aligned.shape, r_aligned.shape)
    (3, 3) (3, 3)  # 都截断为3帧，保证可以对比
    """
    # 计算两个序列的长度，找出最小值，为了对齐数据
    # 使用 len() 获取序列的第一维大小（帧数）
    min_len = min(len(seq1), len(seq2))
    
    # 将两个序列都截断到最小长度
    # seq[:min_len] 保留前 min_len 帧，丢弃超出部分
    # 这样确保两个序列完全对齐，后续可以逐帧对比
    return seq1[:min_len], seq2[:min_len]

def compute_errors(original, replay):
    """
    计算原始轨迹与重放轨迹的误差统计
    
    通过对比两个轨迹数据，计算多个误差指标来评估播放精度。
    每个指标都按维度计算（如机械臂的6个自由度分别统计），便于定位问题。
    
    参数说明：
    original: 原始录制的轨迹数据 (N, D) 其中N是帧数，D是维度（如6表示6自由度）
    replay: 播放过程中采集的实际轨迹数据 (N, D) 必须与original对齐
    
    返回值：
    dict: 包含5种误差统计的字典，每个统计量都是(D,)的数组
        - mean: 绝对误差的平均值 (各维度的平均偏差)
        - std: 误差的标准差 (各维度的波动程度/稳定性)
        - max: 误差的最大值 (各维度的最坏情况)
        - rmse: 均方根误差 (综合考虑误差大小和频率)
        - median: 误差的中位数 (典型情况，不受极端值影响)
    
    误差计算逻辑：
    1. 计算绝对误差 = |original - replay|，得到(N, D)的误差矩阵
    2. 按第0维（帧方向）统计各维度的误差分布
    3. 返回5个统计特征，用于全面评估重放质量
    
    实际应用：
    - 检测在哪些维度上误差最大 (分析最容易出现问题的环节)
    - 评估播放的稳定性 (标准差小说明稳定性好)
    - 了解最坏情况 (最大误差用于安全评估)
    """
    # 计算每一帧上original和replay的绝对误差
    # np.abs() 确保得到正数，便于后续统计
    # errors shape: (N, D) 其中N是帧数，D是数据维度
    errors = np.abs(original - replay)

    # 构建包含5个误差统计量的字典
    stats = {
        # 平均误差：所有帧的误差均值，axis=0表示沿帧方向求平均
        # 越小表示总体重放精度越高
        'mean': np.mean(errors, axis=0),
        
        # 误差标准差：波动程度，axis=0沿帧方向计算
        # 越小表示重放越稳定，重复性越好
        'std': np.std(errors, axis=0),
        
        # 误差最大值：所有帧中的最坏情况
        # 用于评估最坏情况下的精度，关乎安全性
        'max': np.max(errors, axis=0),
        
        # 均方根误差 (RMSE)：综合考虑所有误差的大小
        # 比平均误差对大误差更敏感
        'rmse': np.sqrt(np.mean(errors**2, axis=0)),
        
        # 中位数误差：中间值，对极端值不敏感
        # 反映典型情况下的误差（不受少数大异常的影响）
        'median': np.median(errors, axis=0)
    }

    return stats  # 返回包含所有5个误差统计量的字典

def print_error_summary(stats, data_name, dim_names=None):
    """
    打印格式化的误差统计摘要
    
    将计算得到的误差统计数据以人类可读的表格形式输出到控制台。
    便于快速查看各维度的误差情况，作出性能评估。
    
    参数说明：
    stats: 误差统计字典，包含 mean, std, max, rmse, median 等关键字
    data_name: 数据集名称（用于表头），例如 "机械臂位置" 或 "夹爪位置"
    dim_names: 各维度的名称列表，如 ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
              若为None，则自动生成 ['Dim0', 'Dim1', ...]
    
    输出格式：
    一个格式化的ASCII表格，展示：
    - 每个维度的名称
    - 该维度的 mean（平均误差）
    - 该维度的 std（标准差/稳定性）
    - 该维度的 max（最坏情况）
    - 该维度的 rmse（均方根误差）
    - 该维度的 median（中位数）
    最后一行显示所有维度的整体统计（对所有维度求平均）
    
    实际输出示例：
    ============================================================
    机械臂位置 误差统计
    ============================================================
    维度      | 平均    | 标准差  | 最大    | RMSE    | 中位数
    ----------------------------------------------------------
    X         | 2.45    | 1.23    | 8.90    | 2.71    | 2.10
    Y         | 1.89    | 0.98    | 7.50    | 2.15    | 1.65
    Z         | 3.12    | 1.45    | 9.80    | 3.42    | 2.85
    整体      | 2.49    | 1.22    | 8.73    | 2.76    | 2.20
    ============================================================
    """
    # 打印表格标题分隔线和数据集名称
    print(f"\n{'='*60}")
    print(f"{data_name} 误差统计")
    print(f"{'='*60}")

    # 获取数据的维度数量（从平均误差数组的长度推导）
    # 例如机械臂有6维，夹爪有1维
    n_dims = len(stats['mean'])

    # 如果没有提供维度名称，则自动生成通用名称
    if dim_names is None:
        # 生成 ['Dim0', 'Dim1', ..., 'Dim5'] 这样的通用名称
        dim_names = [f"Dim{i}" for i in range(n_dims)]

    # 打印表头：显示各列的含义
    # :<10 表示左对齐，宽度10个字符
    print(f"{'维度':<10} {'平均':<10} {'标准差':<10} {'最大':<10} {'RMSE':<10} {'中位数':<10}")
    # 分隔线，用于视觉分隔表头和数据
    print("-" * 60)

    # 逐行打印每个维度的误差统计数据
    # enumerate() 获取维度索引和名称对
    for i, name in enumerate(dim_names[:n_dims]):
        # 取出第i个维度的各项统计值，并格式化为3位小数
        # {stats['mean'][i]:<10.3f} 表示左对齐、宽度10、小数3位
        print(f"{name:<10} {stats['mean'][i]:<10.3f} {stats['std'][i]:<10.3f} "
              f"{stats['max'][i]:<10.3f} {stats['rmse'][i]:<10.3f} {stats['median'][i]:<10.3f}")

    # 分隔线，用于与总体统计分隔
    print("-" * 60)
    # 打印整体统计：对所有维度的各个指标求平均（或最大值）
    # mean: 所有维度平均误差的平均值
    # std: 所有维度标准差的平均值
    # max: 所有维度中的最大误差（最坏情况）
    # rmse: 所有维度RMSE的平均值
    # median: 所有维度中位数的平均值
    print(f"{'总体':<10} {np.mean(stats['mean']):<10.3f} {np.mean(stats['std']):<10.3f} "
          f"{np.max(stats['max']):<10.3f} {np.mean(stats['rmse']):<10.3f} {np.mean(stats['median']):<10.3f}")

def plot_comparison(original_data, replay_data, save_dir=None):
    """
    绘制原始轨迹与播放轨迹的对比图
    
    生成三组matplotlib图表，用于可视化重放效果：
    1. 机械臂轨迹对比 (6个子图，分别显示X,Y,Z,RX,RY,RZ)
    2. 夹爪轨迹对比 (1个子图)
    3. 误差时间线 (6个子图，显示各维度的误差变化)
    
    参数说明：
    original_data: 原始录制数据字典，包含 robot_target 和 gripper_target
    replay_data: 播放数据字典，包含 robot_current 和 gripper_current
    save_dir: 保存图表的目录，若为None则直接显示；若指定则保存为PNG文件
    
    输出文件：
    - robot_trajectory_comparison.png: 机械臂位置对比（6个子图）
    - gripper_trajectory_comparison.png: 夹爪位置对比（1个子图）
    - error_timeline.png: 各维度误差变化趋势（6个子图）
    
    绘制原理：
    - 使用matplotlib的subplots创建网格布局
    - 机械臂用3×2网格 (6个维度)，每个子图包含原始目标和实际轨迹曲线
    - 利用plot()在同一坐标系内绘制两条曲线，便于视觉对比
    - alpha参数控制透明度，grid()添加网格便于读值
    """

    # 提取机械臂数据
    orig_robot = original_data.get('robot_target')
    replay_robot = replay_data.get('robot_current')

    # 检查数据存在且完整，然后进行对齐
    if orig_robot is not None and replay_robot is not None:
        # 对齐两个序列到相同长度（处理丢帧情况）
        orig_robot, replay_robot = align_sequences(orig_robot, replay_robot)

        # 创建机械臂对比图：3行2列的子图网格（共6个，对应6个自由度）
        # figsize=(14, 10) 设置图表大小为14×10英寸
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('机械臂轨迹对比 (原始目标 vs 播放实际)', fontsize=14)

        # 定义6个维度的标签（单位）
        dim_names = ['X (mm)', 'Y (mm)', 'Z (mm)', 'RX (deg)', 'RY (deg)', 'RZ (deg)']

        # 遍历6个维度，每个维度占一个子图
        for i in range(6):
            # 计算当前维度对应的行列位置
            # i=0→(0,0), i=1→(0,1), i=2→(1,0), i=3→(1,1), i=4→(2,0), i=5→(2,1)
            row, col = i // 2, i % 2
            ax = axes[row, col]

            # 在同一子图上绘制原始轨迹和实际轨迹两条曲线
            # label用于图例，linewidth设置线宽，alpha控制透明度
            ax.plot(orig_robot[:, i], label='原始目标', linewidth=1.5, alpha=0.8)
            ax.plot(replay_robot[:, i], label='播放实际', linewidth=1.5, alpha=0.8)
            
            # 设置子图标题为维度名称（如 'X (mm)'）
            ax.set_title(dim_names[i])
            # 横轴标签：帧序号
            ax.set_xlabel('帧')
            # 显示图例（左上角）
            ax.legend()
            # 添加网格背景，alpha=0.3设置透明度
            ax.grid(True, alpha=0.3)

        # 调整子图间距，避免标签重叠
        plt.tight_layout()

        # 保存或显示图表
        if save_dir:
            # 若指定了保存目录，将图表保存为PNG文件
            plt.savefig(os.path.join(save_dir, 'robot_trajectory_comparison.png'), dpi=150)
            print(f"保存图表: {os.path.join(save_dir, 'robot_trajectory_comparison.png')}")
        else:
            # 若未指定保存目录，则直接在屏幕上显示
            plt.show()

    # 提取夹爪数据并进行处理
    orig_gripper = original_data.get('gripper_target')
    replay_gripper = replay_data.get('gripper_current')

    # 检查夹爪数据存在且完整，然后进行对齐
    if orig_gripper is not None and replay_gripper is not None:
        # 对齐两个序列到相同长度（处理丢帧情况）
        orig_gripper, replay_gripper = align_sequences(orig_gripper, replay_gripper)

        # 创建夹爪对比图：单个子图显示夹爪位置
        # figsize=(12, 4) 设置为宽条形图表
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # 绘制原始目标和实际位置两条曲线
        ax.plot(orig_gripper, label='原始目标', linewidth=1.5, alpha=0.8)
        ax.plot(replay_gripper, label='播放实际', linewidth=1.5, alpha=0.8)
        
        # 设置图表标题和轴标签
        ax.set_title('夹爪位置对比')
        ax.set_xlabel('帧')
        # 纵轴为夹爪位置，范围0-1000（0=全开，1000=全闭）
        ax.set_ylabel('位置 (0-1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存或显示图表
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'gripper_trajectory_comparison.png'), dpi=150)
            print(f"保存图表: {os.path.join(save_dir, 'gripper_trajectory_comparison.png')}")
        else:
            plt.show()

    # 绘制误差时间线图：显示各维度误差随时间的变化趋势
    if orig_robot is not None and replay_robot is not None:
        # 计算每一帧上机械臂各维度的绝对误差
        # errors shape: (N, 6) 其中N是帧数，6是维度数
        errors = np.abs(orig_robot - replay_robot)

        # 创建误差时间线图
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # 在同一图表上绘制6条曲线，分别代表6个维度的误差
        # enumerate() 遍历维度名称列表，每条线对应一个维度
        for i, name in enumerate(dim_names):
            # 绘制第i维的误差曲线
            # alpha=0.7 设置透明度，便于重叠曲线之间的观察
            ax.plot(errors[:, i], label=name, alpha=0.7)
        
        # 设置标题和轴标签
        ax.set_title('机械臂各维度误差')
        ax.set_xlabel('帧')
        ax.set_ylabel('绝对误差')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存或显示图表
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'robot_errors.png'), dpi=150)
            print(f"保存图表: {os.path.join(save_dir, 'robot_errors.png')}")
        else:
            plt.show()

def main():
    """
    命令行入口函数 - 协调整个评估流程
    
    功能：
    1. 解析命令行参数（输入文件路径、可选的绘图和保存参数）
    2. 验证输入文件是否存在
    3. 加载原始数据集和播放数据集
    4. 对齐两个数据集（处理丢帧）
    5. 计算机械臂和夹爪的误差统计
    6. 打印格式化的误差统计表格
    7. 根据用户选择生成并显示/保存对比图表
    
    命令行使用方式：
    基本用法 - 仅显示误差统计：
        python3 evaluate_playback.py original.hdf5 replay.hdf5
    
    生成并显示图表：
        python3 evaluate_playback.py original.hdf5 replay.hdf5 --plot
    
    生成并保存图表到指定目录：
        python3 evaluate_playback.py original.hdf5 replay.hdf5 --plot --save-dir ./results/
    
    关键变量说明：
    - args.original: 原始录制的HDF5文件路径
    - args.replay: 播放过程中重新录制的HDF5文件路径
    - args.plot: 布尔标志，是否生成图表（--plot传入时为True）
    - args.save_dir: 图表保存目录，若未指定则为None（直接显示）
    """
    # 创建命令行参数解析器
    # description 在用户调用 --help 时显示
    parser = argparse.ArgumentParser(description='对比原始数据集与播放重现数据集')
    
    # 定义必需的位置参数
    parser.add_argument(
        'original',
        type=str,
        help='原始数据集路径 (例如: ./data/episode_0.hdf5)'
    )
    parser.add_argument(
        'replay',
        type=str,
        help='播放录制数据集路径 (例如: ./data/episode_1.hdf5)'
    )
    
    # 定义可选的开关参数
    parser.add_argument(
        '--plot',
        action='store_true',  # 不需要值，有此参数则为True，否则为False
        help='绘制对比图表'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,  # 若未指定则为None
        help='保存图表的目录'
    )

    # 解析命令行参数，将其转换为args对象
    args = parser.parse_args()

    # 验证原始数据集文件是否存在
    if not os.path.exists(args.original):
        print(f"错误: 原始数据集不存在 - {args.original}")
        return  # 提前退出程序

    # 验证播放数据集文件是否存在
    if not os.path.exists(args.replay):
        print(f"错误: 播放数据集不存在 - {args.replay}")
        return  # 提前退出程序

    # 从HDF5文件加载两个数据集
    print("加载数据集...")
    original_data = load_hdf5_data(args.original)
    replay_data = load_hdf5_data(args.replay)

    # 打印原始数据集的信息
    print(f"\n原始数据集: {args.original}")
    print(f"  帧数: {len(original_data.get('robot_target', []))}")
    print(f"  元数据: {original_data.get('attrs', {})}")

    # 打印播放数据集的信息
    print(f"\n播放数据集: {args.replay}")
    print(f"  帧数: {len(replay_data.get('robot_current', []))}")
    print(f"  元数据: {replay_data.get('attrs', {})}")

    # 计算机械臂误差（如果两个数据集都包含相应数据）
    if 'robot_target' in original_data and 'robot_current' in replay_data:
        # 对齐机械臂轨迹到相同长度
        orig_robot, replay_robot = align_sequences(
            original_data['robot_target'],
            replay_data['robot_current']
        )

        # 计算误差统计（包含mean, std, max, rmse, median）
        # 注意：compute_errors返回stats字典（不是robot_errors值）
        robot_stats = compute_errors(orig_robot, replay_robot)
        
        # 定义6个维度的名称，用于表格显示
        dim_names = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
        # 打印格式化的误差统计表格
        print_error_summary(robot_stats, "机械臂位置", dim_names)

    # 计算夹爪误差（如果两个数据集都包含相应数据）
    if 'gripper_target' in original_data and 'gripper_current' in replay_data:
        # 对齐夹爪轨迹到相同长度
        orig_gripper, replay_gripper = align_sequences(
            original_data['gripper_target'],
            replay_data['gripper_current']
        )

        # 计算误差统计（包含mean, std, max, rmse, median）
        # compute_errors返回stats字典，包含5种统计指标
        gripper_stats = compute_errors(orig_gripper, replay_gripper)
        # 打印夹爪误差统计，只有1个维度（Position）
        print_error_summary(gripper_stats, "夹爪位置", ['Position'])

    # 条件绘图：仅在用户指定 --plot 参数时生成图表
    if args.plot:
        # 若指定了保存目录但目录不存在，则创建它
        if args.save_dir and not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # 生成对比图表（包括机械臂、夹爪、误差时间线）
        # 若指定了save_dir，则保存为PNG；否则直接显示
        plot_comparison(original_data, replay_data, args.save_dir)

    # 打印完成提示
    print("\n评估完成！")

if __name__ == "__main__":
    # Python脚本的标准入口：只有当脚本直接运行时才执行main()
    # 若脚本被其他模块导入，则不会执行main()
    main()
