#!/usr/bin/env python3
"""
播放质量分析工具

本模块用于分析机械臂播放轨迹时的执行质量，通过对比目标位置和实际位置，
计算多维度的跟踪误差，包括位置误差、旋转误差、夹爪误差等性能指标，
生成详细的统计分析报告和可视化图表。

核心功能：
1. 加载评估数据（evaluation_X.hdf5 格式的HDF5文件）
2. 计算跟踪误差：
   - 位置误差（笛卡尔空间 X/Y/Z 坐标）
   - 旋转误差（欧拉角表示的姿态误差）
   - 夹爪位置误差（0-1000 范围的开度误差）
3. 统计分析：计算平均值、中位数、标准差、最小值、最大值、百分位数
4. 时序分析：分析误差随时间的变化趋势，识别大误差点
5. 跟踪质量评估：综合评级（优秀/良好/中等/较差）及改进建议
6. 生成可视化图表：时序曲线、直方图、CDF分布等

使用方法：
  # 基本用法：加载数据并分析
  python3 analyze_playback.py playback_eval/evaluation_0.hdf5

  # 生成可视化图表（需要 matplotlib）
  python3 analyze_playback.py playback_eval/evaluation_0.hdf5 --plot

  # 自定义大误差阈值
  python3 analyze_playback.py playback_eval/evaluation_0.hdf5 --pos-threshold 3.0 --rot-threshold 1.5
"""

# 标准库导入：用于 HDF5 文件读写、数值计算、命令行解析、系统操作
import h5py  # HDF5 文件格式的读写库，用于加载二进制评估数据
import numpy as np  # 数值计算库，用于数组操作和统计计算
import argparse  # 命令行参数解析库，解析脚本输入参数
import sys  # 系统相关功能库，用于程序退出和输出
import os  # 操作系统接口库，用于文件路径和目录操作

def normalize_angle(angle):
    """
    将角度归一化到 [-180, 180] 度范围

    处理角度周期性问题：0° 和 360° 是同一个角度
    例如：365° -> 5°, -190° -> 170°

    Args:
        angle: 输入角度（度数）

    Returns:
        归一化后的角度，范围 [-180, 180]
    """
    # 先转换到 [0, 360) 范围
    angle = angle % 360
    # 再转换到 [-180, 180] 范围
    if angle > 180:
        angle -= 360
    return angle

def angular_difference(angle1, angle2):
    """
    计算两个角度之间的最小差值

    考虑角度的周期性，返回最短的角度距离
    例如：5° 和 355° 的差值是 10°，而不是 350°

    Args:
        angle1: 第一个角度（度数）
        angle2: 第二个角度（度数）

    Returns:
        最小角度差值，范围 [0, 180]
    """
    diff = normalize_angle(angle1 - angle2)
    return abs(diff)

class PlaybackAnalyzer:
    """
    播放质量分析器
    
    这个类负责加载、处理和分析机械臂播放评估数据。主要功能包括：
    - 从 HDF5 文件读取目标和实际的机械臂位置/旋转以及夹爪状态
    - 计算多维度的跟踪误差（位置、旋转、夹爪）
    - 执行统计分析（平均值、标准差、百分位数等）
    - 评估跟踪质量等级
    - 识别大误差事件
    - 生成可视化图表
    
    属性说明：
        hdf5_path: 评估数据文件的完整路径（HDF5 格式）
        data: 字典，存储从文件中加载的四种数据
        timestamps: 时间戳数组，记录每一帧的采样时刻
        results: 字典，存储所有计算得到的误差数据
        attrs: 字典，存储文件的元数据属性
    """

    def __init__(self, hdf5_path):
        """
        初始化播放质量分析器
        
        创建分析器实例，设置待分析数据文件的路径，初始化内部数据结构。
        此时不加载文件，只是记录文件路径供后续 load_data() 方法使用。

        Args:
            hdf5_path (str): 评估数据文件的路径，应为 HDF5 格式
                            例如："playback_eval/evaluation_0.hdf5"
        
        Attributes:
            self.hdf5_path: 保存待处理文件的路径
            self.data: 初始化为空字典，之后存储加载的机械臂和夹爪数据
            self.timestamps: 初始化为 None，之后保存时间序列数据
            self.results: 初始化为空字典，之后保存计算的各类误差数据
            self.attrs: 将在 load_data() 中初始化，保存文件元数据
        """
        self.hdf5_path = hdf5_path  # 评估数据文件的完整路径
        self.data = {}  # 字典：存储加载的原始数据（robot_target/actual、gripper_target/actual）
        self.timestamps = None  # 时间戳数组：每一帧的采样时刻（秒）
        self.results = {}  # 字典：存储计算得到的误差数据（pos_errors、rot_errors 等）

    def load_data(self):
        """
        从 HDF5 文件加载评估数据
        
        打开指定的 HDF5 文件，读取以下四类关键数据：
        1. robot/target：机械臂目标位置和姿态（6维向量：[x,y,z,rx,ry,rz]）
        2. robot/actual：机械臂实际位置和姿态（相同维度）
        3. gripper/target：夹爪目标位置（标量或一维数组，范围 0-1000）
        4. gripper/actual：夹爪实际位置（标量或一维数组）
        5. timestamp：每一帧的时间戳（浮点秒数）
        
        同时提取文件级别的元数据（如采样频率、采样时长等）。
        
        返回值说明：
            True：文件成功加载
            False：文件不存在或读取失败
        
        输出：
            打印加载进度和基本统计信息（总帧数、时长、平均频率）
        
        异常处理：
            - 如果文件不存在，返回 False 并打印错误消息
            - 如果读取出错（文件损坏等），捕获异常并返回 False
        """
        print(f"正在加载评估数据: {self.hdf5_path}\n")

        # 检查文件是否存在，避免尝试打开不存在的文件
        if not os.path.exists(self.hdf5_path):
            print(f"错误：文件不存在 {self.hdf5_path}")
            return False

        try:
            # 以读模式打开 HDF5 文件，使用 with 语句确保文件正确关闭
            with h5py.File(self.hdf5_path, 'r') as f:
                # 加载时间戳数组：每一帧对应的采样时刻
                # 转换为 numpy 数组便于后续时间序列处理
                self.timestamps = np.array(f['timestamp'])

                # 加载机械臂数据：
                # robot/target：期望的目标轨迹位置和姿态，格式 [x,y,z,rx,ry,rz]
                # 前3维为笛卡尔空间坐标（毫米），后3维为欧拉角（度数）
                self.data['robot_target'] = np.array(f['robot/target'])
                
                # robot/actual：实际执行的机械臂位置和姿态，同样格式 [x,y,z,rx,ry,rz]
                # 用于对比目标值，计算执行误差
                self.data['robot_actual'] = np.array(f['robot/actual'])

                # 加载夹爪数据：
                # gripper/target：目标夹爪开度，范围通常为 0-1000
                # 0 表示完全关闭，1000 表示完全打开
                self.data['gripper_target'] = np.array(f['gripper/target'])
                
                # gripper/actual：实际夹爪开度，同样范围 0-1000
                # 用于评估夹爪执行精度
                self.data['gripper_actual'] = np.array(f['gripper/actual'])

                # 元数据：提取文件属性，可能包含采样频率、运动时长、机械臂型号等信息
                self.attrs = dict(f.attrs)

            # 加载成功后，打印统计信息帮助用户理解数据规模
            print(f"✓ 加载完成")
            print(f"  总帧数: {len(self.timestamps)}")  # 评估中记录的总数据点数
            print(f"  总时长: {self.timestamps[-1] - self.timestamps[0]:.2f} 秒")  # 评估持续时间
            print(f"  平均频率: {len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0]):.1f} Hz")  # 采样频率
            print()

            return True

        except Exception as e:
            # 捕获任何文件读取异常（权限问题、文件损坏等）
            print(f"加载失败: {e}")
            return False

    def compute_errors(self):
        """
        计算所有维度的跟踪误差
        
        对比目标值（target）和实际值（actual），计算以下六类误差指标：
        
        1. pos_errors：笛卡尔空间位置误差（毫米）
           - 从 X/Y/Z 坐标的欧氏距离计算，单位毫米
           - 公式：sqrt((x_target-x_actual)² + (y_target-y_actual)² + (z_target-z_actual)²)
        
        2. rot_errors：欧拉角姿态误差（度数）
           - 从旋转角的欧氏距离计算，单位度数
           - 公式：sqrt((rx_target-rx_actual)² + (ry_target-ry_actual)² + (rz_target-rz_actual)²)
        
        3. x_errors/y_errors/z_errors：各轴分量误差（毫米）
           - 分别计算 X、Y、Z 三个轴向的独立误差
           - 用于分析误差在不同方向的分布
        
        4. gripper_errors：夹爪位置误差（0-1000 范围）
           - 计算目标和实际夹爪位置的绝对差值
        
        所有计算结果保存在 self.results 字典中，供后续分析使用。
        """
        print("=" * 70)
        print("计算跟踪误差")
        print("=" * 70 + "\n")

        # 提取机械臂目标位置坐标（前3列）：[x_target, y_target, z_target]
        # 单位为毫米
        pos_target = self.data['robot_target'][:, :3]
        
        # 提取机械臂实际位置坐标（前3列）：[x_actual, y_actual, z_actual]
        # 与目标位置对比，计算执行误差
        pos_actual = self.data['robot_actual'][:, :3]
        
        # 计算笛卡尔空间位置误差：每一帧的位置向量欧氏距离
        # np.linalg.norm(..., axis=1) 计算每一行（每一帧）的 L2 范数
        # 结果是一维数组，长度等于数据帧数
        pos_errors = np.linalg.norm(pos_target - pos_actual, axis=1)

        # 提取机械臂目标姿态（后3列）：[rx_target, ry_target, rz_target]
        # 使用欧拉角表示（通常单位为度数）
        rot_target = self.data['robot_target'][:, 3:]

        # 提取机械臂实际姿态（后3列）：[rx_actual, ry_actual, rz_actual]
        rot_actual = self.data['robot_actual'][:, 3:]

        # 计算欧拉角姿态误差：考虑角度周期性
        # 对每个欧拉角分量（rx, ry, rz）分别计算最小角度差
        # 然后计算三个分量的欧氏距离
        rot_diff_rx = np.array([angular_difference(t, a) for t, a in zip(rot_target[:, 0], rot_actual[:, 0])])
        rot_diff_ry = np.array([angular_difference(t, a) for t, a in zip(rot_target[:, 1], rot_actual[:, 1])])
        rot_diff_rz = np.array([angular_difference(t, a) for t, a in zip(rot_target[:, 2], rot_actual[:, 2])])

        # 组合三个分量，计算总旋转误差（欧氏距离）
        rot_errors = np.sqrt(rot_diff_rx**2 + rot_diff_ry**2 + rot_diff_rz**2)

        # 夹爪误差计算：
        # 将二维数组展平为一维（通常夹爪状态只有一个值，但防止维度不一致）
        gripper_target = self.data['gripper_target'].flatten()
        gripper_actual = self.data['gripper_actual'].flatten()
        # 计算绝对误差：|目标值 - 实际值|
        gripper_errors = np.abs(gripper_target - gripper_actual)

        # 分析位置误差的分量分布：
        # 计算目标和实际位置的差向量
        pos_diff = pos_target - pos_actual
        
        # X 轴误差：只看 X 方向的偏离量（毫米）
        x_errors = np.abs(pos_diff[:, 0])
        
        # Y 轴误差：只看 Y 方向的偏离量（毫米）
        y_errors = np.abs(pos_diff[:, 1])
        
        # Z 轴误差：只看 Z 方向的偏离量（毫米）
        z_errors = np.abs(pos_diff[:, 2])

        # 保存所有计算结果到字典，供后续统计和分析使用
        self.results = {
            'pos_errors': pos_errors,  # 综合位置误差（欧氏距离，mm）
            'rot_errors': rot_errors,  # 综合旋转误差（欧氏距离，度）
            'gripper_errors': gripper_errors,  # 夹爪误差（0-1000 范围）
            'x_errors': x_errors,  # X 轴分量误差（mm）
            'y_errors': y_errors,  # Y 轴分量误差（mm）
            'z_errors': z_errors,  # Z 轴分量误差（mm）
        }

        print("✓ 误差计算完成\n")

    def print_statistics(self):
        """
        打印误差统计信息
        
        对所有计算得到的误差数据执行统计分析，包括：
        - 算术平均值：所有误差的平均水平
        - 中位数：50% 分位数，代表典型值
        - 标准差：误差的变异程度（值越小越稳定）
        - 最小值/最大值：误差的范围
        - 95%/99% 分位数：尾部偏差，表示有多少比例的误差超过该值
        
        分组统计对象：
        1. 位置误差（整体和 X/Y/Z 分量）
        2. 旋转误差
        3. 夹爪误差
        
        输出格式统一，便于对比分析。
        """
        print("=" * 70)
        print("跟踪误差统计")
        print("=" * 70 + "\n")

        # ==================== 位置误差统计 ====================
        # 获取之前计算的综合位置误差数组
        pos_errors = self.results['pos_errors']
        
        print("[机械臂位置误差] (mm)")
        # 平均误差：所有帧的位置误差的均值，代表系统的典型精度水平
        print(f"  平均值: {np.mean(pos_errors):.3f} mm")
        
        # 中位数：50% 的误差在此值以下，较少受极值影响
        print(f"  中位数: {np.median(pos_errors):.3f} mm")
        
        # 标准差：衡量误差的波动程度，值越小表示跟踪越稳定
        print(f"  标准差: {np.std(pos_errors):.3f} mm")
        
        # 最小误差：最好的执行时刻的误差值
        print(f"  最小值: {np.min(pos_errors):.3f} mm")
        
        # 最大误差：最差的执行时刻的误差值（重要指标）
        print(f"  最大值: {np.max(pos_errors):.3f} mm")
        
        # 95% 分位数：95% 的误差在此值以下，只有 5% 超过此值（尾部偏差）
        print(f"  95%分位: {np.percentile(pos_errors, 95):.3f} mm")
        
        # 99% 分位数：99% 的误差在此值以下，只有 1% 的极端误差超过此值
        print(f"  99%分位: {np.percentile(pos_errors, 99):.3f} mm")

        # ==================== 位置分量误差 ====================
        # 分别统计 X、Y、Z 三个轴向的误差
        # 有助于识别在特定方向上的系统偏差
        print(f"\n[位置分量误差] (mm)")
        print(f"  X轴: 平均={np.mean(self.results['x_errors']):.3f}, 最大={np.max(self.results['x_errors']):.3f}")
        print(f"  Y轴: 平均={np.mean(self.results['y_errors']):.3f}, 最大={np.max(self.results['y_errors']):.3f}")
        print(f"  Z轴: 平均={np.mean(self.results['z_errors']):.3f}, 最大={np.max(self.results['z_errors']):.3f}")

        # ==================== 旋转误差统计 ====================
        # 获取之前计算的旋转误差数组
        rot_errors = self.results['rot_errors']
        
        print(f"\n[机械臂旋转误差] (deg)")
        # 平均旋转误差：欧拉角偏差的平均水平
        print(f"  平均值: {np.mean(rot_errors):.3f} deg")
        
        # 旋转中位数：典型的旋转误差值
        print(f"  中位数: {np.median(rot_errors):.3f} deg")
        
        # 旋转标准差：旋转稳定性指标
        print(f"  标准差: {np.std(rot_errors):.3f} deg")
        
        print(f"  最小值: {np.min(rot_errors):.3f} deg")
        print(f"  最大值: {np.max(rot_errors):.3f} deg")
        print(f"  95%分位: {np.percentile(rot_errors, 95):.3f} deg")

        # ==================== 夹爪误差统计 ====================
        # 获取之前计算的夹爪误差数组
        gripper_errors = self.results['gripper_errors']
        
        print(f"\n[夹爪位置误差] (0-1000)")
        # 夹爪开度的平均误差（0-1000 的绝对值）
        print(f"  平均值: {np.mean(gripper_errors):.3f}")
        
        # 夹爪开度的中位误差
        print(f"  中位数: {np.median(gripper_errors):.3f}")
        
        # 夹爪开度的误差波动
        print(f"  标准差: {np.std(gripper_errors):.3f}")
        
        print(f"  最小值: {np.min(gripper_errors):.3f}")
        print(f"  最大值: {np.max(gripper_errors):.3f}")
        print(f"  95%分位: {np.percentile(gripper_errors, 95):.3f}")

        print()

    def analyze_tracking_quality(self):
        """
        评估整体跟踪质量等级
        
        基于误差指标的统计值，对机械臂的播放执行能力进行综合评估：
        
        评估维度：
        1. 位置跟踪质量：按误差范围分为优秀/良好/较差三个等级
           - 优秀：< 1.0 mm，适合高精度任务
           - 良好：< 5.0 mm，适合通用工业应用
           - 较差：>= 5.0 mm，需要参数优化
        
        2. 旋转跟踪质量：按欧拉角误差分为三个等级
           - 优秀：< 0.5°，高精度姿态控制
           - 良好：< 2.0°，通用应用
           - 较差：>= 2.0°，需要改进
        
        3. 总体评级：综合位置和旋转误差，给出四级评分
           - 优秀：平均位置误差 < 1.0 mm 且 平均旋转误差 < 0.5°
           - 良好：平均位置误差 < 3.0 mm 且 平均旋转误差 < 1.5°
           - 中等：平均位置误差 < 5.0 mm 且 平均旋转误差 < 2.0°
           - 较差：超过中等标准，需要系统检查
        
        同时输出具体建议，帮助用户改进性能。
        """
        print("=" * 70)
        print("跟踪质量评估")
        print("=" * 70 + "\n")

        # 获取误差数据
        pos_errors = self.results['pos_errors']
        rot_errors = self.results['rot_errors']

        # ==================== 位置跟踪质量评估 ====================
        # 统计优秀级别（< 1.0 mm）的帧占比
        # 这类帧的执行精度高，适合高精度任务
        pos_good = np.sum(pos_errors < 1.0) / len(pos_errors) * 100
        
        # 统计良好级别（< 5.0 mm）的帧占比
        # 累积到此的帧可满足大多数工业应用需求
        pos_acceptable = np.sum(pos_errors < 5.0) / len(pos_errors) * 100
        
        # 统计较差级别（>= 5.0 mm）的帧占比
        # 表示有多少比例的帧执行精度不够理想
        pos_poor = np.sum(pos_errors >= 5.0) / len(pos_errors) * 100

        print("[位置跟踪质量]")
        # 显示优秀帧的百分比和绝对数量
        print(f"  优秀 (<1mm):   {pos_good:.1f}% ({int(pos_good * len(pos_errors) / 100)} 帧)")
        # 显示良好帧的百分比（累积，包括优秀）
        print(f"  良好 (<5mm):   {pos_acceptable:.1f}% ({int(pos_acceptable * len(pos_errors) / 100)} 帧)")
        # 显示较差帧的百分比
        print(f"  较差 (>=5mm):  {pos_poor:.1f}% ({int(pos_poor * len(pos_errors) / 100)} 帧)")

        # ==================== 旋转跟踪质量评估 ====================
        # 统计旋转优秀级别（< 0.5°）的帧占比
        rot_good = np.sum(rot_errors < 0.5) / len(rot_errors) * 100
        
        # 统计旋转良好级别（< 2.0°）的帧占比
        rot_acceptable = np.sum(rot_errors < 2.0) / len(rot_errors) * 100
        
        # 统计旋转较差级别（>= 2.0°）的帧占比
        rot_poor = np.sum(rot_errors >= 2.0) / len(rot_errors) * 100

        print(f"\n[旋转跟踪质量]")
        print(f"  优秀 (<0.5°):  {rot_good:.1f}% ({int(rot_good * len(rot_errors) / 100)} 帧)")
        print(f"  良好 (<2°):    {rot_acceptable:.1f}% ({int(rot_acceptable * len(rot_errors) / 100)} 帧)")
        print(f"  较差 (>=2°):   {rot_poor:.1f}% ({int(rot_poor * len(rot_errors) / 100)} 帧)")

        # ==================== 总体评级逻辑 ====================
        # 根据平均误差确定综合评级等级
        # 评级从高到低：优秀 -> 良好 -> 中等 -> 较差
        
        print(f"\n[总体评级]")
        # 最高等级：位置和旋转误差都很低，适合精密应用
        if np.mean(pos_errors) < 1.0 and np.mean(rot_errors) < 0.5:
            rating = "优秀"  # 评级标签
            comment = "跟踪精度非常高，适合高精度任务"  # 应用建议
        # 次高等级：误差适中，满足大多数工业应用
        elif np.mean(pos_errors) < 3.0 and np.mean(rot_errors) < 1.5:
            rating = "良好"
            comment = "跟踪精度良好，适合大部分任务"
        # 中等等级：误差在可接受范围，可能需要参数调优
        elif np.mean(pos_errors) < 5.0 and np.mean(rot_errors) < 2.0:
            rating = "中等"
            comment = "跟踪精度一般，建议优化控制参数"
        # 最低等级：误差较大，需要深入调查和改进
        else:
            rating = "较差"
            comment = "跟踪精度不足，需要检查系统配置"

        # 输出评级和建议
        print(f"  评级: {rating}")
        print(f"  建议: {comment}")
        print()

    def find_large_errors(self, pos_threshold=5.0, rot_threshold=2.0):
        """
        查找并报告大误差发生的时刻
        
        系统地搜索超过预设阈值的误差点，帮助用户定位问题发生在执行序列的哪些时刻。
        这对于识别轨迹中的难点和调试控制参数非常有帮助。
        
        Args:
            pos_threshold (float): 位置误差阈值，单位毫米（默认 5.0 mm）
                                 超过此值的帧被视为大误差
            rot_threshold (float): 旋转误差阈值，单位度数（默认 2.0°）
                                 超过此值的帧被视为大误差
        
        输出：
            - 位置误差 > pos_threshold 的帧列表
            - 旋转误差 > rot_threshold 的帧列表
            - 每个大误差点的帧号、误差值、时间戳等详细信息
        
        注意：
            如果大误差点较多（超过5个），只显示前5个，以避免输出过长。
        """
        print("=" * 70)
        print("大误差分析")
        print("=" * 70 + "\n")

        # 获取之前计算的误差数据
        pos_errors = self.results['pos_errors']
        rot_errors = self.results['rot_errors']

        # ==================== 查找位置大误差 ====================
        # 找出所有位置误差超过阈值的帧的索引
        # np.where() 返回满足条件的索引数组
        large_pos_idx = np.where(pos_errors > pos_threshold)[0]
        
        print(f"[位置误差 >{pos_threshold}mm 的帧]")
        
        if len(large_pos_idx) > 0:
            # 发现了大误差点
            print(f"  发现 {len(large_pos_idx)} 个大误差点")
            print(f"  前5个位置:")
            
            # 遍历前 5 个大误差点（或全部，如果少于 5 个）
            for i, idx in enumerate(large_pos_idx[:5]):
                # idx：帧号
                # pos_errors[idx]：该帧的位置误差值（mm）
                # self.timestamps[idx] - self.timestamps[0]：相对于序列开始的时刻（秒）
                print(f"    #{i+1} 帧{idx}: 误差={pos_errors[idx]:.2f}mm, "
                      f"时间={self.timestamps[idx]-self.timestamps[0]:.2f}s")
        else:
            # 未发现大误差，说明位置跟踪质量很好
            print(f"  ✓ 未发现位置大误差")

        # ==================== 查找旋转大误差 ====================
        # 同样查找旋转误差超过阈值的帧
        large_rot_idx = np.where(rot_errors > rot_threshold)[0]
        
        print(f"\n[旋转误差 >{rot_threshold}° 的帧]")
        
        if len(large_rot_idx) > 0:
            # 发现了旋转大误差点
            print(f"  发现 {len(large_rot_idx)} 个大误差点")
            print(f"  前5个位置:")
            
            # 遍历前 5 个大误差点
            for i, idx in enumerate(large_rot_idx[:5]):
                # idx：帧号
                # rot_errors[idx]：该帧的旋转误差值（度）
                # self.timestamps[idx] - self.timestamps[0]：相对时刻
                print(f"    #{i+1} 帧{idx}: 误差={rot_errors[idx]:.2f}°, "
                      f"时间={self.timestamps[idx]-self.timestamps[0]:.2f}s")
        else:
            # 未发现大误差
            print(f"  ✓ 未发现旋转大误差")

        print()

    def plot_results(self, output_dir=None):
        """
        生成详细的可视化分析图表
        
        使用 matplotlib 创建一组 6 个子图，全面展示跟踪误差的分布和趋势：
        
        图表 1 (左上)：位置误差时序曲线
            - X 轴：时间（秒）
            - Y 轴：位置误差（毫米）
            - 显示平均值和 95% 分位数参考线
            - 有助于观察误差是否存在时间趋势
        
        图表 2 (右上)：位置误差分布直方图
            - 显示误差值的频率分布
            - 可识别误差是否集中在某个范围
            - 平均值参考线指示误差中心
        
        图表 3 (左中)：旋转误差时序曲线
            - 与位置误差类似，展示旋转误差随时间的变化
            - 帮助识别旋转控制的稳定性
        
        图表 4 (右中)：夹爪误差时序曲线
            - 展示夹爪执行精度
            - 与机械臂位姿误差一起分析
        
        图表 5 (左下)：XYZ 分量误差对比
            - 三条曲线分别表示 X、Y、Z 轴的误差分量
            - 用于识别某个特定方向上的系统偏差
            - 例如，若 Z 轴误差明显大于 X/Y，可能存在垂直方向的控制问题
        
        图表 6 (右下)：位置误差累积分布函数 (CDF)
            - 显示有多少比例的数据点在各个误差值以下
            - X 轴为误差值，Y 轴为百分比
            - 95% 分位数的垂直线帮助快速判断尾部偏差
        
        Args:
            output_dir (str, optional): 图表保存目录
                                      默认为评估数据文件所在的目录
                                      输出文件名为：evaluation_X_analysis.png
        
        依赖：
            需要安装 matplotlib 库，可用 pip install matplotlib 安装
        
        输出：
            在指定目录生成名为 {basename}_analysis.png 的图像文件
        """
        try:
            # 尝试导入 matplotlib，如果未安装会抛出异常
            import matplotlib.pyplot as plt
        except ImportError:
            # 用户未安装 matplotlib，提示安装方法
            print("错误：需要安装 matplotlib 来生成图表")
            print("安装命令: pip install matplotlib")
            return

        # 设置输出目录，默认使用输入文件所在的目录
        if output_dir is None:
            output_dir = os.path.dirname(self.hdf5_path)

        print("=" * 70)
        print("生成可视化图表")
        print("=" * 70 + "\n")

        # 计算相对时间序列：以序列开始时刻为零点，单位秒
        # 便于理解哪些时刻出现大误差
        time_series = self.timestamps - self.timestamps[0]

        # 创建 3 行 2 列的子图布局，总共 6 个图表
        # figsize=(15, 12) 指定整体图表大小
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        # 设置整体标题
        fig.suptitle('播放质量评估', fontsize=16, fontweight='bold')

        # ==================== 图表 1：位置误差时序 ====================
        ax = axes[0, 0]  # 第 1 行第 1 列
        # 绘制位置误差随时间的变化曲线（蓝色，透明度 0.7）
        ax.plot(time_series, self.results['pos_errors'], 'b-', linewidth=0.5, alpha=0.7)
        # 添加平均值参考线（红色虚线）
        ax.axhline(y=np.mean(self.results['pos_errors']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(self.results["pos_errors"]):.2f}mm')
        # 添加 95% 分位数参考线（橙色虚线），表示有 5% 的误差超过此值
        ax.axhline(y=np.percentile(self.results['pos_errors'], 95), color='orange', linestyle='--',
                   label=f'95%: {np.percentile(self.results["pos_errors"], 95):.2f}mm')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (mm)')
        ax.set_title('Position Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ==================== 图表 2：位置误差直方图 ====================
        ax = axes[0, 1]  # 第 1 行第 2 列
        # 绘制位置误差的频率分布，分为 50 个柱子
        ax.hist(self.results['pos_errors'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        # 显示平均值
        ax.axvline(x=np.mean(self.results['pos_errors']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(self.results["pos_errors"]):.2f}mm')
        ax.set_xlabel('Position Error (mm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Position Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ==================== 图表 3：旋转误差时序 ====================
        ax = axes[1, 0]  # 第 2 行第 1 列
        # 绘制旋转误差随时间的变化（绿色）
        ax.plot(time_series, self.results['rot_errors'], 'g-', linewidth=0.5, alpha=0.7)
        # 显示平均旋转误差
        ax.axhline(y=np.mean(self.results['rot_errors']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(self.results["rot_errors"]):.2f}°')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rotation Error (deg)')
        ax.set_title('Rotation Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ==================== 图表 4：夹爪误差时序 ====================
        ax = axes[1, 1]  # 第 2 行第 2 列
        # 绘制夹爪误差时序（品红色）
        ax.plot(time_series, self.results['gripper_errors'], 'm-', linewidth=0.5, alpha=0.7)
        # 显示平均夹爪误差
        ax.axhline(y=np.mean(self.results['gripper_errors']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(self.results["gripper_errors"]):.2f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gripper Error')
        ax.set_title('Gripper Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ==================== 图表 5：XYZ 分量误差对比 ====================
        ax = axes[2, 0]  # 第 3 行第 1 列
        # 用三条曲线分别显示 X、Y、Z 方向的位置误差
        # 有助于识别系统在某个特定方向上的偏差趋势
        ax.plot(time_series, self.results['x_errors'], 'r-', linewidth=0.5, alpha=0.6, label='X')
        ax.plot(time_series, self.results['y_errors'], 'g-', linewidth=0.5, alpha=0.6, label='Y')
        ax.plot(time_series, self.results['z_errors'], 'b-', linewidth=0.5, alpha=0.6, label='Z')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Component Error (mm)')
        ax.set_title('Position Error Components (X, Y, Z)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ==================== 图表 6：误差累积分布函数 (CDF) ====================
        ax = axes[2, 1]  # 第 3 行第 2 列
        # 对位置误差进行排序，用于绘制 CDF
        sorted_pos_errors = np.sort(self.results['pos_errors'])
        # 计算累积百分比：第 i 个点表示有 i/total*100% 的数据点误差不超过该值
        cdf = np.arange(1, len(sorted_pos_errors) + 1) / len(sorted_pos_errors) * 100
        # 绘制 CDF 曲线
        ax.plot(sorted_pos_errors, cdf, 'b-', linewidth=2)
        # 显示 95% 分位数位置
        ax.axvline(x=np.percentile(self.results['pos_errors'], 95), color='orange',
                   linestyle='--', label='95% percentile')
        # 添加水平参考线表示 95% 点
        ax.axhline(y=95, color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('Position Error (mm)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Position Error CDF')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 调整子图间距，避免标签重叠
        plt.tight_layout()

        # 生成输出文件名：原文件名 + '_analysis.png'
        base_name = os.path.splitext(os.path.basename(self.hdf5_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_analysis.png")
        
        # 保存图表到文件
        # dpi=150：分辨率 150 像素/英寸，平衡清晰度和文件大小
        # bbox_inches='tight'：自动调整边界，避免标签被裁剪
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存到: {output_path}")

        # 可选：显示图表窗口（如果需要交互查看，取消注释）
        # plt.show()

    def run(self, generate_plot=False):
        """
        执行完整的分析流程
        
        按顺序执行以下步骤：
        1. 加载 HDF5 评估数据文件
        2. 计算各类跟踪误差
        3. 打印统计分析结果（平均值、标准差等）
        4. 评估整体跟踪质量等级
        5. 查找大误差点及其发生时刻
        6. （可选）生成可视化图表
        
        Args:
            generate_plot (bool): 是否生成 matplotlib 图表
                                 默认 False，仅生成文本报告
                                 若为 True，需要安装 matplotlib
        
        Returns:
            bool: 分析成功返回 True，失败返回 False
                 失败通常因为数据文件不存在或格式错误
        
        输出：
            在控制台打印详细的分析报告
            若 generate_plot=True，还会保存图表文件
        """
        print("=" * 70)
        print("播放质量分析")
        print("=" * 70 + "\n")

        # 步骤 1：加载评估数据
        # 如果加载失败（文件不存在等），直接返回 False 停止分析
        if not self.load_data():
            return False

        # 步骤 2：计算所有维度的跟踪误差
        self.compute_errors()

        # 步骤 3：打印详细的统计信息
        self.print_statistics()

        # 步骤 4：进行质量评估和评级
        self.analyze_tracking_quality()

        # 步骤 5：查找大误差发生的位置
        self.find_large_errors()

        # 步骤 6：（可选）生成可视化图表
        if generate_plot:
            self.plot_results()

        # 打印结束标记
        print("=" * 70)
        print("分析完成")
        print("=" * 70)

        return True

def main():
    """
    主程序入口
    
    解析命令行参数，创建 PlaybackAnalyzer 实例，运行分析流程。
    
    命令行参数：
        evaluation_file (positional): 必需，评估数据文件路径
                                     格式应为 HDF5（.hdf5 或 .h5 扩展名）
                                     示例：playback_eval/evaluation_0.hdf5
        
        --plot: 可选标志，是否生成可视化图表
               添加此标志会调用 matplotlib 生成 6 个子图的分析图表
               示例：--plot
        
        --pos-threshold: 可选参数，位置大误差阈值（毫米），默认 5.0
                        超过此值的帧会被标记为大误差
                        示例：--pos-threshold 3.0
        
        --rot-threshold: 可选参数，旋转大误差阈值（度数），默认 2.0
                        超过此值的帧会被标记为大误差
                        示例：--rot-threshold 1.5
    
    使用示例：
        # 基本用法
        python3 analyze_playback.py playback_eval/evaluation_0.hdf5
        
        # 生成图表
        python3 analyze_playback.py playback_eval/evaluation_0.hdf5 --plot
        
        # 自定义阈值
        python3 analyze_playback.py playback_eval/evaluation_0.hdf5 --pos-threshold 3.0 --rot-threshold 1.5
        
        # 综合示例
        python3 analyze_playback.py playback_eval/evaluation_0.hdf5 --plot --pos-threshold 2.0
    
    退出代码：
        0：分析成功完成
        1：分析失败（通常因为文件问题）
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='分析播放评估数据')
    
    # 位置参数：评估数据文件路径（必需）
    parser.add_argument('evaluation_file', type=str,
                       help='评估数据文件路径 (例如: playback_eval/evaluation_0.hdf5)')
    
    # 可选标志：是否生成图表
    parser.add_argument('--plot', action='store_true',
                       help='生成可视化图表')
    
    # 可选参数：位置误差大值阈值
    parser.add_argument('--pos-threshold', type=float, default=5.0,
                       help='位置大误差阈值(mm)，默认5.0')
    
    # 可选参数：旋转误差大值阈值
    parser.add_argument('--rot-threshold', type=float, default=2.0,
                       help='旋转大误差阈值(度)，默认2.0')

    # 解析用户输入的命令行参数
    args = parser.parse_args()

    # 创建分析器实例，传入评估数据文件路径
    analyzer = PlaybackAnalyzer(args.evaluation_file)

    # 运行分析流程
    # 注意：find_large_errors() 方法中的阈值参数暂未在此处使用
    # 实际调用中会使用默认值，若需自定义需要修改代码
    success = analyzer.run(generate_plot=args.plot)

    # 根据分析结果返回适当的退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    """
    程序入口点
    
    当这个脚本被直接执行（而不是被导入为模块）时，执行 main() 函数。
    这是 Python 脚本的标准做法，确保仅在直接运行时执行主程序逻辑，
    而在被其他代码导入时不会自动执行。
    
    执行方式：
        python3 analyze_playback.py <args>
    """
    main()
