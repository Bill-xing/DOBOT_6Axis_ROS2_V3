#!/usr/bin/env python3
"""
HDF5 Robot Data Visualizer - 机器人数据可视化工具

功能：
1. 读取 HDF5 格式的机器人录制数据
2. 生成带有机器人状态仪表盘的视频（RGB + 深度图）
3. 验证时间戳对齐质量和数据完整性

主要特点：
- 支持高质量视频编码（使用 ffmpeg）
- 实时显示机器人位姿、夹爪状态、位置误差
- 时间戳对齐质量验证和丢帧检测
- 支持 RGB 和深度图双视频输出
  # 1. 查看帮助
  python show_vla.py --help

  # 2. 生成高质量视频（推荐）
  python show_vla.py data/episode_0.hdf5

  # 3. 仅验证时间戳质量
  python show_vla.py data/episode_0.hdf5 --verify-only
"""

import h5py
import cv2
import numpy as np
import argparse
import os
import subprocess
import tempfile
import shutil
from tqdm import tqdm

def draw_text_with_bg(img, text, position, font_scale=0.6, thickness=1, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    在图像上绘制带半透明背景的文字（提高可读性）

    参数：
        img: 目标图像（会被直接修改）
        text: 要绘制的文字内容
        position: 文字位置 (x, y)，左下角坐标
        font_scale: 字体缩放比例，默认 0.6
        thickness: 字体粗细，默认 1
        text_color: 文字颜色 BGR 格式，默认白色
        bg_color: 背景颜色 BGR 格式，默认黑色
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    # 边界检测：防止文字超出图像范围
    h, w = img.shape[:2]
    if x + text_w > w: x = w - text_w
    if y - text_h < 0: y = text_h + 5

    # 提取文字区域并创建半透明背景
    sub_img = img[y-text_h-baseline:y+baseline, x:x+text_w]
    if sub_img.size == 0: return

    # 半透明背景效果（原图50% + 背景色50%）
    white_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    img[y-text_h-baseline:y+baseline, x:x+text_w] = res

    # 绘制抗锯齿文字
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_dashboard(img, robot_current, robot_target, gripper_current, gripper_target):
    """
    在图像上绘制机器人状态仪表盘

    显示内容：
    1. 机器人当前位置（绿色）vs 目标位置（橙色）
    2. 机器人当前姿态（绿色）vs 目标姿态（橙色）
    3. 位置误差（黄色）- 欧几里得距离
    4. 夹爪当前开度进度条（绿色渐变）
    5. 夹爪目标开度进度条（橙色渐变）

    参数：
        img: 目标图像（会被直接修改）
        robot_current: [x, y, z, rx, ry, rz] 当前位姿 (mm和度)
        robot_target: [x, y, z, rx, ry, rz] 目标位姿 (mm和度)
        gripper_current: [pos] 夹爪当前开度 (0-1000)
        gripper_target: [pos] 夹爪目标开度 (0-1000)
    """
    h, w = img.shape[:2]
    y_start = 30
    line_height = 25

    # ==== 显示机械臂位置 (X, Y, Z) ====
    # 当前位置（绿色）
    pos_curr_str = f"Pos(Curr): X:{robot_current[0]:.0f} Y:{robot_current[1]:.0f} Z:{robot_current[2]:.0f}"
    draw_text_with_bg(img, pos_curr_str, (10, y_start), text_color=(0, 255, 0))

    # 目标位置（橙色）
    pos_targ_str = f"Pos(Targ): X:{robot_target[0]:.0f} Y:{robot_target[1]:.0f} Z:{robot_target[2]:.0f}"
    draw_text_with_bg(img, pos_targ_str, (10, y_start + line_height), text_color=(255, 165, 0))

    # 位置误差（黄色）- 计算3D欧几里得距离
    pos_error = np.linalg.norm(robot_current[:3] - robot_target[:3])
    error_str = f"Pos Error: {pos_error:.2f}mm"
    draw_text_with_bg(img, error_str, (10, y_start + line_height * 2), text_color=(255, 255, 0))

    # ==== 显示机械臂姿态 (RX, RY, RZ) ====
    # 当前姿态（绿色）
    rot_curr_str = f"Rot(Curr): RX:{robot_current[3]:.1f} RY:{robot_current[4]:.1f} RZ:{robot_current[5]:.1f}"
    draw_text_with_bg(img, rot_curr_str, (10, y_start + line_height * 3), text_color=(0, 255, 0))

    # 目标姿态（橙色）
    rot_targ_str = f"Rot(Targ): RX:{robot_target[3]:.1f} RY:{robot_target[4]:.1f} RZ:{robot_target[5]:.1f}"
    draw_text_with_bg(img, rot_targ_str, (10, y_start + line_height * 4), text_color=(255, 165, 0))

    # ==== 夹爪状态进度条（双层显示）====
    bar_h = 20  # 进度条高度
    bar_w = int(w * 0.7)  # 进度条宽度（屏幕宽度的70%）
    bar_x = int((w - bar_w) / 2)  # 居中显示
    bar_y = h - 70  # 距离底部70像素

    # --- 夹爪当前开度进度条 ---
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)  # 灰色背景
    val_curr = float(gripper_current[0])
    ratio_curr = max(0.0, min(1.0, val_curr / 1000.0))  # 归一化到 [0, 1]
    fill_w_curr = int(bar_w * ratio_curr)
    # 渐变色：从蓝色（关闭）到绿色（打开）
    color_curr = (0, int(255 * ratio_curr), int(255 * (1 - ratio_curr)))
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w_curr, bar_y + bar_h), color_curr, -1)
    label_curr = f"Gripper Curr: {int(val_curr)}"
    draw_text_with_bg(img, label_curr, (bar_x, bar_y - 10), font_scale=0.5, text_color=(0, 255, 0))

    # --- 夹爪目标开度进度条 ---
    bar_y_target = bar_y + bar_h + 10  # 第二个进度条位置
    cv2.rectangle(img, (bar_x, bar_y_target), (bar_x + bar_w, bar_y_target + bar_h), (50, 50, 50), -1)
    val_targ = float(gripper_target[0])
    ratio_targ = max(0.0, min(1.0, val_targ / 1000.0))
    fill_w_targ = int(bar_w * ratio_targ)
    color_targ = (0, int(255 * ratio_targ), int(255 * (1 - ratio_targ)))
    cv2.rectangle(img, (bar_x, bar_y_target), (bar_x + fill_w_targ, bar_y_target + bar_h), color_targ, -1)
    label_targ = f"Gripper Targ: {int(val_targ)}"
    draw_text_with_bg(img, label_targ, (bar_x, bar_y_target - 10), font_scale=0.5, text_color=(255, 165, 0))

def process_depth_image(depth_raw):
    """
    处理深度图并转换为可视化的彩色图

    处理流程：
    1. 过滤无效深度值（depth=0）
    2. 归一化到 0-255 范围
    3. 应用 JET 伪彩色映射（蓝色=近，红色=远）

    参数：
        depth_raw: 原始深度图 (H, W) uint16, 单位mm

    返回：
        depth_color: 彩色深度图 (H, W, 3) uint8 BGR格式
    """
    # 创建掩码：过滤无效深度值（depth=0）
    mask = depth_raw > 0
    if np.sum(mask) == 0:  # 如果没有有效深度值，返回黑色图像
        return np.zeros((*depth_raw.shape, 3), dtype=np.uint8)

    # 计算有效深度值的范围
    min_val = np.min(depth_raw[mask])
    max_val = np.max(depth_raw[mask])

    # 归一化到 0-255 范围
    depth_norm = np.zeros_like(depth_raw, dtype=np.float32)
    if max_val > min_val:
        depth_norm[mask] = (depth_raw[mask] - min_val) / (max_val - min_val) * 255

    depth_norm = depth_norm.astype(np.uint8)

    # 应用 JET 颜色映射（蓝色=近，红色=远）
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return depth_color

def prepare_frame(frame, is_rgb_source=True):
    """
    数据清洗和预处理 - 统一图像格式

    处理步骤：
    1. 转换维度：Channel-First (C,H,W) -> Channel-Last (H,W,C)
    2. 数据类型：确保 uint8 格式
    3. 颜色空间：RGB -> BGR（仅对彩色图像，OpenCV标准）
    4. 内存布局：确保内存连续（提高性能）

    参数：
        frame: 输入图像数组
        is_rgb_source: 是否为RGB图像源（True表示需要RGB->BGR转换）

    返回：
        frame: 处理后的图像 (H, W, C) uint8 BGR格式（彩色图）或 (H, W) uint8（灰度图/深度图）
    """
    # 1. 处理维度转换：Channel-First (C, H, W) -> Channel-Last (H, W, C)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    elif frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0]  # 深度图：(1, H, W) -> (H, W)

    # 2. 确保数据类型为 uint8
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # 3. 颜色空间转换：RGB -> BGR（仅针对彩色图像）
    if is_rgb_source and frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 4. 确保内存连续（提高OpenCV性能）
    frame = np.ascontiguousarray(frame)

    return frame

def encode_video_with_ffmpeg(frame_dir, output_path, fps=30, quality='high'):
    """
    使用 ffmpeg 命令行工具编码高质量视频

    质量选项（quality参数）：
    - 'lossless': H.264 无损编码（最大文件，CRF=0）
    - 'high': 高质量编码（推荐，CRF=17，视觉无损）
    - 'medium': 中等质量编码（CRF=23，平衡质量和文件大小）

    CRF (Constant Rate Factor) 说明：
    - 0 = 无损，18 = 视觉无损，23 = 默认，51 = 最差质量
    - CRF 越低，质量越高，文件越大

    参数：
        frame_dir: 存储帧的临时目录路径（PNG格式）
        output_path: 输出视频文件路径（.mp4）
        fps: 视频帧率，默认30
        quality: 质量档位 {'lossless', 'high', 'medium'}

    返回：
        bool: 编码成功返回True，失败返回False

    依赖：
        需要系统安装 ffmpeg：
        - Ubuntu/Debian: sudo apt-get install ffmpeg
        - CentOS/RHEL: sudo yum install ffmpeg
    """
    # 检查 ffmpeg 是否已安装并可用
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  CentOS/RHEL: sudo yum install ffmpeg")
        return False

    # 根据质量档位选择编码参数
    if quality == 'lossless':
        # 无损 H.264 编码（CRF=0，最大文件）
        codec_params = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '0']
    elif quality == 'high':
        # 高质量编码（CRF=17，视觉无损，推荐）
        codec_params = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '17']
    else:
        # 中等质量编码（CRF=23，平衡质量和大小）
        codec_params = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23']

    # 构建 ffmpeg 命令
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),  # 输入帧率
        '-i', os.path.join(frame_dir, 'frame_%06d.png'),  # 输入图像序列（frame_000001.png, frame_000002.png, ...）
    ] + codec_params + [
        '-pix_fmt', 'yuv420p',  # 像素格式（确保兼容性）
        '-y',  # 覆盖已存在的输出文件
        output_path
    ]

    try:
        # 执行 ffmpeg 命令
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg encoding failed: {e.stderr.decode()}")
        return False

def verify_timestamps(hdf5_path):
    """
    验证时间戳对齐质量和数据完整性

    检查项目：
    1. 同步方法元数据（sync_method, sync_tolerance_ms等）
    2. 数据集时长和平均帧率
    3. 帧间时间间隔统计（均值、标准差、最小值、最大值）
    4. 丢帧检测（时间间隔超过正常值1.5倍）

    参数：
        hdf5_path: HDF5文件路径

    输出：
        打印详细的时间戳统计报告到控制台
    """
    print("\n" + "="*60)
    print("TIMESTAMP ALIGNMENT VERIFICATION")
    print("="*60)

    with h5py.File(hdf5_path, 'r') as f:
        # 检查同步方法元数据
        if 'sync_method' in f.attrs:
            print(f"Sync Method: {f.attrs['sync_method']}")
            if 'sync_tolerance_ms' in f.attrs:
                print(f"Sync Tolerance: {f.attrs['sync_tolerance_ms']} ms")
            if 'camera_frequency_hz' in f.attrs:
                print(f"Camera Frequency: {f.attrs['camera_frequency_hz']} Hz")
            if 'robot_frequency_hz' in f.attrs:
                print(f"Robot Frequency: {f.attrs['robot_frequency_hz']} Hz")
        else:
            print("Warning: No synchronization metadata found!")

        # 读取时间戳数据
        total_frames = f['timestamp'].shape[0]
        timestamps = f['timestamp'][:]

        # 基本统计
        print(f"\nTotal Frames: {total_frames}")
        print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"Average Frame Rate: {total_frames / (timestamps[-1] - timestamps[0]):.2f} Hz")

        # 帧间时间间隔分析
        dt = np.diff(timestamps)
        print(f"\nFrame Interval Statistics:")
        print(f"  Mean: {np.mean(dt)*1000:.2f} ms")
        print(f"  Std:  {np.std(dt)*1000:.2f} ms")
        print(f"  Min:  {np.min(dt)*1000:.2f} ms")
        print(f"  Max:  {np.max(dt)*1000:.2f} ms")

        # 丢帧检测（假设30Hz，间隔应为33.3ms）
        expected_interval = 1.0 / 30  # 30Hz
        dropped_frames = np.sum(dt > expected_interval * 1.5)
        if dropped_frames > 0:
            print(f"\nWarning: Detected {dropped_frames} potential dropped frames!")
        else:
            print(f"\n✓ No dropped frames detected")

    print("="*60 + "\n")

def visualize(hdf5_path, output_dir=None, quality='high', verify_only=False):
    """
    主可视化函数 - 读取HDF5数据并生成视频

    工作流程：
    1. 验证时间戳对齐质量（调用 verify_timestamps）
    2. （可选）仅验证模式，跳过视频生成
    3. 读取 HDF5 数据集
    4. 为每帧添加机器人状态仪表盘
    5. 保存帧到临时目录（PNG格式，无损）
    6. 使用 ffmpeg 编码为高质量视频
    7. 清理临时文件

    输出文件：
    - {episode_name}_rgb.mp4: RGB视频 + 状态仪表盘
    - {episode_name}_depth.mp4: 深度图视频 + 状态仪表盘（如果有）

    参数：
        hdf5_path: HDF5 文件路径（例如：episode_0.hdf5）
        output_dir: 输出目录，默认None（与输入文件同目录）
        quality: 视频质量 {'lossless', 'high', 'medium'}
        verify_only: 仅验证时间戳，不生成视频（默认False）

    示例：
        visualize('episode_0.hdf5')  # 生成高质量视频
        visualize('episode_0.hdf5', quality='lossless')  # 生成无损视频
        visualize('episode_0.hdf5', verify_only=True)  # 仅验证时间戳
    """
    # 检查文件是否存在
    if not os.path.exists(hdf5_path):
        print(f"Error: File {hdf5_path} not found.")
        return

    # 验证时间戳对齐质量
    verify_timestamps(hdf5_path)

    # 如果是仅验证模式，直接返回
    if verify_only:
        print("Verification only mode - skipping video generation.")
        return

    # 确定输出目录和文件名
    file_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(hdf5_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_out_path = os.path.join(output_dir, f"{file_name}_rgb.mp4")
    depth_out_path = os.path.join(output_dir, f"{file_name}_depth.mp4")

    print(f"Processing: {hdf5_path}")

    # 创建临时目录存储帧（PNG格式）
    temp_rgb_dir = tempfile.mkdtemp(prefix='rgb_frames_')
    temp_depth_dir = None

    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 打印数据集结构
            print("\nDataset structure:")
            print(f"  observations/images/color: {f['observations/images/color'].shape}")
            if 'observations/images/depth' in f:
                print(f"  observations/images/depth: {f['observations/images/depth'].shape}")
            print(f"  observations/robot_current: {f['observations/robot_current'].shape}")
            print(f"  actions/robot_target: {f['actions/robot_target'].shape}")
            print(f"  observations/gripper_current: {f['observations/gripper_current'].shape}")
            print(f"  actions/gripper_target: {f['actions/gripper_target'].shape}")

            images = f['observations/images/color']

            # 检测原始图像分辨率（不做修改，保持原始分辨率）
            first_frame = images[0]
            if first_frame.shape[0] == 3:  # CHW格式
                h, w = first_frame.shape[1], first_frame.shape[2]
            else:  # HWC格式
                h, w = first_frame.shape[0], first_frame.shape[1]

            print(f"\nOriginal Resolution: {w}x{h}")
            print(f"Quality Setting: {quality}")

            # 检查是否有深度图
            has_depth = 'observations/images/depth' in f
            if has_depth:
                temp_depth_dir = tempfile.mkdtemp(prefix='depth_frames_')

            # 读取数据集
            robot_current = f['observations/robot_current']
            robot_target = f['actions/robot_target']
            gripper_current = f['observations/gripper_current']
            gripper_target = f['actions/gripper_target']
            total_frames = images.shape[0]

            # ==== 步骤1：生成所有帧（带仪表盘）====
            print("\nStep 1/2: Generating frames...")
            for i in tqdm(range(total_frames), desc="Writing Frames"):
                # --- RGB 处理 ---
                raw_rgb = images[i]
                frame_rgb = prepare_frame(raw_rgb, is_rgb_source=True)
                # 添加机器人状态仪表盘
                draw_dashboard(frame_rgb, robot_current[i], robot_target[i],
                             gripper_current[i], gripper_target[i])

                # 保存为 PNG（无损）
                frame_path = os.path.join(temp_rgb_dir, f'frame_{i:06d}.png')
                success = cv2.imwrite(frame_path, frame_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not success:
                    print(f"\nWarning: Failed to write frame {i}")

                # --- Depth 处理 ---
                if has_depth:
                    raw_depth = f['observations/images/depth'][i]
                    # 处理深度图维度
                    if raw_depth.ndim == 3 and raw_depth.shape[0] == 1:
                        raw_depth = raw_depth[0]

                    # 转换为彩色深度图
                    depth_color = process_depth_image(raw_depth)
                    frame_depth = prepare_frame(depth_color, is_rgb_source=False)
                    # 添加机器人状态仪表盘
                    draw_dashboard(frame_depth, robot_current[i], robot_target[i],
                                 gripper_current[i], gripper_target[i])

                    depth_frame_path = os.path.join(temp_depth_dir, f'frame_{i:06d}.png')
                    cv2.imwrite(depth_frame_path, frame_depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            # ==== 步骤2：使用 ffmpeg 编码视频 ====
            print("Step 2/2: Encoding video with ffmpeg...")

            # 编码 RGB 视频
            if encode_video_with_ffmpeg(temp_rgb_dir, rgb_out_path, fps=30, quality=quality):
                file_size = os.path.getsize(rgb_out_path) / (1024 * 1024)  # MB
                print(f"✓ Saved: {rgb_out_path} ({file_size:.1f} MB)")
            else:
                print(f"✗ Failed to encode RGB video")

            # 编码深度图视频（如果有）
            if has_depth:
                if encode_video_with_ffmpeg(temp_depth_dir, depth_out_path, fps=30, quality=quality):
                    file_size = os.path.getsize(depth_out_path) / (1024 * 1024)  # MB
                    print(f"✓ Saved: {depth_out_path} ({file_size:.1f} MB)")
                else:
                    print(f"✗ Failed to encode depth video")

            print("\nDone!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

    finally:
        # 清理临时文件
        print("Cleaning up temporary files...")
        if os.path.exists(temp_rgb_dir):
            shutil.rmtree(temp_rgb_dir)
        if temp_depth_dir and os.path.exists(temp_depth_dir):
            shutil.rmtree(temp_depth_dir)

if __name__ == "__main__":
    """
    命令行入口程序

    使用方法：
        python show_vla.py episode_0.hdf5                      # 默认：高质量视频
        python show_vla.py episode_0.hdf5 --quality lossless  # 无损视频
        python show_vla.py episode_0.hdf5 --out ./videos/     # 指定输出目录
        python show_vla.py episode_0.hdf5 --verify-only       # 仅验证时间戳
    """
    parser = argparse.ArgumentParser(
        description='可视化 HDF5 机器人数据，生成带状态仪表盘的视频并验证时间戳对齐',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 生成高质量视频（推荐）
  python show_vla.py data/episode_0.hdf5

  # 生成无损视频（文件较大）
  python show_vla.py data/episode_0.hdf5 --quality lossless

  # 指定输出目录
  python show_vla.py data/episode_0.hdf5 --out ./videos/

  # 仅验证时间戳对齐质量，不生成视频
  python show_vla.py data/episode_0.hdf5 --verify-only

输出文件：
  - {episode_name}_rgb.mp4: RGB视频 + 机器人状态仪表盘
  - {episode_name}_depth.mp4: 深度图视频 + 机器人状态仪表盘（如果数据包含深度图）
        """
    )

    parser.add_argument("file", type=str, help="HDF5 文件路径（例如：episode_0.hdf5）")
    parser.add_argument("--out", type=str, default=None, help="输出目录（默认：与输入文件同目录）")
    parser.add_argument("--quality", type=str, default='high',
                       choices=['lossless', 'high', 'medium'],
                       help="视频质量：lossless（无损，最大文件）, high（推荐，视觉无损）, medium（中等质量）")
    parser.add_argument("--verify-only", action='store_true',
                       help="仅验证时间戳对齐，不生成视频")
    args = parser.parse_args()

    visualize(args.file, args.out, quality=args.quality, verify_only=args.verify_only)
