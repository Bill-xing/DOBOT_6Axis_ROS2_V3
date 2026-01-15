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
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    h, w = img.shape[:2]
    if x + text_w > w: x = w - text_w
    if y - text_h < 0: y = text_h + 5

    sub_img = img[y-text_h-baseline:y+baseline, x:x+text_w]
    if sub_img.size == 0: return

    white_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    img[y-text_h-baseline:y+baseline, x:x+text_w] = res
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_dashboard(img, robot_current, robot_target, gripper_current, gripper_target):
    """
    绘制机器人状态仪表盘

    参数:
    - robot_current: [x, y, z, rx, ry, rz] 当前位姿
    - robot_target: [x, y, z, rx, ry, rz] 目标位姿
    - gripper_current: [pos] 夹爪当前位置 (0-1000)
    - gripper_target: [pos] 夹爪目标位置 (0-1000)
    """
    h, w = img.shape[:2]
    y_start = 30
    line_height = 25

    # 机械臂位置 - 当前 vs 目标
    pos_curr_str = f"Pos(Curr): X:{robot_current[0]:.0f} Y:{robot_current[1]:.0f} Z:{robot_current[2]:.0f}"
    draw_text_with_bg(img, pos_curr_str, (10, y_start), text_color=(0, 255, 0))

    pos_targ_str = f"Pos(Targ): X:{robot_target[0]:.0f} Y:{robot_target[1]:.0f} Z:{robot_target[2]:.0f}"
    draw_text_with_bg(img, pos_targ_str, (10, y_start + line_height), text_color=(255, 165, 0))

    # 计算位置误差
    pos_error = np.linalg.norm(robot_current[:3] - robot_target[:3])
    error_str = f"Pos Error: {pos_error:.2f}mm"
    draw_text_with_bg(img, error_str, (10, y_start + line_height * 2), text_color=(255, 255, 0))

    # 机械臂姿态
    rot_curr_str = f"Rot(Curr): RX:{robot_current[3]:.1f} RY:{robot_current[4]:.1f} RZ:{robot_current[5]:.1f}"
    draw_text_with_bg(img, rot_curr_str, (10, y_start + line_height * 3), text_color=(0, 255, 0))

    rot_targ_str = f"Rot(Targ): RX:{robot_target[3]:.1f} RY:{robot_target[4]:.1f} RZ:{robot_target[5]:.1f}"
    draw_text_with_bg(img, rot_targ_str, (10, y_start + line_height * 4), text_color=(255, 165, 0))

    # 夹爪进度条 - 双层显示(当前+目标)
    bar_h = 20
    bar_w = int(w * 0.7)
    bar_x = int((w - bar_w) / 2)
    bar_y = h - 70

    # 绘制夹爪当前状态
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    val_curr = float(gripper_current[0])
    ratio_curr = max(0.0, min(1.0, val_curr / 1000.0))
    fill_w_curr = int(bar_w * ratio_curr)
    color_curr = (0, int(255 * ratio_curr), int(255 * (1 - ratio_curr)))
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w_curr, bar_y + bar_h), color_curr, -1)
    label_curr = f"Gripper Curr: {int(val_curr)}"
    draw_text_with_bg(img, label_curr, (bar_x, bar_y - 10), font_scale=0.5, text_color=(0, 255, 0))

    # 绘制夹爪目标状态
    bar_y_target = bar_y + bar_h + 10
    cv2.rectangle(img, (bar_x, bar_y_target), (bar_x + bar_w, bar_y_target + bar_h), (50, 50, 50), -1)
    val_targ = float(gripper_target[0])
    ratio_targ = max(0.0, min(1.0, val_targ / 1000.0))
    fill_w_targ = int(bar_w * ratio_targ)
    color_targ = (0, int(255 * ratio_targ), int(255 * (1 - ratio_targ)))
    cv2.rectangle(img, (bar_x, bar_y_target), (bar_x + fill_w_targ, bar_y_target + bar_h), color_targ, -1)
    label_targ = f"Gripper Targ: {int(val_targ)}"
    draw_text_with_bg(img, label_targ, (bar_x, bar_y_target - 10), font_scale=0.5, text_color=(255, 165, 0))

def process_depth_image(depth_raw):
    mask = depth_raw > 0
    if np.sum(mask) == 0:
        return np.zeros((*depth_raw.shape, 3), dtype=np.uint8)

    min_val = np.min(depth_raw[mask])
    max_val = np.max(depth_raw[mask])

    depth_norm = np.zeros_like(depth_raw, dtype=np.float32)
    if max_val > min_val:
        depth_norm[mask] = (depth_raw[mask] - min_val) / (max_val - min_val) * 255

    depth_norm = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return depth_color

def prepare_frame(frame, is_rgb_source=True):
    """
    数据清洗核心 - 不再改变分辨率
    1. 处理 Channel First/Last
    2. 确保 uint8
    3. RGB -> BGR (OpenCV使用BGR)
    4. 确保内存连续
    """
    # 1. 处理维度 (C, H, W) -> (H, W, C)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)
    elif frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0] # Depth (1, H, W) -> (H, W)

    # 2. 确保 uint8
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # 3. 颜色转换 (仅针对 RGB 图像源)
    if is_rgb_source and frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 4. 确保内存连续
    frame = np.ascontiguousarray(frame)

    return frame

def encode_video_with_ffmpeg(frame_dir, output_path, fps=30, quality='high'):
    """
    使用 ffmpeg 命令行工具编码高质量视频

    quality options:
    - 'lossless': 无损编码 (最大文件)
    - 'high': 高质量 (推荐，视觉无损)
    - 'medium': 中等质量
    """
    # 检查 ffmpeg 是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  CentOS/RHEL: sudo yum install ffmpeg")
        return False

    # 根据质量选择编码参数
    if quality == 'lossless':
        # 无损 H.264 编码
        codec_params = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '0']
    elif quality == 'high':
        # 高质量编码 (CRF 17, 视觉无损)
        codec_params = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '17']
    else:
        # 中等质量
        codec_params = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23']

    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(frame_dir, 'frame_%06d.png'),
    ] + codec_params + [
        '-pix_fmt', 'yuv420p',  # 确保兼容性
        '-y',
        output_path
    ]

    try:
        # 显示 ffmpeg 输出以便调试
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg encoding failed: {e.stderr.decode()}")
        return False

def verify_timestamps(hdf5_path):
    """
    验证时间戳对齐质量
    检查是否使用了ApproximateTimeSynchronizer以及对齐质量
    """
    print("\n" + "="*60)
    print("TIMESTAMP ALIGNMENT VERIFICATION")
    print("="*60)

    with h5py.File(hdf5_path, 'r') as f:
        # 检查元数据
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

        # 检查是否有调试时间戳信息 (仅在第一帧检查)
        total_frames = f['timestamp'].shape[0]
        timestamps = f['timestamp'][:]

        print(f"\nTotal Frames: {total_frames}")
        print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"Average Frame Rate: {total_frames / (timestamps[-1] - timestamps[0]):.2f} Hz")

        # 分析帧间时间间隔
        dt = np.diff(timestamps)
        print(f"\nFrame Interval Statistics:")
        print(f"  Mean: {np.mean(dt)*1000:.2f} ms")
        print(f"  Std:  {np.std(dt)*1000:.2f} ms")
        print(f"  Min:  {np.min(dt)*1000:.2f} ms")
        print(f"  Max:  {np.max(dt)*1000:.2f} ms")

        # 检测丢帧
        expected_interval = 1.0 / 30  # 30Hz
        dropped_frames = np.sum(dt > expected_interval * 1.5)
        if dropped_frames > 0:
            print(f"\nWarning: Detected {dropped_frames} potential dropped frames!")
        else:
            print(f"\n✓ No dropped frames detected")

    print("="*60 + "\n")

def visualize(hdf5_path, output_dir=None, quality='high', verify_only=False):
    if not os.path.exists(hdf5_path):
        print(f"Error: File {hdf5_path} not found.")
        return

    # 验证时间戳
    verify_timestamps(hdf5_path)

    if verify_only:
        print("Verification only mode - skipping video generation.")
        return

    file_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(hdf5_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_out_path = os.path.join(output_dir, f"{file_name}_rgb.mp4")
    depth_out_path = os.path.join(output_dir, f"{file_name}_depth.mp4")

    print(f"Processing: {hdf5_path}")

    # 创建临时目录存储帧
    temp_rgb_dir = tempfile.mkdtemp(prefix='rgb_frames_')
    temp_depth_dir = None

    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 检查数据格式
            print("\nDataset structure:")
            print(f"  observations/images/color: {f['observations/images/color'].shape}")
            if 'observations/images/depth' in f:
                print(f"  observations/images/depth: {f['observations/images/depth'].shape}")
            print(f"  observations/robot_current: {f['observations/robot_current'].shape}")
            print(f"  actions/robot_target: {f['actions/robot_target'].shape}")
            print(f"  observations/gripper_current: {f['observations/gripper_current'].shape}")
            print(f"  actions/gripper_target: {f['actions/gripper_target'].shape}")

            images = f['observations/images/color']

            # 读取原始尺寸 - 不做任何修改
            first_frame = images[0]
            if first_frame.shape[0] == 3: # CHW
                h, w = first_frame.shape[1], first_frame.shape[2]
            else: # HWC
                h, w = first_frame.shape[0], first_frame.shape[1]

            print(f"\nOriginal Resolution: {w}x{h}")
            print(f"Quality Setting: {quality}")

            has_depth = 'observations/images/depth' in f
            if has_depth:
                temp_depth_dir = tempfile.mkdtemp(prefix='depth_frames_')

            robot_current = f['observations/robot_current']
            robot_target = f['actions/robot_target']
            gripper_current = f['observations/gripper_current']
            gripper_target = f['actions/gripper_target']
            total_frames = images.shape[0]

            print("\nStep 1/2: Generating frames...")
            for i in tqdm(range(total_frames), desc="Writing Frames"):
                # --- RGB 处理 ---
                raw_rgb = images[i]
                frame_rgb = prepare_frame(raw_rgb, is_rgb_source=True)
                draw_dashboard(frame_rgb, robot_current[i], robot_target[i],
                             gripper_current[i], gripper_target[i])

                # 使用 PNG 无损保存
                frame_path = os.path.join(temp_rgb_dir, f'frame_{i:06d}.png')
                success = cv2.imwrite(frame_path, frame_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not success:
                    print(f"\nWarning: Failed to write frame {i}")

                # --- Depth 处理 ---
                if has_depth:
                    raw_depth = f['observations/images/depth'][i]
                    if raw_depth.ndim == 3 and raw_depth.shape[0] == 1:
                        raw_depth = raw_depth[0]

                    depth_color = process_depth_image(raw_depth)
                    frame_depth = prepare_frame(depth_color, is_rgb_source=False)
                    draw_dashboard(frame_depth, robot_current[i], robot_target[i],
                                 gripper_current[i], gripper_target[i])

                    depth_frame_path = os.path.join(temp_depth_dir, f'frame_{i:06d}.png')
                    cv2.imwrite(depth_frame_path, frame_depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            print("Step 2/2: Encoding video with ffmpeg...")

            # 使用 ffmpeg 高质量编码
            if encode_video_with_ffmpeg(temp_rgb_dir, rgb_out_path, fps=30, quality=quality):
                file_size = os.path.getsize(rgb_out_path) / (1024 * 1024)  # MB
                print(f"✓ Saved: {rgb_out_path} ({file_size:.1f} MB)")
            else:
                print(f"✗ Failed to encode RGB video")

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
    parser = argparse.ArgumentParser(description='Visualize VLA HDF5 robot data and verify temporal alignment')
    parser.add_argument("file", type=str, help="Path to .hdf5 file")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--quality", type=str, default='high',
                       choices=['lossless', 'high', 'medium'],
                       help="Video quality: lossless (largest file), high (recommended), medium")
    parser.add_argument("--verify-only", action='store_true',
                       help="Only verify timestamps without generating video")
    args = parser.parse_args()

    visualize(args.file, args.out, quality=args.quality, verify_only=args.verify_only)
