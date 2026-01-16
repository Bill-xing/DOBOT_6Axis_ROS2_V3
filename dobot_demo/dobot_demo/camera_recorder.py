#!/usr/bin/env python3
"""
相机录制程序 - 带GUI界面
基于 pick_by_camera4.py 的相机驱动
功能：实时显示、录制视频、调整参数
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime


class CameraRecorder:
    """相机录制类"""

    def __init__(self, device_index=0, resolution=(1920, 1080)):
        """
        初始化相机录制器

        Args:
            device_index: 相机设备索引 (/dev/videoX)
            resolution: 相机分辨率 (width, height)
        """
        self.device_index = device_index
        self.dev = f"/dev/video{device_index}"
        self.width, self.height = resolution

        # 初始化相机
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # 相机参数
        self.current_exp = 160  # 降低曝光值，避免过曝
        self.current_focus = 230
        self.setup_hardware()

        # 录制相关
        self.is_recording = False
        self.video_writer = None
        self.output_dir = "/home/hit/camera_recordings"
        self.current_filename = None
        self.frame_count = 0
        self.start_time = None

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 界面参数
        self.window_name = "Camera Recorder"
        self.display_width = 1280
        self.display_height = 720

        print(f"Camera initialized: {self.width}x{self.height}")
        print(f"Output directory: {self.output_dir}")

    def setup_hardware(self):
        """设置相机硬件参数（曝光、对焦）"""
        try:
            os.system(f"v4l2-ctl -d {self.dev} -c auto_exposure=1")
            os.system(f"v4l2-ctl -d {self.dev} -c exposure_time_absolute={self.current_exp}")
            os.system(f"v4l2-ctl -d {self.dev} -c focus_automatic_continuous=0")
            os.system(f"v4l2-ctl -d {self.dev} -c focus_absolute={self.current_focus}")
            print(f"Hardware setup: Exposure={self.current_exp}, Focus={self.current_focus}")
        except Exception as e:
            print(f"Warning: Failed to setup hardware: {e}")

    def set_exposure(self, val):
        """设置曝光值 (1-10000)"""
        self.current_exp = max(1, min(10000, val))
        os.system(f"v4l2-ctl -d {self.dev} -c exposure_time_absolute={self.current_exp}")
        print(f"Exposure set to: {self.current_exp}")

    def set_focus(self, val):
        """设置对焦值 (0-1023)"""
        self.current_focus = max(0, min(1023, val))
        os.system(f"v4l2-ctl -d {self.dev} -c focus_absolute={self.current_focus}")
        print(f"Focus set to: {self.current_focus}")

    def start_recording(self):
        """开始录制"""
        if self.is_recording:
            print("Already recording!")
            return False

        # 生成文件名（时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = os.path.join(self.output_dir, f"recording_{timestamp}.avi")

        # 创建 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            self.current_filename,
            fourcc,
            60.0,  # FPS
            (self.width, self.height)
        )

        if not self.video_writer.isOpened():
            print("Error: Failed to open video writer!")
            return False

        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        print(f"\n>>> Recording started: {self.current_filename}")
        return True

    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        elapsed_time = time.time() - self.start_time
        print(f">>> Recording stopped")
        print(f"    File: {self.current_filename}")
        print(f"    Frames: {self.frame_count}")
        print(f"    Duration: {elapsed_time:.2f}s")
        print(f"    Avg FPS: {self.frame_count/elapsed_time:.2f}\n")

    def draw_ui(self, frame):
        """在帧上绘制UI信息"""
        # 创建显示画面（缩放到显示尺寸）
        display_frame = cv2.resize(frame, (self.display_width, self.display_height))

        # 状态栏背景
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_width, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

        # 标题
        cv2.putText(display_frame, "Camera Recorder", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 录制状态
        if self.is_recording:
            status_text = "RECORDING"
            status_color = (0, 0, 255)  # Red
            elapsed = time.time() - self.start_time
            time_text = f"Time: {elapsed:.1f}s | Frames: {self.frame_count}"

            # 闪烁的录制指示器
            if int(elapsed * 2) % 2 == 0:  # 每0.5秒闪烁
                cv2.circle(display_frame, (200, 75), 15, status_color, -1)
        else:
            status_text = "READY"
            status_color = (0, 255, 0)  # Green
            time_text = "Press 'R' to start recording"

        cv2.putText(display_frame, status_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.putText(display_frame, time_text, (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 相机参数
        param_text = f"Exposure: {self.current_exp} | Focus: {self.current_focus}"
        cv2.putText(display_frame, param_text, (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 控制说明（右下角）
        instructions = [
            "Controls:",
            "R - Start/Stop Recording",
            "+/- - Adjust Exposure",
            "*/8 - Adjust Focus (Up)",
            "//2 - Adjust Focus (Down)",
            "Q/ESC - Quit"
        ]

        y_offset = self.display_height - 20 - (len(instructions) * 25)
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction,
                       (self.display_width - 280, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return display_frame

    def run(self):
        """主运行循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)

        print("\n" + "="*60)
        print("Camera Recorder Started")
        print("="*60)
        print("\nControls:")
        print("  R        : Start/Stop Recording")
        print("  +/-      : Adjust Exposure")
        print("  * or 8   : Increase Focus")
        print("  / or 2   : Decrease Focus")
        print("  Q or ESC : Quit")
        print("\n" + "="*60 + "\n")

        running = True

        try:
            while running:
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame!")
                    break

                # 如果正在录制，保存帧
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)
                    self.frame_count += 1

                # 绘制UI
                display_frame = self.draw_ui(frame)

                # 显示
                cv2.imshow(self.window_name, display_frame)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                    running = False
                    if self.is_recording:
                        self.stop_recording()

                elif key in [ord('r'), ord('R')]:  # R - 开始/停止录制
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()

                elif key in [ord('+'), ord('=')]:  # + - 增加曝光
                    self.set_exposure(self.current_exp + 20)

                elif key == ord('-'):  # - - 减少曝光
                    self.set_exposure(self.current_exp - 20)

                elif key in [ord('*'), ord('8')]:  # * or 8 - 增加对焦
                    self.set_focus(self.current_focus + 10)

                elif key in [ord('/'), ord('2')]:  # / or 2 - 减少对焦
                    self.set_focus(self.current_focus - 10)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            # 清理
            if self.is_recording:
                self.stop_recording()

            self.cap.release()
            cv2.destroyAllWindows()
            print("\nCamera Recorder stopped.")


def main():
    """主函数"""
    # Full HD分辨率 (1920x1080), 60 FPS
    # 曝光：440, 对焦：240

    recorder = CameraRecorder(
        device_index=0,
        resolution=(1920, 1080)  # Full HD
    )

    recorder.run()


if __name__ == "__main__":
    main()
