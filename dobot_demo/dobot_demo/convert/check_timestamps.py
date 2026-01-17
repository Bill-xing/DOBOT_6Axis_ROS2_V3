#!/usr/bin/env python3
"""
检查 HDF5 文件的时间戳质量
"""
import h5py
import numpy as np
from pathlib import Path
import sys

def check_timestamps(hdf5_path):
    """检查 HDF5 文件的时间戳"""
    print(f"\n{'='*60}")
    print(f"Checking: {hdf5_path}")
    print(f"{'='*60}\n")

    with h5py.File(hdf5_path, 'r') as f:
        # 检查是否有时间戳
        if 'timestamp' not in f:
            print("❌ No 'timestamp' dataset found!")
            return False

        timestamps = f['timestamp'][:]
        n_frames = len(timestamps)

        print(f"Total frames: {n_frames}")
        print(f"\nTimestamp statistics:")
        print(f"  First: {timestamps[0]}")
        print(f"  Last: {timestamps[-1]}")
        print(f"  Min: {np.min(timestamps)}")
        print(f"  Max: {np.max(timestamps)}")
        print(f"  Range: {np.max(timestamps) - np.min(timestamps):.6f} seconds")

        # 检查是否所有时间戳都相同
        if np.all(timestamps == timestamps[0]):
            print(f"\n❌ ERROR: All timestamps are identical ({timestamps[0]})!")
            return False

        # 计算帧间隔
        dt = np.diff(timestamps)
        print(f"\nFrame intervals:")
        print(f"  Mean: {np.mean(dt)*1000:.2f} ms")
        print(f"  Std: {np.std(dt)*1000:.2f} ms")
        print(f"  Min: {np.min(dt)*1000:.2f} ms")
        print(f"  Max: {np.max(dt)*1000:.2f} ms")

        # 检查异常值
        zero_intervals = np.sum(dt == 0)
        if zero_intervals > 0:
            print(f"\n⚠️  Warning: {zero_intervals} frames have zero time interval!")

        # 检查负值
        negative_intervals = np.sum(dt < 0)
        if negative_intervals > 0:
            print(f"\n❌ ERROR: {negative_intervals} frames have negative time intervals!")
            return False

        print(f"\n✓ Timestamps appear valid")
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_timestamps.py <hdf5_file_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        check_timestamps(path)
    elif path.is_dir():
        hdf5_files = sorted(path.glob("episode_*.hdf5"))
        print(f"Found {len(hdf5_files)} episode files\n")

        for hdf5_file in hdf5_files:
            valid = check_timestamps(hdf5_file)
            if not valid:
                print(f"\n⚠️  {hdf5_file.name} has timestamp issues!")
    else:
        print(f"Error: {path} not found")

if __name__ == "__main__":
    main()
