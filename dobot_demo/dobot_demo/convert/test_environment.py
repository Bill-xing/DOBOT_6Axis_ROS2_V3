#!/usr/bin/env python3
"""测试 LeRobot 转换环境配置是否正确"""

import sys

def test_imports():
    """测试所有必要的库是否可以导入"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    all_passed = True

    # 核心依赖
    core_libs = [
        ('h5py', 'h5py'),
        ('numpy', 'numpy'),
        ('pyarrow', 'pyarrow'),
        ('opencv', 'cv2'),
        ('pillow', 'PIL'),
        ('tqdm', 'tqdm'),
    ]

    for name, import_name in core_libs:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:15} {version}")
        except ImportError as e:
            print(f"✗ {name:15} FAILED: {e}")
            all_passed = False

    # 可选依赖
    print("\n--- Optional libraries ---")
    optional_libs = [
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
    ]

    for name, import_name in optional_libs:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:15} {version}")
        except ImportError:
            print(f"○ {name:15} not installed (optional)")

    return all_passed


def test_parquet():
    """测试 Parquet 读写"""
    print("\n" + "=" * 60)
    print("Testing Parquet I/O...")
    print("=" * 60)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import tempfile
        import os

        # 创建测试数据
        data = {
            'episode_index': pa.array([0, 0, 0], type=pa.int64()),
            'frame_index': pa.array([0, 1, 2], type=pa.int64()),
            'observation.state': pa.array(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3,
                type=pa.list_(pa.float32(), 6)
            ),
            'action': pa.array(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]] * 3,
                type=pa.list_(pa.float32(), 7)
            ),
        }

        table = pa.Table.from_pydict(data)

        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name

        try:
            pq.write_table(table, temp_path)
            print(f"✓ Written parquet file")

            # 读取
            table_read = pq.read_table(temp_path)
            print(f"✓ Read parquet successfully")
            print(f"  Columns: {table_read.column_names}")
            print(f"  Rows: {table_read.num_rows}")

            return True
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        print(f"✗ Parquet test failed: {e}")
        return False


def test_video():
    """测试视频编码"""
    print("\n" + "=" * 60)
    print("Testing video encoding...")
    print("=" * 60)

    try:
        import cv2
        import numpy as np
        import tempfile
        import os

        # 创建测试视频
        width, height = 640, 480
        fps = 30
        n_frames = 10

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                print("✗ Failed to open video writer")
                return False

            for i in range(n_frames):
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                writer.write(frame)

            writer.release()
            print(f"✓ Created test video ({n_frames} frames)")

            # 读取视频
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                print("✗ Failed to open video file")
                return False

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            print(f"✓ Read video successfully")
            print(f"  Frames: {frame_count}")
            print(f"  FPS: {video_fps}")

            if frame_count != n_frames:
                print(f"⚠ Warning: Frame count mismatch ({frame_count} != {n_frames})")

            return True

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        print(f"✗ Video test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hdf5():
    """测试 HDF5 读写"""
    print("\n" + "=" * 60)
    print("Testing HDF5 I/O...")
    print("=" * 60)

    try:
        import h5py
        import numpy as np
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            temp_path = f.name

        try:
            # 写入
            with h5py.File(temp_path, 'w') as f:
                f.create_dataset('observations/robot_current', data=np.random.randn(10, 6))
                f.create_dataset('actions/robot_target', data=np.random.randn(10, 6))
                f.attrs['fps'] = 30

            print(f"✓ Written HDF5 file")

            # 读取
            with h5py.File(temp_path, 'r') as f:
                print(f"✓ Read HDF5 successfully")
                print(f"  Datasets: {list(f.keys())}")
                print(f"  Attributes: {dict(f.attrs)}")

            return True

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        print(f"✗ HDF5 test failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("LeRobot Convert Environment Test")
    print("=" * 60)
    print()

    results = []

    # 测试导入
    results.append(('Import test', test_imports()))

    # 测试 HDF5
    results.append(('HDF5 test', test_hdf5()))

    # 测试 Parquet
    results.append(('Parquet test', test_parquet()))

    # 测试视频
    results.append(('Video test', test_video()))

    # 总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✅ All tests passed! Environment is ready.")
        print("\nYou can now run:")
        print("  python convert_to_lerobot.py --input ./data --output ./lerobot_dataset")
        return 0
    else:
        print("\n❌ Some tests failed. Please check your environment.")
        print("\nTry reinstalling dependencies:")
        print("  conda env update -f environment.yml")
        return 1


if __name__ == "__main__":
    sys.exit(main())
