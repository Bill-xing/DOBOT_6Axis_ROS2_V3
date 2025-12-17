# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
录制数据集。机器人的动作可以通过遥操作或策略生成。

示例：

```shell
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, camera_index: 0, width: 640, height: 480}}" \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the cube" \
    # <- 如果您想遥操作录制或在回合之间使用策略，Teleop是可选的 \
    # --teleop.type=so100_leader \
    # --teleop.port=/dev/tty.usbmodem58760431551 \
    # --teleop.id=blue \
    # <- 如果您想使用策略录制，Policy是可选的 \
    # --policy.path=${HF_USER}/my_policy \
```

使用双臂so100录制的示例：
```shell
lerobot-record \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.num_episodes=25 \
  --dataset.single_task="Grab and handover the red cube to the other arm"
```
"""

import logging
import time
import numpy as np
import torch
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from collections import deque

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    xlerobot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    xlerobot_vr,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.xlerobot_vr.xlerobot_vr import XLerobotVRTeleop, init_vr_listener
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

GLOBAL_BACK_GOAL = np.array([-99.1578, -67.1670, 21.1723, 99.1829, 5.7592, 0.8043, 99.8230, -67.0739, 21.0242, 99.1322, -0.6326, 0.4778])
GLOBAL_OPEN_GOAL = np.array([-99.1578, -67.1670, 21.1723, 99.1829, 5.7592, 36, 99.8230, -67.0739, 21.0242, 99.1322, -0.6326, 36])
def reset_follower_position(robot, target_position, steps=50, delay=0.015, start_position=None):
    """
    平滑地将机器人移动到目标位置并生成可录制的动作序列。

    Args:
        robot: 机器人对象（必须具有bus1和bus2属性）。
        target_position: 目标位置数组 [左臂关节数据..., 右臂关节数据...]。
        steps: 轨迹步数（默认：150）。
        delay: 每步延迟时间（毫秒）（默认：15毫秒）。

    Returns:
        list: 动作序列，其中每个元素是一个动作字典。
    """
    # Read the current position

    left_current_position_dict = robot.bus1.sync_read("Present_Position")
    right_current_position_dict = robot.bus2.sync_read("Present_Position")
    if start_position is not None:
        left_current_position, right_current_position = start_position[0:6], start_position[6:12]
    else:
        left_current_position = np.array(
            [left_current_position_dict[name] for name in left_current_position_dict], dtype=np.float32
        )
        right_current_position = np.array(
            [right_current_position_dict[name] for name in right_current_position_dict], dtype=np.float32
        )

    left_target_position, right_target_position = target_position[0:6], target_position[6:12]
    
    if start_position is None:
        left_trajectory = torch.from_numpy(
            np.linspace(left_current_position, np.concatenate((left_target_position, left_current_position[-2:])), steps)
        )
        right_trajectory = torch.from_numpy(
            np.linspace(right_current_position[:-3], right_target_position, steps)
        )
    else:
        left_trajectory = torch.from_numpy(
            np.linspace(left_current_position, left_target_position, steps)
        )
        right_trajectory = torch.from_numpy(
            np.linspace(right_current_position, right_target_position, steps)
        )
    
    # generate action sequence
    action_sequence = []
    left_current_position_dict = {f"{k}.pos" for k in left_current_position_dict}
    left_current_position_dict = [
        'left_arm_shoulder_pan.pos',
        'left_arm_shoulder_lift.pos',
        'left_arm_elbow_flex.pos',
        'left_arm_wrist_flex.pos',
        'left_arm_wrist_roll.pos',
        'left_arm_gripper.pos',
    ]
    right_current_position_dict = [
        'right_arm_shoulder_pan.pos',
        'right_arm_shoulder_lift.pos',
        'right_arm_elbow_flex.pos',
        'right_arm_wrist_flex.pos',
        'right_arm_wrist_roll.pos',
        'right_arm_gripper.pos'
    ]
    if start_position is None:
        left_current_position_dict += ['head_motor_1.pos', 'head_motor_2.pos']
    for left_pose, right_pose in zip(left_trajectory, right_trajectory):
        left_action_dict = dict(zip(left_current_position_dict, left_pose, strict=False))
        right_action_dict = dict(zip(right_current_position_dict, right_pose, strict=False))
        if start_position is None:
            head_motor_dict = {}
        else:
            head_motor_dict = {
                'head_motor_1.pos': 8.982,
                'head_motor_2.pos': 28.8703,
            }
        base_action_dict = {
            "x.vel": 0,
            "y.vel": 0,
            "theta.vel": 0,
        }
        
        action_dict = {**left_action_dict, **head_motor_dict, **right_action_dict, **base_action_dict}
        action_sequence.append(action_dict)
    
    return action_sequence

def queue_reset_actions(action_queue, robot, target_position, steps=50, start_position=None):
    """
    Add a reset action sequence to the queue.
    
    Args:
        action_queue: The action queue.
        robot: The robot object.
        target_position: The target position.
        steps: Number of trajectory steps.
    """

    action_sequence = reset_follower_position(robot, target_position, steps, start_position=start_position)
    action_queue.extend(action_sequence)
    logging.info(f"Queued {len(action_sequence)} reset actions")


@dataclass
class DatasetRecordConfig:
    """
    数据集录制配置类 - 定义机器人数据集录制的所有参数

    这个类包含了LeRobot数据集录制的完整配置，包括：
    1. 数据集基本信息（名称、任务描述、存储位置）
    2. 录制参数（帧率、回合长度、数量）
    3. 视频编码设置（编码方式、质量控制）
    4. 性能优化参数（进程数、线程数）
    5. 集成设置（Hugging Face Hub上传）

    配置设计原则：
    - 提供合理的默认值，降低使用门槛
    - 参数分组清晰，便于理解和配置
    - 包含详细的参数说明和调优建议
    - 支持多种使用场景和性能需求

    数据集标准：
    - 遵循LeRobot数据集格式规范
    - 兼容Hugging Face Hub生态
    - 支持离线和在线使用
    - 可扩展的数据格式定义
    """

    """
    === 数据集基本信息 ===

    repo_id: 数据集标识符
    格式: '{用户名}/{数据集名称}' (例如: 'OpenRobot/dataset_example')
    命名规范:
    - 使用有意义的名称描述数据集内容
    - 包含机器人型号和任务类型
    - 避免特殊字符和空格
    - 遵循语义化版本控制(可选)

    数据集类型说明:
    - 演示数据集: 展示特定技能或任务
    - 训练数据集: 用于机器学习模型训练
    - 评估数据集: 用于算法性能测试
    - 基准数据集: 标准化的对比测试集
    """
    repo_id: str

    """
    single_task: 任务描述
    目的: 简洁准确地描述录制期间机器人执行的任务

    好的示例:
    - "Pick up the red cube and place it in the blue box"
    - "Pour water from pitcher to cup"
    - "Screw the bolt into the threaded hole"
    - "Press the button on the control panel"
    - "Fold the piece of paper in half"

    描述要求:
    - 具体明确：说明具体的物体和动作
    - 简洁明了：避免模糊不清的描述
    - 可操作性：描述能够实际执行的任务
    - 标准术语：使用标准的机器人术语
    """
    single_task: str

    """
    root: 数据集存储根目录
    作用: 指定数据集文件在本地文件系统中的存储位置

    目录结构说明:
    root/
    ├── data/           # 图像和视频文件
    │   ├── episode_000/
    │   │   ├── frame_000000.jpg
    │   │   ├── frame_000001.jpg
    │   │   └── ...
    │   └── episode_001/
    ├── info.json         # 数据集元信息
    ├── episodes.jsonl     # 回合索引文件
    └── videos/            # 编码后的视频文件

    路径类型:
    - None: 使用默认位置 (./datasets/{repo_id})
    - str: 绝对或相对路径
    - Path: pathlib.Path对象

    存储考虑:
    - 确保有足够的磁盘空间
    - 考虑数据集大小增长趋势
    - 选择快速存储设备提高I/O性能
    - 定期备份重要数据集
    """
    root: str | Path | None = None

    """
    fps: 录制帧率(帧/秒)
    影响因素: 运动精度、数据大小、处理性能

    常见帧率选择:
    - 10-15fps: 适用于低速任务，节省存储空间
    - 25-30fps: 标准视频帧率，平衡精度和大小
    - 50-60fps: 高精度任务，需要更多存储空间
    - 100+fps: 超高速任务，通常用于特殊研究

    性能权衡:
    ✓ 高fps: 更精确的时间分辨率，更好的动作同步
    ✓ 低fps: 减少存储需求，提高录制稳定性
    ⚠  极高fps: 可能影响系统整体性能
    ⚠  极低fps: 可能丢失快速动作细节

    帧率匹配建议:
    - 匹配摄像头硬件能力
    - 考虑机器人控制频率
    - 满足下游任务需求
    """
    fps: int = 30

    """
    episode_time_s: 回合录制时长(秒)
    定义每个数据集录制的最大时间长度

    时间规划考虑:
    - 任务复杂性: 简单任务使用较短时间
    - 多样性需求: 较长时间增加任务变化
    - 存储限制: 总时间影响数据集大小
    - 处理效率: 长时间减少回合切换开销

    常见时长参考:
    - 简单操作: 15-30秒
    - 中等复杂: 45-60秒
    - 复杂任务: 90-120秒
    - 长时间序列: 180秒+

    优化策略:
    - 预计任务完成时间，避免不必要的空白录制
    - 考虑重置时间对整体效率的影响
    - 为意外情况预留缓冲时间
    """
    episode_time_s: int | float = 60

    """
    reset_time_s: 环境重置时间(秒)
    机器人回合之间重置环境所需的时间

    重置操作内容:
    - 机器人返回安全位置
    - 重置工作台上的物体
    - 恢复初始环境状态
    - 等待系统稳定

    重置时间组成:
    - 机械运动时间: 5-10秒
    - 物体调整时间: 5-15秒
    - 等待稳定时间: 3-5秒
    - 缓冲时间: 2-5秒

    优化建议:
    - 自动化重置流程提高效率
    - 并行化操作减少等待时间
    - 优化重置路径减少运动时间
    - 预先准备重置所需物体
    """
    reset_time_s: int | float = 10

    """
    num_episodes: 录制回合总数
    定义整个数据集录制包含的回合数量

    数量规划原则:
    - 基础验证: 10-20个回合验证代码功能
    - 概念验证: 50-100个回合证明概念可行性
    - 性能基准: 200-500个回合建立性能基准
    - 训练数据: 1000+个回合提供足够训练数据
    - 完整系统: 5000+个回合覆盖所有边界情况

    影响因素:
    - 机器学习算法需求量
    - 数据多样性要求
    - 存储空间限制
    - 录制时间和成本
    - 数据质量控制
    """
    num_episodes: int = 50

    """
    video: 视频编码开关
    控制是否将图像帧编码为视频文件

    编码优势:
    ✓ 减少文件数量，便于管理
    ✓ 提高传输效率，减少网络开销
    ✓ 标准化格式，兼容性好
    ✓ 支持流式播放和随机访问

    不编码优势:
    ✓ 保留原始质量，无压缩损失
    ✓ 便于单帧处理和分析
    ✓ 减少编码计算开销
    ✓ 支持多种图像格式

    使用建议:
    - 大型数据集: 启用视频编码减少存储需求
    - 高精度要求: 保留原始图像
    - 实时处理: 选择适合的编码方式
    - 存储敏感: 仔细评估编码参数
    """
    video: bool = True

    """
    push_to_hub: Hub上传开关
    控制是否将数据集上传到Hugging Face Hub

    Hub平台优势:
    - 版本控制和协作
    - 全球访问和分享
    - 标准化数据集格式
    - 与transformers等库集成
    - 自动化的数据集卡片生成

    上传考虑:
    - 网络带宽和稳定性
    - 数据集大小和上传时间
    - 隐私和许可要求
    - Hub存储配额和成本
    - 访问权限管理

    使用场景:
    ✓ 公开数据集: 促进研究和应用发展
    ✓ 团队协作: 团队内数据共享和版本管理
    ✓ 备份存储: 云端数据安全和可访问性
    - 私有数据: 考虑企业内部解决方案
    """
    push_to_hub: bool = False

    """
    private: 私有数据集开关
    控制数据集在Hub上的可见性

    私有数据集特点:
    - 仅限指定用户访问
    - 适合商业和研究用途
    - 保持数据机密性
    - 支持精细权限管理

    公开数据集特点:
    - 全球研究人员可访问
    - 促进学术交流和合作
    - 提高数据和算法影响力
    - 建立开放科学文化

    选择标准:
    - 数据包含敏感信息 → 私有
    - 商业应用场景 → 私有
    - 学术研究目的 → 公开
    - 促进开源发展 → 公开
    """
    private: bool = False

    """
    tags: 数据集标签列表
    为数据集添加分类和描述标签，便于发现和使用

    标签分类:
    - 机器人类型: "so100", "so101", "xlerobot"
    - 任务类型: "pick-and-place", "pouring", "assembly"
    - 环境类型: "table-top", "kitchen", "factory"
    - 控制方式: "teleoperation", "autonomous"
    - 应用领域: "robotics", "manipulation", "automation"
    - 数据集类型: "demo", "benchmark", "tutorial"

    最佳实践:
    - 使用标准化标签提高可发现性
    - 标签数量适中，避免信息过载
    - 结合使用通用标签和特定标签
    - 定期更新和维护标签准确性
    """
    tags: list[str] | None = None

    """
    === 图像处理性能参数 ===

    num_image_writer_processes: 图像写入进程数
    控制用于保存PNG图像的后台进程数量

    进程vs线程选择:
    进程方式:
    ✓ 内存隔离：单个进程崩溃不影响其他
    ✓ 多核利用：充分利用CPU多核资源
    ✓ 缓存独立：避免Python GIL限制
    ⚠ 资源开销：进程创建和销毁成本高
    ⚠ 复杂性：进程间通信和管理复杂

    线程方式:
    ✓ 资源节省：轻量级创建和销毁
    ✓ 通信简单：共享内存数据访问
    ✓ 调试便利：错误处理和调试更简单
    ⚠ GIL限制：Python多线程性能受限
    ⚠ 内存竞争：可能影响主线程性能

    系统建议:
    - 小规模(≤2摄像头): 0进程，4-8线程
    - 中等规模(3-4摄像头): 1-2进程，4-6线程
    - 大规模(≥5摄像头): 2-4进程，3-5线程

    调优策略:
    1. 从0进程开始，逐步增加线程数
    2. 监控CPU使用率和内存消耗
    3. 测试不同配置下的帧率稳定性
    4. 根据硬件配置选择最优方案
    """
    num_image_writer_processes: int = 0

    """
    num_image_writer_threads_per_camera: 每摄像头线程数
    每个摄像头用于并行写入图像的线程数量

    线程作用:
    - 并行编码：同时编码多个PNG文件
    - 队列缓冲：管理待写入的图像队列
    - 负载均衡：分散磁盘I/O压力
    - 异步处理：不阻塞主控制循环

    线程数影响:
    ✓ 增加线程数 → 提高并行处理能力
    ✓ 适度并行 → 提高整体写入速度
    ⚠ 线程过多 → 竞争CPU和I/O资源
    ⚠ 线程过多 → 增加内存使用和管理开销
    ⚠ 线程过多 → 可能影响主循环性能

    调优指导:
    - CPU密集型：线程数 ≈ CPU核心数
    - I/O密集型：线程数 > CPU核心数
    - 混合负载：平衡CPU和I/O需求
    - 实际测试：根据具体硬件调整
    """
    num_image_writer_threads_per_camera: int = 4

    """
    video_encoding_batch_size: 视频编码批次大小
    控制在开始视频编码前累积的回合数量

    批次处理优势:
    ✓ 优化编码效率：减少编码器启动/停止开销
    ✓ 资源利用优化：批量处理提高硬件利用率
    ✓ 内存管理优化：更好地控制内存峰值
    ✓ 网络传输优化：减少小文件传输开销

    批次大小影响:
    ✓ 增大批次 → 提高编码效率
    ✓ 增大批次 → 减少文件数量
    ⚠ 增大批次 → 增加内存使用
    ⚠ 增大批次 → 延迟视频可用性
    ⚠ 增大批次 → 临时存储空间需求

    选择策略:
    - 实时应用: 1-2个回合，快速可用
    - 离线处理: 5-10个回合，平衡效率
    - 大规模批处理: 20+个回合，最大化效率
    - 内存受限: 1个回合，最小化内存使用
    """
    video_encoding_batch_size: int = 1

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # 是否使用遥操作器控制机器人
    teleop: TeleoperatorConfig | None = None
    # 是否使用策略控制机器人
    policy: PreTrainedConfig | None = None
    # 在屏幕上显示所有摄像头
    display_data: bool = False
    # 使用语音合成来朗读事件。
    play_sounds: bool = True
    # 在现有数据集上恢复录制。
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    action_queue: deque | None = None,
):
    """
    机器人数据录制主循环函数

    这是机器人数据集录制的核心控制循环，负责协调机器人控制、数据采集和事件处理。
    该函数实现了实时控制回路，以固定频率执行观察-决策-动作的完整循环。

    核心功能:
    1. 实时控制循环：以指定FPS执行控制回路
    2. 多种控制模式：支持策略控制、遥操作控制和混合控制
    3. 数据采集录制：同步采集传感器数据并保存到数据集
    4. 事件响应处理：处理用户输入和系统事件
    5. 时间精确控制：确保控制频率的稳定性

    控制流程:
    1. 获取VR事件更新
    2. 检查退出和重置事件
    3. 读取机器人传感器数据
    4. 根据控制模式生成动作指令
    5. 发送动作到机器人执行
    6. 录制数据到数据集
    7. 精确时间控制维持频率

    Args:
        robot (Robot): 机器人实例，提供观察获取和动作发送接口
        events (dict): 事件字典，用于跨线程通信和状态同步
        fps (int): 控制频率，每秒执行的控制循环次数
        dataset (LeRobotDataset | None): 数据集实例，为None时只控制不录制
        teleop (Teleoperator | list[Teleoperator] | None): 遥操作器实例或列表
        policy (PreTrainedPolicy | None): 预训练策略实例
        control_time_s (int | None): 单次录制时长（秒），None表示无限时长
        single_task (str | None): 任务描述字符串，用于数据集标注
        display_data (bool): 是否实时显示数据，用于调试和监控
        action_queue (deque | None): 动作队列，用于预编程动作序列

    Note:
        - 函数使用装饰器确保图像写入进程的安全退出
        - 支持多遥操作器配置（如键盘+VR控制器组合）
        - 实现了精确的时间控制以保证控制频率稳定性
        - 异常处理确保单帧失败不影响整体录制流程
    """
    # 验证数据集帧率与请求帧率的一致性
    # 这是重要的数据完整性检查，确保录制数据的帧率符合预期
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # 初始化多遥操作器支持
    # 支持组合控制：一个键盘遥操作器 + 一个机械臂遥操作器
    # 这种配置适用于复杂机器人系统，如移动机械臂（LeKiwi）
    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        # 从遥操作器列表中查找键盘控制器
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)

        # 从遥操作器列表中查找机械臂控制器
        # 支持多种机械臂遥操作器类型
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so100_leader.SO100Leader,    # SO100机械臂领导者
                        so101_leader.SO101Leader,    # SO101机械臂领导者
                        koch_leader.KochLeader,      # Koch机械臂领导者
                    ),
                )
            ),
            None,
        )

        # 验证多遥操作器配置的有效性
        # 当前只支持LeKiwi机器人的双遥操作器配置
        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # 重置策略状态，确保策略从干净状态开始
    # 这对于确保录制数据的一致性和可重现性很重要
    if policy is not None:
        policy.reset()

    # 初始化动作队列，用于存储预编程的动作序列
    # 队列可以用于自动化动作序列，如初始化、重置等
    if action_queue is None:
        action_queue = deque()

    # 初始化时间控制变量
    # timestamp: 当前回合计时器，用于检查录制时长限制
    # start_episode_t: 回合开始时间戳，用于精确计算经过时间
    timestamp = 0
    start_episode_t = time.perf_counter()

    # 主控制循环 - 以指定频率持续执行直到达到时间限制
    while timestamp < control_time_s:
        # 记录循环开始时间，用于精确的频率控制
        start_loop_t = time.perf_counter()

        # 获取并处理VR事件更新
        # VR事件包括用户输入、系统状态变化等
        current_events = teleop.get_vr_events()
        events.update(current_events)

        # 检查提前退出事件
        # 用户可以通过VR控制器或键盘触发提前退出
        if events["exit_early"]:
            events["exit_early"] = False  # 重置事件状态
            log_say("Exit early")  # 语音提示
            time.sleep(1)  # 短暂延迟确保状态稳定
            break

        # 获取机器人传感器观察数据
        # 包括关节角度、末端执行器位置、摄像头图像等
        try:
            observation = robot.get_observation()
        except TimeoutError as e:
            # 处理摄像头超时错误，常见于USB带宽不足或硬件故障
            logging.warning(f"Camera timeout: {e}. Skipping this frame.")
            continue  # 跳过当前帧，继续下一轮循环

        # 构建数据集记录帧格式（如果需要录制或使用策略）
        # 将原始观察数据转换为数据集标准格式
        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        # 动作生成策略1：优先使用预编程动作队列
        # 队列中的动作通常是自动化序列，如初始化、重置等
        if action_queue:
            action = action_queue.popleft()  # 取出队列中的下一个动作
            if action == {}: # 空字典作为重置到零位的标志
                action = teleop.move_to_zero_position(robot)

        # 动作生成策略2：使用预训练策略推理
        # 基于深度学习的策略根据当前观察预测最优动作
        elif policy is not None:
            # 使用神经网络策略预测动作
            action_values = predict_action(
                observation_frame,                               # 观察数据
                policy,                                          # 策略模型
                get_safe_torch_device(policy.config.device),    # 计算设备（CPU/GPU）
                policy.config.use_amp,                          # 是否使用混合精度
                task=single_task,                               # 任务描述
                robot_type=robot.robot_type,                   # 机器人类型
            )
            # 将张量动作值转换为字典格式
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}

        # 动作生成策略3：单个遥操作器控制
        # 用户通过VR控制器、键盘等设备实时控制
        elif policy is None and isinstance(teleop, Teleoperator):
            action = teleop.get_action(observation, robot)

        # 动作生成策略4：多遥操作器组合控制
        # 适用于复杂机器人系统，如移动机械臂
        elif policy is None and isinstance(teleop, list):
            # TODO(pepijn, steven): 清理记录循环以用于多个机器人（可能使用pipeline）
            # 获取机械臂遥操作器的动作指令
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

            # 获取键盘遥操作器的动作指令（通常用于移动底盘）
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)

            # 合并机械臂和底盘的动作指令
            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

        # 无控制源的情况：跳过动作生成
        # 这种情况通常在环境重置且没有遥操作设备时发生
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue  # 跳过当前循环迭代

        # 注意：动作可能会被机器人的安全限制裁剪（如最大相对目标限制）
        # 因此实际发送的动作会被保存到数据集中，以确保数据的一致性

        # 处理重置到零位事件
        # 用户可以通过VR控制器触发机器人回到零位姿态
        if events["reset_position"]:
            logging.info("Rest to the zero position of robot")
            log_say("Reset position")  # 语音提示
            action = teleop.move_to_zero_position(robot)  # 执行零位重置动作
            # 重置事件状态，避免重复触发
            teleop.vr_event_handler.events['reset_position'] = False
            events["reset_position"] = False

        # 处理返回后台位置事件
        # 将机器人移动到预设的后台位置，通常用于安全停放或准备位置
        elif events["back_position"]:
            logging.info("Back to the backet position of robot")
            log_say("Back backet position")  # 语音提示

            # 检查动作队列容量，避免过度填充
            if len(action_queue) < 10:
                # 执行三阶段动作序列：
                # 1. 移动到后台位置（GLOBAL_BACK_GOAL）
                queue_reset_actions(action_queue, robot, GLOBAL_BACK_GOAL, steps=30)

                # 2. 从后台位置移动到打开位置（GLOBAL_OPEN_GOAL）
                queue_reset_actions(action_queue, robot, GLOBAL_OPEN_GOAL, steps=10, start_position=GLOBAL_BACK_GOAL)

                # 3. 从打开位置回到零位
                queue_reset_actions(action_queue, robot, np.zeros(12), steps=30, start_position=GLOBAL_OPEN_GOAL)

                # 添加重置标志，触发零位重置
                action_queue.append({})  # reset to zero flag

            # 重置事件状态
            teleop.vr_event_handler.events['back_position'] = False
            events["back_position"] = False  # reset event state

        # 发送动作到机器人执行
        # send_action方法可能应用安全限制，因此返回实际执行的动作
        sent_action = robot.send_action(action)

        # 将数据保存到数据集（如果启用录制）
        if dataset is not None:
            # 构建动作帧格式
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")

            # 合并观察帧、动作帧和任务标签，形成完整的数据帧
            frame = {**observation_frame, **action_frame, "task": single_task}

            # 将数据帧添加到数据集缓冲区
            dataset.add_frame(frame)

        # 实时数据显示（用于调试和监控）
        if display_data:
            log_rerun_data(observation, action)

        # 精确时间控制：计算循环执行时间并等待到下一帧时间点
        dt_s = time.perf_counter() - start_loop_t  # 计算本轮循环耗时
        busy_wait(1 / fps - dt_s)                  # 等待剩余时间，维持恒定帧率

        # 更新回合计时器
        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    """
    机器人数据集录制主函数

    这是整个数据录制系统的入口点，负责初始化所有组件并协调整个录制过程。
    该函数支持完整的录制生命周期，包括设备连接、数据录制、保存和上传。

    主要功能:
    1. 系统初始化：日志、显示、机器人、遥操作器
    2. 数据集配置：创建新数据集或恢复现有录制
    3. 录制循环管理：多回合数据采集和控制
    4. 资源管理：设备连接、进程管理、安全退出
    5. 数据处理：视频编码、特征提取、保存上传

    录制流程:
    1. 初始化所有系统组件
    2. 设置数据集（新建或恢复）
    3. 连接机器人和控制设备
    4. 启动录制循环直到达到指定回合数
    5. 保存数据并可选择上传到Hugging Face Hub

    Args:
        cfg (RecordConfig): 完整的录制配置，包含机器人、数据集、遥操作等设置

    Returns:
        LeRobotDataset: 录制完成的机器人数据集对象

    Note:
        - 支持断点续录功能
        - 自动处理设备连接和错误恢复
        - 集成Hugging Face Hub用于数据集共享
    """
    # 初始化日志系统，输出详细配置信息
    init_logging()
    logging.info(pformat(asdict(cfg)))  # 以格式化方式输出完整配置

    # 初始化数据显示系统（如果启用）
    # 用于实时监控录制过程和调试
    if cfg.display_data:
        init_rerun(session_name="recording")

    # 根据配置创建机器人和遥操作器实例
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # 构建数据集特征描述
    # 将硬件特性映射为数据集格式，支持动作和观察数据
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    # 数据集初始化：支持恢复模式或新建模式
    if cfg.resume:
        # 恢复现有数据集录制
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        # 启动图像写入进程（如果机器人有摄像头）
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        # 验证现有数据集与当前机器人的兼容性
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)

    else:
        # 创建新的空数据集或加载已存在的回合
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)  # 验证数据集名称有效性
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,                        # 数据集仓库ID
            cfg.dataset.fps,                           # 录制帧率
            root=cfg.dataset.root,                     # 本地存储路径
            robot_type=robot.name,                     # 机器人类型
            features=dataset_features,                 # 数据特征定义
            use_videos=cfg.dataset.video,              # 是否使用视频格式
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    # 加载预训练策略（如果配置了策略控制）
    # 策略可以用于自动演示、数据增强或半自动录制
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    # 连接机器人硬件设备
    # 建立与机器人控制器的通信连接
    robot.connect()

    # 连接遥操作设备（如果配置了遥操作）
    if teleop is not None:
        teleop.connect(robot=robot)      # 将遥操作器与机器人关联
        teleop.send_feedback()           # 发送反馈信息（如触觉反馈）

    # 根据遥操作器类型选择相应的事件监听器
    if isinstance(teleop, XLerobotVRTeleop):
        # 使用VR控制器监听器
        listener, events = init_vr_listener(teleop)
        logging.info("🎮 Using VR to control recording status")
    else:
        # 使用键盘监听器（默认选项）
        listener, events = init_keyboard_listener()
        logging.info("⌨️ Using keyboard to control recording status")

    # 使用视频编码管理器确保视频文件正确处理
    # 这个上下文管理器处理视频编码进程的生命周期
    with VideoEncodingManager(dataset):
        recorded_episodes = 0  # 已录制回合计数器

        # 主录制循环：持续录制直到达到目标回合数或用户停止
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            # 语音提示开始录制新回合
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            time.sleep(3)  # 给用户3秒准备时间

            # 执行数据录制循环
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop=teleop,
                policy=policy,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            # 环境重置阶段：给用户时间手动重置环境
            # 跳过最后一个录制回合的重置（避免不必要的等待）
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset environment", cfg.play_sounds)
                # 执行环境重置循环（不录制数据）
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    dataset=None,  # 不录制数据到数据集
                )

            # 处理重新录制当前回合的事件
            # 用户可以删除刚录制的回合并重新录制
            if events["rerecord_episode"]:
                log_say("Delete Again record", cfg.play_sounds)  # 语音提示
                # 重置所有相关事件状态
                events["rerecord_episode"] = False
                events["exit_early"] = False
                teleop.vr_event_handler.events['rerecord_episode'] = False

                # 清除当前回合的缓冲数据
                dataset.clear_episode_buffer()
                time.sleep(10)  # 给用户10秒时间准备重新录制
                continue  # 跳过保存，重新开始录制当前回合

            # 保存当前录制回合
            log_say("Saving Episode", cfg.play_sounds, blocking=True)  # 阻塞式语音提示
            dataset.save_episode()
            recorded_episodes += 1  # 增加已录制回合计数

    # 录制完成，语音提示停止录制
    log_say("Stop recording", cfg.play_sounds, blocking=True)

    # 安全断开所有设备连接
    robot.disconnect()  # 断开机器人连接
    if teleop is not None:
        teleop.disconnect()  # 断开遥操作器连接

    # 停止事件监听器（如果在非无头模式下运行）
    if not is_headless() and listener is not None:
        listener.stop()

    # 上传数据集到Hugging Face Hub（如果启用）
    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    # 语音提示程序退出
    log_say("Exiting", cfg.play_sounds)
    return dataset  # 返回录制完成的数据集对象


def main():
    """
    程序主入口点

    这是record.py脚本的主函数，直接调用record()函数开始数据录制过程。
    使用命令行参数来配置录制选项。
    """
    record()  # 开始录制过程


if __name__ == "__main__":
    # 当脚本被直接执行时运行主函数
    # 支持命令行参数：python record.py --help 查看所有选项
    main()
