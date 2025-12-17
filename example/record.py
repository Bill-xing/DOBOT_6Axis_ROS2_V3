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
    # 数据集标识符。按照惯例，它应该匹配'{hf_username}/{dataset_name}'（例如 `lerobot/test`）。
    repo_id: str
    # 录制期间执行的任务的简短但准确的描述（例如"拾取乐高积木并将其放入右侧的盒子中"）。
    single_task: str
    # 数据集存储的根目录（例如 'dataset/path'）。
    root: str | Path | None = None
    # 限制每秒帧数。
    fps: int = 30
    # 每个回合数据录制的秒数。
    episode_time_s: int | float = 60
    # 每个回合后重置环境的秒数。
    reset_time_s: int | float = 10
    # 要录制的回合数。
    num_episodes: int = 50
    # 将数据集中的帧编码为视频
    video: bool = True
    # 上传数据集到Hugging Face hub。
    push_to_hub: bool = False
    # 上传到Hugging Face hub上的私有存储库。
    private: bool = False
    # 在hub上为您的数据集添加标签。
    tags: list[str] | None = None
    # 处理将帧保存为PNG的子进程数。设置为0则仅使用线程；
    # 设置为≥1以使用子进程，每个子进程使用线程写入图像。最佳进程数
    # 和线程数取决于您的系统。我们建议每个摄像头4个线程，0个进程。
    # 如果fps不稳定，请调整线程数。如果仍然不稳定，请尝试使用1个或更多子进程。
    num_image_writer_processes: int = 0
    # 每个摄像头将帧作为png图像写入磁盘的线程数。
    # 太多线程可能导致主线程被阻塞，从而导致遥操作fps不稳定。
    # 线程不足可能导致摄像头fps低。
    num_image_writer_threads_per_camera: int = 4
    # 批量编码视频前要录制的回合数
    # 设置为1进行立即编码（默认行为），或设置更高值进行批量编码
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
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so100_leader.SO100Leader,
                        so101_leader.SO101Leader,
                        koch_leader.KochLeader,
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    # Init action queue
    if action_queue is None:
        action_queue = deque()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        current_events = teleop.get_vr_events()
        events.update(current_events)


        if events["exit_early"]:
            events["exit_early"] = False
            log_say("Exit early")
            time.sleep(1)
            break

        try:
            observation = robot.get_observation()
        except TimeoutError as e:
            logging.warning(f"Camera timeout: {e}. Skipping this frame.")
            continue  # skip current

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
        
        # 如果队列不为None，使用队列中的动作
        if action_queue:
            action = action_queue.popleft()
            if action == {}: # 重置标志
                action = teleop.move_to_zero_position(robot)

        elif policy is not None:
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        elif policy is None and isinstance(teleop, Teleoperator):
            action = teleop.get_action(observation, robot)
        elif policy is None and isinstance(teleop, list):
            # TODO(pepijn, steven): 清理记录循环以用于多个机器人（可能使用pipeline）
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)

            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.

        if events["reset_position"]:
            logging.info("Rest to the zero position of robot")
            log_say("Reset position")
            action = teleop.move_to_zero_position(robot)
            teleop.vr_event_handler.events['reset_position'] = False
            events["reset_position"] = False 
        elif events["back_position"]:
            logging.info("Back to the backet position of robot")
            log_say("Back backet position")
            if len(action_queue) < 10:  
                # Place in back and reset
                queue_reset_actions(action_queue, robot, GLOBAL_BACK_GOAL, steps=30)
                queue_reset_actions(action_queue, robot, GLOBAL_OPEN_GOAL, steps=10, start_position=GLOBAL_BACK_GOAL)
                queue_reset_actions(action_queue, robot, np.zeros(12), steps=30, start_position=GLOBAL_OPEN_GOAL)
                action_queue.append({}) # reset to zero flag
            teleop.vr_event_handler.events['back_position'] = False
            events["back_position"] = False  # reset event state
            
        sent_action = robot.send_action(action)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation, action)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    robot.connect()
    if teleop is not None:
        teleop.connect(robot=robot)
        teleop.send_feedback()

    # Select acording teleoperator
    if isinstance(teleop, XLerobotVRTeleop):
        # Use VR listener
        listener, events = init_vr_listener(teleop)
        logging.info("🎮 Using VR to control recording status")
    else:
        # Use keyboard listener
        listener, events = init_keyboard_listener()
        logging.info("⌨️ Using keyboard to control recording status")

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            time.sleep(3)
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

            # 执行几秒钟不录制，给时间手动重置环境
            # 跳过要录制的最后一个回合的重置
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset environment", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    dataset=None,
                )

            if events["rerecord_episode"]:
                log_say("Delete Again record", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                teleop.vr_event_handler.events['rerecord_episode'] = False

                dataset.clear_episode_buffer()
                time.sleep(10)
                continue

            log_say("Saving Episode", cfg.play_sounds, blocking=True)

            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
