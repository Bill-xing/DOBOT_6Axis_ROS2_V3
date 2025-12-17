# 在主机上运行
'''python
PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot
'''

# 运行遥操作:
'''python
PYTHONPATH=src python -m examples.xlerobot.teleoperate_Keyboard
'''

import time
import numpy as np
import math

from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
# from lerobot.robots.xlerobot import XLerobotClient, XLerobotClientConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig

# 按键映射 (语义动作: 按键)
LEFT_KEYMAP = {
    'shoulder_pan+': 'q', 'shoulder_pan-': 'e',
    'wrist_roll+': 'r', 'wrist_roll-': 'f',
    'gripper+': 't', 'gripper-': 'g',
    'x+': 'w', 'x-': 's', 'y+': 'a', 'y-': 'd',
    'pitch+': 'z', 'pitch-': 'x',
    'reset': 'c',
    # 头部电机
    "head_motor_1+": "<", "head_motor_1-": ">",
    "head_motor_2+": ",", "head_motor_2-": ".",

    'triangle': 'y',  # 矩形轨迹键
}
RIGHT_KEYMAP = {
    'shoulder_pan+': '7', 'shoulder_pan-': '9',
    'wrist_roll+': '/', 'wrist_roll-': '*',
    'gripper+': '+', 'gripper-': '-',
    'x+': '8', 'x-': '2', 'y+': '4', 'y-': '6',
    'pitch+': '1', 'pitch-': '3',
    'reset': '0',

    'triangle': 'Y',  # 矩形轨迹键
}

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

# 头部电机映射
HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class RectangularTrajectory:
    """
    在x-y平面上生成具有正弦速度曲线的矩形轨迹。
    矩形被分为4个线段，每段都有平滑的加速/减速。
    """
    def __init__(self, width=0.06, height=0.06, segment_duration=0.91):
        """
        初始化矩形轨迹参数。

        Args:
            width: 矩形宽度(米)
            height: 矩形高度(米)
            segment_duration: 每条线段的时间(秒)
        """
        self.width = width
        self.height = height
        self.segment_duration = segment_duration
        self.total_duration = 4 * segment_duration

    def get_trajectory_point(self, current_x, current_y, t):
        """
        获取时间t时矩形轨迹的目标x,y位置。

        Args:
            current_x: 起始x位置
            current_y: 起始y位置
            t: 轨迹开始后的时间(0到total_duration)

        Returns:
            tuple: (target_x, target_y)
        """
        # 确定我们在哪个线段
        segment = int(t / self.segment_duration)
        segment_t = t % self.segment_duration

        # 标准化线段时间(0到1)
        normalized_t = segment_t / self.segment_duration

        # 正弦速度曲线: 平滑加速和减速
        # s(t) = 0.5 * (1 - cos(π * t)) 给出平滑的0到1过渡
        smooth_t = 0.5 * (1 - math.cos(math.pi * normalized_t))

        # 定义相对于起始位置的矩形角点
        corners = [
            (current_x, current_y),                           # 起点(左下)
            (current_x + self.width, current_y),              # 右下
            (current_x + self.width, current_y + self.height), # 右上
            (current_x, current_y + self.height),             # 左上
            (current_x, current_y)                            # 回到起点
        ]

        # 将线段限制在有效范围
        segment = max(0, min(3, segment))

        # 在当前角点和下一个角点之间插值
        start_corner = corners[segment]
        end_corner = corners[segment + 1]

        target_x = start_corner[0] + smooth_t * (end_corner[0] - start_corner[0])
        target_y = start_corner[1] + smooth_t * (end_corner[1] - start_corner[1])

        return target_x, target_y

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=0.81):
        self.kp = kp
        self.degree_step = 1
        # 初始化头部电机位置
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_keys(self, key_state):
        if key_state.get('head_motor_1+'):
            self.target_positions["head_motor_1"] += self.degree_step
            print(f"[头部] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_1-'):
            self.target_positions["head_motor_1"] -= self.degree_step
            print(f"[头部] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_2+'):
            self.target_positions["head_motor_2"] += self.degree_step
            print(f"[头部] head_motor_2: {self.target_positions['head_motor_2']}")
        if key_state.get('head_motor_2-'):
            self.target_positions["head_motor_2"] -= self.degree_step
            print(f"[头部] head_motor_2: {self.target_positions['head_motor_2']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action

class SimpleTeleopArm:
    def __init__(self, kinematics, joint_map, initial_obs, prefix="left", kp=0.81):
        self.kinematics = kinematics
        self.joint_map = joint_map
        self.prefix = prefix  # To distinguish left and right arm
        self.kp = kp
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        # Set the degree step and xy step
        self.degree_step = 3
        self.xy_step = 0.0081
        # Set target positions to zero for P control
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        
        # 矩形轨迹实例
        self.rectangular_trajectory = RectangularTrajectory(
            width=0.06,          # 6cm宽矩形
            height=0.06,         # 6cm高矩形
            segment_duration=1.01 # 每条线段1.01秒
        )

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] 移动到零位: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()  # 使用副本避免引用问题

        # 将运动学变量重置为初始状态
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0

        # 不让handle_keys重新计算wrist_flex - 显式设置
        self.target_positions["wrist_flex"] = 0.0

        action = self.p_control_action(robot)
        robot.send_action(action)

    def execute_rectangular_trajectory(self, robot, fps=30):
        """
        在x-y平面上执行阻塞式矩形轨迹。

        Args:
            robot: 发送动作的机器人实例
            fps: 控制循环频率
        """
        print(f"[{self.prefix}] 启动矩形轨迹...")
        print(f"[{self.prefix}] 矩形: {self.rectangular_trajectory.width:.3f}m x {self.rectangular_trajectory.height:.3f}m")
        print(f"[{self.prefix}] 持续时间: {self.rectangular_trajectory.total_duration:.3f}s总计")

        # 存储起始位置
        start_x = self.current_x
        start_y = self.current_y

        # 执行轨迹
        start_time = time.time()
        dt = 1.0 / fps

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # 检查轨迹是否完成
            if elapsed_time >= self.rectangular_trajectory.total_duration:
                print(f"[{self.prefix}] 矩形轨迹完成!")
                break

            # 从轨迹获取目标位置
            target_x, target_y = self.rectangular_trajectory.get_trajectory_point(
                start_x, start_y, elapsed_time
            )

            # 更新当前位置
            self.current_x = target_x
            self.current_y = target_y

            # 计算逆运动学
            try:
                joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
                self.target_positions["shoulder_lift"] = joint2
                self.target_positions["elbow_flex"] = joint3

                # 更新wrist_flex耦合
                self.target_positions["wrist_flex"] = (
                    -self.target_positions["shoulder_lift"]
                    -self.target_positions["elbow_flex"]
                    + self.pitch
                )

                # 获取动作
                action = self.p_control_action(robot)

                # 确定哪个手臂在执行并发送适当的动作结构
                if self.prefix == "left":
                    # 发送左臂动作，其他组件为空动作
                    robot_action = {**action, **{}, **{}, **{}}
                elif self.prefix == "right":
                    # 发送右臂动作，其他组件为空动作
                    robot_action = {**{}, **action, **{}, **{}}

                # 发送动作到机器人
                robot.send_action(robot_action)

                # 获取观察值并记录数据
                obs = robot.get_observation()
                log_rerun_data(obs, robot_action)

            except Exception as e:
                print(f"[{self.prefix}] 在x={self.current_x:.4f}, y={self.current_y:.4f}时IK失败: {e}")
                break

            # 保持控制频率
            # busy_wait(dt)

        print(f"[{self.prefix}] 轨迹执行完成。")

    def handle_keys(self, key_state):
        # Joint increments
        if key_state.get('shoulder_pan+'):
            self.target_positions["shoulder_pan"] += self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('shoulder_pan-'):
            self.target_positions["shoulder_pan"] -= self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('wrist_roll+'):
            self.target_positions["wrist_roll"] += self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('wrist_roll-'):
            self.target_positions["wrist_roll"] -= self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('gripper+'):
            self.target_positions["gripper"] += self.degree_step
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('gripper-'):
            self.target_positions["gripper"] -= self.degree_step
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('pitch+'):
            self.pitch += self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")
        if key_state.get('pitch-'):
            self.pitch -= self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")

        # XY plane (IK)
        moved = False
        if key_state.get('x+'):
            self.current_x += self.xy_step
            moved = True
            print(f"[{self.prefix}] x+: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('x-'):
            self.current_x -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x-: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('y+'):
            self.current_y += self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y+: {self.current_y:.4f}")
        if key_state.get('y-'):
            self.current_y -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y-: {self.current_y:.4f}")
        if moved:
            joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            self.target_positions["shoulder_lift"] = joint2
            self.target_positions["elbow_flex"] = joint3
            print(f"[{self.prefix}] shoulder_lift: {joint2}, elbow_flex: {joint3}")

        # Wrist flex is always coupled to pitch and the other two
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            -self.target_positions["elbow_flex"]
            + self.pitch
        )
        # print(f"[{self.prefix}] wrist_flex: {self.target_positions['wrist_flex']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action
    

def main():
    # 遥操作参数
    FPS = 50
    # ip = "192.168.1.123"  # 用于zmq连接
    ip = "localhost"  # 用于本地/有线连接
    robot_name = "my_xlerobot_pc"

    # 用于zmq连接
    # robot_config = XLerobotClientConfig(remote_ip=ip, id=robot_name)
    # robot = XLerobotClient(robot_config)

    # 用于本地/有线连接
    robot_config = XLerobotConfig()
    robot = XLerobot(robot_config)

    try:
        robot.connect()
        print(f"[主程序] 成功连接到机器人")
    except Exception as e:
        print(f"[主程序] 连接机器人失败: {e}")
        print(robot_config)
        print(robot)
        return

    init_rerun(session_name="xlerobot_teleop_v2")

    # 初始化键盘实例
    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)
    keyboard.connect()

    # 初始化手臂和头部实例
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(kin_left, LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = SimpleTeleopArm(kin_right, RIGHT_JOINT_MAP, obs, prefix="right")
    head_control = SimpleHeadControl(obs)

    # 启动时将双臂和头部移动到零位
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)

    try:
        while True:
            pressed_keys = set(keyboard.get_action().keys())
            left_key_state = {action: (key in pressed_keys) for action, key in LEFT_KEYMAP.items()}
            right_key_state = {action: (key in pressed_keys) for action, key in RIGHT_KEYMAP.items()}

            # 处理左臂矩形轨迹(y键)
            if left_key_state.get('triangle'):
                print("[主程序] 左臂矩形轨迹触发!")
                left_arm.execute_rectangular_trajectory(robot, fps=FPS)
                continue

            # 处理右臂矩形轨迹(Y键)
            if right_key_state.get('triangle'):
                print("[主程序] 右臂矩形轨迹触发!")
                right_arm.execute_rectangular_trajectory(robot, fps=FPS)
                continue

            # 处理左臂重置
            if left_key_state.get('reset'):
                left_arm.move_to_zero_position(robot)
                continue

            # 处理右臂重置
            if right_key_state.get('reset'):
                right_arm.move_to_zero_position(robot)
                continue

            # 用'?'处理头部电机重置
            if '?' in pressed_keys:
                head_control.move_to_zero_position(robot)
                continue

            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)
            head_control.handle_keys(left_key_state)  # 头部由左臂键映射控制

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)
            head_action = head_control.p_control_action(robot)

            # 底座动作
            keyboard_keys = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

            action = {**left_action, **right_action, **head_action, **base_action}
            robot.send_action(action)

            obs = robot.get_observation()
            # print(f"[主程序] 观测值: {obs}")
            log_rerun_data(obs, action)
            # busy_wait(1.0 / FPS)
    finally:
        robot.disconnect()
        keyboard.disconnect()
        print("遥操作结束。")

if __name__ == "__main__":
    main()
