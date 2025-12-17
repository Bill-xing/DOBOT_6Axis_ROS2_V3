#!/usr/bin/env python3
"""
简化的SO100/SO101机器人键盘控制程序
修复了动作格式转换问题
使用P控制，键盘只改变目标关节角度
"""

import time
import logging
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 关节标定系数配置 - 需要根据实际机器人进行手动调整
#
# 配置格式：[关节名称, 零位偏移量(度), 缩放因子]
#
# 零位偏移量说明：
#   - 正值：物理零位在传感器零位之后（需要减去偏移）
#   - 负值：物理零位在传感器零位之前（需要加上偏移）
#   - 零值：传感器零位与物理零位对齐
#
# 缩放因子说明：
#   - 大于1：传感器读数偏小，需要放大（scale=1.05表示放大5%）
#   - 小于1：传感器读数偏大，需要缩小（scale=0.97表示缩小3%）
#   - 等于1：传感器读数准确，无需缩放
#
# 各关节具体说明：
# - shoulder_pan：基座旋转关节，控制机械臂水平旋转
# - shoulder_lift：肩部抬升关节，控制机械臂上下运动
# - elbow_flex：肘部弯曲关节，控制大臂与小臂角度
# - wrist_flex：手腕弯曲关节，控制手腕上下弯曲
# - wrist_roll：手腕旋转关节，控制手腕自转
# - gripper：夹爪关节，控制夹爪开合

JOINT_CALIBRATION = [
    # 基座旋转关节
    ['shoulder_pan', 6.0, 1.0],      # 零位偏移+6度，无缩放（常见于基座安装误差）

    # 肩部抬升关节
    ['shoulder_lift', 2.0, 0.97],     # 零位偏移+2度，缩小3%（考虑重力影响）

    # 肘部弯曲关节
    ['elbow_flex', 0.0, 1.05],        # 无零位偏移，放大5%（传动比误差）

    # 手腕弯曲关节
    ['wrist_flex', 0.0, 0.94],        # 无零位偏移，缩小6%（机械间隙补偿）

    # 手腕旋转关节 - 注意：此关节缩放因子为0.5
    ['wrist_roll', 0.0, 0.5],        # 无零位偏移，缩小50%（齿轮比特殊设计）

    # 夹爪关节
    ['gripper', 0.0, 1.0],           # 无零位偏移，无缩放（标准配置）
]

def apply_joint_calibration(joint_name, raw_position):
    """
    应用关节标定系数，修正原始传感器读数以获得更准确的关节位置

    这个函数解决了机械臂制造、安装过程中产生的误差问题，通过标定系数
    将原始传感器数据转换为实际物理角度。

    标定公式：标定后位置 = (原始位置 - 零位偏移) × 缩放因子

    Args:
        joint_name (str): 关节名称，如'shoulder_pan', 'elbow_flex'等
        raw_position (float): 从机器人传感器读取的原始位置值（度）

    Returns:
        float: 标定后的位置值（度），如果未找到对应关节数据则返回原始值

    Example:
        >>> # 假设elbow_flex关节的标定系数为 [0.0, 1.05]
        >>> raw_pos = 30.0  # 传感器原始读数30度
        >>> calibrated_pos = apply_joint_calibration('elbow_flex', raw_pos)
        >>> print(f"标定后位置: {calibrated_pos:.1f}度")
        标定后位置: 31.5度
    """

    # 遍历所有关节数据，查找匹配的关节配置
    for joint_cal in JOINT_CALIBRATION:

        # joint_cal格式：[关节名称, 零位偏移量, 缩放因子]
        # 例如：['elbow_flex', 0.0, 1.05]

        if joint_cal[0] == joint_name:

            # 提取标定参数
            offset = joint_cal[1]  # 零位偏移量（度）
            scale = joint_cal[2]   # 缩放因子（无量纲）

            """
            标定计算详解：

            1. 减去零位偏移：修正机械零位误差
               - 如果offset=2.0，表示当传感器读数为0时，实际关节位置为2度
               - 需要从原始读数中减去这个偏移量

            2. 乘以缩放因子：修正传动比误差
               - 如果scale=1.05，表示传感器读数比实际角度小5%
               - 需要乘以缩放因子放大读数

            实际应用场景：
            - offset > 0：机械零位偏正，传感器0点在物理零位之前
            - offset < 0：机械零位偏负，传感器0点在物理零位之后
            - scale > 1：传感器读数偏小，需要放大
            - scale < 1：传感器读数偏大，需要缩小
            """

            # 应用标定公式
            calibrated_position = (raw_position - offset) * scale

            # 调试信息（可选，生产环境可注释掉）
            # if abs(calibrated_position - raw_position) > 0.1:
            #     logger.info(f"关节 {joint_name}: 原始={raw_position:.2f}°, "
            #                f"标定={calibrated_position:.2f}°, "
            #                f"偏移={offset:.2f}°, 缩放={scale:.3f}")

            return calibrated_position

    # 如果没有找到对应的关节配置
    # 这种情况可能发生在：
    # 1. 新添加的关节未在配置中定义
    # 2. 关节名称拼写错误
    # 3. 配置文件缺失或损坏

    # 记录警告信息（可选）
    # logger.warning(f"未找到关节 '{joint_name}' 的标定配置，使用原始值")

    return raw_position  # 返回未修改的原始位置值


def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    使用P控制缓慢移动机器人到零位

    Args:
        robot: 机器人实例
        duration: 移动到零位所需时间(秒)
        kp: 比例增益
    """
    print("使用P控制缓慢移动机器人到零位...")

    # 获取当前机器人状态
    current_obs = robot.get_observation()

    # 提取当前关节位置
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value

    # 零位目标
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }

    # 计算控制步数
    control_freq = 50  # 50Hz控制频率
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    print(f"将在{duration}秒内使用P控制移动到零位，控制频率: {control_freq}Hz，比例增益: {kp}")

    for step in range(total_steps):
        # 获取当前机器人状态
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # 应用标定系数
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value

        # P控制计算
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos

                # P控制: 输出 = Kp * 误差
                control_output = kp * error

                # 转换控制输出为位置命令
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        # 发送动作到机器人
        if robot_action:
            robot.send_action(robot_action)

        # 显示进度
        if step % (control_freq // 2) == 0:  # 每0.5秒显示一次进度
            progress = (step / total_steps) * 100
            print(f"移动到零位进度: {progress:.1f}%")

        time.sleep(step_time)

    print("机器人已移动到零位")

def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    使用P控制返回起始位置

    Args:
        robot: 机器人实例
        start_positions: 起始关节位置字典
        kp: 比例增益
        control_freq: 控制频率(Hz)
    """
    print("返回起始位置...")

    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # 最大5秒

    for step in range(max_steps):
        # 获取当前机器人状态
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value  # 不应用标定系数

        # P控制计算
        robot_action = {}
        total_error = 0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)

                # P控制: 输出 = Kp * 误差
                control_output = kp * error

                # 转换控制输出为位置命令
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        # 发送动作到机器人
        if robot_action:
            robot.send_action(robot_action)

        # 检查是否到达起始位置
        if total_error < 2.0:  # 如果总误差小于2度，认为已到达
            print("已返回起始位置")
            break

        time.sleep(control_period)

    print("返回起始位置完成")

def p_control_loop(robot, keyboard, target_positions, start_positions, kp=0.5, control_freq=50):
    """
    P控制循环

    Args:
        robot: 机器人实例
        keyboard: 键盘实例
        target_positions: 目标关节位置字典
        start_positions: 起始关节位置字典
        kp: 比例增益
        control_freq: 控制频率(Hz)
    """
    control_period = 1.0 / control_freq

    print(f"启动P控制循环，控制频率: {control_freq}Hz，比例增益: {kp}")

    while True:
        try:
            # 获取键盘输入
            keyboard_action = keyboard.get_action()

            if keyboard_action:
                # 处理键盘输入，更新目标位置
                for key, value in keyboard_action.items():
                    if key == 'x':
                        # 退出程序，先返回起始位置
                        print("检测到退出命令，返回起始位置...")
                        return_to_start_position(robot, start_positions, 0.2, control_freq)
                        return

                    # 关节控制映射
                    joint_controls = {
                        'q': ('shoulder_pan', -1),    # 关节1减小
                        'a': ('shoulder_pan', 1),     # 关节1增大
                        'w': ('shoulder_lift', -1),   # 关节2减小
                        's': ('shoulder_lift', 1),    # 关节2增大
                        'e': ('elbow_flex', -1),      # 关节3减小
                        'd': ('elbow_flex', 1),       # 关节3增大
                        'r': ('wrist_flex', -1),      # 关节4减小
                        'f': ('wrist_flex', 1),       # 关节4增大
                        't': ('wrist_roll', -1),      # 关节5减小
                        'g': ('wrist_roll', 1),       # 关节5增大
                        'y': ('gripper', -1),         # 关节6减小
                        'h': ('gripper', 1),          # 关节6增大
                    }

                    if key in joint_controls:
                        joint_name, delta = joint_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"更新目标位置 {joint_name}: {current_target} -> {new_target}")

            # 获取当前机器人状态
            current_obs = robot.get_observation()

            # 提取当前关节位置
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # 应用标定系数
                    calibrated_value = apply_joint_calibration(motor_name, value)
                    current_positions[motor_name] = calibrated_value

            # P控制计算
            robot_action = {}
            for joint_name, target_pos in target_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos

                    # P控制: 输出 = Kp * 误差
                    control_output = kp * error

                    # 转换控制输出为位置命令
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position

            # 发送动作到机器人
            if robot_action:
                robot.send_action(robot_action)

            time.sleep(control_period)

        except KeyboardInterrupt:
            print("用户中断程序")
            break
        except Exception as e:
            print(f"P控制循环错误: {e}")
            traceback.print_exc()
            break

def main():
    """主函数"""
    print("LeRobot 简化键盘控制示例 (P控制)")
    print("="*50)

    try:
        # 导入必要的模块
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

        # 获取端口
        port = input("请输入SO100机器人USB端口 (例如: /dev/ttyACM0): ").strip()

        # 如果直接按回车，使用默认端口
        if not port:
            port = "/dev/ttyACM0"
            print(f"使用默认端口: {port}")
        else:
            print(f"连接到端口: {port}")

        # 配置机器人
        robot_config = SO100FollowerConfig(port=port)
        robot = SO100Follower(robot_config)

        # 配置键盘
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)

        # 连接设备
        robot.connect()
        keyboard.connect()

        print("设备连接成功!")

        # 询问是否重新标定
        while True:
            calibrate_choice = input("是否要重新标定机器人? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes']:
                print("开始重新标定...")
                robot.calibrate()
                print("标定完成!")
                break
            elif calibrate_choice in ['n', 'no']:
                print("使用之前的标定文件")
                break
            else:
                print("请输入 y 或 n")

        # 读取起始关节角度
        print("读取起始关节角度...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                start_positions[motor_name] = int(value)  # 不应用标定系数

        print("起始关节角度:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")

        # 移动到零位
        move_to_zero_position(robot, duration=3.0)

        # 初始化目标位置为当前位置(整数)
        target_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
          }


        print("键盘控制说明:")
        print("- Q/A: 关节1 (shoulder_pan) 减小/增大")
        print("- W/S: 关节2 (shoulder_lift) 减小/增大")
        print("- E/D: 关节3 (elbow_flex) 减小/增大")
        print("- R/F: 关节4 (wrist_flex) 减小/增大")
        print("- T/G: 关节5 (wrist_roll) 减小/增大")
        print("- Y/H: 关节6 (gripper) 减小/增大")
        print("- X: 退出程序 (先返回起始位置)")
        print("- ESC: 退出程序")
        print("="*50)
        print("注意: 机器人将持续移动到目标位置")

        # 启动P控制循环
        p_control_loop(robot, keyboard, target_positions, start_positions, kp=0.5, control_freq=50)

        # 断开连接
        robot.disconnect()
        keyboard.disconnect()
        print("程序结束")

    except Exception as e:
        print(f"程序执行失败: {e}")
        traceback.print_exc()
        print("请检查:")
        print("1. 机器人是否正确连接")
        print("2. USB端口是否正确")
        print("3. 是否有足够的权限访问USB设备")
        print("4. 机器人是否正确配置")

if __name__ == "__main__":
    main() 