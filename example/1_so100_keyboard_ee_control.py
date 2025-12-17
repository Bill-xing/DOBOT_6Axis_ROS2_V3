#!/usr/bin/env python3
"""
简化的SO100/SO101机器人键盘控制程序
修复了动作格式转换问题
使用P控制，键盘只改变目标关节角度
支持末端执行器坐标控制
"""

import time
import logging
import traceback
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 关节标定系数 - 手动编辑
# 格式: [关节名称, 零位偏移量(度), 缩放因子]
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      # 关节1: 零位偏移量, 缩放因子
    ['shoulder_lift', 2.0, 0.97],     # 关节2: 零位偏移量, 缩放因子
    ['elbow_flex', 0.0, 1.05],        # 关节3: 零位偏移量, 缩放因子
    ['wrist_flex', 0.0, 0.94],        # 关节4: 零位偏移量, 缩放因子
    ['wrist_roll', 0.0, 0.5],        # 关节5: 零位偏移量, 缩放因子
    ['gripper', 0.0, 1.0],           # 关节6: 零位偏移量, 缩放因子
]

def apply_joint_calibration(joint_name, raw_position):
    """
    应用关节标定系数

    Args:
        joint_name: 关节名称
        raw_position: 原始位置值

    Returns:
        calibrated_position: 标定后的位置值
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # 零位偏移量
            scale = joint_cal[2]   # 缩放因子
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # 如果找不到标定系数，返回原始值

def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    计算2连杆机械臂的逆运动学，考虑关节偏移

    Parameters:
        x: 末端执行器x坐标
        y: 末端执行器y坐标
        l1: 上臂长度 (默认 0.1159 m)
        l2: 下臂长度 (默认 0.1350 m)

    Returns:
        joint2, joint3: URDF文件中定义的关节角度(弧度)
    """
    # 计算关节2和关节3在theta1和theta2中的偏移
    theta1_offset = math.atan2(0.028, 0.11257)  # 关节2=0时的theta1偏移
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # 关节3=0时的theta2偏移

    # 计算从原点到目标点的距离
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2  # 最大可达距离

    # 如果目标点超出最大工作空间，缩放到边界
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max

    # 如果目标点小于最小工作空间(|l1-l2|)，缩放它
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min

    # 使用余弦定理计算theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)

    # 计算theta2（肘部角度）
    theta2 = math.pi - math.acos(cos_theta2)

    # 计算theta1（肩部角度）
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    # 将theta1和theta2转换为关节2和关节3角度
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset

    # 确保角度在URDF限制范围内
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))

    # 从弧度转换为度数
    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    joint2_deg = 90-joint2_deg
    joint3_deg = joint3_deg-90

    return joint2_deg, joint3_deg

def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    使用P控制缓慢移动机器人到零位

    Args:
        robot: 机器人实例
        duration: 移动到零位所需时间(秒)
        kp: 比例增益
    """
    print("Using P control to slowly move robot to zero position...")
    
    # Get current robot state
    current_obs = robot.get_observation()
    
    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value
    
    # Zero position targets
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"Will use P control to move to zero position in {duration} seconds, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value
        
        # P control calculation
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                
                # P control: output = Kp * error
                control_output = kp * error
                
                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)
        
        # Show progress
        if step % (control_freq // 2) == 0:  # Show progress every 0.5 seconds
            progress = (step / total_steps) * 100
            print(f"Moving to zero position progress: {progress:.1f}%")
        
        time.sleep(step_time)
    
    print("Robot has moved to zero position")

def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    Use P control to return to start position
    
    Args:
        robot: robot instance
        start_positions: start joint position dictionary
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    print("Returning to start position...")
    
    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds
    
    for step in range(max_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value  # Don't apply calibration coefficients
        
        # P control calculation
        robot_action = {}
        total_error = 0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)
                
                # P control: output = Kp * error
                control_output = kp * error
                
                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)
        
        # Check if reached start position
        if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
            print("Returned to start position")
            break
        
        time.sleep(control_period)
    
    print("Return to start position completed")

def p_control_loop(robot, keyboard, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50):
    """
    P控制循环

    Args:
        robot: 机器人实例
        keyboard: 键盘实例
        target_positions: 目标关节位置字典
        start_positions: 起始关节位置字典
        current_x: 当前x坐标
        current_y: 当前y坐标
        kp: 比例增益
        control_freq: 控制频率(Hz)
    """
    control_period = 1.0 / control_freq

    # 初始化俯仰控制变量
    pitch = 0.0  # 初始俯仰调整
    pitch_step = 1  # 俯仰调整步长

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
                        't': ('wrist_roll', -1),      # 关节5减小
                        'g': ('wrist_roll', 1),       # 关节5增大
                        'y': ('gripper', -1),         # 关节6减小
                        'h': ('gripper', 1),          # 关节6增大
                    }

                    # x,y坐标控制
                    xy_controls = {
                        'w': ('x', -0.004),  # x减小
                        's': ('x', 0.004),   # x增大
                        'e': ('y', -0.004),  # y减小
                        'd': ('y', 0.004),   # y增大
                    }

                    # 俯仰控制
                    if key == 'r':
                        pitch += pitch_step
                        print(f"增加俯仰调整: {pitch:.3f}")
                    elif key == 'f':
                        pitch -= pitch_step
                        print(f"减少俯仰调整: {pitch:.3f}")
                    
                    if key in joint_controls:
                        joint_name, delta = joint_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"Update target position {joint_name}: {current_target} -> {new_target}")
                    
                    elif key in xy_controls:
                        coord, delta = xy_controls[key]
                        if coord == 'x':
                            current_x += delta
                            # 计算关节2和关节3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions['shoulder_lift'] = joint2_target
                            target_positions['elbow_flex'] = joint3_target
                            print(f"更新x坐标: {current_x:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_y += delta
                            # 计算关节2和关节3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions['shoulder_lift'] = joint2_target
                            target_positions['elbow_flex'] = joint3_target
                            print(f"更新y坐标: {current_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")

            # 应用俯仰调整到wrist_flex
            # 基于shoulder_lift和elbow_flex计算wrist_flex目标位置
            if 'shoulder_lift' in target_positions and 'elbow_flex' in target_positions:
                target_positions['wrist_flex'] = - target_positions['shoulder_lift'] - target_positions['elbow_flex'] + pitch
                # 显示当前俯仰值（每100步显示一次避免屏幕刷屏）
                if hasattr(p_control_loop, 'step_counter'):
                    p_control_loop.step_counter += 1
                else:
                    p_control_loop.step_counter = 0

                if p_control_loop.step_counter % 100 == 0:
                    print(f"当前俯仰调整: {pitch:.3f}, wrist_flex目标: {target_positions['wrist_flex']:.3f}")

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
                    
                    # P control: output = Kp * error
                    control_output = kp * error
                    
                    # Convert control output to position command
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # Send action to robot
            if robot_action:
                robot.send_action(robot_action)
            
            time.sleep(control_period)
            
        except KeyboardInterrupt:
            print("User interrupted program")
            break
        except Exception as e:
            print(f"P control loop error: {e}")
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
        port = input("请输入SO100机器人的USB端口 (例如: /dev/ttyACM0): ").strip()

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

        # 读取初始关节角度
        print("读取初始关节角度...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                start_positions[motor_name] = int(value)  # 不应用标定系数

        print("初始关节角度:")
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

        # 初始化x,y坐标控制
        x0, y0 = 0.1629, 0.1131
        current_x, current_y = x0, y0
        print(f"初始化末端执行器位置: x={current_x:.4f}, y={current_y:.4f}")


        print("键盘控制说明:")
        print("- Q/A: 关节1 (shoulder_pan) 减小/增大")
        print("- W/S: 控制末端执行器x坐标 (关节2+3)")
        print("- E/D: 控制末端执行器y坐标 (关节2+3)")
        print("- R/F: 俯仰调整增大/减小 (影响wrist_flex)")
        print("- T/G: 关节5 (wrist_roll) 减小/增大")
        print("- Y/H: 关节6 (gripper) 减小/增大")
        print("- X: 退出程序 (先返回起始位置)")
        print("- ESC: 退出程序")
        print("="*50)
        print("注意: 机器人将持续移动到目标位置")

        # 启动P控制循环
        p_control_loop(robot, keyboard, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50)

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