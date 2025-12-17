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

def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    计算2连杆机械臂的逆运动学，考虑关节偏移

    逆运动学是根据末端执行器的笛卡尔坐标(x,y)计算对应关节角度的过程。
    这里使用几何法求解2R机械臂的逆运动学问题。

    Parameters:
        x (float): 末端执行器在基座坐标系中的x坐标(米)
        y (float): 末端执行器在基座坐标系中的y坐标(米)
        l1 (float): 上臂长度(第一连杆长度)，默认0.1159米
        l2 (float): 下臂长度(第二连杆长度)，默认0.1350米

    Returns:
        tuple: (joint2_deg, joint3_deg) - 关节2和关节3的角度(度数)

    机械臂结构说明:
        - 基座: 坐标原点(0,0)
        - 关节1(shoulder_pan): 控制水平旋转，不参与此2D IK计算
        - 关节2(shoulder_lift): 第一个旋转关节，对应theta1
        - 关节3(elbow_flex): 第二个旋转关节，对应theta2
        - 末端执行器: 位于(x,y)坐标

    算法步骤:
        1. 验证目标点是否在工作空间内
        2. 使用几何关系计算关节角度
        3. 应用机械偏移补偿
        4. 角度限制和单位转换
    """

    """
    机械臂几何参数说明：

    连杆偏移参数（来自实际机械设计）：
    - theta1_offset: 关节2轴线相对于理想位置的偏移
      0.028: 垂直偏移量(米)
      0.11257: 水平偏移量(米)
      偏移角度 = atan2(垂直偏移, 水平偏移)

    - theta2_offset: 关节3轴线相对于关节2的额外偏移
      0.0052: 垂直偏移量(米)
      0.1349: 水平偏移量(米)

    这些偏移反映了实际机械结构中的几何误差和设计约束。
    """

    # 计算关节2和关节3在理想几何模型中的偏移角度
    # 这些偏移角度用于补偿实际机械结构中的几何误差
    theta1_offset = math.atan2(0.028, 0.11257)  # 关节2轴线偏移角（弧度）
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # 关节3累积偏移角（弧度）

    # 计算目标点到基座原点的距离
    r = math.sqrt(x**2 + y**2)

    # 定义机械臂的工作空间范围
    r_max = l1 + l2  # 最大可达距离：两连杆完全伸展
    r_min = abs(l1 - l2)  # 最小可达距离：两连杆完全折叠

    """
    工作空间检查和边界处理：

    工作空间是指机械臂末端能够到达的所有点的集合。
    对于2R机械臂，工作空间是一个环形区域：
    - 内径: |l1 - l2|
    - 外径: l1 + l2

    如果目标点在工作空间外，需要特殊处理：
    1. 超出外径：缩放到边界圆上
    2. 小于内径：缩放到内径圆上（如果r>0）
    """

    # 检查目标点是否超出最大工作空间
    if r > r_max:
        # 计算缩放因子，将目标点投影到工作空间边界
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max
        print(f"目标点超出工作空间，已缩放到边界: r={r:.3f}m")

    # 检查目标点是否小于最小工作空间（且不在原点）
    if r < r_min and r > 0:
        # 缩放最小距离到工作空间内径
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min
        print(f"目标点过近，已调整到最小距离: r={r:.3f}m")

    """
    逆运动学几何求解：

    使用余弦定理求解2R机械臂：

    设:
    - r: 目标点到原点的距离
    - l1, l2: 两连杆长度
    - theta1: 第一关节角度
    - theta2: 第二关节角度

    根据余弦定理：
    cos(theta2) = (l1² + l2² - r²) / (2*l1*l2)

    注意：这里使用负号是因为我们计算的是外角
    """

    # 使用余弦定理计算第二关节角度（肘关节）
    # 公式推导：根据余弦定理 c² = a² + b² - 2ab*cos(C)
    # 其中 c=r, a=l1, b=l2, C=theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)

    # 数值稳定性检查：确保cos_theta2在有效范围内[-1, 1]
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    # 计算theta2（肘部关节角度）
    # theta2 = π - arccos(cos_theta2) 选择肘部向下的解
    theta2 = math.pi - math.acos(cos_theta2)

    """
    计算第一关节角度（肩关节）：

    使用几何关系：
    - beta: 目标点相对于x轴的角度
    - gamma: 由第二连杆相对于第一连杆的偏移角度

    theta1 = beta + gamma

    其中：
    - beta = atan2(y, x) (目标点方向角)
    - gamma = atan2(l2*sin(theta2), l1 + l2*cos(theta2)) (连杆夹角)
    """

    # 计算目标点的方向角
    beta = math.atan2(y, x)

    # 计算连杆之间的夹角
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))

    # 第一关节角度 = 方向角 + 连杆夹角
    theta1 = beta + gamma

    # 应用机械偏移补偿
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset

    """
    关节角度限制：

    每个关节都有物理运动限制，由机械设计和安全要求决定：
    - joint2 (shoulder_lift): [-0.1, 3.45] 弧度 ≈ [-5.7°, 197.7°]
    - joint3 (elbow_flex): [-0.2, π] 弧度 ≈ [-11.5°, 180°]

    这些限制来自于URDF文件中的关节定义。
    """

    # 应用关节角度限制（安全范围检查）
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))

    # 从弧度转换为度数
    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    """
    坐标系转换：

    理论计算得到的角度需要转换为实际控制系统使用的角度：

    1. joint2转换: 90° - joint2_deg
       - 可能是为了匹配控制器的零位定义
       - 将垂直向上作为0度参考

    2. joint3转换: joint3_deg - 90°
       - 可能是为了匹配肘关节的正方向定义
       - 将完全伸直作为0度参考
    """

    # 应用坐标系转换以匹配实际控制器
    joint2_deg = 90 - joint2_deg
    joint3_deg = joint3_deg - 90

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
    P控制循环 - 末端执行器坐标控制版本

    这是整个控制程序的核心循环，集成了：
    1. P控制器实现平滑运动
    2. 逆运动学计算（笛卡尔坐标→关节角度）
    3. 键盘输入处理
    4. 俯仰角补偿计算
    5. 安全退出机制

    Args:
        robot: SO100机器人实例
        keyboard: 键盘遥操作器实例
        target_positions: 目标关节位置字典 {关节名: 角度}
        start_positions: 安全起始位置字典（用于退出时返回）
        current_x: 当前末端执行器x坐标(米)
        current_y: 当前末端执行器y坐标(米)
        kp: P控制比例增益，控制响应速度和稳定性
        control_freq: 控制循环频率(Hz)，影响控制精度

    P控制原理:
        - 控制输出 = Kp × 误差
        - 误差 = 目标位置 - 当前位置
        - 新位置 = 当前位置 + 控制输出

    控制特点:
        - 高频控制（50Hz）确保平滑运动
        - 实时逆运动学计算实现笛卡尔空间控制
        - 自动俯仰补偿保持末端执行器姿态
        - 增量式键盘控制便于精确定位
    """

    # 计算控制周期（秒）
    control_period = 1.0 / control_freq

    """
    俯仰控制参数说明：

    俯仰(pitch)是指末端执行器的倾斜角度控制。
    在实际应用中，为了保持抓取物体的稳定性，
    末端执行器的姿态需要根据机械臂配置进行补偿。

    参数说明：
    - pitch: 当前俯仰角调整值（度数）
    - pitch_step: 每次键盘输入的俯仰调整步长（度数）

    补偿原理：
    当机械臂运动时，为了保持末端执行器相对于地面的角度不变，
    需要根据shoulder_lift和elbow_flex的变化量来调整wrist_flex关节。
    """

    # 初始化俯仰控制变量
    pitch = 0.0  # 初始俯仰调整（度数），0表示末端垂直向下
    pitch_step = 1  # 俯仰调整步长（度数），每次按键改变1度

    print(f"启动P控制循环，控制频率: {control_freq}Hz，比例增益: {kp}")
    print("控制模式: 末端执行器笛卡尔坐标控制 + 俯仰补偿")
    
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