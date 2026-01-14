"""
===================================================================================
DOBOT 6轴机械臂遥操作数据采集脚本 V4 - 带垂直抓取功能
===================================================================================

版本特性：
-----------
1. **带时间戳的夹爪控制**：使用 PointStamped 消息，包含精确时间戳
2. **独立夹爪线程**：GripperManager 在独立线程中运行，实现：
   - 按下时：开环快速响应（虚拟值控制）
   - 松开后：自动回读硬件真实值，确保数据同步
3. **数据录制功能**：支持 O/P/L 键控制录制开始/保存/丢弃
4. **垂直抓取**：Z 键触发自动抓取序列（下降-闭合-上升）
5. **线程安全设计**：SharedMouseState 实现多线程安全的输入状态管理

控制说明：
-----------
- 鼠标移动：XY 平面平移
- 空格/Ctrl：Z 轴升降（按住 Alt 加速）
- WASD：姿态旋转（Rx, Ry）
- 滚轮：Rz 旋转
- 鼠标左键/右键：夹爪闭合/张开
- O/P/L：录制开始/保存/丢弃
- Z：触发垂直抓取
- X：退出抓取或快速张开
- ESC：退出程序

架构说明：
-----------
主线程 (TeleopController.run)：
    - 100Hz 控制循环
    - 处理鼠标/键盘输入
    - 发送 ServoP 指令控制机械臂
    - 管理垂直抓取状态机

夹爪线程 (GripperManager.run)：
    - 独立运行，与主线程并行
    - 监听鼠标按键状态
    - 自动模式下接管夹爪控制
    - 松开后自动回读硬件位置

ROS Spin 线程：
    - 后台处理 ROS 2 回调
    - 确保服务调用和消息发布正常工作

注意事项：
-----------
- 垂直抓取参数（V_Z1, V_Z2, V_G_OPEN 等）需根据实际环境调整
- 夹爪速度影响因素：GRIPPER_STEP, V_RG, 硬件 Force 设置
- 建议先在安全位置测试 Z 键抓取功能
===================================================================================
"""

import rclpy
from rclpy.node import Node
import time
import threading
import re
import sys
import math
import os
import tkinter as tk
from queue import Queue, Empty

# 引入消息类型
from std_msgs.msg import Int32
from geometry_msgs.msg import PointStamped 
from dobot_msgs_v3.srv import *
from pynput import keyboard, mouse

class DobotRosWrapper(Node):
    """
    ============================================================================
    DOBOT ROS2 通信封装类
    ============================================================================

    功能说明：
    ---------
    1. 封装所有 DOBOT ROS2 服务客户端（Service Client）
    2. 提供同步和异步两种服务调用方式
    3. 发布夹爪状态消息（带时间戳）
    4. 发布数据录制控制指令

    服务列表：
    ---------
    - EnableRobot, ClearError, SpeedFactor：机器人控制
    - MovJ, MovL：点到点运动
    - ServoJ, ServoP：伺服模式（高频实时控制）
    - Sync：等待运动完成
    - GetPose, GetAngle：获取当前位姿
    - ModbusCreate, ModbusClose：Modbus 连接管理
    - SetHoldRegs, GetHoldRegs：夹爪寄存器读写

    发布话题：
    ---------
    - /gripper/command_update (PointStamped)：夹爪目标位置 + 时间戳
    - /recorder/command (Int32)：录制指令（0=丢弃, 1=开始, 2=保存）
    ============================================================================
    """
    def __init__(self, node_name="dobot_api_client"):
        super().__init__(node_name)
        
        # ========== 机器人基础控制服务 ==========
        # 使能机器人（必须在运动前调用，上电后机器人处于未使能状态）
        self.cli_enable = self.create_client(EnableRobot, '/dobot_bringup_v3/srv/EnableRobot')

        # 清除错误状态（当机器人报警时调用，清除后需重新使能）
        self.cli_clear_error = self.create_client(ClearError, '/dobot_bringup_v3/srv/ClearError')

        # 设置速度比例（0-100%，影响所有运动指令的执行速度）
        self.cli_speed_factor = self.create_client(SpeedFactor, '/dobot_bringup_v3/srv/SpeedFactor')

        # ========== 点到点运动服务 ==========
        # MovJ: 关节空间运动（Joint Move）
        #   - 每个关节独立插值到目标角度
        #   - 轨迹不是直线，但速度快、能耗低
        #   - 适用场景：非精密路径的快速移动
        self.cli_mov_j = self.create_client(MovJ, '/dobot_bringup_v3/srv/MovJ')

        # MovL: 笛卡尔空间直线运动（Linear Move）
        #   - 末端执行器沿直线路径运动
        #   - 保证轨迹精度，但可能较慢
        #   - 适用场景：需要精确路径的操作（如画直线、切割）
        self.cli_mov_l = self.create_client(MovL, '/dobot_bringup_v3/srv/MovL')

        # ========== 伺服模式（实时控制）==========
        # ServoJ: 关节空间伺服控制
        #   - 高频发送关节角度目标（30-100Hz）
        #   - 每次调用产生微小增量
        #   - 适用场景：拖动示教、力控、实时轨迹跟踪
        self.cli_servo_j = self.create_client(ServoJ, '/dobot_bringup_v3/srv/ServoJ')

        # ServoP: 笛卡尔空间伺服控制（本脚本使用）
        #   - 高频发送笛卡尔位姿目标（30-100Hz）
        #   - 适合鼠标/键盘实时遥操作
        #   - 自动处理逆运动学，用户无需关心关节角度
        self.cli_servo_p = self.create_client(ServoP, '/dobot_bringup_v3/srv/ServoP')

        # Sync: 等待运动完成
        #   - 阻塞等待当前运动队列执行完毕
        #   - 用于 MovJ/MovL 之后，确保到位后再执行下一步
        #   - ServoJ/ServoP 通常不需要 Sync
        self.cli_sync = self.create_client(Sync, '/dobot_bringup_v3/srv/Sync')

        # ========== 状态查询服务 ==========
        # GetPose: 获取当前笛卡尔位姿
        #   - 返回: [x, y, z, rx, ry, rz] (mm, deg)
        #   - 用于初始化、监控、记录轨迹
        self.cli_get_pose = self.create_client(GetPose, '/dobot_bringup_v3/srv/GetPose')

        # GetAngle: 获取当前关节角度
        #   - 返回: [j1, j2, j3, j4, j5, j6] (deg)
        #   - 用于关节空间控制、避障判断
        self.cli_get_angle = self.create_client(GetAngle, '/dobot_bringup_v3/srv/GetAngle')

        # ========== Modbus 通信服务（夹爪控制）==========
        # ModbusCreate: 创建 Modbus 连接
        #   - 用于连接夹爪、传感器等外设
        #   - 返回连接 ID，后续读写操作需要此 ID
        self.cli_modbus_create = self.create_client(ModbusCreate, '/dobot_bringup_v3/srv/ModbusCreate')

        # ModbusClose: 关闭 Modbus 连接
        #   - 释放端口资源
        #   - 初始化时先关闭旧连接，防止占用
        self.cli_modbus_close = self.create_client(ModbusClose, '/dobot_bringup_v3/srv/ModbusClose')

        # SetHoldRegs: 写入保持寄存器
        #   - 用于控制夹爪位置、力度等参数
        #   - 支持同步（阻塞）和异步（非阻塞）两种模式
        self.cli_set_hold_regs = self.create_client(SetHoldRegs, '/dobot_bringup_v3/srv/SetHoldRegs')

        # GetHoldRegs: 读取保持寄存器
        #   - 用于读取夹爪实时位置、状态等反馈
        #   - 返回寄存器数值
        self.cli_get_hold_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs')

        # 使用 PointStamped 发布夹爪状态（x字段存数值，header存时间戳）
        self.pub_gripper_update = self.create_publisher(PointStamped, "/gripper/command_update", 5)
        
        # 数据收集指令发布器
        self.pub_record_cmd = self.create_publisher(Int32, "/recorder/command", 10)

        self.get_logger().info("Waiting for Dobot services...")
        if not self.cli_mov_j.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Dobot services not available yet.")

    def call_service(self, client, request):
        """
        同步调用 ROS2 服务（阻塞等待结果）

        参数：
        ----
        client: ROS2 服务客户端
        request: 服务请求对象

        返回：
        ----
        服务响应对象，失败时返回 None

        说明：
        ----
        - 使用忙等待（busy-wait）直到服务返回结果
        - 适用于需要立即获取结果的场景（如查询位姿）
        - 会阻塞当前线程，不适合高频控制
        """
        if not client.service_is_ready():
            return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.001)  # 1ms 轮询间隔
        return future.result()

    def call_service_async_no_wait(self, client, request):
        """
        异步调用 ROS2 服务（非阻塞，不等待结果）

        参数：
        ----
        client: ROS2 服务客户端
        request: 服务请求对象

        说明：
        ----
        - 发送请求后立即返回，不等待结果
        - 适用于高频控制场景（如夹爪连续运动）
        - 可能导致指令积压，需要控制调用频率
        """
        if client.service_is_ready():
            client.call_async(request)

class GripperComm:
    """
    ============================================================================
    DH-Robotics AG 系列夹爪 Modbus-RTU 通信类
    ============================================================================

    功能说明：
    ---------
    - 只负责底层 Modbus 协议通信，不包含控制逻辑
    - 通过 DOBOT 控制器的 Modbus 服务转发指令
    - 支持寄存器读写操作

    关键寄存器地址（十六进制 -> 十进制）：
    ---------------------------------------
    0x0100 (256)  - 初始化寄存器（写1=初始化, 写165=重新标定）
    0x0101 (257)  - 力度/速度设置（20-100%）
    0x0103 (259)  - 位置控制（0-1000，0=闭合，1000=张开）
    0x0201 (513)  - 运动状态（0=运动中, 1=到位, 2=夹持, 3=掉落）
    0x0202 (514)  - 实时位置反馈（0-1000）

    连接参数：
    ---------
    - IP: 127.0.0.1（本地回环，通过 DOBOT 控制器转发）
    - Port: 60000
    - Slave ID: 1
    - Mode: RTU (is_rtu=1)
    ============================================================================
    """
    def __init__(self, wrapper):
        self.wrapper = wrapper  # 引用 DobotRosWrapper 实例
        self.id = 0  # Modbus 连接 ID（由 ModbusCreate 返回）
        self.init_connection()

    def init_connection(self):
        # 关闭旧连接
        for i in range(1, 5):
            req = ModbusClose.Request()
            req.index = i
            self.wrapper.call_service(self.wrapper.cli_modbus_close, req)
        
        # 创建连接
        req = ModbusCreate.Request()
        req.ip = "127.0.0.1"      # 本地回环地址
        req.port = 60000           # DOBOT Modbus 端口
        req.slave_id = 1           # 夹爪从机地址
        req.is_rtu = 1             # RTU 模式
        res = self.wrapper.call_service(self.wrapper.cli_modbus_create, req)
        
        if res and res.res == 0:
            match = re.search(r'(\d+)', str(res.index))
            self.id = int(match.group(1)) if match else int(res.index)
            print(f"[SUCCESS] Gripper Connected, ID: {self.id}")
        else:
            print(f"[ERROR] Gripper ModbusCreate Failed! Response: {res}")
            self.id = 0
        
        if self.id > 0:
            self.write_reg(256, 1, "1", wait=True) # Enable
            self.write_reg(257, 1, "60", wait=True) # Force/Speed

    def write_reg(self, addr, count, val_str, wait=False):
        """
        写入保持寄存器

        参数：
        ----
        addr: 寄存器地址（十进制）
        count: 寄存器数量（通常为1）
        val_str: 要写入的值（字符串格式）
        wait: 是否等待写入完成（True=阻塞，False=非阻塞）
        """
        if self.id <= 0: return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_str  # 保持字符串类型（ROS消息定义要求）
        if wait:
            self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)
        else:
            self.wrapper.call_service_async_no_wait(self.wrapper.cli_set_hold_regs, req)

    def read_reg(self, addr):
        """
        读取保持寄存器

        参数：
        ----
        addr: 寄存器地址（十进制）

        返回：
        ----
        寄存器值（整数），失败时返回 None
        """
        if self.id <= 0: return None
        req = GetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = 1
        res = self.wrapper.call_service(self.wrapper.cli_get_hold_regs, req)
        if res and res.res == 0:
            try:
                return int(res.value)
            except:
                pass
        return None

class GripperManager(threading.Thread):
    """
    ============================================================================
    夹爪智能管理线程 - 核心创新架构
    ============================================================================

    设计理念：
    ---------
    传统方案的问题：
        - 直接在主线程中读写夹爪 → Modbus 通信延迟导致控制卡顿
        - 纯发送指令无反馈 → 录制数据与实际硬件状态不一致

    本方案的创新：
        1. **独立线程运行**：完全解耦，不阻塞主控制循环
        2. **两阶段控制**：
           - 阶段A（按下时）：开环快速响应，虚拟值增量控制
           - 阶段B（松开后）：自动回读硬件真实值，修正数据
        3. **自动模式**：支持脚本控制（如垂直抓取序列）

    工作流程（手动模式）：
    ---------------------
    1. 用户按住鼠标左键/右键
       └─> 虚拟位置 += GRIPPER_STEP
       └─> 立即发布 PointStamped 消息（带时间戳）
       └─> 发送 Modbus 写指令（非阻塞）

    2. 用户松开鼠标
       └─> 等待 20ms（让硬件稳定）
       └─> 循环读取寄存器 514（实时位置）
       └─> 发布真实值修正录制数据
       └─> 连续2次读数一致 → 退出回读

    工作流程（自动模式）：
    ---------------------
    - 由主线程调用 set_auto_mode(True, target_pos)
    - 线程以 50Hz 驱动夹爪到目标位置
    - 用于垂直抓取等自动序列

    关键变量：
    ---------
    current_pos: 当前虚拟位置（0-1000）
    target_auto_pos: 自动模式目标位置
    was_pressed: 状态标志，用于检测松开事件
    auto_mode: 自动控制标志

    发布消息格式（PointStamped）：
    -----------------------------
    header.stamp: ROS 时间戳
    header.frame_id: "gripper"
    point.x: 夹爪位置值（0-1000）
    ============================================================================
    """
    def __init__(self, wrapper, mouse_state):
        super().__init__(daemon=True)
        self.wrapper = wrapper              # ROS Wrapper 引用
        self.comm = GripperComm(wrapper)    # 底层通信对象
        self.mouse_state = mouse_state      # 共享鼠标状态（线程安全）
        self.running = True                 # 线程运行标志

        # 控制参数
        self.current_pos = 1000.0           # 当前虚拟位置（初始张开）
        self.GRIPPER_STEP = 80              # 每次增量（影响响应速度）

        # 状态机标志
        self.was_pressed = False            # 上一帧是否有按键按下

        # 自动模式
        self.auto_mode = False              # 是否处于自动控制模式
        self.target_auto_pos = 1000.0       # 自动模式目标位置

        # 初始化：读取硬件当前位置并同步
        init_val = self.comm.read_reg(514)
        if init_val is not None:
            self.current_pos = float(init_val)
            self.publish_state(self.current_pos)

    def publish_state(self, pos):
        """
        发布带时间戳的夹爪状态消息

        参数：
        ----
        pos: 夹爪位置（0-1000）

        消息格式：
        ---------
        PointStamped 用于存储夹爪位置 + 时间戳
        - header.stamp: ROS 时间戳（用于数据同步）
        - header.frame_id: "gripper"（标识来源）
        - point.x: 夹爪位置值（float）
        """
        msg = PointStamped()
        msg.header.stamp = self.wrapper.get_clock().now().to_msg()
        msg.header.frame_id = "gripper"
        msg.point.x = float(pos)
        self.wrapper.pub_gripper_update.publish(msg)

    def set_auto_mode(self, enabled, target_pos=None):
        """
        切换自动/手动模式（线程安全）

        参数：
        ----
        enabled: True=自动模式，False=手动模式
        target_pos: 自动模式的目标位置（可选）

        说明：
        ----
        - 自动模式下，线程接管夹爪控制，忽略鼠标输入
        - 用于垂直抓取等自动序列
        - 主线程调用此方法，夹爪线程读取标志
        """
        self.auto_mode = enabled
        if target_pos is not None:
            self.target_auto_pos = float(target_pos)

    def move(self, place):
        """
        立即移动到指定位置（非阻塞）

        参数：
        ----
        place: 目标位置（0-1000）

        说明：
        ----
        - 更新虚拟位置
        - 发送非阻塞 Modbus 指令
        - 不发布消息（由调用者决定）
        """
        self.current_pos = place
        self.comm.write_reg(259, 1, str(int(self.current_pos)), wait=False)

    def run(self):
        """
        ====================================================================
        夹爪控制线程主循环 - 核心状态机
        ====================================================================

        循环逻辑：
        ---------
        while running:
            if 自动模式:
                → 驱动到 target_auto_pos（50Hz）
                → 发布状态消息
                → 跳过手动控制逻辑

            elif 鼠标按下:
                → [阶段A] 开环快速响应
                → 计算新位置（增量控制）
                → 发送 Modbus 写指令（非阻塞）
                → 发布虚拟位置消息
                → 标记 was_pressed = True

            elif 鼠标松开 且 was_pressed:
                → [阶段B] 硬件状态回读
                → 等待 20ms 让硬件稳定
                → 循环读取寄存器 514
                → 发布真实位置修正数据
                → 检测稳定性（连续2次读数一致）
                → 标记 was_pressed = False

            else:
                → 空闲状态，低频休眠

        关键设计：
        ---------
        1. **非阻塞写入**：按下时用 wait=True 确保指令到达
           （wait=False 会导致指令丢失）

        2. **稳定性检测**：松开后读取直到连续2次数值一致
           （误差容忍 5 个单位）

        3. **中断保护**：回读期间检测用户再次按下，立即退出

        4. **频率控制**：
           - 按下时：50ms/次（20Hz）
           - 回读时：20ms/次（50Hz）
           - 空闲时：10ms/次（100Hz）
        ====================================================================
        """
        read_cnt = 0
        while self.running:
            # ========== 优先级1: 自动模式 ==========
            if self.auto_mode:
                # 自动模式：由脚本逻辑控制位置（如垂直抓取）
                self.current_pos = self.target_auto_pos
                self.comm.write_reg(259, 1, str(int(self.current_pos)), wait=False)
                self.publish_state(self.current_pos)
                time.sleep(0.02)  # 50Hz 控制频率
                continue

            read_cnt += 1
            # 获取当前按键状态（线程安全读取共享变量）
            left_pressed = self.mouse_state.left_pressed
            right_pressed = self.mouse_state.right_pressed
            is_pressed = left_pressed or right_pressed

            # ========== 优先级2: 鼠标按下（开环控制）==========
            if is_pressed:
                # === 阶段A: 持续按下，仅发送写指令 ===
                self.was_pressed = True
                changed = False

                # 计算新位置（增量控制）
                if left_pressed:  # 闭合（减小值）
                    self.current_pos = max(0.0, self.current_pos - self.GRIPPER_STEP * 0.5)
                    changed = True
                elif right_pressed:  # 张开（增大值）
                    self.current_pos = min(1000.0, self.current_pos + self.GRIPPER_STEP * 0.5)
                    changed = True

                if changed:
                    # 发送 Modbus 写指令（wait=True 确保指令到达）
                    self.comm.write_reg(259, 1, str(int(self.current_pos)), wait=True)
                    # 发布虚拟位置消息（带时间戳）
                    self.publish_state(self.current_pos)

                # 保持指令频率约 20Hz（避免过载）
                time.sleep(0.05)

            # ========== 优先级3: 鼠标松开（闭环回读）==========
            else:
                read_cnt = 0
                # === 阶段B: 松开按键，同步真实状态 ===
                if self.was_pressed:
                    # 刚松开，等待 20ms 后开始读取（让硬件稳定）
                    time.sleep(0.02)

                    # 进入回读循环（Blocking Read）
                    stable_count = 0
                    last_read_val = -1

                    # 尝试读取直到稳定（最多 1秒，防止死循环）
                    for t_after in range(50):
                        # 检测用户是否再次按下，立即退出回读
                        if self.mouse_state.left_pressed or self.mouse_state.right_pressed or self.auto_mode:
                            break

                        # 控制读取频率（每5次循环读1次）
                        if t_after % 5 != 4:
                            break

                        # 读取硬件真实位置
                        real_val = self.comm.read_reg(514)  # 寄存器 514: 实时位置
                        if real_val is not None:
                            # 更新虚拟位置并发布真实值（修正录制数据）
                            self.current_pos = float(real_val)
                            self.publish_state(self.current_pos)

                            # 检查是否稳定（连续读数一致）
                            if abs(real_val - last_read_val) < 5:  # 误差容忍 5 个单位
                                stable_count += 1
                            else:
                                stable_count = 0

                            # 连续2次读数一致，认为稳定
                            if stable_count > 2:
                                break

                            last_read_val = real_val

                        time.sleep(0.02)  # 20ms 读取间隔

                    self.was_pressed = False

                # 空闲状态，低频休眠
                time.sleep(0.01)

class SharedMouseState:
    """
    ============================================================================
    线程安全的鼠标状态容器
    ============================================================================

    设计目的：
    ---------
    - pynput 回调运行在独立线程
    - 主控制循环和夹爪线程需要访问鼠标状态
    - 使用 threading.Lock 确保线程安全

    共享变量：
    ---------
    dx, dy: 鼠标累积位移（相对屏幕中心）
    scroll_dy: 滚轮累积滚动量
    left_pressed: 左键是否按下
    right_pressed: 右键是否按下

    方法说明：
    ---------
    update_move(): pynput 回调调用，累加位移
    get_and_clear_move(): 主线程读取并清零位移
    update_scroll(): pynput 回调调用，累加滚轮
    get_and_clear_scroll(): 主线程读取并清零滚轮

    线程安全：
    ---------
    所有读写操作都使用 with self.lock 保护
    ============================================================================
    """
    def __init__(self):
        self.dx = 0                      # 累积X位移
        self.dy = 0                      # 累积Y位移
        self.scroll_dy = 0               # 累积滚轮
        self.left_pressed = False        # 左键状态
        self.right_pressed = False       # 右键状态
        self.lock = threading.Lock()     # 线程锁

    def update_move(self, dx, dy):
        with self.lock:
            self.dx += dx
            self.dy += dy
    
    def get_and_clear_move(self):
        with self.lock:
            dx, dy = self.dx, self.dy
            self.dx, self.dy = 0, 0
            return dx, dy

    def update_scroll(self, dy):
        with self.lock:
            self.scroll_dy += dy

    def get_and_clear_scroll(self):
        with self.lock:
            dy = self.scroll_dy
            self.scroll_dy = 0
            return dy

class Robot:
    """
    ============================================================================
    机器人控制器封装类（简化版）
    ============================================================================

    功能说明：
    ---------
    - 初始化 ROS2 节点和通信
    - 提供高层运动控制接口（ServoP, get_pose）
    - 管理 ROS Spin 线程

    初始化流程：
    -----------
    1. 初始化 rclpy
    2. 创建 DobotRosWrapper 节点
    3. 启动后台 Spin 线程
    4. 使能机器人
    5. 设置速度因子
    6. 清除错误

    注意：
    -----
    - 此类不包含夹爪控制（由 GripperManager 独立处理）
    - ServoP 适用于实时控制，MovJ/MovL 适用于点到点运动
    ============================================================================
    """
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = DobotRosWrapper()

        # ROS Spin 线程（后台处理回调）
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        # 初始化机器人
        self.node.call_service(self.node.cli_speed_factor, SpeedFactor.Request(ratio=100))
        self.node.call_service(self.node.cli_clear_error, ClearError.Request())
        self.enable_robot()

        print(">.< 机器人服务已连接 >.<")

    def enable_robot(self):
        """使能机器人（必须在运动前调用）"""
        self.node.call_service(self.node.cli_enable, EnableRobot.Request())

    def ServoP(self, x, y, z, rx, ry, rz):
        """
        笛卡尔空间伺服控制（实时控制）

        参数：
        ----
        x, y, z: 位置 (mm)
        rx, ry, rz: 姿态角 (度)

        说明：
        ----
        - 适合高频控制（30-100Hz）
        - 每次调用产生微小增量
        - 实现平滑轨迹跟踪
        """
        req = ServoP.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        self.node.call_service(self.node.cli_servo_p, req)

    def get_pose(self):
        """
        获取当前笛卡尔位姿

        返回：
        ----
        [x, y, z, rx, ry, rz] 或 None（失败时）
        """
        res = self.node.call_service(self.node.cli_get_pose, GetPose.Request())
        if res and hasattr(res, 'res'):
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", res.pose)
            if len(matches) >= 6:
                return [float(x) for x in matches[:6]]
        return None

    def close(self):
        """关闭 ROS 连接"""
        rclpy.shutdown()

class TeleopController:
    """
    ============================================================================
    遥操作主控制器 - 统筹所有输入和运动控制
    ============================================================================

    架构说明：
    ---------
    1. **主线程** (run 方法)：
       - 100Hz 控制循环
       - 处理键盘/鼠标输入
       - 发送 ServoP 指令控制机械臂
       - 管理垂直抓取状态机

    2. **夹爪线程** (GripperManager)：
       - 独立运行，处理夹爪控制
       - 自动回读硬件状态

    3. **ROS Spin 线程**：
       - 后台处理 ROS 回调

    4. **pynput 监听线程**：
       - 捕获键盘/鼠标事件
       - 更新 SharedMouseState

    控制映射：
    ---------
    鼠标：
        - 移动 → XY 平面平移
        - 滚轮 → Rz 旋转
        - 左键 → 夹爪闭合
        - 右键 → 夹爪张开

    键盘：
        - Space/Ctrl → Z轴升降
        - Alt + Space/Ctrl → 加速升降
        - WASD → 姿态旋转 (Rx, Ry)
        - O/P/L → 录制控制
        - Z → 垂直抓取
        - X → 退出抓取/快速张开
        - ESC → 退出程序

    垂直抓取状态机：
    ----------------
    状态0（空闲）: 正常遥操作
    状态1（下降）: 张开夹爪 + 下降到 Z1
    状态2（抓取）: 闭合夹爪 + 上升到 Z2
    完成后回到状态0

    关键参数：
    ---------
    LOOP_RATE: 主循环频率 (100Hz)
    MOUSE_SENSITIVITY: 鼠标灵敏度 (0.1 mm/pixel)
    KEY_XYZ_STEP: 键盘Z轴步长 (0.3 mm/tick)
    KEY_XYZ_STEP_FAST: 加速步长 (0.6 mm/tick)
    KEY_ROT_STEP: 旋转步长 (0.3 deg/tick)
    V_Z1, V_Z2: 垂直抓取高度范围
    V_G_OPEN: 抓取时夹爪张开值
    V_SPEED_UP/DOWN: 抓取时Z轴速度
    V_RG: 抓取时夹爪闭合速度
    ============================================================================
    """
    def __init__(self):
        self.robot = Robot()
        self.mouse_state = SharedMouseState()
        
        # 启动独立的夹爪控制线程
        self.gripper_worker = GripperManager(self.robot.node, self.mouse_state)
        self.gripper_worker.start()

        # 参数
        self.LOOP_RATE = 100.0                      # 主控制循环频率 (Hz)，决定了 ServoP 指令发送频率
        self.MOUSE_SENSITIVITY = 0.1                # 鼠标灵敏度系数 (mm/pixel)，控制鼠标移动转换为XY轴位移的比例
        self.SCROLL_SENSITIVITY = 2.0               # 滚轮灵敏度系数 (deg/scroll_unit)，控制滚轮转换为Rz旋转的比例
        self.KEY_XYZ_STEP = 0.3                     # 键盘控制Z轴的单步长度 (mm/tick)，按Space/Ctrl时的移动速度
        self.KEY_XYZ_STEP_FAST = 0.6                # 键盘控制Z轴的加速步长 (mm/tick)，按Alt+Space/Ctrl时的移动速度（双倍）
        self.KEY_ROT_STEP = 0.3                     # 键盘控制姿态旋转的单步长度 (deg/tick)，按WASD键时的旋转速度

        # [新增] 垂直抓取 (Vertical Grab) 参数
        # 请根据实际环境修改这些值！
        self.V_G_OPEN = 600.0  # 夹爪张开范围 g
        self.V_Z1 = 212.         # 底部高度 z1 (绝对坐标，单位mm) - 请确保该高度不会撞击
        self.V_Z2 = 245.0       # 顶部高度 z2 (抓取后上升到的高度)
        self.V_SPEED_UP = 3.         # 上升速率 (mm/tick, 100Hz下约等于 50mm/s)
        self.V_SPEED_DOWN = 3. # 下降速率 (mm/tick)
        self.V_RG = 180.0         # 夹爪闭合速率 rg (单位/tick, 100Hz下 800单位/s)

        self.vgrab_state = 0    # 0=Idle, 1=Descending, 2=Ascending+Closing

        self.running = True
        self.target_pose = [0.0] * 6
        self.keys_pressed = set()
        
        self._init_pose()
        
        # 输入监听
        self.kb_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release, suppress=True)
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move, 
            on_click=self._on_mouse_click, 
            on_scroll=self._on_scroll,
            suppress=True
        )
        self.kb_listener.start()
        self.mouse_listener.start()

        # 屏幕中心初始化
        root = tk.Tk()
        self.center_x = root.winfo_screenwidth() // 2
        self.center_y = root.winfo_screenheight() // 2
        root.destroy()
        
        self.mouse_controller = mouse.Controller()
        self.mouse_controller.position = (self.center_x, self.center_y)
        
        try: os.system("xdotool getactivewindow windowactivate")
        except: pass

        print("=== 系统就绪 ===")
        print(" [Main Loop]    : 负责机械臂运动 (ServoP)")
        print(" [Gripper Thread]: 负责夹爪控制与回读")
        print(" [O/P/L]        : 录制控制")
        print(" [Z]            : 触发垂直抓取序列")

    def _init_pose(self):
        """
        初始化机械臂位姿

        流程：
        ----
        1. 读取当前笛卡尔位姿
        2. 设置为目标位姿（避免启动时突然移动）
        3. 如果读取失败，重试最多5次
        4. 失败后退出程序（避免未知初始状态）

        说明：
        ----
        - 确保 ServoP 从当前位置开始，避免跳变
        - target_pose 作为控制循环的目标值
        """
        for _ in range(5):
            pose = self.robot.get_pose()
            if pose:
                self.target_pose = pose
                return
            time.sleep(0.5)
        print("错误：无法获取机械臂初始位置！")
        sys.exit(1)

    # --- 输入回调 (运行在 pynput 线程) ---
    def _on_mouse_move(self, x, y):
        """
        鼠标移动回调（运行在 pynput 线程）

        无限鼠标原理：
        ------------
        1. 计算鼠标相对屏幕中心的位移 (dx, dy)
        2. 累加到共享状态 mouse_state
        3. 立即将鼠标重置回屏幕中心
        4. 重复循环，实现"无限移动"效果

        为什么这样设计：
        --------------
        - pynput.suppress=True 屏蔽系统鼠标显示
        - 鼠标每次移动后回中，永远不会碰到屏幕边缘
        - 允许用户持续旋转机械臂，不受屏幕范围限制

        线程安全：
        ---------
        - 通过 SharedMouseState.update_move() 保证线程安全
        """
        dx = x - self.center_x
        dy = y - self.center_y
        if dx != 0 or dy != 0:
            self.mouse_state.update_move(dx, dy)
            self.mouse_controller.position = (self.center_x, self.center_y)

    def _on_mouse_click(self, x, y, button, pressed):
        """
        鼠标点击回调（运行在 pynput 线程）

        说明：
        ----
        - 如果正在自动抓取，忽略鼠标点击（防止干扰）
        - 更新共享状态，供 GripperManager 读取
        """
        if self.vgrab_state != 0:
            return
            
        # 更新共享状态，供 GripperManager 读取
        if button == mouse.Button.left:
            self.mouse_state.left_pressed = pressed
            print(f"[DEBUG] 左键 {'按下' if pressed else '松开'}")
        elif button == mouse.Button.right:
            self.mouse_state.right_pressed = pressed
            print(f"[DEBUG] 右键 {'按下' if pressed else '松开'}")

    def _on_scroll(self, x, y, dx, dy):
        """
        鼠标滚轮回调（运行在 pynput 线程）

        说明：
        ----
        - 累加滚轮值到共享状态
        - 主线程读取后用于 Rz 旋转
        """
        self.mouse_state.update_scroll(dy)

    def _on_key_press(self, key):
        """
        键盘按下回调（运行在 pynput 线程）

        处理逻辑：
        ---------
        1. ESC 键 → 停止程序
        2. O/P/L 键 → 发布录制指令
        3. Z 键 → 启动垂直抓取
        4. 其他按键 → 添加到 keys_pressed 集合
        """
        if key == keyboard.Key.esc:
            self.running = False
        try:
            if hasattr(key, 'char'):
                char_key = key.char.lower()
                self.keys_pressed.add(char_key)

                # 录制指令发布
                msg = Int32()
                if char_key == 'o':
                    msg.data = 1
                    self.robot.node.pub_record_cmd.publish(msg)
                    print("Rec Start")
                elif char_key == 'p':
                    msg.data = 2
                    self.robot.node.pub_record_cmd.publish(msg)
                    print("Rec Save")
                elif char_key == 'l':
                    msg.data = 0
                    self.robot.node.pub_record_cmd.publish(msg)
                    print("Rec Discard")

                # Z 键触发垂直抓取
                elif char_key == 'z':
                    print(">>> 开始垂直抓取序列")
                    self.vgrab_state = 1  # 进入下降阶段
                    # 开启夹爪自动模式，设置目标为张开
                    self.gripper_worker.set_auto_mode(True, self.V_G_OPEN)

            else:
                self.keys_pressed.add(key)
        except:
            pass

    def _on_key_release(self, key):
        try:
            if hasattr(key, 'char'): self.keys_pressed.discard(key.char.lower())
            else: self.keys_pressed.discard(key)
        except: pass

    # --- 主循环 (Main Process) ---
    def run(self):
        """
        ====================================================================
        主控制循环 - 100Hz 实时控制
        ====================================================================

        循环结构：
        ---------
        1. 检查垂直抓取状态机
           ├─ 状态1: 下降到底部
           └─ 状态2: 闭合并上升

        2. 手动遥操作模式
           ├─ 读取鼠标/键盘输入
           ├─ 计算目标位姿
           └─ 发送 ServoP 指令

        3. 循环频率控制
           └─ 动态调整 time_rate 补偿延迟

        时间补偿机制：
        -------------
        - 正常情况：sleep 保持 100Hz，time_rate = 1.0
        - 延迟情况：跳过 sleep，time_rate > 1.0 加速补偿
        - 保证控制平滑性，即使偶尔掉帧也能补偿

        ServoP 指令：
        -----------
        - 笛卡尔空间伺服控制
        - 适合高频实时控制（30-100Hz）
        - 每次指令微小增量，实现平滑运动
        ====================================================================
        """
        time_rate = 1.0  # 时间补偿系数（正常为1.0）

        while self.running:
            start_time = time.time()

            # ========== 优先级1: 垂直抓取状态机 ==========
            if self.vgrab_state != 0:
                # X 键紧急退出抓取序列
                if 'x' in self.keys_pressed:
                    print(">>> 手动退出抓取")
                    self.vgrab_state = 0
                    self.gripper_worker.set_auto_mode(False)  # 交还夹爪控制权

                # 清除积压的鼠标输入（防止抓取完成后突然移动）
                self.mouse_state.get_and_clear_move()
                self.mouse_state.get_and_clear_scroll()

                # === 状态1: 下降阶段 ===
                if self.vgrab_state == 1:
                    # 确保夹爪目标是张开状态（准备抓取）
                    self.gripper_worker.target_auto_pos = self.V_G_OPEN

                    # 下降 Z 轴（以 V_SPEED_DOWN 速率）
                    if self.target_pose[2] > self.V_Z1:
                        self.target_pose[2] -= self.V_SPEED_DOWN
                        self.target_pose[2] = max(self.target_pose[2], self.V_Z1)  # 限制最低值
                    else:
                        # 到达底部，切换到上升阶段
                        print(">>> 到达底部，开始闭合上升")
                        self.vgrab_state = 2

                # === 状态2: 上升+闭合阶段 ===
                elif self.vgrab_state == 2:
                    # 上升 Z 轴（带渐进加速，避免抖动）
                    if abs(self.target_pose[2] - self.V_Z1) <= self.V_SPEED_UP * 2:
                        # 刚起步，减速避免冲击
                        self.target_pose[2] += self.V_SPEED_UP / 2
                    else:
                        # 正常上升
                        self.target_pose[2] += self.V_SPEED_UP

                    # 同时闭合夹爪（以 V_RG 速率）
                    new_grip = self.gripper_worker.target_auto_pos - self.V_RG
                    self.gripper_worker.target_auto_pos = max(0.0, new_grip)

                    # 检查结束条件
                    if self.target_pose[2] > self.V_Z2:
                        print(">>> 抓取结束，恢复手动控制")
                        self.vgrab_state = 0
                        self.gripper_worker.set_auto_mode(False)  # 交还夹爪控制权

            # ========== 优先级2: 手动遥操作模式 ==========
            else:
                # === 读取鼠标输入 ===
                dx, dy = self.mouse_state.get_and_clear_move()
                scroll_dy = self.mouse_state.get_and_clear_scroll()

                # === XY 平面控制（鼠标移动）===
                self.target_pose[0] += dx * self.MOUSE_SENSITIVITY   # X 轴
                self.target_pose[1] -= dy * self.MOUSE_SENSITIVITY   # Y 轴（屏幕坐标系取反）

                # === Z 轴控制（键盘 Space/Ctrl）===
                if keyboard.Key.space in self.keys_pressed:
                    # 检测是否按下 Alt 加速
                    step = self.KEY_XYZ_STEP_FAST if (keyboard.Key.alt in self.keys_pressed) else self.KEY_XYZ_STEP
                    self.target_pose[2] += step * time_rate  # 乘以 time_rate 补偿延迟

                if keyboard.Key.ctrl in self.keys_pressed or keyboard.Key.ctrl_l in self.keys_pressed:
                    step = self.KEY_XYZ_STEP_FAST if (keyboard.Key.alt in self.keys_pressed) else self.KEY_XYZ_STEP
                    self.target_pose[2] -= step * time_rate

                # === 姿态旋转控制（WASD 键）===
                if 'a' in self.keys_pressed:
                    self.target_pose[3] += self.KEY_ROT_STEP * time_rate  # Rx
                if 'd' in self.keys_pressed:
                    self.target_pose[3] -= self.KEY_ROT_STEP * time_rate
                if 'w' in self.keys_pressed:
                    self.target_pose[4] -= self.KEY_ROT_STEP * time_rate  # Ry
                if 's' in self.keys_pressed:
                    self.target_pose[4] += self.KEY_ROT_STEP * time_rate

                # === 快速张开夹爪（X 键）===
                if 'x' in self.keys_pressed:
                    self.gripper_worker.move(self.V_G_OPEN)

                # === Rz 旋转（滚轮）===
                if scroll_dy != 0:
                    self.target_pose[5] += scroll_dy * self.SCROLL_SENSITIVITY

            # ========== 发送机械臂指令 ==========
            # 无论哪种模式，都发送 ServoP 指令
            self.robot.ServoP(*self.target_pose)

            # ========== 循环频率控制 ==========
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.LOOP_RATE) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                time_rate = 1.0  # 重置时间补偿
            else:
                # 循环超时，计算补偿系数
                time_rate = elapsed / (1.0 / self.LOOP_RATE) + 1

        # 清理退出
        self.gripper_worker.running = False  # 停止夹爪线程
        self.gripper_worker.join()           # 等待线程结束
        self.robot.close()                   # 关闭 ROS

if __name__ == "__main__":
    """
    ============================================================================
    程序入口
    ============================================================================

    使用方法：
    ---------
    1. 确保 DOBOT 驱动节点已启动：
       $ ros2 launch dobot_bringup_v3 bringup_v3.launch.py

    2. （可选）启动数据录制节点（如果需要录制数据）：
       $ ros2 run your_package data_recorder_node

    3. 运行此脚本：
       $ python3 data_collector4.py

    4. 使用鼠标/键盘控制机械臂，按 O/P/L 录制数据

    注意事项：
    ---------
    1. **垂直抓取参数调整**：
       - V_Z1, V_Z2: 根据实际工作台高度调整
       - V_G_OPEN: 根据物体尺寸调整夹爪张开值
       - V_SPEED_UP/DOWN: 调整速度避免碰撞

    2. **安全测试**：
       - 首次运行建议降低速度
       - 先在安全位置测试 Z 键抓取
       - 确认机械臂工作空间边界

    3. **性能优化**：
       - 如果夹爪响应慢，增大 GRIPPER_STEP
       - 如果控制卡顿，检查 ROS 服务延迟
       - 如果录制数据不准，检查时间戳同步

    故障排查：
    ---------
    - "Gripper ModbusCreate Failed" → 检查夹爪连接和端口配置
    - "Dobot services not available" → 检查驱动节点是否启动
    - 控制不响应 → 检查 pynput 是否有权限（sudo 或 X11）
    - 夹爪位置跳变 → 检查 Modbus 通信稳定性

    ============================================================================
    """
    controller = TeleopController()
    controller.run()