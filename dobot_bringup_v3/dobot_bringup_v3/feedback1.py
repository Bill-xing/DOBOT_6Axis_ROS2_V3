#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 循环读取夹爪状态发布，卡顿

import rclpy                                     # ROS2 Python接口库
from rclpy.time import Time
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor                    
from dobot_msgs_v3.msg import ToolVectorActual
from sensor_msgs.msg import JointState  
from dobot_msgs_v3.srv import GetHoldRegs # 引入服务类型
import socket
import numpy as np
import os




MyType = np.dtype([('len',np.int64,), ('digital_input_bits',np.uint64,), ('digital_output_bits',
    np.uint64,), ('robot_mode',np.uint64,), ('time_stamp',np.uint64,), ( 'time_stamp_reserve_bit', np.uint64,),
    ('test_value',np.uint64,), ('test_value_keep_bit', np.float64,), ('speed_scaling',np.float64,), ('linear_momentum_norm',np.float64,),
    ( 'v_main',np.float64,), ('v_robot',np.float64,), ('i_robot',np.float64,), ('i_robot_keep_bit1',np.float64,), ( 'i_robot_keep_bit2',np.float64,),
    ('tool_accelerometer_values', np.float64, (3, )),
    ('elbow_position', np.float64, (3, )),
    ('elbow_velocity', np.float64, (3, )),
    ('q_target', np.float64, (6, )),
    ('qd_target', np.float64, (6, )),
    ('qdd_target', np.float64, (6, )),
    ('i_target', np.float64, (6, )),
    ('m_target', np.float64, (6, )),
    ('q_actual', np.float64, (6, )),
    ('qd_actual', np.float64, (6, )),
    ('i_actual', np.float64, (6, )),
    ('actual_TCP_force', np.float64, (6, )),
    ('tool_vector_actual', np.float64, (6, )),
    ('TCP_speed_actual', np.float64, (6, )),
    ('TCP_force', np.float64, (6, )),
    ('Tool_vector_target', np.float64, (6, )),
    ('TCP_speed_target', np.float64, (6, )),
    ('motor_temperatures', np.float64, (6, )),
    ('joint_modes', np.float64, (6, )),
    ('v_actual', np.float64, (6, )),
    ('hand_type', np.byte, (4,)),
    ('user', np.byte,),
    ('tool', np.byte,),
    ('run_queued_cmd', np.byte,),
    ('pause_cmd_flag', np.byte,),
    ('velocity_ratio', np.int8,),
    ('acceleration_ratio', np.int8,),
    ('jerk_ratio', np.int8,),
    ('xyz_velocity_ratio', np.int8,),
    ('r_velocity_ratio', np.int8,),
    ('xyz_acceleration_ratio', np.int8,),
    ('r_acceleration_ratio', np.int8,),
    ('xyz_jerk_ratio', np.int8,),
    ('r_jerk_ratio', np.int8,),
    ('brake_status', np.int8,),
    ('enable_status', np.int8,),
    ('drag_status', np.int8,),
    ('running_status', np.int8,),
    ('error_status',np.int8,),
    ('jog_status', np.int8,),
    ('robot_type', np.int8,),
    ('drag_button_signal', np.int8,),
    ('enable_button_signal', np.int8,),
    ('record_button_signal', np.int8,),
    ('reappear_button_signal', np.int8,),
    ('jaw_button_signal', np.int8,),
    ('six_force_online', np.int8,),
    ('reserve2', np.int8, (82,)),
    ('m_actual', np.float64, (6,)),
    ('load', np.float64,),
    ('center_x', np.float64,),
    ('center_y', np.float64,),
    ('center_z', np.float64,),
    ('user1', np.float64, (6,)),
    ('Tool1', np.float64, (6,)),
    ('trace_index', np.float64,),
    ('six_force_value', np.float64, (6,)),
    ('target_quaternion', np.float64, (4,)),
    ('actual_quaternion', np.float64, (4,)),
    ('reserve3',np.int8, (24,))
     ])



class fankuis():
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket_feedback = 0
        with open('./log_feed.txt', 'w') as f:
            pass

        if self.port == 30005 or self.port == 30004:
                self.socket_feedback = socket.socket()
                self.socket_feedback.settimeout(1)
                self.socket_feedback.connect((self.ip, self.port))
        else:
            print("Connect to feedback server need use port 30003 !")
    
    def feed(self):
        try:
            self.socket_feedback.setblocking(True)  # 需要先设置为非阻塞, 使用select超时机制清空
            self.all = self.socket_feedback.recv(10240)
            data = self.all[0:1440]
            a = np.frombuffer(data, dtype=MyType)
            # if hex((a['test_value'][0])) == '0x123456789abcdef':
            #     tool_v = a['tool_vector_actual'][0]
            #     tool_j = a['q_actual'][0]
            # return [tool_v,tool_j]
            return a
        except:
            return []

"""
创建一个发布者节点
"""
class PublisherNode(Node):
    
    def __init__(self, name):
        super().__init__(name)
        self.IP = str(os.getenv("IP_address"))
        
        # 使用 ReentrantCallbackGroup 允许并发执行 (UDP接收和Modbus请求互不阻塞)
        self.cb_group = ReentrantCallbackGroup()

        # 1. 机械臂相关发布器
        self.pub_tool = self.create_publisher(ToolVectorActual, "dobot_msgs_v3/msg/ToolVectorActual", 10)
        self.pub_joint_robot = self.create_publisher(JointState, "joint_states_robot", 10)
        
        # 2. 夹爪相关发布器 (新增)
        self.pub_gripper = self.create_publisher(JointState, "gripper/state", 10)
        
        # 3. 夹爪服务客户端 (连接到 dobot_bringup)
        self.cli_get_regs = self.create_client(GetHoldRegs, '/dobot_bringup_v3/srv/GetHoldRegs', callback_group=self.cb_group)
        self.gripper_slave_id = 1 # 默认 Modbus ID

        # 4. 时间同步变量
        self.time_offset_ns = None
        self.last_robot_time_ms = 0

        # 连接 UDP
        self.connect_udp()
        
        # 等待 Modbus 服务上线 (非阻塞等待，避免卡死初始化)
        # 注意：这里假设终端1的 dobot_bringup 服务已经由同一 launch 文件启动
        
        # 5. 定时器
        # 机械臂状态定时器 (4ms / 250Hz) - 读取 UDP
        self.timer_robot = self.create_timer(0.004, self.robot_timer_callback, callback_group=self.cb_group)
        
        # 夹爪轮询定时器 (17ms / 60Hz) - 请求 Modbus
        self.timer_gripper = self.create_timer(0.017, self.gripper_timer_callback, callback_group=self.cb_group)

    def connect_udp(self):
        try:
           self.get_logger().info(f"Connecting to Robot Feedback UDP: {self.IP}:30004")
           self.feed_v = fankuis(self.IP, 30004)
           self.get_logger().info("UDP Connection Succeeded")
        except Exception as e:
            self.get_logger().error(f"UDP Connection Failed: {e}")

    def robot_timer_callback(self):
        """处理机械臂 UDP 数据"""
        try:
            actual = self.feed_v.feed() # 获取解析后的 numpy 数据
            
            if len(actual) != 0:
                # --- 时间戳对齐逻辑 ---
                robot_time_ms = float(actual[0]['time_stamp'])
                current_host_time = self.get_clock().now()

                # 计算或校准 Offset
                # 如果是第一次，或者检测到机械臂时间跳变（重启），重置 offset
                if self.time_offset_ns is None or abs(robot_time_ms - self.last_robot_time_ms) > 5000:
                    robot_time_ns = int(robot_time_ms * 1_000_000)
                    host_time_ns = current_host_time.nanoseconds
                    self.time_offset_ns = host_time_ns - robot_time_ns
                    self.get_logger().info(f"[Time Sync] Robot Offset Initialized: {self.time_offset_ns} ns")
                
                self.last_robot_time_ms = robot_time_ms
                
                # 计算最终对齐时间
                aligned_time_ns = int((robot_time_ms * 1_000_000) + self.time_offset_ns)
                aligned_timestamp = Time(nanoseconds=aligned_time_ns).to_msg()
                # ---------------------

                # 发布关节数据
                msg_joint = JointState()
                msg_joint.header.stamp = aligned_timestamp
                msg_joint.header.frame_id = 'joint_states'
                msg_joint.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
                
                q_target = actual[0]['q_actual']
                msg_joint.position = [float(x * 3.14159265358979 / 180) for x in q_target]
                self.pub_joint_robot.publish(msg_joint)

                # 发布末端位姿 (无header字段，直接赋值)
                msg_tool = ToolVectorActual()
                tool_vec = actual[0]['tool_vector_actual']
                msg_tool.x, msg_tool.y, msg_tool.z = tool_vec[0], tool_vec[1], tool_vec[2]
                msg_tool.rx, msg_tool.ry, msg_tool.rz = tool_vec[3], tool_vec[4], tool_vec[5]
                self.pub_tool.publish(msg_tool)
                
        except Exception as e:
            # 偶尔的丢包忽略，频繁报错才打印
            pass

    def gripper_timer_callback(self):
        """处理夹爪 Modbus 轮询"""
        if not self.cli_get_regs.service_is_ready():
            return # 服务还没准备好，跳过

        # 记录发送时间 t1
        t1 = self.get_clock().now()

        # 构造请求：读取夹爪实时位置 (假设地址 514 / 0x0202)
        req = GetHoldRegs.Request()
        req.index = self.gripper_slave_id
        req.addr = 514 
        req.count = 1
        # req.val_type = "U16" # 根据实际情况填写，如果默认处理可留空

        # 异步调用
        future = self.cli_get_regs.call_async(req)
        # 使用 lambda 传递 t1 到回调
        future.add_done_callback(lambda f: self.gripper_response_callback(f, t1))

    def gripper_response_callback(self, future, t1):
        try:
            response = future.result()
            # 记录接收时间 t2
            t2 = self.get_clock().now()
            
            # --- 时间戳对齐逻辑 ---
            # 夹爪状态的真实发生时间估算为请求和响应的中间时刻
            t_mid_ns = int((t1.nanoseconds + t2.nanoseconds) / 2)
            timestamp = Time(nanoseconds=t_mid_ns).to_msg()
            # ---------------------

            if response.res == 0: # 0 表示成功
                # 解析返回值，通常是字符串 "{500}" 或 "500"
                try:
                    val_str = str(response.value).strip("{}")
                    position_raw = float(val_str)
                    
                    # 归一化位置 0-1000 -> 0.0-1.0 (可选，视 MoveIt 配置而定)
                    # 或者保持 0-1000，但在 URDF 中定义对应的 limit
                    position = position_raw 

                    msg = JointState()
                    msg.header.stamp = timestamp
                    msg.header.frame_id = "gripper_base_link" # 根据你的 URDF 修改
                    msg.name = ["gripper_finger_joint"] # 你的 URDF 中夹爪关节的名字
                    msg.position = [position]
                    
                    self.pub_gripper.publish(msg)
                except ValueError:
                    pass
        except Exception as e:
            self.get_logger().warn(f"Gripper read failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode("dobot_feedback")
    
    # 关键：使用多线程执行器，确保 Timer 和 Service Client 回调能并行处理
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()
