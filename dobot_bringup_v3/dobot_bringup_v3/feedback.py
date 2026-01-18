#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy                                     # ROS2 Python接口库
from rclpy.node import Node                      
from dobot_msgs_v3.msg import ToolVectorActual
from dobot_msgs_v3.msg import ToolVectorTarget
from sensor_msgs.msg import JointState  
import socket
import numpy as np
import os
import time

# 数据结构定义保持不变
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

        if self.port == 30005 or self.port == 30004:
                self.socket_feedback = socket.socket()
                self.socket_feedback.settimeout(1)
                self.socket_feedback.connect((self.ip, self.port))
        else:
            print("Connect to feedback server need use port 30003 !")
    
    def feed(self):
        try:
            self.socket_feedback.setblocking(True) 
            self.all = self.socket_feedback.recv(10240)
            data = self.all[0:1440]
            a = np.frombuffer(data, dtype=MyType)
            if hex((a['test_value'][0])) == '0x123456789abcdef':
                tool_v = a['tool_vector_actual'][0]
                tool_t = a['Tool_vector_target'][0]
                tool_j = a['q_actual'][0]
            return [tool_v, tool_j, tool_t]
        except:
            return ["NG1"]

class PublisherNode(Node):
    
    def __init__(self, name):
        super().__init__(name)                                    
        self.IP = str(os.getenv("IP_address"))
        self.connect()
        # 初始化发布器
        self.pub = self.create_publisher(ToolVectorActual, "dobot_msgs_v3/msg/ToolVectorActual", 10)
        self.pub2 = self.create_publisher(JointState, "joint_states_robot", 10)
        # 补全 ToolVectorTarget 发布器
        self.pub3 = self.create_publisher(ToolVectorTarget, "dobot_msgs_v3/msg/ToolVectorTarget", 10)
        
        self.timer = self.create_timer(0.01, self.timer_callback)

    def connect(self):
        try:
           self.get_logger().info("connection:30004")
           self.feed_v = fankuis(self.IP,30004)
           self.get_logger().info("connection succeeded:30004")
        except:
            self.get_logger().info("Connection failed!!!")

    def timer_callback(self):
        actual = self.feed_v.feed()
        
        if len(actual) != 1:
            current_time = self.get_clock().now().to_msg()
            
            # 1. 处理 ToolVectorActual (msg)
            msg = ToolVectorActual()
            msg.header.stamp = current_time
            msg.header.frame_id = 'base_link'
            msg.x = actual[0][0]
            msg.y = actual[0][1]
            msg.z = actual[0][2]
            msg.rx = actual[0][3]
            msg.ry = actual[0][4]
            msg.rz = actual[0][5]

            # 2. 处理 JointState (msg2)
            msg2 = JointState()
            msg2.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            msg2.header.stamp = current_time
            msg2.header.frame_id = 'joint_states'
            q_actual_val = actual[1]
            joint_a = []
            for ii in q_actual_val:
                joint_a.append(float(ii * 3.14159 / 180))
            msg2.position = joint_a

            # 3. 补全 ToolVectorTarget (msg3)
            # 这里的格式与 ToolVectorActual 完全一致，数据取自 actual[2] (即 tool_t)
            msg3 = ToolVectorTarget()
            msg3.header.stamp = current_time
            msg3.header.frame_id = 'base_link'
            msg3.x = actual[2][0]
            msg3.y = actual[2][1]
            msg3.z = actual[2][2]
            msg3.rx = actual[2][3]
            msg3.ry = actual[2][4]
            msg3.rz = actual[2][5]

            # 发布所有消息
            self.pub.publish(msg)
            self.pub2.publish(msg2) 
            self.pub3.publish(msg3)
        
def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode("dobot_feedback")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()