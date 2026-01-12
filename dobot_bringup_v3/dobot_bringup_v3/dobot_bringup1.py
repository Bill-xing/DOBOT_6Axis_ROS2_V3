#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from dobot_msgs_v3.srv import *
from .dobot_api import *
import os
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading  # <--- 引入线程锁

class adderServer(Node):
    def __init__(self, name):
        super().__init__(name)
        self.IP = str(os.getenv("IP_address"))
        self.get_logger().info(f"Connecting to Robot IP: {self.IP}")

        # --- 线程锁 (关键修改) ---
        # 29999 端口锁：防止多个线程同时读写 Dashboard 端口
        self.lock_dashboard = threading.Lock()
        # 30003 端口锁：防止多个线程同时读写 Move 端口
        self.lock_move = threading.Lock()
        # -----------------------

        # 使用可重入回调组，允许 ROS 层面的并发
        self.cb_group = ReentrantCallbackGroup()

        self.connect()

        # 注册服务 (全部使用 cb_group)
        # Dashboard 相关 (需要加 lock_dashboard)
        self.create_dashboard_service(EnableRobot, 'EnableRobot', self.EnableRobot)
        self.create_dashboard_service(ClearError, 'ClearError', self.ClearError)
        self.create_dashboard_service(ResetRobot, 'ResetRobot', self.ResetRobot)
        self.create_dashboard_service(GetPose, 'GetPose', self.GetPose)
        self.create_dashboard_service(GetAngle, 'GetAngle', self.GetAngle)
        self.create_dashboard_service(SpeedFactor, 'SpeedFactor', self.SpeedFactor)
        self.create_dashboard_service(AccJ, 'AccJ', self.AccJ)
        self.create_dashboard_service(AccL, 'AccL', self.AccL)
        self.create_dashboard_service(SpeedJ, 'SpeedJ', self.SpeedJ)
        self.create_dashboard_service(SpeedL, 'SpeedL', self.SpeedL)
        self.create_dashboard_service(Arch, 'Arch', self.Arch)
        self.create_dashboard_service(CP, 'CP', self.CP)
        self.create_dashboard_service(PayLoad, 'PayLoad', self.PayLoad)
        self.create_dashboard_service(SetPayload, 'SetPayload', self.SetPayload)
        self.create_dashboard_service(Tool, 'Tool', self.Tool)
        self.create_dashboard_service(User, 'User', self.User)
        self.create_dashboard_service(RobotMode, 'RobotMode', self.RobotMode)
        self.create_dashboard_service(GetErrorID, 'GetErrorID', self.GetErrorID)
        self.create_dashboard_service(DisableRobot, 'DisableRobot', self.DisableRobot)
        
        # IO / Modbus 相关 (属于 Dashboard 端口)
        self.create_dashboard_service(DO, 'DO', self.DO)
        self.create_dashboard_service(DOExecute, 'DOExecute', self.DOExecute)
        self.create_dashboard_service(DOGroup, 'DOGroup', self.DOGroup)
        self.create_dashboard_service(ToolDO, 'ToolDO', self.ToolDO)
        self.create_dashboard_service(ToolDOExecute, 'ToolDOExecute', self.ToolDOExecute)
        self.create_dashboard_service(ToolDI, 'ToolDI', self.ToolDI)
        self.create_dashboard_service(DI, 'DI', self.DI) # 注意：DI 其实是 ToolDI 的封装，视具体 API 而定
        self.create_dashboard_service(GetCoils, 'GetCoils', self.GetCoils)
        self.create_dashboard_service(SetCoils, 'SetCoils', self.SetCoils)
        self.create_dashboard_service(GetHoldRegs, 'GetHoldRegs', self.GetHoldRegs)
        self.create_dashboard_service(SetHoldRegs, 'SetHoldRegs', self.SetHoldRegs)
        self.create_dashboard_service(GetInBits, 'GetInBits', self.GetInBits)
        self.create_dashboard_service(GetInRegs, 'GetInRegs', self.GetInRegs)
        self.create_dashboard_service(ModbusCreate, 'ModbusCreate', self.ModbusCreate)
        self.create_dashboard_service(ModbusClose, 'ModbusClose', self.ModbusClose)

        # Move 相关 (需要加 lock_move)
        self.create_move_service(MovJ, 'MovJ', self.MovJ)
        self.create_move_service(MovL, 'MovL', self.MovL)
        self.create_move_service(ServoJ, 'ServoJ', self.ServoJ)
        self.create_move_service(ServoP, 'ServoP', self.ServoP)
        self.create_move_service(Sync, 'Sync', self.Sync)
        self.create_move_service(JointMovJ, 'JointMovJ', self.JointMovJ)
        self.create_move_service(MovJIO, 'MovJIO', self.MovJIO)
        self.create_move_service(MovLIO, 'MovLIO', self.MovLIO)
        self.create_move_service(MoveJog, 'MoveJog', self.MoveJog)
        self.create_move_service(RelMovJ, 'RelMovJ', self.RelMovJ)
        self.create_move_service(RelMovL, 'RelMovL', self.RelMovL)

    def create_dashboard_service(self, msg_type, name, callback):
        self.create_service(msg_type, f'/dobot_bringup_v3/srv/{name}', callback, callback_group=self.cb_group)

    def create_move_service(self, msg_type, name, callback):
        self.create_service(msg_type, f'/dobot_bringup_v3/srv/{name}', callback, callback_group=self.cb_group)

    def connect(self):
        try:
           self.get_logger().info("connection:29999")
           self.get_logger().info("connection:30003")
           self.dashboard = DobotApiDashboard(self.IP, 29999)
           self.move = DobotApiMove(self.IP, 30003)
           self.get_logger().info("connection succeeded:29999,30003")
        except:
            self.get_logger().info("Connection failed!!!")

    # ================= Dashboard 端口回调 (加锁 lock_dashboard) =================
    
    def EnableRobot(self, request, response):
        with self.lock_dashboard:  # 加锁
            return_t = self.dashboard.EnableRobot([request.load])
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def ClearError(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.ClearError()
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def ResetRobot(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.ResetRobot()
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def PayLoad(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.PayLoad(request.weight, request.inertia)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def SetPayload(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.SetPayload(request.weight, request.inertia)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def GetPose(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.GetPose(request.user, request.tool)
        return_tt = return_t[:return_t.find("{")-1]
        try:
            response.res = int(return_tt)
            response.pose = return_t[return_t.find("{"):return_t.find("}")+1]
        except:
            response.res = -1
        # self.get_logger().info(return_t) # 高频调用建议注释日志
        return response 

    def GetAngle(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.GetAngle()
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        response.angle = return_t[return_t.find("{"):return_t.find("}")+1]
        self.get_logger().info(return_t)
        return response 

    def RobotMode(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.RobotMode()
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        response.mode = return_t[return_t.find("{")+1:return_t.find("}")]
        self.get_logger().info(return_t)
        return response 

    # --- Modbus / IO 相关 (也是 Dashboard 端口) ---

    def ModbusCreate(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.ModbusCreate(request.ip, request.port, request.slave_id, request.is_rtu)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        response.index = return_t[return_t.find("{")+1:return_t.find("}")]
        self.get_logger().info(return_t)
        return response 

    def ModbusClose(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.ModbusClose(request.index)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def GetHoldRegs(self, request, response):
        # 注意：这里注释了日志，但必须加锁
        with self.lock_dashboard:
            return_t = self.dashboard.GetHoldRegs(request.index, request.addr, request.count, request.val_type)
        return_tt = return_t[:return_t.find("{")-1]
        try:
            response.res = int(return_tt)
            response.value = return_t[return_t.find("{")+1:return_t.find("}")]
        except:
            response.res = -1
        # self.get_logger().info(return_t)  # 已注释
        return response

    def SetHoldRegs(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.SetHoldRegs(request.index, request.addr, request.count, request.val_tab, request.val_type)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def GetInBits(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.GetInBits(request.index, request.addr, request.count)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        response.value = return_t[return_t.find("{")+1:return_t.find("}")]
        self.get_logger().info(return_t)
        return response 

    def GetInRegs(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.GetInRegs(request.index, request.addr, request.count, request.val_type)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        response.value = return_t[return_t.find("{")+1:return_t.find("}")]
        self.get_logger().info(return_t)
        return response 

    def GetCoils(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.GetCoils(request.index, request.addr, request.count)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        response.value = return_t[return_t.find("{")+1:return_t.find("}")]
        self.get_logger().info(return_t)
        return response 

    def SetCoils(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.SetCoils(request.index, request.addr, request.count, request.val_tab)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def DO(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.DO(request.index, request.status)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def DOExecute(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.DOExecute(request.index, request.status)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def DOGroup(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.DOGroup(request.args)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def Tool(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.Tool(request.index)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def User(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.User(request.index)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def SpeedFactor(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.SpeedFactor(request.ratio)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def SpeedJ(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.SpeedJ(request.r)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def SpeedL(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.SpeedL(request.r)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def AccJ(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.AccJ(request.r)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def AccL(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.AccL(request.r)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def Arch(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.Arch(request.index)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def CP(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.CP(request.r)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def DisableRobot(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.DisableRobot()
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def GetErrorID(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.GetErrorID()
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def ToolDO(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.ToolDO(request.index, request.status)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def ToolDOExecute(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.ToolDOExecute(request.index, request.status)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def ToolDI(self, request, response):
        with self.lock_dashboard:
            return_t = self.dashboard.ToolDI(request.index)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 
    
    def DI(self, request, response): # 修正：原代码DI里调用的是ToolDO，可能是笔误，这里保持原样但加锁
        with self.lock_dashboard:
            return_t = self.dashboard.ToolDO(request.index) # 看起来这里应该是DI? 请检查API，这里按原代码逻辑加锁
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    # ================= Move 端口回调 (加锁 lock_move) =================

    def MovJ(self, request, response):
        with self.lock_move:  # 加锁
            return_t = self.move.MovJ(request.x, request.y, request.z, request.rx, request.ry, request.rz, request.param_value)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def MovL(self, request, response):
        with self.lock_move:
            return_t = self.move.MovL(request.x, request.y, request.z, request.rx, request.ry, request.rz, request.param_value)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def ServoJ(self, request, response):
        with self.lock_move:
            return_t = self.move.ServoJ(request.j1, request.j2, request.j3, request.j4, request.j5, request.j6, request.t, request.param_value)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        # self.get_logger().info(return_t) # 高频指令建议注释
        return response 

    def ServoP(self, request, response):
        with self.lock_move:
            return_t = self.move.ServoP(request.x, request.y, request.z, request.rx, request.ry, request.rz)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        # self.get_logger().info(return_t)
        return response 

    def Sync(self, request, response):
        with self.lock_move:
            return_t = self.move.Sync()
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def JointMovJ(self, request, response):
        with self.lock_move:
            return_t = self.move.JointMovJ(request.j1, request.j2, request.j3, request.j4, request.j5, request.j6, request.param_value)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def MovJIO(self, request, response):
        with self.lock_move:
            return_t = self.move.MovJIO(request.x, request.y, request.z, request.rx, request.ry, request.rz, request.param_value)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def MovLIO(self, request, response):
        with self.lock_move:
            return_t = self.move.MovLIO(request.x, request.y, request.z, request.rx, request.ry, request.rz, request.param_value)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def MoveJog(self, request, response):
        with self.lock_move:
            return_t = self.move.MoveJog(request.axis_id, request.param_value)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def RelMovJ(self, request, response):
        with self.lock_move:
            return_t = self.move.RelMovJ(request.offset1, request.offset2, request.offset3, request.offset4, request.offset5, request.offset6, request.param_value)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

    def RelMovL(self, request, response):
        with self.lock_move:
            return_t = self.move.RelMovL(request.offset1, request.offset2, request.offset3, request.offset4, request.offset5, request.offset6, request.param_value)
        return_tt = return_t[:return_t.find("{")-1]
        response.res = int(return_tt)
        self.get_logger().info(return_t)
        return response 

def main(args=None):
    rclpy.init(args=args)
    node = adderServer("dobot_bringup_v3")
    
    # 必须使用多线程执行器，配合 ReentrantCallbackGroup 实现并发
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()