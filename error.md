
使能
```bash
ros2 service call /dobot_bringup_v3/srv/EnableRobot dobot_msgs_v3/srv/EnableRobot "{load: 5.0}"
```


设置速度因子（没卵用）
```bash
ros2 service call /dobot_bringup_v3/srv/SpeedFactor dobot_msgs_v3/srv/SpeedFactor "{ratio: 30}"
```


清除残留错误
```bash
ros2 service call /dobot_bringup_v3/srv/ClearError dobot_msgs_v3/srv/ClearError "{}"
```

模式验证
‵‵`
ros2 service call /dobot_bringup_v3/srv/RobotMode dobot_msgs_v3/srv/RobotMode "{}"
```
如果返回的 `mode` 是 **5** 或 **7**，说明机器人已就绪。


```bash
ros2 service call /dobot_bringup_v3/srv/MovJ dobot_msgs_v3/srv/MovJ "{x: 61, y: -397, z: 200, rx: -180.0, ry: 0.0, rz: 60.0}"
```

APP中：
查看错误
安全等级至多为2
tcp模式

终端2中要
```bash
export DISPLAY=:1
```
否则显示不出来



/home/hit/dobot_ws/install/dobot_bringup_v3/lib/python3.10/site-packages/dobot_bringup_v3/dobot_api.py
```
    def ServoJ(self, j1, j2, j3, j4,j5,j6,t,*dynParams):
        string = "ServoJ({:f},{:f},{:f},{:f},{:f},{:f},t={:f}".format(
            j1,j2,j3,j4,j5,j6,t)
        for params in dynParams[0]:
             string =string+ ","+ str(params)
        string =string+ ")" 
        print(string) 
        string = f"ServoJ({j1},{j2},{j3},{j4},{j5},{j6})"
        return self.sendRecvMsg(string)
```
这个函数底层有bug，t=0.1,lookahead_time=50,gain=500这几个参数传不进去

```
    def SetHoldRegs(self, id, addr, count, table, type=None):
        if type is not None:
          # 原先dddds
          string = "SetHoldRegs({:d},{:d},{:d},{:s},{:s})".format(
            id, addr, count, table, type)
        else:
          string = "SetHoldRegs({:d},{:d},{:d},{:d})".format(
            id, addr, count, table)
        return self.sendRecvMsg(string)
```

添加到输入组并刷新控制台
sudo usermod -aG input $USER
newgrp input