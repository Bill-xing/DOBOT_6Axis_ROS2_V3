conda activate dobot_collect
cd ~/dobot_ws
source install/local_setup.sh
ros2 launch dobot_bringup_v3 dobot_bringup_ros2.launch.py



conda activate dobot_collect
cd ~/dobot_ws
source install/local_setup.sh
ros2 launch dobot_moveit dobot_moveit.launch.py



conda activate dobot_collect
source ~/ros2_ws/install/setup.bash
ros2 launch orbbec_camera astra2.launch.py 



cd ~
conda activate dobot_collect
python /home/hit/dobot_ws/build/dobot_demo/dobot_demo/recorder.py



一定要打开全新终端 全屏运行
conda activate dobot_collect
python /home/hit/dobot_ws/build/dobot_demo/dobot_demo/data_collector4.py
cd ~/data
python show.py ./episode_3.hdf5 