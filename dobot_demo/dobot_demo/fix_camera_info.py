import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

class CameraInfoFixer(Node):
    def __init__(self):
        super().__init__('camera_info_fixer')

        # 声明参数，允许通过 launch file 重映射话题
        self.declare_parameter('input_topic', '/camera/color/camera_info_raw')
        self.declare_parameter('output_topic', '/camera/color/camera_info')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        # 订阅原始话题
        self.subscription = self.create_subscription(
            CameraInfo, input_topic, self.listener_callback, 10)

        # 发布修正后的话题
        self.publisher = self.create_publisher(
            CameraInfo, output_topic, 10)

        self.fixed_count = 0
        self.get_logger().info(f'Camera Info Fixer started:')
        self.get_logger().info(f'  Input:  {input_topic}')
        self.get_logger().info(f'  Output: {output_topic}')

    def listener_callback(self, msg):
        original_size = len(msg.d)

        # 核心操作：确保 plumb_bob 模型只有 5 个畸变参数
        if msg.distortion_model == 'plumb_bob' and len(msg.d) != 5:
            msg.d = list(msg.d[:5])  # 转换为 list 确保可变
            self.fixed_count += 1

            if self.fixed_count == 1:  # 只打印第一次修正的信息
                self.get_logger().warn(
                    f'Fixed distortion parameters: {original_size} -> 5 parameters')
                self.get_logger().info(f'D = {msg.d}')

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoFixer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()