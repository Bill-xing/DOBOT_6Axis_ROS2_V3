""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: base_link -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "base_link",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "-0.777506",
                "--y",
                "-0.330606",
                "--z",
                "0.455513",
                "--qx",
                "-0.489589",
                "--qy",
                "0.703917",
                "--qz",
                "-0.421769",
                "--qw",
                "0.294811",
                # "--roll",
                # "2.56618",
                # "--pitch",
                # "0.97559",
                # "--yaw",
                # "2.23762",
            ],
        ),
    ]
    return LaunchDescription(nodes)
