""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: dummy_link -> camera_color_optical_frame """
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
                "dummy_link",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "-56.9025",
                "--y",
                "-79.3526",
                "--z",
                "42.2258",
                "--qx",
                "-0.434044",
                "--qy",
                "0.315847",
                "--qz",
                "-0.000607426",
                "--qw",
                "0.843709",
                # "--roll",
                # "2.09547",
                # "--pitch",
                # "2.57887",
                # "--yaw",
                # "-2.81274",
            ],
        ),
    ]
    return LaunchDescription(nodes)
