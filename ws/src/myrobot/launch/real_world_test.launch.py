import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    myplan_share = get_package_share_directory("myplan")
    demo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(myplan_share, "launch", "demo.launch.py")
        )
    )

    real_arm_interface = Node(
        package="myrobot",
        executable="magnet_moveit_real_arm_interface",
        output="screen",
    )

    stm32_serial_interface = Node(
        package="myrobot",
        executable="magnet_serial_with_ST",
        output="screen",
    )

    return LaunchDescription(
        [
            demo_launch,
            real_arm_interface,
            stm32_serial_interface,
        ]
    )
