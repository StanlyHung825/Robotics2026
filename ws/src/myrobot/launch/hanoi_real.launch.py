import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Include MoveIt / RViz2 simulation launch file from myplan package
    myplan_share = get_package_share_directory('myplan')
    demo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(myplan_share, 'launch', 'demo.launch.py')
        )
    )

    # 2. Hanoi Planner node (handles planning service)
    hanoi_planner = Node(
        package='myrobot',
        executable='hanoi_planner',
        name='hanoi_planner',
        output='screen'
    )

    # 3. USB Camera driver node
    usb_cam = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[{'video_device': '/dev/video4'}]
    )

    # 4. Hanoi Vision Tracking node
    hanoi_vision = Node(
        package='myrobot',
        executable='hanoi_vision',
        name='hanoi_vision',
        output='screen'
    )

    hanoi_status_to_sim = Node(
        package='myrobot',
        executable='hanoi_status_to_sim',
        name='hanoi_status_to_sim',
        output='screen'
    )

    # 5. Voice GPT parser node
    voicegpt = Node(
        package='voicegpt',
        executable='voicegpt_node',
        name='voicegpt_node',
        output='screen'
    )

    magnet_serial = Node(
        package='myrobot',
        executable='magnet_serial_with_ST',
        name='magnet_serial',
        output='screen'
    )

    magnet_moveit = Node(
        package='myrobot',
        executable='magnet_moveit_real_arm_interface',
        name='magnet_moveit',
        output='screen'
    )

    return LaunchDescription([
        demo_launch,
        hanoi_planner,
        usb_cam,
        hanoi_vision,
        hanoi_status_to_sim,
        voicegpt,
        magnet_serial,
        magnet_moveit
    ])
