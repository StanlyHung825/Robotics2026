from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    voicegpt = Node(
        package='voicegpt',
        executable='voicegpt_node',
        name='voicegpt_node',
        output='screen', 
    )

    turtlesim = Node(
        package='turtlesim',
        executable='turtlesim_node',
        name='turtlesim',
        output='screen',
    )

    turtle_node = Node(
        package='voicegpt',        
        executable='turtlenode',       
        name='turtlenode',
        output='screen',
    )

    return LaunchDescription([
        voicegpt,
        turtlesim,
        turtle_node,
    ])