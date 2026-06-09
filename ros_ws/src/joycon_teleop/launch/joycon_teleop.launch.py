#!/usr/bin/env python3
"""
Launch file for Joy-Con teleoperation system
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, conditions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory("joycon_teleop")
    rviz_config = os.path.join(pkg_dir, "rviz", "joycon_teleop.rviz")

    # Declare launch arguments
    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate", default_value="50.0", description="Publishing rate in Hz"
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="true", description="Start RViz visualization"
    )

    # Joy-Con teleoperation node
    joycon_node = Node(
        package="joycon_teleop",
        executable="joycon_teleop_node",
        name="joycon_teleop_node",
        output="screen",
        parameters=[
            {
                "publish_rate": LaunchConfiguration("publish_rate"),
                "world_frame": "world",
                "stick_velocity_scale": 0.5,
                "position_limits": [2.0, 2.0, 2.0],
            }
        ],
    )

    # RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        condition=conditions.IfCondition(LaunchConfiguration("use_rviz")),
    )

    # Static transform publisher for world frame
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_broadcaster",
        arguments=["0", "0", "0", "0", "0", "0", "map", "world"],
    )

    return LaunchDescription(
        [
            publish_rate_arg,
            use_rviz_arg,
            joycon_node,
            static_tf_node,
            rviz_node,
        ]
    )
