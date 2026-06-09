#!/usr/bin/env python3
"""
Multi-robot Franka bringup with Joy-Con teleoperation and camera launch.
"""

import os
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from system_bringup.launch_utils import load_yaml


def _as_bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "on", "yes")


def _parse_profile(profile_string: str, default=(640, 480, 30)):
    if not profile_string:
        return default
    text = str(profile_string).lower().replace(" ", "")
    try:
        width, height, fps = text.split("x")
        return int(width), int(height), int(fps)
    except ValueError:
        return default


def _get_nested(entry, keys, default=None):
    current = entry
    for key in keys.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _format_namespace(namespace):
    if not namespace:
        return ""
    return "/" + namespace.strip("/")


def _load_realsense_nodes(context):
    launch_realsense = LaunchConfiguration("launch_realsense")
    realsense_config = LaunchConfiguration("realsense_config")
    if not _as_bool(context.perform_substitution(launch_realsense)):
        return []

    config_path = Path(
        os.path.expanduser(context.perform_substitution(realsense_config))
    ).resolve()
    if not config_path.exists():
        raise RuntimeError(f"RealSense config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as cfg_file:
        rs_config = yaml.safe_load(cfg_file) or {}

    cameras = rs_config.get("cameras", [])
    if not cameras:
        raise RuntimeError(f"No cameras defined in config file {config_path}")
    common = rs_config.get("common_parameters", {}) or {}
    nodes = []
    for camera in cameras:
        camera_name = camera.get("camera_name") or camera.get("name")
        serial_no = camera.get("serial_no") or camera.get("serial")
        if not camera_name or not serial_no:
            raise RuntimeError(
                f"Camera entry must include camera_name and serial_no: {camera}"
            )
        color_profile = _parse_profile(
            camera.get("rgb_camera", {}).get(
                "profile", camera.get("rgb_camera.profile")
            )
            or common.get("rgb_camera.profile", "640x480x30")
        )
        depth_profile = _parse_profile(
            camera.get("depth_module", {}).get(
                "profile", camera.get("depth_module.profile")
            )
            or common.get("depth_module.profile", "640x480x30"),
            default=color_profile,
        )
        align_cfg = _get_nested(camera, "align_depth.enable")
        if align_cfg is None:
            align_cfg = camera.get(
                "align_depth", common.get("align_depth.enable", True)
            )

        clip_distance = camera.get("parameters", {}).get(
            "clip_distance",
            common.get("clip_distance", 3.0),
        )
        if clip_distance is None:
            clip_distance = 0.0
        align_depth = bool(align_cfg if align_cfg is not None else True)

        params = {
            "serial": str(serial_no),
            "camera_name": str(camera_name),
            "camera_namespace": camera.get(
                "camera_namespace", common.get("camera_namespace", "camera")
            ),
            "base_frame_id": camera.get("base_frame_id", f"{camera_name}_link"),
            "color_frame_id": camera.get(
                "color_frame_id", f"{camera_name}_color_optical_frame"
            ),
            "depth_frame_id": camera.get(
                "depth_frame_id", f"{camera_name}_depth_optical_frame"
            ),
            "color_width": color_profile[0],
            "color_height": color_profile[1],
            "color_fps": color_profile[2],
            "depth_width": depth_profile[0],
            "depth_height": depth_profile[1],
            "depth_fps": depth_profile[2],
            "publish_depth": bool(
                camera.get("enable_depth", common.get("enable_depth", True))
            ),
            "publish_color": bool(
                camera.get("enable_color", common.get("enable_color", True))
            ),
            "publish_compressed": bool(
                camera.get("publish_compressed", common.get("publish_compressed", True))
            ),
            "align_depth": align_depth,
            "clip_distance": float(clip_distance),
            "initial_reset": bool(
                camera.get("parameters", {}).get("initial_reset", False)
            ),
        }
        node_name = camera.get("node_name", f"{camera_name}_realsense")
        nodes.append(
            Node(
                package="cameras",
                executable="realsense",
                name=node_name,
                output="screen",
                parameters=[params],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    camera.get("log_level", "WARN"),
                ],
            )
        )
    return nodes


def _load_robot_nodes(context):
    config_file = LaunchConfiguration("robot_config_file").perform(context)
    controllers_yaml = LaunchConfiguration("controllers_yaml").perform(context)
    controller_name = LaunchConfiguration("controller_name").perform(context)
    launch_pose_error = LaunchConfiguration("launch_pose_error_broadcaster")
    pose_error_rate = LaunchConfiguration("pose_error_rate")
    gripper_config_path = LaunchConfiguration("gripper_config").perform(context)
    configs = load_yaml(config_file) or {}
    gripper_cfg = load_yaml(gripper_config_path) or {}
    gripper_params = gripper_cfg.get("franka_gripper_bridge", {}).get(
        "ros__parameters", {}
    )

    nodes = []
    for _, config in configs.items():
        namespace = str(config["namespace"])
        ns_prefix = _format_namespace(namespace)
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("system_bringup"),
                            "launch",
                            "franka.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "arm_id": str(config["arm_id"]),
                    "arm_prefix": str(config["arm_prefix"]),
                    "namespace": str(namespace),
                    "urdf_file": str(config["urdf_file"]),
                    "robot_ip": str(config["robot_ip"]),
                    "load_gripper": str(config["load_gripper"]),
                    "use_fake_hardware": str(config["use_fake_hardware"]),
                    "fake_sensor_commands": str(config["fake_sensor_commands"]),
                    "joint_state_rate": str(config["joint_state_rate"]),
                    "controllers_yaml": controllers_yaml,
                    "controller_name": controller_name,
                }.items(),
            )
        )
        # nodes.append(
        #     Node(
        #         package="controller_manager",
        #         executable="spawner",
        #         namespace=namespace,
        #         arguments=[controller_name, "--controller-manager-timeout", "30"],
        #         parameters=[controllers_yaml],
        #         output="screen",
        #     )
        # )
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                namespace=namespace,
                arguments=["pose_broadcaster", "--controller-manager-timeout", "30"],
                parameters=[controllers_yaml],
                output="screen",
            )
        )
        nodes.append(
            Node(
                package="system_bringup",
                executable="pose_error_broadcaster",
                name="pose_error_broadcaster",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "current_pose_topic": "current_pose",
                        "target_pose_topic": "target_pose",
                        "output_topic": "pose_error",
                        "publish_rate": pose_error_rate,
                    }
                ],
                condition=IfCondition(launch_pose_error),
            )
        )
        nodes.append(
            Node(
                package="system_bringup",
                executable="franka_gripper_bridge",
                name="franka_gripper_bridge",
                namespace=namespace,
                output="screen",
                parameters=[
                    gripper_params,
                    {
                        "command_topic": (
                            f"{ns_prefix}/gripper/gripper_position_controller/commands"
                            if ns_prefix
                            else "/gripper/gripper_position_controller/commands"
                        ),
                        "gripper_namespace": (
                            f"{ns_prefix}/franka_gripper"
                            if ns_prefix
                            else "/franka_gripper"
                        ),
                        "normalized_joint_state_topic": (
                            f"{ns_prefix}/gripper/joint_states"
                            if ns_prefix
                            else "/gripper/joint_states"
                        ),
                    },
                ],
            )
        )

    return nodes


def _load_teleop_nodes(context):
    teleop_config_path = LaunchConfiguration("teleop_config").perform(context)
    teleop_cfg = load_yaml(teleop_config_path) or {}
    mode = str(teleop_cfg.get("mode", "single")).lower()
    default_side = str(teleop_cfg.get("default_side", "right")).lower()
    base_frame = str(teleop_cfg.get("base_frame", "fr3_link0"))
    mapping = teleop_cfg.get("joycon_mapping", {}) or {}
    dataset_params = (teleop_cfg.get("dataset_recorder", {}) or {}).get(
        "ros__parameters", {}
    ) or {}

    joycon_params = teleop_cfg.get("joycon_single_teleop_node", {}).get(
        "ros__parameters", {}
    )

    nodes = []
    sides = []
    if mode == "dual":
        sides = ["left", "right"]
    else:
        sides = [default_side]

    for side in sides:
        namespace = mapping.get(side, "")
        ns_prefix = _format_namespace(namespace)
        robot_pose_topic = f"{ns_prefix}/current_pose" if ns_prefix else "/current_pose"
        target_pose_topic = f"{ns_prefix}/target_pose" if ns_prefix else "/target_pose"
        gripper_state_topic = (
            f"{ns_prefix}/gripper/joint_states"
            if ns_prefix
            else "/gripper/joint_states"
        )
        gripper_command_topic = (
            f"{ns_prefix}/gripper/gripper_position_controller/commands"
            if ns_prefix
            else "/gripper/gripper_position_controller/commands"
        )

        nodes.append(
            Node(
                package="joycon_teleop",
                executable="joycon_single_teleop_node",
                name=f"joycon_single_teleop_node_{side}",
                output="screen",
                parameters=[
                    joycon_params,
                    {
                        "joycon_side": side,
                        "world_frame": base_frame,
                        "robot_pose_topic": robot_pose_topic,
                        "target_pose_topic": target_pose_topic,
                        "gripper_state_topic": gripper_state_topic,
                        "gripper_command_topic": gripper_command_topic,
                    },
                ],
            )
        )

    dataset_recorder_node = Node(
        condition=IfCondition(LaunchConfiguration("record_dataset")),
        package="system_bringup",
        executable="dataset_recorder",
        name="dataset_recorder",
        output="screen",
        parameters=[
            dataset_params,
            {
                "dataset_name": LaunchConfiguration("dataset_name"),
                "camera_config": LaunchConfiguration("realsense_config"),
            },
        ],
    )
    nodes.append(dataset_recorder_node)
    nodes.append(OpaqueFunction(function=_load_realsense_nodes))

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("system_bringup"), "config", "franka.config.yaml"]
                ),
                description="Path to the robot configuration file to load",
            ),
            DeclareLaunchArgument(
                "controllers_yaml",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("system_bringup"),
                        "config",
                        "controller",
                        "predictive_controllers.yaml",
                    ]
                ),
                description="Path to controllers.yaml",
            ),
            DeclareLaunchArgument(
                "controller_name",
                default_value="cartesian_impedance_controller",
                description="Name of the controller to spawn",
            ),
            DeclareLaunchArgument(
                "launch_pose_error_broadcaster",
                default_value="false",
                description="Launch pose error broadcaster node per robot.",
            ),
            DeclareLaunchArgument(
                "pose_error_rate",
                default_value="100.0",
                description="Publish rate for pose error broadcaster (Hz).",
            ),
            DeclareLaunchArgument(
                "gripper_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("system_bringup"), "config", "gripper.yaml"]
                ),
                description="Gripper bridge configuration file",
            ),
            DeclareLaunchArgument(
                "teleop_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("system_bringup"), "config", "teleoperation.yaml"]
                ),
                description="Teleoperation config (joycon mapping, mode)",
            ),
            DeclareLaunchArgument(
                "record_dataset",
                default_value="false",
                description="Enable dataset recorder node for demonstrations",
            ),
            DeclareLaunchArgument(
                "dataset_name",
                default_value="Test",
                description="Name of the dataset folder to append trajectories to",
            ),
            DeclareLaunchArgument(
                "realsense_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("system_bringup"),
                        "config",
                        "realsense_cameras.yaml",
                    ]
                ),
                description="YAML file describing RealSense cameras to launch",
            ),
            DeclareLaunchArgument(
                "launch_realsense",
                default_value="true",
                description="Launch RealSense camera drivers defined in the config file",
            ),
            OpaqueFunction(function=_load_robot_nodes),
            OpaqueFunction(function=_load_teleop_nodes),
        ]
    )
