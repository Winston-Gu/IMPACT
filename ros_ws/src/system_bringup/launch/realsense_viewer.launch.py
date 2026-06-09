"""Launch RealSense cameras from config and open image_view."""

import os
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    realsense_config_default = PathJoinSubstitution(
        [
            FindPackageShare("system_bringup"),
            "config",
            "realsense_cameras.yaml",
        ]
    )

    realsense_config_arg = DeclareLaunchArgument(
        "realsense_config",
        default_value=realsense_config_default,
        description="YAML file describing RealSense cameras to launch",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/front_camera/color/image_rect_raw",
        description="Image topic to visualize with image_view",
    )
    image_topic_secondary_arg = DeclareLaunchArgument(
        "image_topic_secondary",
        default_value="/camera/side_camera/color/image_rect_raw",
        description="Second image topic to visualize with image_view",
    )

    realsense_config = LaunchConfiguration("realsense_config")
    image_topic = LaunchConfiguration("image_topic")
    image_topic_secondary = LaunchConfiguration("image_topic_secondary")

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

    def _load_realsense_nodes(context):
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
            crop_cfg = camera.get("parameters", {}) or {}

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
                    camera.get(
                        "publish_compressed", common.get("publish_compressed", True)
                    )
                ),
                "align_depth": align_depth,
                "clip_distance": float(clip_distance),
                "initial_reset": bool(
                    camera.get("parameters", {}).get("initial_reset", False)
                ),
                "crop_x": int(crop_cfg.get("crop_x", 0)),
                "crop_y": int(crop_cfg.get("crop_y", 0)),
                "crop_width": int(crop_cfg.get("crop_width", 0)),
                "crop_height": int(crop_cfg.get("crop_height", 0)),
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

    realsense_nodes = OpaqueFunction(function=_load_realsense_nodes)

    image_view = Node(
        package="image_view",
        executable="image_view",
        name="image_view",
        remappings=[("image", image_topic)],
        output="screen",
    )
    image_view_secondary = Node(
        package="image_view",
        executable="image_view",
        name="image_view_secondary",
        remappings=[("image", image_topic_secondary)],
        output="screen",
    )

    return LaunchDescription(
        [
            realsense_config_arg,
            image_topic_arg,
            image_topic_secondary_arg,
            realsense_nodes,
            image_view,
            image_view_secondary,
        ]
    )
