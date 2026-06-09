import xacro
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetEnvironmentVariable,
    Shutdown,
    UnsetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from system_bringup.launch_utils import load_yaml


def _as_bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "on", "yes")


def _format_namespace(namespace: str) -> str:
    if not namespace:
        return ""
    return "/" + namespace.strip("/")


def _resolve_mujoco_model(model: str, context) -> str:
    if not model:
        return model
    if "/" in model:
        return model
    name = model if model.endswith(".xml") else f"{model}.xml"
    return PathJoinSubstitution(
        [FindPackageShare("sim_bringup"), "assets", name]
    ).perform(context)


def generate_robot_nodes(context):
    namespace = LaunchConfiguration("namespace").perform(context)
    controllers_yaml = LaunchConfiguration("controllers_yaml").perform(context)
    mujoco_overrides = PathJoinSubstitution(
        [
            FindPackageShare("sim_bringup"),
            "config",
            "mujoco_overrides.yaml",
        ]
    ).perform(context)
    controller_name = LaunchConfiguration("controller_name").perform(context)
    load_gripper = _as_bool(LaunchConfiguration("load_gripper").perform(context))
    use_fake_hardware = LaunchConfiguration("use_fake_hardware").perform(context)
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands").perform(context)
    mujoco_model = _resolve_mujoco_model(
        LaunchConfiguration("mujoco_model").perform(context), context
    )
    launch_joycon = _as_bool(LaunchConfiguration("launch_joycon").perform(context))
    joycon_side = LaunchConfiguration("joycon_side").perform(context)
    joycon_world_frame = LaunchConfiguration("joycon_world_frame").perform(context)
    joycon_publish_rate = float(
        LaunchConfiguration("joycon_publish_rate").perform(context)
    )

    urdf_path = PathJoinSubstitution(
        [
            FindPackageShare("sim_bringup"),
            "assets",
            "urdf",
            LaunchConfiguration("urdf_file"),
        ]
    ).perform(context)

    enable_third_person_view = _as_bool(
        LaunchConfiguration("enable_third_person_view").perform(context)
    )
    if enable_third_person_view:
        camera_names = "front_camera,side_camera,third_person_view"
        camera_frame_ids = (
            "front_color_optical_frame,side_color_optical_frame,"
            "third_person_view_color_optical_frame"
        )
    else:
        camera_names = "front_camera,side_camera"
        camera_frame_ids = "front_color_optical_frame,side_color_optical_frame"

    robot_description = xacro.process_file(
        urdf_path,
        mappings={
            "ros2_control": "true",
            "arm_id": LaunchConfiguration("arm_id").perform(context),
            "arm_prefix": LaunchConfiguration("arm_prefix").perform(context),
            "robot_ip": "",
            "hand": str(load_gripper).lower(),
            "use_fake_hardware": use_fake_hardware,
            "fake_sensor_commands": fake_sensor_commands,
            "mujoco_model": mujoco_model,
            "camera_names": camera_names,
            "camera_frame_ids": camera_frame_ids,
        },
    ).toprettyxml(indent="  ")

    joint_topic = (
        "joint_states" if not namespace else f"/{namespace.strip('/')}/joint_states"
    )

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=namespace,
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            namespace=namespace,
            parameters=[
                controllers_yaml,
                mujoco_overrides,
                {"robot_description": robot_description},
            ],
            remappings=[("joint_states", joint_topic)],
            output="screen",
            on_exit=Shutdown(),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=["joint_state_broadcaster", "--controller-manager-timeout", "30"],
            parameters=[controllers_yaml, mujoco_overrides],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=["pose_broadcaster", "--controller-manager-timeout", "30"],
            parameters=[controllers_yaml, mujoco_overrides],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=[controller_name, "--controller-manager-timeout", "30"],
            parameters=[controllers_yaml, mujoco_overrides],
            output="screen",
        ),
    ]

    if load_gripper:
        ns_prefix = _format_namespace(namespace)
        fake_gripper_params = {
            "arm_id": LaunchConfiguration("arm_id").perform(context),
        }
        arm_prefix = LaunchConfiguration("arm_prefix").perform(context)
        if arm_prefix:
            joint_prefix = f"{arm_prefix}_"
        else:
            joint_prefix = ""
        fake_gripper_params["joint_names"] = [
            f"{joint_prefix}{fake_gripper_params['arm_id']}_finger_joint1",
            f"{joint_prefix}{fake_gripper_params['arm_id']}_finger_joint2",
        ]
        gripper_namespace = (
            f"{namespace.strip('/')}/franka_gripper" if namespace else "franka_gripper"
        )
        nodes.append(
            Node(
                package="sim_bringup",
                executable="fake_franka_gripper",
                name="franka_gripper",
                namespace=gripper_namespace,
                output="screen",
                parameters=[
                    fake_gripper_params,
                    {
                        "mujoco_target_topic": (
                            f"{ns_prefix}/mujoco_gripper/target_width"
                            if ns_prefix
                            else "/mujoco_gripper/target_width"
                        ),
                        "command_topic": (
                            f"{ns_prefix}/gripper/gripper_position_controller/commands"
                            if ns_prefix
                            else "/gripper/gripper_position_controller/commands"
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

    if launch_joycon:
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
                name=f"joycon_single_teleop_node_{joycon_side}",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "joycon_side": joycon_side,
                        "publish_rate": joycon_publish_rate,
                        "world_frame": joycon_world_frame,
                        "robot_pose_topic": robot_pose_topic,
                        "target_pose_topic": target_pose_topic,
                        "gripper_state_topic": gripper_state_topic,
                        "gripper_command_topic": gripper_command_topic,
                    }
                ],
            )
        )

    return nodes


def generate_randomization_env(context):
    config_path = LaunchConfiguration("randomization_config").perform(context)
    cfg = load_yaml(config_path) or {}
    rand_cfg = cfg.get("randomization", {}) if isinstance(cfg, dict) else {}

    def _get_value(name: str):
        override = LaunchConfiguration(name).perform(context)
        if override not in ("", None):
            return override
        return str(rand_cfg.get(name, ""))

    randomize_scene = _as_bool(LaunchConfiguration("randomize_scene").perform(context))
    actions = []
    if randomize_scene:
        actions.append(SetEnvironmentVariable("MUJOCO_RANDOMIZE_SCENE", "1"))
        seed = _get_value("random_seed")
        if seed:
            actions.append(SetEnvironmentVariable("MUJOCO_RANDOM_SEED", seed))
        actions.extend(
            [
                SetEnvironmentVariable("MUJOCO_CUBE_X_MIN", _get_value("cube_x_min")),
                SetEnvironmentVariable("MUJOCO_CUBE_X_MAX", _get_value("cube_x_max")),
                SetEnvironmentVariable("MUJOCO_CUBE_Y_MIN", _get_value("cube_y_min")),
                SetEnvironmentVariable("MUJOCO_CUBE_Y_MAX", _get_value("cube_y_max")),
                SetEnvironmentVariable(
                    "MUJOCO_CUBE_MASS_MIN", _get_value("cube_mass_min")
                ),
                SetEnvironmentVariable(
                    "MUJOCO_CUBE_MASS_MAX", _get_value("cube_mass_max")
                ),
                SetEnvironmentVariable(
                    "MUJOCO_CUBE_SPAWN_CLEARANCE", _get_value("cube_spawn_clearance")
                ),
                SetEnvironmentVariable(
                    "MUJOCO_BASKET_X_MIN", _get_value("basket_x_min")
                ),
                SetEnvironmentVariable(
                    "MUJOCO_BASKET_X_MAX", _get_value("basket_x_max")
                ),
                SetEnvironmentVariable(
                    "MUJOCO_BASKET_Y_MIN", _get_value("basket_y_min")
                ),
                SetEnvironmentVariable(
                    "MUJOCO_BASKET_Y_MAX", _get_value("basket_y_max")
                ),
            ]
        )
    else:
        actions.append(SetEnvironmentVariable("MUJOCO_RANDOMIZE_SCENE", "0"))
        actions.extend(
            [
                UnsetEnvironmentVariable("MUJOCO_RANDOM_SEED"),
                UnsetEnvironmentVariable("MUJOCO_CUBE_X_MIN"),
                UnsetEnvironmentVariable("MUJOCO_CUBE_X_MAX"),
                UnsetEnvironmentVariable("MUJOCO_CUBE_Y_MIN"),
                UnsetEnvironmentVariable("MUJOCO_CUBE_Y_MAX"),
                UnsetEnvironmentVariable("MUJOCO_CUBE_MASS_MIN"),
                UnsetEnvironmentVariable("MUJOCO_CUBE_MASS_MAX"),
                UnsetEnvironmentVariable("MUJOCO_CUBE_SPAWN_CLEARANCE"),
                UnsetEnvironmentVariable("MUJOCO_BASKET_X_MIN"),
                UnsetEnvironmentVariable("MUJOCO_BASKET_X_MAX"),
                UnsetEnvironmentVariable("MUJOCO_BASKET_Y_MIN"),
                UnsetEnvironmentVariable("MUJOCO_BASKET_Y_MAX"),
            ]
        )
    return actions


def generate_launch_description():
    default_model = PathJoinSubstitution(
        [FindPackageShare("sim_bringup"), "assets", "fr3_table_scene.xml"]
    )
    randomization_config = PathJoinSubstitution(
        [FindPackageShare("sim_bringup"), "config", "randomization.yaml"]
    )

    launch_args = [
        DeclareLaunchArgument(
            "arm_id", default_value="fr3", description="Arm identifier"
        ),
        DeclareLaunchArgument(
            "arm_prefix", default_value="", description="Prefix for arm topics"
        ),
        DeclareLaunchArgument(
            "namespace", default_value="", description="Namespace for the robot"
        ),
        DeclareLaunchArgument(
            "urdf_file",
            default_value="fr3_mujoco.urdf.xacro",
            description="URDF xacro within sim_bringup/assets/urdf for MuJoCo hardware plugin",
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
            description="Controller manager configuration",
        ),
        DeclareLaunchArgument(
            "controller_name",
            default_value="cartesian_impedance_controller",
            description="Primary controller to spawn",
        ),
        DeclareLaunchArgument(
            "load_gripper",
            default_value="false",
            description="Load Franka gripper nodes",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Use mock_components/GenericSystem instead of real hardware",
        ),
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="Enable fake sensor commands when using fake hardware",
        ),
        DeclareLaunchArgument(
            "mujoco_model",
            default_value=default_model,
            description=(
                "MuJoCo XML scene file to load (short names resolve to "
                "sim_bringup/assets/<name>.xml)"
            ),
        ),
        DeclareLaunchArgument(
            "viewer_rate", default_value="200.0", description="Viewer update rate (Hz)"
        ),
        DeclareLaunchArgument(
            "gripper_config",
            default_value=PathJoinSubstitution(
                [FindPackageShare("system_bringup"), "config", "gripper.yaml"]
            ),
            description="Gripper bridge configuration file",
        ),
        DeclareLaunchArgument(
            "launch_joycon",
            default_value="false",
            description="Launch Joy-Con teleoperation node",
        ),
        DeclareLaunchArgument(
            "joycon_side",
            default_value="right",
            description="Which Joy-Con (left/right) to use",
        ),
        DeclareLaunchArgument(
            "joycon_world_frame",
            default_value="fr3_link0",
            description="Base/world frame used by Joy-Con teleop",
        ),
        DeclareLaunchArgument(
            "joycon_publish_rate",
            default_value="50.0",
            description="Joy-Con teleop publish rate (Hz)",
        ),
        DeclareLaunchArgument(
            "randomization_config",
            default_value=randomization_config,
            description="Randomization config YAML for scene objects",
        ),
        DeclareLaunchArgument(
            "randomize_scene",
            default_value="false",
            description="Randomize cube/basket pose and mass on startup",
        ),
        DeclareLaunchArgument(
            "enable_third_person_view",
            default_value="true",
            description="Enable third-person camera publishing",
        ),
        DeclareLaunchArgument(
            "random_seed",
            default_value="",
            description="Seed for scene randomization (blank for random)",
        ),
        DeclareLaunchArgument("cube_x_min", default_value="", description="Cube X min"),
        DeclareLaunchArgument("cube_x_max", default_value="", description="Cube X max"),
        DeclareLaunchArgument("cube_y_min", default_value="", description="Cube Y min"),
        DeclareLaunchArgument("cube_y_max", default_value="", description="Cube Y max"),
        DeclareLaunchArgument(
            "cube_mass_min", default_value="", description="Cube mass min (kg)"
        ),
        DeclareLaunchArgument(
            "cube_mass_max", default_value="", description="Cube mass max (kg)"
        ),
        DeclareLaunchArgument(
            "cube_spawn_clearance",
            default_value="",
            description="Cube Z clearance above table surface (m)",
        ),
        DeclareLaunchArgument(
            "basket_x_min", default_value="", description="Basket X min"
        ),
        DeclareLaunchArgument(
            "basket_x_max", default_value="", description="Basket X max"
        ),
        DeclareLaunchArgument(
            "basket_y_min", default_value="", description="Basket Y min"
        ),
        DeclareLaunchArgument(
            "basket_y_max", default_value="", description="Basket Y max"
        ),
    ]

    return LaunchDescription(
        launch_args
        + [
            SetEnvironmentVariable("__GLX_VENDOR_LIBRARY_NAME", "nvidia"),
            SetEnvironmentVariable("__NV_PRIME_RENDER_OFFLOAD", "1"),
            SetEnvironmentVariable("MUJOCO_GL", "glx"),
            OpaqueFunction(function=generate_randomization_env),
            OpaqueFunction(function=generate_robot_nodes),
        ]
    )
