from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _declare_arg(name: str, default: str, description: str):
    return DeclareLaunchArgument(name, default_value=default, description=description)


def generate_launch_description():
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("sim_bringup"),
                    "launch",
                    "franka_mujoco.launch.py",
                ]
            )
        ),
        launch_arguments={
            "load_gripper": LaunchConfiguration("load_gripper"),
            "mujoco_model": LaunchConfiguration("mujoco_model"),
            "controller_name": LaunchConfiguration("controller_name"),
            "randomize_scene": LaunchConfiguration("randomize_scene"),
            "enable_third_person_view": LaunchConfiguration("enable_third_person_view"),
            "cube_x_min": LaunchConfiguration("cube_x_min"),
            "cube_x_max": LaunchConfiguration("cube_x_max"),
            "cube_y_min": LaunchConfiguration("cube_y_min"),
            "cube_y_max": LaunchConfiguration("cube_y_max"),
            "cube_mass_min": LaunchConfiguration("cube_mass_min"),
            "cube_mass_max": LaunchConfiguration("cube_mass_max"),
            "basket_x_min": LaunchConfiguration("basket_x_min"),
            "basket_x_max": LaunchConfiguration("basket_x_max"),
            "basket_y_min": LaunchConfiguration("basket_y_min"),
            "basket_y_max": LaunchConfiguration("basket_y_max"),
        }.items(),
    )

    return LaunchDescription(
        [
            _declare_arg("episodes", "5", "Number of pick-and-place episodes to run"),
            _declare_arg(
                "log_name",
                "",
                "Video log name (logs/{log_name}); blank uses timestamp",
            ),
            _declare_arg("dataset_name", "Sim", "Dataset name for sim recordings"),
            _declare_arg(
                "output_root",
                "datasets/raw",
                "Dataset output root. Relative paths resolve from the launch working directory.",
            ),
            _declare_arg(
                "record_dataset",
                "true",
                "Record dataset per episode in simulation",
            ),
            _declare_arg(
                "apply_z_offset",
                "true",
                "Apply mass-based Z offset during lift/transit",
            ),
            _declare_arg(
                "dataset_config",
                PathJoinSubstitution(
                    [
                        FindPackageShare("sim_bringup"),
                        "config",
                        "sim_dataset_recorder.yaml",
                    ]
                ),
                "Dataset recorder config YAML for simulation",
            ),
            _declare_arg(
                "enable_third_person_view",
                "true",
                "Enable third-person camera publishing and recording",
            ),
            _declare_arg("load_gripper", "true", "Load Franka gripper nodes"),
            _declare_arg("mujoco_model", "", "MuJoCo XML scene file to load"),
            _declare_arg(
                "controller_name",
                "cartesian_impedance_controller",
                "Primary controller to spawn",
            ),
            _declare_arg(
                "randomize_scene",
                "true",
                "Randomize cube/basket pose and mass on reset",
            ),
            _declare_arg("cube_x_min", "", "Cube X min"),
            _declare_arg("cube_x_max", "", "Cube X max"),
            _declare_arg("cube_y_min", "", "Cube Y min"),
            _declare_arg("cube_y_max", "", "Cube Y max"),
            _declare_arg("cube_mass_min", "", "Cube mass min (kg)"),
            _declare_arg("cube_mass_max", "", "Cube mass max (kg)"),
            _declare_arg("basket_x_min", "", "Basket X min"),
            _declare_arg("basket_x_max", "", "Basket X max"),
            _declare_arg("basket_y_min", "", "Basket Y min"),
            _declare_arg("basket_y_max", "", "Basket Y max"),
            base_launch,
            Node(
                package="sim_bringup",
                executable="pick_place_demo",
                name="pick_place_demo",
                output="screen",
                parameters=[
                    {
                        "episodes": LaunchConfiguration("episodes"),
                        "record_dataset": LaunchConfiguration("record_dataset"),
                        "apply_z_offset": LaunchConfiguration("apply_z_offset"),
                    }
                ],
            ),
            Node(
                package="sim_bringup",
                executable="dataset_recorder",
                name="dataset_recorder",
                output="screen",
                condition=IfCondition(LaunchConfiguration("record_dataset")),
                parameters=[
                    LaunchConfiguration("dataset_config"),
                    {
                        "dataset_name": LaunchConfiguration("dataset_name"),
                        "output_root": LaunchConfiguration("output_root"),
                    },
                ],
            ),
            Node(
                package="sim_bringup",
                executable="video_recorder",
                name="video_recorder",
                output="screen",
                condition=IfCondition(LaunchConfiguration("enable_third_person_view")),
                parameters=[
                    {
                        "log_name": LaunchConfiguration("log_name"),
                        "image_topic": "/camera/third_person_view/color/image_rect_raw",
                        "episode_start_topic": "/episode/start",
                        "episode_end_topic": "/episode/end",
                        "output_root": "logs",
                        "fps": 30.0,
                        "fourcc": "avc1",
                    }
                ],
            ),
        ]
    )
