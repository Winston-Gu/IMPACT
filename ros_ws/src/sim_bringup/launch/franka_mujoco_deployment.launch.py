#!/usr/bin/env python3
"""MuJoCo bringup with diffusion policy deployment (manual rollout)."""

import os
from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _declare_arg(name: str, default: str, description: str):
    return DeclareLaunchArgument(name, default_value=default, description=description)


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "impact").is_dir():
            return parent
    return Path(__file__).resolve().parents[0]


def _resolve_benchmark_dir(
    checkpoint_path: str,
    benchmark_name: str,
    benchmark_root: str,
    benchmark_subdir: str,
    benchmark_run: str,
) -> tuple[Path, str, str]:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    base_dir = ckpt_path.parent
    if base_dir.name == "checkpoints":
        base_dir = base_dir.parent
    run_name = benchmark_run.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    name = benchmark_name or run_name
    root = Path(benchmark_root).expanduser()
    if not root.is_absolute():
        root = base_dir / root
    subdir = benchmark_subdir.strip()
    target = root / name / run_name
    if subdir:
        target = target / subdir
    return target, name, run_name


def _launch_policy_node(context):
    repo_root_arg = LaunchConfiguration("repo_root").perform(context)
    repo_root = Path(repo_root_arg).expanduser() if repo_root_arg else _find_repo_root()
    checkpoint_path = LaunchConfiguration("checkpoint_path").perform(context)
    benchmark_name = LaunchConfiguration("benchmark_name").perform(context)
    benchmark_root = LaunchConfiguration("benchmark_root").perform(context)
    benchmark_subdir = LaunchConfiguration("benchmark_subdir").perform(context)
    benchmark_run = LaunchConfiguration("benchmark_run").perform(context)
    benchmark_dir, bench_name, run_name = _resolve_benchmark_dir(
        checkpoint_path,
        benchmark_name,
        benchmark_root,
        benchmark_subdir,
        benchmark_run,
    )
    benchmark_csv = benchmark_dir / "benchmark.csv"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                os.fspath(repo_root.resolve()),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
    }
    python_executable = LaunchConfiguration("python_executable").perform(context)
    if not python_executable:
        virtual_env = os.environ.get("VIRTUAL_ENV", "")
        python_executable = (
            os.fspath(Path(virtual_env) / "bin" / "python") if virtual_env else "python3"
        )
    set_benchmark_dir = SetLaunchConfiguration(
        "benchmark_dir", os.fspath(benchmark_dir)
    )
    policy_node = ExecuteProcess(
        cmd=[
            python_executable,
            "-m",
            "sim_bringup.policy_deployment",
            "--config",
            LaunchConfiguration("policy_config"),
            "--checkpoint",
            LaunchConfiguration("checkpoint_path"),
            "--ros-args",
            "-p",
            f"benchmark_csv_path:={benchmark_csv}",
            "-p",
            f"episodes:={LaunchConfiguration('episodes').perform(context)}",
            "-p",
            f"episode_duration:={LaunchConfiguration('episode_duration').perform(context)}",
            "-p",
            f"enable_episode_timer:={LaunchConfiguration('enable_episode_timer').perform(context)}",
        ],
        env=env,
        output="screen",
    )
    shutdown_on_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=policy_node,
            on_exit=[Shutdown()],
        )
    )
    return [set_benchmark_dir, policy_node, shutdown_on_exit]


def generate_launch_description():
    randomization_config = PathJoinSubstitution(
        [FindPackageShare("sim_bringup"), "config", "randomization.yaml"]
    )
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
            "randomization_config": LaunchConfiguration("randomization_config"),
            "randomize_scene": LaunchConfiguration("randomize_scene"),
            "random_seed": LaunchConfiguration("random_seed"),
            "enable_third_person_view": LaunchConfiguration("enable_third_person_view"),
            "cube_x_min": LaunchConfiguration("cube_x_min"),
            "cube_x_max": LaunchConfiguration("cube_x_max"),
            "cube_y_min": LaunchConfiguration("cube_y_min"),
            "cube_y_max": LaunchConfiguration("cube_y_max"),
            "cube_mass_min": LaunchConfiguration("cube_mass_min"),
            "cube_mass_max": LaunchConfiguration("cube_mass_max"),
            "cube_spawn_clearance": LaunchConfiguration("cube_spawn_clearance"),
            "basket_x_min": LaunchConfiguration("basket_x_min"),
            "basket_x_max": LaunchConfiguration("basket_x_max"),
            "basket_y_min": LaunchConfiguration("basket_y_min"),
            "basket_y_max": LaunchConfiguration("basket_y_max"),
        }.items(),
    )

    return LaunchDescription(
        [
            _declare_arg(
                "checkpoint_path",
                "",
                "Diffusion policy checkpoint path (required).",
            ),
            _declare_arg(
                "policy_config",
                "mujoco_side_frontcamera",
                "IMPACT deploy config name or path.",
            ),
            _declare_arg(
                "benchmark_name",
                "",
                "Benchmark run name (defaults to timestamp).",
            ),
            _declare_arg(
                "benchmark_root",
                "logs/benchmark",
                "Benchmark output root (relative to checkpoint dir if not absolute).",
            ),
            _declare_arg(
                "benchmark_subdir",
                "",
                "Benchmark subfolder under benchmark name/run (optional).",
            ),
            _declare_arg(
                "benchmark_run",
                "",
                "Benchmark run timestamp (defaults to now if blank).",
            ),
            _declare_arg(
                "repo_root",
                "",
                "Repo root to add to PYTHONPATH (auto-detected if blank).",
            ),
            _declare_arg(
                "python_executable",
                "",
                "Python executable for policy deployment. Defaults to $VIRTUAL_ENV/bin/python when a venv is active, otherwise python3.",
            ),
            _declare_arg("load_gripper", "true", "Load Franka gripper nodes"),
            _declare_arg(
                "mujoco_model",
                "",
                "MuJoCo XML scene file to load (short names resolve to sim_bringup/assets/<name>.xml)",
            ),
            _declare_arg(
                "controller_name",
                "cartesian_impedance_controller",
                "Primary controller to spawn",
            ),
            _declare_arg(
                "enable_third_person_view",
                "false",
                "Publish third-person camera images",
            ),
            _declare_arg(
                "log_name",
                "",
                "Video log name (logs/{log_name}); blank uses timestamp",
            ),
            _declare_arg(
                "episodes",
                "1",
                "Number of rollout episodes to run",
            ),
            _declare_arg(
                "episode_duration",
                "15.0",
                "Duration (seconds) for each rollout episode",
            ),
            _declare_arg(
                "enable_episode_timer",
                "true",
                "Publish episode start/end markers for rollout",
            ),
            _declare_arg(
                "video_output_root",
                "logs",
                "Output root for third-person recordings",
            ),
            _declare_arg("video_fps", "30.0", "Third-person recorder FPS"),
            _declare_arg("video_fourcc", "avc1", "Third-person video codec"),
            _declare_arg(
                "randomize_scene",
                "true",
                "Randomize cube/basket pose and mass on reset",
            ),
            _declare_arg(
                "randomization_config",
                randomization_config,
                "Randomization config YAML for scene objects",
            ),
            _declare_arg(
                "random_seed",
                "",
                "Seed for scene randomization (blank for random)",
            ),
            _declare_arg("cube_x_min", "", "Cube X min"),
            _declare_arg("cube_x_max", "", "Cube X max"),
            _declare_arg("cube_y_min", "", "Cube Y min"),
            _declare_arg("cube_y_max", "", "Cube Y max"),
            _declare_arg("cube_mass_min", "", "Cube mass min (kg)"),
            _declare_arg("cube_mass_max", "", "Cube mass max (kg)"),
            _declare_arg(
                "cube_spawn_clearance",
                "",
                "Cube Z clearance above table surface (m)",
            ),
            _declare_arg("basket_x_min", "", "Basket X min"),
            _declare_arg("basket_x_max", "", "Basket X max"),
            _declare_arg("basket_y_min", "", "Basket Y min"),
            _declare_arg("basket_y_max", "", "Basket Y max"),
            base_launch,
            OpaqueFunction(function=_launch_policy_node),
            Node(
                package="sim_bringup",
                executable="video_recorder",
                name="video_recorder",
                output="screen",
                condition=IfCondition(LaunchConfiguration("enable_third_person_view")),
                parameters=[
                    {
                        "log_name": "videos",
                        "image_topic": "/camera/third_person_view/color/image_rect_raw",
                        "episode_start_topic": "/episode/start",
                        "episode_end_topic": "/episode/end",
                        "output_root": LaunchConfiguration("benchmark_dir"),
                        "fps": LaunchConfiguration("video_fps"),
                        "fourcc": LaunchConfiguration("video_fourcc"),
                    }
                ],
            ),
        ]
    )
