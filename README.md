# IMPACT: Learning Internal-Model Predictive Control for Forceful Robotic Manipulation

<p align="center">
  <img src="assets/IMPACT_Teaser.gif" width="600">
</p>

This repository contains the training, simulation, teleoperation, and deployment code for IMPACT. It includes:

- `impact/`: diffusion-policy training, datasets, models, normalizers, and checkpoint utilities.
- `config/`: post-processing and deployment YAML configs.
- `ros_ws/src/`: ROS 2 packages for Franka bringup, MuJoCo simulation, predictive controllers, Joy-Con teleoperation, RealSense cameras, and policy deployment.
- `scripts/`: dataset inspection, post-processing, visualization, benchmark, and cluster helper scripts.

Datasets, checkpoints, ROS build products, logs, and local vendor checkouts are intentionally not included.

## Requirements

The full robot/simulation stack is intended for Ubuntu 22.04 with ROS 2 Humble. Training-only workflows can run anywhere the Python dependencies and CUDA/PyTorch stack are available.

Install the main system dependencies:

```bash
sudo apt update
sudo apt install -y \
  curl \
  git \
  python3-pip \
  python3-vcstool \
  ros-humble-desktop \
  ros-dev-tools \
  ros-humble-xacro \
  ros-humble-franka-msgs \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-generate-parameter-library \
  ros-humble-pinocchio \
  libglfw3-dev \
  libopencv-dev \
  python3-pynput \
  python3-numpy \
  python3-opencv
```

Install `uv` if it is not already available, then create the Python environment from the lockfile:

```bash
uv sync
source .venv/bin/activate
```

## MuJoCo

Install MuJoCo and expose its library path:

```bash
mkdir -p /tmp/mujoco-install "$HOME/mujoco"
curl -fL \
  https://github.com/google-deepmind/mujoco/releases/download/3.8.0/mujoco-3.8.0-linux-x86_64.tar.gz \
  -o /tmp/mujoco-install/mujoco-3.8.0-linux-x86_64.tar.gz
tar -xzf /tmp/mujoco-install/mujoco-3.8.0-linux-x86_64.tar.gz -C "$HOME/mujoco"

export MUJOCO_DIR="$HOME/mujoco/mujoco-3.8.0"
export LD_LIBRARY_PATH="$MUJOCO_DIR/lib:$LD_LIBRARY_PATH"
```

Add the two `export` lines to your shell profile if you use MuJoCo regularly.

## ROS Workspace

Import external ROS dependencies and build the workspace:

```bash
source /opt/ros/humble/setup.zsh

cd ros_ws
mkdir -p vendor
vcs import . < vendor.repos

rosdep update
rosdep install --from-paths src vendor --ignore-src -r -y
colcon build --symlink-install

source install/setup.zsh
cd ..
```

If your lab already provides compatible Franka ROS packages system-wide, you can ignore or remove the matching directories under `ros_ws/vendor/`. Otherwise, keep the imported vendor packages and build them with the workspace.

## Simulation Data Collection

From the repository root:

```bash
source /opt/ros/humble/setup.zsh
source ros_ws/install/setup.zsh
source .venv/bin/activate

ros2 launch sim_bringup franka_mujoco_MP.launch.py \
  mujoco_model:=scene_hammer_basket \
  episodes:=2 \
  record_dataset:=true \
  dataset_name:=DemoSim \
  output_root:=datasets/raw \
  controller_name:=cartesian_impedance_controller \
  cube_mass_min:=0.2 \
  cube_mass_max:=1.0
```

Raw episodes are written to `datasets/raw/<DATASET_NAME>/`.

## Post-process Datasets

Align raw episodes and export a training pickle:

```bash
uv run python scripts/postprocess/align_timesteps.py \
  --dataset-name DemoSim \
  --config MJ_side_front

uv run python scripts/postprocess/export_trajectories.py \
  --dataset-name DemoSim \
  --config MJ_side_front \
  --stream
```

The exported dataset is written to `datasets/exported/DemoSim/DemoSim_<N>.pkl`.

You can inspect raw or exported datasets with:

```bash
uv run python scripts/check_datasets.py datasets/exported/DemoSim/DemoSim_2.pkl
uv run python scripts/dataset_viewer/viewer_server.py --dataset datasets/exported/DemoSim/DemoSim_2.pkl
```

## Train a Policy

Override `dataset.dataset_path`; the default config is only a placeholder.

```bash
HYDRA_FULL_ERROR=1 uv run python -m impact.train \
  experiment_name=DemoSim \
  training.device=cuda:0 \
  training.num_epochs=1000 \
  dataset.dataset_path=datasets/exported/DemoSim/DemoSim_2.pkl
```

Outputs are written under `logs/<experiment_name>/<timestamp>/`. Checkpoints are under `checkpoints/` inside that run directory.

Weights & Biases logging is enabled by default in `impact/config/logging/logging.yaml`. Use `logging.use_wandb=false` if you do not want online logging:

```bash
uv run python -m impact.train \
  logging.use_wandb=false \
  dataset.dataset_path=datasets/exported/DemoSim/DemoSim_2.pkl
```

For Slurm, set `PROJECT_ROOT` when submitting if the scheduler runs from a different working directory. The helper defaults to a conda/mamba environment named `impact`; override it with `CONDA_ENV_NAME` if your cluster uses another name:

```bash
sbatch --export=PROJECT_ROOT="$PWD",CONDA_ENV_NAME=impact scripts/train.slurm
```

## Simulation Deployment and Benchmarking

Run a trained checkpoint in MuJoCo:

```bash
source /opt/ros/humble/setup.zsh
source ros_ws/install/setup.zsh
source .venv/bin/activate

ros2 launch sim_bringup franka_mujoco_deployment.launch.py \
  checkpoint_path:=logs/DemoSim/<run>/checkpoints/latest.ckpt \
  mujoco_model:=scene_hammer_basket \
  randomize_scene:=true \
  cube_mass_min:=0.2 \
  cube_mass_max:=1.0 \
  enable_third_person_view:=true \
  episodes:=3 \
  episode_duration:=50.0 \
  benchmark_name:=DemoBenchmark
```

For mass-sweep benchmarks, edit `scripts/run_mass_sweep.sh` first. In particular, set the `checkpoints`, `checkpoint_names`, `controllers`, mass bins, and optional `resume_root`.

## Real Robot Bringup

Before running on hardware, update:

- `ros_ws/src/system_bringup/config/franka.config.yaml`: `robot_ip`, `namespace`, and robot model/URDF fields.
- `ros_ws/src/system_bringup/config/realsense_cameras.yaml`: RealSense `serial_no` values. Use `rs-enumerate-devices -s` to find them.
- `ros_ws/src/system_bringup/config/teleoperation.yaml`: namespace mapping, dataset output root, and recorded topics.
- `config/deploy/side_frontcamera.yaml`: CUDA device and ROS topic names for real-world policy deployment.

Bring up a Franka arm:

```bash
source /opt/ros/humble/setup.zsh
source ros_ws/install/setup.zsh

ros2 launch system_bringup franka.launch.py \
  robot_ip:=<ROBOT_IP> \
  arm_id:=fr3 \
  use_fake_hardware:=false \
  load_gripper:=true
```

Launch teleoperation and optional dataset recording:

```bash
./scripts/joycon_connect.sh

ros2 launch system_bringup franka_teleopration.launch.py \
  record_dataset:=true \
  dataset_name:=DemoReal \
  launch_realsense:=true
```

Dataset recording controls:

- `SPACE`: start or stop recording.
- `DELETE` or `BACKSPACE`: delete the last recorded episode.

Run real-world policy deployment after setting the config and checkpoint:

```bash
ros2 launch system_bringup franka_deployment.launch.py \
  checkpoint_path:=logs/DemoReal/<run>/checkpoints/latest.ckpt \
  policy_config:=side_frontcamera
```

Use conservative controller gains and validate on fake hardware or simulation before commanding a physical robot.

## Paths and Machine-specific Settings

The repository avoids personal absolute paths. Users still need to set these project-specific values:

- Dataset path for training: pass `dataset.dataset_path=...` or edit `impact/config/dataset/dataset.yaml`.
- Checkpoint path for deployment: pass `checkpoint_path:=...`.
- Dataset output root: pass `output_root:=...` or edit the recorder configs.
- MuJoCo install path: set `MUJOCO_DIR` and `LD_LIBRARY_PATH`.
- CUDA device: set `training.device=...` and the `device` field in `config/deploy/*.yaml`.
- Franka robot IP and namespace: edit `ros_ws/src/system_bringup/config/franka.config.yaml`.
- RealSense serial numbers: edit `ros_ws/src/system_bringup/config/realsense_cameras.yaml`.
- Benchmark helper paths: edit `scripts/run_mass_sweep.sh` checkpoint arrays and optional `resume_root`.
- Slurm root: set `PROJECT_ROOT` when submitting `scripts/train.slurm` if auto-detection is not correct.

## Development

Install dev tools with `uv sync --group dev`, then run:

```bash
pre-commit install
pre-commit run --all-files
```

Useful sanity checks:

```bash
uv run python -m compileall impact scripts
cd ros_ws && colcon build --symlink-install
```

## License

Main project code is released under Apache-2.0; see `LICENSE`. Package-level exceptions, if any, are noted in the corresponding ROS package manifests and source comments.
