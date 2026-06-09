# AGENTS.md

This file gives coding agents project-specific context for IMPACT. The public user-facing instructions live in `README.md`; keep this file focused on implementation guidance, project boundaries, and release hygiene.

## Project Overview

IMPACT studies internal-model predictive control for forceful robotic manipulation. The code trains an image-conditioned diffusion policy from demonstrations, then deploys the policy with robot controllers that can compensate for forceful or out-of-distribution interaction conditions such as lifting heavier objects.

The current Python package name is `impact`. Do not reintroduce deprecated package or project names in imports, configs, docs, ROS node names, Slurm scripts, or examples.

## Repository Layout

- `impact/`: Python training and inference package.
  - `impact/train.py`: Hydra entrypoint for policy training.
  - `impact/config/`: Hydra configs for dataset, logging, dataloaders, normalizers, workspace, policy, and optimizer.
  - `impact/dataset/`: exported demonstration dataset loaders and replay-buffer utilities.
  - `impact/model/`: diffusion, vision encoder, and tensor utility modules.
  - `impact/policy/`: diffusion image policy implementation.
  - `impact/workspace/`: training loop, checkpointing, logging, validation, and runner orchestration.
  - `impact/deploy/`: ROS-facing policy loading and deployment helpers.
- `config/`: runtime configs outside Hydra, mainly deployment and post-processing YAML files.
- `scripts/`: dataset checks, post-processing, viewers, benchmark helpers, Joy-Con pairing helpers, and Slurm helper.
- `ros_ws/src/`: ROS 2 Humble workspace packages.
  - `system_bringup`: real Franka bringup, RealSense launch, teleoperation, dataset recording, gripper bridge, pose-error broadcasting.
  - `sim_bringup`: MuJoCo data collection, simulation deployment, benchmarking, video recording, fake gripper, scripted pick-place episodes.
  - `predictive_controller`: C++ controllers, including Cartesian impedance and predictive controller variants.
  - `mujoco_sim`: MuJoCo hardware interface and rendering integration.
  - `joycon_teleop`: Joy-Con input library and teleoperation nodes.
  - `cameras`: RealSense camera node.
- `ros_ws/vendor.repos`: external Franka ROS dependencies imported into `ros_ws/vendor/`.
- `assets/`: public README assets only.

## Development Rules

- Prefer existing module boundaries. Training code belongs under `impact/`; robot/simulation runtime code belongs under the relevant ROS package.
- Keep examples and defaults repo-relative where possible. User-specific values must be launch/config parameters or documented README settings.
- Do not add datasets, checkpoints, logs, ROS build products, vendored dependency checkouts, virtual environments, or generated caches to git.
- Do not use private absolute paths. Acceptable user-set paths include documented parameters such as `dataset.dataset_path`, `checkpoint_path`, `output_root`, `MUJOCO_DIR`, and `PROJECT_ROOT`.
- Keep Python package imports rooted at `impact`.
- Keep Hydra `_target_` values synchronized with actual import paths.
- Preserve robot safety. Any change to controllers, deployment loops, action scaling, gripper commands, reset logic, or collision behavior should be treated as high risk and tested in simulation or fake hardware first.
- Be conservative with controller gains, frame names, topic remaps, and action interpretation. Small changes can cause unsafe robot motion.
- Do not change licensing metadata casually. The top-level repo license is Apache-2.0; package-level exceptions are declared in ROS package manifests and source comments.

## Common Workflows

Python environment:

```bash
uv sync
source .venv/bin/activate
```

Training entrypoint:

```bash
HYDRA_FULL_ERROR=1 uv run python -m impact.train \
  experiment_name=DemoSim \
  training.device=cuda:0 \
  dataset.dataset_path=datasets/exported/DemoSim/DemoSim_2.pkl
```

Dataset post-processing:

```bash
uv run python scripts/postprocess/align_timesteps.py --dataset-name DemoSim --config MJ_side_front
uv run python scripts/postprocess/export_trajectories.py --dataset-name DemoSim --config MJ_side_front --stream
```

ROS workspace build on Ubuntu 22.04 with ROS 2 Humble:

```bash
source /opt/ros/humble/setup.zsh
cd ros_ws
mkdir -p vendor
vcs import . < vendor.repos
rosdep install --from-paths src vendor --ignore-src -r -y
colcon build --symlink-install
```

## Verification Checklist

Run the checks that match the files you changed:

```bash
uv lock --check
uv run python -m compileall impact scripts
uv run python -c "import impact; import impact.train; print(impact.__name__)"
git diff --check
```

For ROS changes, additionally run `colcon build --symlink-install` from `ros_ws` in a ROS Humble environment. For controller or deployment changes, validate in simulation before hardware.

Before release or PR handoff, scan for stale project names, private absolute paths, secrets, placeholder comments, and generated artifacts. Keep the scan pattern outside committed docs so the docs do not match their own release check.

```bash
find . -name '.DS_Store' -o -name '__pycache__' -o -name '*.pyc' -o -name '.venv'
```

`TODO`-style comments should not describe release blockers. Generated files found by the `find` command should be removed unless they are intentionally ignored local artifacts.

## User-specific Configuration

These values are expected to vary by user and should remain configurable:

- `dataset.dataset_path` for training.
- `checkpoint_path` for deployment and benchmarking.
- `output_root` for dataset recorders.
- `MUJOCO_DIR` and `LD_LIBRARY_PATH`.
- CUDA device in `training.device` and `config/deploy/*.yaml`.
- Franka robot IP and namespace in `ros_ws/src/system_bringup/config/franka.config.yaml`.
- RealSense serial numbers in `ros_ws/src/system_bringup/config/realsense_cameras.yaml`.
- Slurm `PROJECT_ROOT` and `CONDA_ENV_NAME` for `scripts/train.slurm`.
- Benchmark checkpoint arrays and `resume_root` in `scripts/run_mass_sweep.sh`.
