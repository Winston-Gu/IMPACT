#!/bin/bash
#SBATCH --job-name=impact
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

export HYDRA_FULL_ERROR=1
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-impact}"
cd "${PROJECT_ROOT}"

mamba run -n "${CONDA_ENV_NAME}" python -m impact.train \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  logging.wandb.name=1215ClusterSanity
