#!/usr/bin/env bash
set -euo pipefail

model="scene_hammer_basket"
episodes="50"
duration="60.0"
benchmark_name="BenchmarkMassEpisode50"
randomize_scene="true"
enable_third_person_view="false"
benchmark_root="$(pwd)/logs/benchmark"
benchmark_run="$(date +%Y%m%d_%H%M%S)"
resume="false"
# resume_root="<repo-root>/logs/benchmark/BenchmarkMassEpisode50/YYYYMMDD_HHMMSS"
parallel_jobs="8"
ros_domain_base="10"
stagger_start_s="5"

checkpoints=(
  "logs/<experiment>/<run>/checkpoints/latest.ckpt"
)
checkpoint_names=(
  "policy"
)

controllers=(
  "cartesian_impedance_controller"
  "cerebellum_predictive_controller"
)
controller_names=(
  "cart"
  "cereb"
)

bin_list=(
  "0.10-0.30"
  "0.30-0.50"
  "0.50-1.00"
  "1.00-2.00"
  "2.00-4.00"
  "4.00-6.00"
  "6.00-8.00"
  "8.00-10.00"
)

total_jobs=$(( ${#checkpoints[@]} * ${#controllers[@]} * ${#bin_list[@]} ))
job_idx=0

if [[ "${resume}" == "true" ]]; then
  if [[ -z "${resume_root}" ]]; then
    echo "resume_root is required when resume=true"
    exit 1
  fi
  if [[ ! -d "${resume_root}" ]]; then
    echo "resume_root not found: ${resume_root}"
    exit 1
  fi
  benchmark_root="$(dirname "$(dirname "${resume_root}")")"
  benchmark_name="$(basename "$(dirname "${resume_root}")")"
  benchmark_run="$(basename "${resume_root}")"
fi

for i in "${!checkpoints[@]}"; do
  checkpoint="${checkpoints[$i]}"
  checkpoint_name="${checkpoint_names[$i]}"
  for j in "${!controllers[@]}"; do
    controller="${controllers[$j]}"
    controller_name="${controller_names[$j]}"
    if [[ "${checkpoint_name}" == "mix" && "${controller_name}" == "cereb" ]]; then
      continue
    fi
    for bin in "${bin_list[@]}"; do
      lo="${bin%-*}"
      hi="${bin#*-}"
      combo_dir="${checkpoint_name}_${controller_name}/${lo}-${hi}"
      if [[ "${resume}" == "true" ]]; then
        csv_path="${benchmark_root}/${benchmark_name}/${benchmark_run}/${combo_dir}/benchmark.csv"
        if [[ -f "${csv_path}" ]]; then
          completed=$(awk 'END{print NR-1}' "${csv_path}")
          if (( completed >= episodes )); then
            continue
          fi
          remaining=$(( episodes - completed ))
        else
          remaining="${episodes}"
        fi
      else
        remaining="${episodes}"
      fi
      domain_id=$(( ros_domain_base + job_idx ))
      job_idx=$(( job_idx + 1 ))
      (
        export ROS_DOMAIN_ID="${domain_id}"
        sleep $(( (job_idx - 1) % parallel_jobs * stagger_start_s ))
        ros2 launch sim_bringup franka_mujoco_deployment.launch.py \
          checkpoint_path:="${checkpoint}" \
          mujoco_model:="${model}" \
          randomize_scene:="${randomize_scene}" \
          cube_mass_min:="${lo}" \
          cube_mass_max:="${hi}" \
          enable_third_person_view:="${enable_third_person_view}" \
          episodes:="${remaining}" \
          episode_duration:="${duration}" \
          controller_name:="${controller}" \
          benchmark_name:="${benchmark_name}" \
          benchmark_root:="${benchmark_root}" \
          benchmark_subdir:="${combo_dir}" \
          benchmark_run:="${benchmark_run}"
      ) &

      if (( $(jobs -pr | wc -l) >= parallel_jobs )); then
        wait -n
      fi
    done
    wait
  done
done

wait
