#!/usr/bin/env python3
"""Export aligned episodes into a single pickle for diffusion-policy training."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from data_transforms import (
    transform_float_array,
    transform_gripper_state,
    transform_image_compressed,
    transform_joint_position,
    transform_joint_state,
    transform_pose,
)
from loguru import logger

TransformFn = Callable[[Any, dict[str, Any]], np.ndarray | None]

TRANSFORM_REGISTRY: dict[str, TransformFn] = {
    "pose": transform_pose,
    "joint_position": transform_joint_position,
    "joint_state": transform_joint_state,
    "gripper_state": transform_gripper_state,
    "float_array": transform_float_array,
    "image_compressed": transform_image_compressed,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aligned episodes into one pickle."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config name or path. If a name is provided, it is resolved under "
            "config/postprocess/<name>.yaml."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name override (otherwise read from config.dataset_name).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output pickle path (otherwise uses trajectory_export.output_path under output_root).",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of episode indices to export (e.g. 0 1 5).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Write a streaming pickle (header + per-episode pickles) to reduce "
            "memory usage during export."
        ),
    )
    return parser.parse_args()


def _forward_fill(
    samples: list[np.ndarray | None], feature_name: str
) -> list[np.ndarray]:
    first_valid = next((s for s in samples if s is not None), None)
    if first_valid is None:
        raise ValueError(f"No valid samples found for feature '{feature_name}'.")

    filled: list[np.ndarray] = []
    last = first_valid
    for sample in samples:
        if sample is not None:
            last = sample
        filled.append(last)
    return filled


def _process_feature(
    name: str,
    key: str,
    samples: list[Any],
    cfg: dict[str, Any],
) -> np.ndarray:
    transform_name = cfg.get("transform")
    if not transform_name:
        raise ValueError(f"Feature '{name}' must define a transform.")
    transform_fn = TRANSFORM_REGISTRY.get(transform_name)
    if transform_fn is None:
        raise ValueError(f"Unknown transform '{transform_name}' for feature '{name}'.")

    processed = [transform_fn(sample, cfg) for sample in samples]
    filled = _forward_fill(processed, name)
    try:
        stacked = np.stack(filled, axis=0)
    except ValueError as exc:
        raise ValueError(f"Failed to stack feature '{name}': {exc}") from exc
    return stacked


def main() -> None:
    args = parse_args()
    config_arg = args.config
    if not config_arg:
        logger.warning("Missing --config. Please provide a config name or path.")
        raise ValueError("Missing required --config.")
    config_path = Path(config_arg).expanduser()
    if not config_path.suffix:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "config/postprocess"
            / f"{config_arg}.yaml"
        )
    config_path = config_path.resolve()

    with config_path.open("r") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {config_path} must be a mapping.")
    config = config.get("export", config)
    dataset_name = args.dataset_name
    if not dataset_name:
        raise ValueError("Please set dataset_name in the config or via --dataset-name.")

    # Load Transforms Map

    transform_entries = config.get("transforms", [])
    transform_map: dict[str, dict[str, Any]] = {}
    for entry in transform_entries:
        if not isinstance(entry, dict):
            raise ValueError(
                "Each transform entry must be a mapping with a 'name' field."
            )
        name = entry.get("name")
        if not name:
            raise ValueError("Transform entries must include a 'name'.")
        if name in transform_map:
            raise ValueError(f"Duplicate transform definition for '{name}'.")
        transform_map[name] = {k: v for k, v in entry.items() if k != "name"}

    input_root = Path("datasets/aligned").expanduser().resolve()
    dataset_dir = input_root / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Aligned dataset directory not found: {dataset_dir}")

    if args.episodes is None:
        episode_paths = sorted(dataset_dir.glob("ep_*.pkl"))
    else:
        explicit = [dataset_dir / f"ep_{idx:05d}.pkl" for idx in args.episodes]
        episode_paths = [p for p in explicit if p.exists()]
    if not episode_paths:
        raise FileNotFoundError(
            f"No episodes found in {dataset_dir} for selection {args.episodes}."
        )

    len_episodes = len(episode_paths)

    output_root = Path("datasets/exported").expanduser().resolve()
    output_path = f"{dataset_name}_{len_episodes}.pkl"
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = output_root / dataset_name / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obs_cfgs: list[Any] = config.get("observations", [])
    act_cfgs: list[Any] = config.get("actions", [])
    if not obs_cfgs or not act_cfgs:
        raise ValueError(
            "Both 'observations' and 'actions' must be defined in the config."
        )

    trajectories = []
    episode_lengths: list[int] = []
    episode_names: list[str] = []
    stream_tmp_path: Path | None = None
    stream_handle = None
    if args.stream:
        stream_tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        stream_handle = stream_tmp_path.open("wb")

    for idx, ep_path in enumerate(sorted(episode_paths)):
        with ep_path.open("rb") as handle:
            episode = pickle.load(handle)

        data_block = episode.get("data", {})

        # Derive episode length from the first available observation key.
        length: int | None = None

        processed_cache: dict[str, np.ndarray] = {}

        def _get_feature(feature_name: str) -> np.ndarray:
            if feature_name in processed_cache:
                return processed_cache[feature_name]
            base_cfg = transform_map.get(feature_name)
            if base_cfg is None:
                raise KeyError(f"Transform '{feature_name}' is undefined.")
            data_key = base_cfg.get("key")
            if not data_key:
                raise ValueError(
                    f"Transform '{feature_name}' missing 'key' for episode {ep_path.name}"
                )
            samples = data_block.get(data_key)
            if samples is None:
                raise KeyError(
                    f"Channel '{data_key}' missing in episode {ep_path.name}"
                )
            comp_cfg = {"name": feature_name, **base_cfg}
            comp_arr = _process_feature(feature_name, data_key, samples, comp_cfg)
            processed_cache[feature_name] = comp_arr
            return comp_arr

        def _maybe_delta(feature_name: str, base: np.ndarray) -> np.ndarray:
            base_cfg = transform_map.get(feature_name) or {}
            delta_with = base_cfg.get("delta_with")
            if not delta_with:
                return base
            ref = _get_feature(delta_with)
            if ref.shape != base.shape:
                raise ValueError(
                    f"Delta ref '{delta_with}' shape {ref.shape} does not match "
                    f"'{feature_name}' shape {base.shape} in {ep_path.name}"
                )
            delta_indices = base_cfg.get("delta_indices")
            if delta_indices is None:
                return base - ref
            delta_indices = [int(idx) for idx in delta_indices]
            out = base.copy()
            out[:, delta_indices] = base[:, delta_indices] - ref[:, delta_indices]
            return out

        observations: dict[str, np.ndarray] = {}
        for obs_cfg in obs_cfgs:
            name = obs_cfg.get("name")
            key_names = obs_cfg.get("keys", [])
            if not name:
                raise ValueError("Observation entry missing 'name'.")
            if not key_names:
                raise ValueError(f"Observation '{name}' must specify at least one key.")

            components: list[np.ndarray] = []
            for key_name in key_names:
                comp_arr = _get_feature(key_name)
                if length is None:
                    length = len(comp_arr)
                if len(comp_arr) != length:
                    raise ValueError(
                        f"Feature '{name}' component '{key_name}' length mismatch in {ep_path.name}: {len(comp_arr)} vs {length}"
                    )
                comp_arr = _maybe_delta(key_name, comp_arr)
                components.append(comp_arr)

            if length is None or length == 0:
                raise ValueError(
                    f"Episode {ep_path.name} has no frames for observation '{name}'."
                )

            if len(components) == 1:
                feature = components[0]
            else:
                if any(arr.ndim != 2 for arr in components):
                    raise ValueError(
                        f"Observation '{name}' requires 2D components to concatenate in {ep_path.name}."
                    )
                feature = np.concatenate(components, axis=1)

            observations[name] = feature

        if length is None:
            raise ValueError(f"Episode {ep_path.name} has no observation frames.")

        actions: dict[str, np.ndarray] = {}
        for act_cfg in act_cfgs:
            name = act_cfg.get("name")
            key_names = act_cfg.get("keys", [])
            if not name:
                raise ValueError("Action entry missing 'name'.")
            if not key_names:
                raise ValueError(f"Action '{name}' must specify at least one key.")

            components: list[np.ndarray] = []
            for key_name in key_names:
                comp_arr = _get_feature(key_name)
                if len(comp_arr) != length:
                    raise ValueError(
                        f"Action '{name}' component '{key_name}' length mismatch in {ep_path.name}: {len(comp_arr)} vs {length}"
                    )
                comp_arr = _maybe_delta(key_name, comp_arr)
                components.append(comp_arr)

            if len(components) == 1:
                feature = components[0]
            else:
                if any(arr.ndim != 2 for arr in components):
                    raise ValueError(
                        f"Action '{name}' requires 2D components to concatenate in {ep_path.name}."
                    )
                feature = np.concatenate(components, axis=1)

            actions[name] = feature

        traj = {
            "episode": ep_path.name,
            "num_steps": length,
            "observations": observations,
            "actions": actions,
        }
        if stream_handle is None:
            trajectories.append(traj)
        else:
            pickle.dump(
                {"observations": observations, "actions": actions},
                stream_handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        episode_lengths.append(length)
        episode_names.append(ep_path.name)
        print(
            f"[{idx + 1}/{len(episode_paths)}] Processed {ep_path.name} ({length} steps)"
        )
    if stream_handle is not None:
        stream_handle.close()
    episode_ends = np.cumsum(episode_lengths)

    metadata_blob = {
        "dataset_name": dataset_name,
        "config_path": str(config_path),
        "source_dir": str(dataset_dir),
        "episodes": episode_names,
        "num_steps": episode_lengths,
        "observation_features": [cfg["name"] for cfg in obs_cfgs],
        "action_features": [cfg["name"] for cfg in act_cfgs],
        "observation_feature_keys": {
            cfg["name"]: cfg.get("keys", []) for cfg in obs_cfgs
        },
        "action_feature_keys": {cfg["name"]: cfg.get("keys", []) for cfg in act_cfgs},
    }

    if args.stream:
        metadata_blob["format"] = "stream_v1"
        header_blob = {
            **metadata_blob,
            "format": "stream_v1",
            "episode_ends": episode_ends.tolist(),
            "num_episodes": len(episode_lengths),
        }
        if stream_tmp_path is None:
            raise ValueError("Streaming output requested but temporary file missing.")
        with output_path.open("wb") as handle:
            pickle.dump(header_blob, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with stream_tmp_path.open("rb") as tmp_handle:
                shutil.copyfileobj(tmp_handle, handle)
        stream_tmp_path.unlink(missing_ok=True)
    else:
        data_blob = {
            "trajectories": [
                {"observations": t["observations"], "actions": t["actions"]}
                for t in trajectories
            ],
            "episode_ends": episode_ends.tolist(),
        }
        with output_path.open("wb") as handle:
            pickle.dump(data_blob, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_blob, handle, indent=2)

    logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")
    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
