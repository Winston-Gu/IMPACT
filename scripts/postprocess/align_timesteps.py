#!/usr/bin/env python3
"""Post-process raw dataset episodes into uniformly sampled trajectories."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from utils import ChannelConfig, align_stream, build_processing_config, build_time_grid


def filter_incomplete_group(
    timeline: list[int],
    data_map: dict[str, list[Any | None]],
    timestamp_map: dict[str, list[int | None]],
    group_cfgs: list[ChannelConfig],
    drop_incomplete: bool,
) -> list[int]:
    if not drop_incomplete:
        return timeline

    required_keys = [cfg.key for cfg in group_cfgs if cfg.required]
    if not required_keys:
        return timeline

    keep_indices = [
        idx
        for idx in range(len(timeline))
        if all(data_map[key][idx] is not None for key in required_keys)
    ]

    if len(keep_indices) == len(timeline):
        return timeline

    # Update timeline and channel data in-place for the affected group.
    new_timeline = [timeline[i] for i in keep_indices]
    for cfg in group_cfgs:
        key = cfg.key
        data_map[key] = [data_map[key][i] for i in keep_indices]
        timestamp_map[key] = [timestamp_map[key][i] for i in keep_indices]
    return new_timeline


def process_episode(
    episode_path: Path,
    output_path: Path,
    processing_cfg: dict,
) -> tuple[int, int]:
    with episode_path.open("rb") as handle:
        episode = pickle.load(handle)

    data = episode.get("data", {})
    timestamps = episode.get("timestamps", {})

    channel_cfgs = processing_cfg["channels"]
    target_frequency_hz = processing_cfg["target_frequency_hz"]

    # Single global timeline spanning all channels.
    series_list = []
    for cfg in channel_cfgs:
        series = [ts for ts in timestamps.get(cfg.key, []) if ts is not None]
        if series:
            series_list.append(series)
    if not series_list:
        raise ValueError("No timestamps available to define alignment bounds.")
    start_ns = min(series[0] for series in series_list)
    end_ns = max(series[-1] for series in series_list)

    timeline = build_time_grid(start_ns, end_ns, target_frequency_hz)

    aligned_data: dict[str, list[Any | None]] = {}
    aligned_ts: dict[str, list[int | None]] = {}

    for cfg in channel_cfgs:
        samples = data.get(cfg.key, [])
        series_ts = timestamps.get(cfg.key, [])
        aligned_samples, used_ts = align_stream(timeline, samples, series_ts, cfg)
        aligned_data[cfg.key] = aligned_samples
        aligned_ts[cfg.key] = used_ts

    # Optionally drop frames where any required channel is missing.
    timeline = filter_incomplete_group(
        timeline,
        aligned_data,
        aligned_ts,
        channel_cfgs,
        drop_incomplete=processing_cfg.get("drop_incomplete_frames", False),
    )

    processed_episode = {
        "data": aligned_data,
        "timestamps": aligned_ts,
        "all_timestamps": timeline,
    }

    with output_path.open("wb") as handle:
        pickle.dump(processed_episode, handle)

    original_frames = len(episode.get("all_timestamps", []))
    processed_frames = len(timeline)
    return original_frames, processed_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process a raw dataset into uniformly sampled episodes."
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
        required=True,
        help="Name of the dataset directory under datasets/raw and datasets/processed.",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=[-1],
        help="Use -1 for all episodes or provide integer indices (e.g. 0 1 5).",
    )
    args = parser.parse_args()

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
    config = config.get("align", config)

    dataset_root = Path("datasets").resolve()
    input_dir = dataset_root / "raw" / args.dataset_name
    output_dir = dataset_root / "aligned" / args.dataset_name

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_paths: list[Path]
    if args.episodes == [-1]:
        episode_paths = sorted(input_dir.glob("ep_*.pkl"))
    else:
        episode_paths = [input_dir / f"ep_{idx:05d}.pkl" for idx in args.episodes]

    episode_paths = [p for p in episode_paths if p.exists()]
    if not episode_paths:
        raise FileNotFoundError(f"No episodes found matching selection in {input_dir}")

    logger.info(f"Processing {len(episode_paths)} episodes from {input_dir}")

    with episode_paths[0].open("rb") as handle:
        sample_episode = pickle.load(handle)
    available_keys = sample_episode.get("data", {}).keys()
    processing_cfg = build_processing_config(config, list(available_keys))

    for episode_path in episode_paths:
        output_path = output_dir / episode_path.name
        before, after = process_episode(episode_path, output_path, processing_cfg)
        logger.info(
            f"{episode_path.name}: {before} -> {after} frames (output: {output_path.relative_to(output_dir)})"
        )


if __name__ == "__main__":
    main()
