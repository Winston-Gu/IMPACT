#!/usr/bin/env python3
"""Quick inspection tool for recorded dataset episodes."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from statistics import mean, median

NANOSECONDS_IN_SECOND = 1_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a dataset episode pickle.")
    parser.add_argument(
        "dataset",
        help="Dataset name (directory inside the datasets folder)",
    )
    parser.add_argument(
        "episode",
        help="Episode identifier. Accepts a zero-based index (e.g. 5), "
        "a filename like ep_00005.pkl, or a relative path inside the dataset directory.",
    )
    parser.add_argument(
        "--datasets-root",
        default="datasets",
        help="Root directory containing datasets/ (default: %(default)s)",
    )
    return parser.parse_args()


def resolve_episode_path(root: Path, dataset: str, episode_arg: str) -> Path:
    dataset_dir = root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    candidate = Path(episode_arg)
    if candidate.suffix == ".pkl":
        # Treat direct filename relative to dataset directory unless absolute.
        if candidate.is_absolute():
            return candidate
        return dataset_dir / candidate

    if candidate.is_absolute():
        return candidate

    # Interpret numeric episode index.
    if episode_arg.isdigit():
        episode_idx = int(episode_arg)
        filename = f"ep_{episode_idx:05d}.pkl"
        return dataset_dir / filename

    # Fallback to relative path under dataset dir.
    return dataset_dir / episode_arg


def main() -> None:
    args = parse_args()
    datasets_root = Path(args.datasets_root).expanduser().resolve()
    episode_path = resolve_episode_path(datasets_root, args.dataset, args.episode)

    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")

    with episode_path.open("rb") as handle:
        data = pickle.load(handle)

    print(f"Loaded episode: {episode_path}")
    top_level_keys = list(data.keys())
    print(f"Top-level keys: {top_level_keys}")
    data_keys = list(data["data"].keys())
    print(f"Data keys: {data_keys}")

    timestamps = data.get("timestamps", {})
    channels = set(data.get("data", {}).keys()) | set(timestamps.keys())
    for key in sorted(channels):
        value = data.get("data", {}).get(key)
        ts = timestamps.get(key)
        length = len(value) if isinstance(value, list) else "?"
        freq_desc = describe_frequency(ts)
        print(f"- {key}: {length} entries; {freq_desc}")


def describe_frequency(timestamps: list[int] | None) -> str:
    if not timestamps:
        return "frequency N/A (no timestamps)"
    if len(timestamps) < 2:
        return "frequency N/A (only one timestamp)"

    deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if b > a]
    if not deltas:
        return "frequency N/A (non-increasing timestamps)"

    median_dt_ns = median(deltas)
    mean_dt_ns = mean(deltas)
    duration_ns = timestamps[-1] - timestamps[0]
    overall_hz = (
        (len(timestamps) - 1) * NANOSECONDS_IN_SECOND / duration_ns
        if duration_ns > 0
        else None
    )

    median_hz = NANOSECONDS_IN_SECOND / median_dt_ns
    mean_hz = NANOSECONDS_IN_SECOND / mean_dt_ns

    parts = [f"median ~{median_hz:.2f} Hz", f"mean ~{mean_hz:.2f} Hz"]
    if overall_hz is not None:
        parts.append(f"span ~{overall_hz:.2f} Hz")
    return ", ".join(parts)


if __name__ == "__main__":
    main()
