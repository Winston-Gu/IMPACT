#!/usr/bin/env python
"""
Dump videos from a diffusion-policy zarr replay buffer.

Usage:
    python scripts/data/dump_zarr_videos.py --zarr-path datasets/cache.zarr --out-dir outputs/zarr_videos

This script expects the zarr layout produced by FrankaPickDataset:
  - group "data/front_camera": (T,H,W,3) uint8 stacked over all episodes
  - group "data/side_camera": (T,H,W,3) uint8
  - group "meta/episode_ends": 1D array of cumulative time indices

It writes one video per episode per camera as MP4 files named:
  episode_{idx:04d}_front_camera.mp4
  episode_{idx:04d}_side_camera.mp4
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import zarr


def load_arrays(zarr_path: Path, keys):
    root = zarr.open_group(str(zarr_path), mode="r")
    data_group = root["data"]
    arrays = {k: data_group[k][:] for k in keys if k in data_group}
    episode_ends = root["meta"]["episode_ends"][:]
    return arrays, episode_ends


def dump_videos(arrays, episode_ends, out_dir: Path, fps: int = 20):
    out_dir.mkdir(parents=True, exist_ok=True)
    starts = np.concatenate([[0], episode_ends[:-1]])
    ends = episode_ends
    for ep_idx, (s, e) in enumerate(zip(starts, ends)):
        for cam_key, frames in arrays.items():
            epi_frames = frames[s:e]  # (T,H,W,3)
            if epi_frames.size == 0:
                continue
            fname = out_dir / f"episode_{ep_idx:04d}_{cam_key}.mp4"
            writer = imageio.get_writer(fname, fps=fps)
            for f in epi_frames:
                writer.append_data(f)
            writer.close()
            print(
                f"Wrote {fname} with {epi_frames.shape[0]} frames at {epi_frames.shape[1]}x{epi_frames.shape[2]}"
            )


def main():
    parser = argparse.ArgumentParser(description="Dump zarr episodes to videos.")
    parser.add_argument(
        "--zarr-path", type=str, required=True, help="Path to zarr directory (cache)."
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Output directory for videos."
    )
    parser.add_argument("--fps", type=int, default=20, help="FPS for output videos.")
    parser.add_argument(
        "--cams",
        type=str,
        nargs="+",
        default=["front_camera", "side_camera"],
        help="Camera keys to export.",
    )
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not zarr_path.exists():
        raise FileNotFoundError(f"Missing zarr path: {zarr_path}")

    arrays, episode_ends = load_arrays(zarr_path, args.cams)
    if len(arrays) == 0:
        raise ValueError(
            f"No matching camera keys found in {zarr_path}/data for {args.cams}"
        )
    if episode_ends.size == 0:
        raise ValueError("episode_ends is empty; no episodes to dump.")

    dump_videos(arrays, episode_ends, out_dir=out_dir, fps=args.fps)


if __name__ == "__main__":
    main()
