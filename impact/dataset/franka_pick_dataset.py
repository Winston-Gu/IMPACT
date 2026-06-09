import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import zarr
from filelock import FileLock
from loguru import logger as loguru_logger
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from impact.dataset.base_dataset import BaseImageDataset
from impact.utils.normalizers import LinearNormalizer, SingleFieldLinearNormalizer
from impact.utils.pytorch_util import dict_apply
from impact.utils.replay_buffer import ReplayBuffer
from impact.utils.sampler import SequenceSampler, downsample_mask, get_val_mask


def _resize_video(frames: np.ndarray, hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if hw is None:
        return frames
    height, width = hw
    resized = np.empty((frames.shape[0], height, width, 3), dtype=frames.dtype)
    for idx, img in enumerate(frames):
        resized[idx] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def _build_replay_buffer(
    dataset_path: Path,
    shape_meta: dict,
    image_hw: Optional[Tuple[int, int]],
    store,
    load_episodes_limit: Optional[int] = None,
) -> ReplayBuffer:
    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)

    obs_shape_meta = shape_meta["obs"]
    rgb_keys = [
        k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "rgb"
    ]
    lowdim_keys = [
        k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "low_dim"
    ]

    buffer = ReplayBuffer.create_empty_zarr(storage=store)

    def _add_episode(traj: Dict[str, Any], idx: int, total: Optional[int]) -> None:
        total_label = str(total) if total is not None else "?"
        print(f"[FrankaPickDataset] Loading episode {idx + 1}/{total_label}")
        obs = traj["observations"]
        act = traj["actions"]
        episode: Dict[str, np.ndarray] = {}

        if "actions" in act:
            episode["action"] = np.asarray(act["actions"], dtype=np.float32)
        else:
            raise KeyError("Expected key 'actions' inside trajectory['actions'].")

        for key in lowdim_keys:
            if key not in obs:
                continue
            episode[key] = np.asarray(obs[key], dtype=np.float32)

        for key in rgb_keys:
            if key not in obs:
                continue
            episode[key] = _resize_video(np.asarray(obs[key], dtype=np.uint8), image_hw)

        buffer.add_episode(episode)

    if isinstance(payload, dict) and "trajectories" in payload:
        trajectories = payload["trajectories"]
        if load_episodes_limit is not None:
            trajectories = trajectories[:load_episodes_limit]

        for idx, traj in enumerate(trajectories):
            _add_episode(traj, idx, len(trajectories))
    elif isinstance(payload, dict) and payload.get("format") == "stream_v1":
        num_episodes = payload.get("num_episodes")
        if load_episodes_limit is not None and num_episodes is not None:
            total = min(load_episodes_limit, num_episodes)
        else:
            total = load_episodes_limit or num_episodes
        idx = 0
        with open(dataset_path, "rb") as f_stream:
            pickle.load(f_stream)
            while True:
                if num_episodes is not None and idx >= num_episodes:
                    break
                if load_episodes_limit is not None and idx >= load_episodes_limit:
                    break
                try:
                    traj = pickle.load(f_stream)
                except EOFError:
                    break
                _add_episode(traj, idx, total)
                idx += 1
        if num_episodes is not None and idx < num_episodes:
            raise ValueError(
                f"Streamed dataset ended early: expected {num_episodes}, got {idx}."
            )
    else:
        raise ValueError(f"Unsupported dataset format in {dataset_path}")

    return buffer


class FrankaPickDataset(BaseImageDataset):
    """
    Dataset loader for the exported Franka pick dataset.
    Supports caching to an on-disk zarr store to avoid reprocessing.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = 2,
        n_latency_steps: int = 0,
        val_ratio: float = 0.1,
        max_train_episodes: Optional[int] = None,
        resize_image_hw: Optional[Sequence[int]] = (256, 256),
        use_disk_cache: bool = True,
        cache_path: Optional[str] = None,
        seed: int = 42,
        load_episodes_limit: Optional[int] = None,
        normalizer: Optional[dict] = None,
    ):
        dataset_path = Path(dataset_path).expanduser().resolve()
        assert dataset_path.is_file(), f"Missing dataset: {dataset_path}"

        loguru_logger.info(f"Loading dataset from {dataset_path}")

        cache_path = self._compute_cache_path(
            dataset_path, shape_meta, resize_image_hw, cache_path
        )
        cache_store = None
        replay_buffer = None

        if use_disk_cache:
            cache_store = zarr.DirectoryStore(str(cache_path))
            lock = FileLock(str(cache_path) + ".lock")
            with lock:
                need_rebuild = True
                if cache_path.exists() and cache_path.is_dir():
                    try:
                        cache_group = zarr.group(store=cache_store)
                        # verify structure
                        if (
                            "meta" in cache_group
                            and "episode_ends" in cache_group["meta"]
                        ):
                            replay_buffer = ReplayBuffer.create_from_group(cache_group)
                            need_rebuild = False
                    except Exception:
                        need_rebuild = True
                if need_rebuild:
                    # clean incomplete cache and rebuild
                    loguru_logger.info(f"Rebuilding cache at {cache_path}")
                    if cache_path.exists():
                        shutil.rmtree(cache_path, ignore_errors=True)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_store = zarr.DirectoryStore(str(cache_path))
                    replay_buffer = _build_replay_buffer(
                        dataset_path=dataset_path,
                        shape_meta=shape_meta,
                        image_hw=tuple(resize_image_hw) if resize_image_hw else None,
                        store=cache_store,
                        load_episodes_limit=load_episodes_limit,
                    )
        else:
            cache_store = zarr.storage.MemoryStore()
            replay_buffer = _build_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                image_hw=tuple(resize_image_hw) if resize_image_hw else None,
                store=cache_store,
                load_episodes_limit=load_episodes_limit,
            )

        obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = [
            k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "rgb"
        ]
        self.lowdim_keys = [
            k
            for k, v in obs_shape_meta.items()
            if v.get("type", "low_dim") == "low_dim"
        ]

        key_first_k = {
            k: n_obs_steps
            for k in self.rgb_keys + self.lowdim_keys
            if n_obs_steps is not None
        }

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, max_train_episodes, seed=seed)
        train_eps = np.nonzero(train_mask)[0].tolist()
        val_eps = np.nonzero(val_mask)[0].tolist()
        loguru_logger.info(
            "FrankaPickDataset split -> train episodes: {}, val episodes: {}",
            train_eps,
            val_eps,
        )

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.normalizer_cfg = normalizer

    @staticmethod
    def _compute_cache_path(
        dataset_path: Path,
        shape_meta: dict,
        resize_image_hw: Optional[Sequence[int]],
        cache_path: Optional[str],
    ) -> Path:
        if cache_path is not None:
            return Path(cache_path)
        stem = dataset_path.stem
        resize_tag = "fullres"
        if resize_image_hw:
            resize_tag = f"{resize_image_hw[0]}x{resize_image_hw[1]}"
        fingerprint = hashlib.md5(
            json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()[:8]
        cache_name = f"{stem}_{resize_tag}_{fingerprint}.zarr"
        return dataset_path.parent / cache_name

    def get_validation_dataset(self):
        val_set = object.__new__(FrankaPickDataset)
        val_set.__dict__ = self.__dict__.copy()
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        cfg = self.normalizer_cfg or {}
        keys_cfg = cfg.get("keys", {})
        dtype = getattr(torch, cfg.get("dtype", "float32"), torch.float32)

        def _fit_field(data, field_cfg, key_name: str):
            if "mode" not in field_cfg:
                raise ValueError(f"Missing normalizer mode for key '{key_name}'.")
            mode = field_cfg["mode"]
            if mode == "identity":
                return SingleFieldLinearNormalizer.create_identity(dtype=dtype)
            if mode == "image":
                input_min = float(field_cfg.get("input_min", 0.0))
                input_max = float(field_cfg.get("input_max", 1.0))
                output_min = float(field_cfg.get("output_min", -1.0))
                output_max = float(field_cfg.get("output_max", 1.0))
                mean = field_cfg.get("mean", None)
                std = field_cfg.get("std", None)
                if mean is None:
                    mean = (input_min + input_max) / 2.0
                if std is None:
                    std = (input_max - input_min) / np.sqrt(12)
                input_range = input_max - input_min
                scale = (output_max - output_min) / input_range
                offset = output_min - scale * input_min
                stats = {
                    "min": np.array([input_min], dtype=np.float32),
                    "max": np.array([input_max], dtype=np.float32),
                    "mean": np.array([mean], dtype=np.float32),
                    "std": np.array([std], dtype=np.float32),
                }
                return SingleFieldLinearNormalizer.create_manual(
                    scale=np.array([scale], dtype=np.float32),
                    offset=np.array([offset], dtype=np.float32),
                    input_stats_dict=stats,
                )
            if mode == "limits":
                output_min = field_cfg.get("output_min", -1.0)
                output_max = field_cfg.get("output_max", 1.0)
                return SingleFieldLinearNormalizer.create_fit(
                    data,
                    mode=mode,
                    output_min=output_min,
                    output_max=output_max,
                    dtype=dtype,
                )
            return SingleFieldLinearNormalizer.create_fit(
                data,
                mode=mode,
                dtype=dtype,
            )

        normalizer = LinearNormalizer()
        action_cfg = cfg.get("action", {})
        normalizer["action"] = _fit_field(
            self.replay_buffer["action"], action_cfg, "action"
        )

        lowdim_cfg = cfg.get("lowdim", {})
        rgb_cfg = cfg.get("rgb", {})

        for key in self.lowdim_keys:
            field_cfg = keys_cfg.get(key, lowdim_cfg)
            normalizer[key] = _fit_field(self.replay_buffer[key], field_cfg, key)

        for key in self.rgb_keys:
            field_cfg = keys_cfg.get(key, rgb_cfg)
            normalizer[key] = _fit_field(self.replay_buffer[key], field_cfg, key)

        return normalizer

    def get_all_actions(self) -> np.ndarray:
        return self.replay_buffer["action"][:]

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict: Dict[str, np.ndarray] = {}
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps :]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
        return torch_data
