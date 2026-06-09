from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

NANOSECONDS_IN_SECOND = 1_000_000_000


def ns_from_ms(value_ms: float | int | None) -> int | None:
    if value_ms is None:
        return None
    return int(float(value_ms) * 1_000_000)


@dataclass(frozen=True)
class ChannelConfig:
    key: str
    required: bool
    max_abs_diff_ns: int | None
    strategy: str = "nearest"  # nearest | hold | future
    allow_future: bool = True
    initial_value: Any | None = None


def normalize_channel_configs(
    cfg: dict, available_keys: Iterable[str]
) -> tuple[list[ChannelConfig], dict]:
    defaults = cfg.get("defaults", {})
    default_required = defaults.get("required", True)
    default_diff_ns = ns_from_ms(defaults.get("max_time_diff_ms"))
    default_strategy = defaults.get("strategy", "nearest")
    default_allow_future = defaults.get("allow_future", True)
    default_initial_value = defaults.get("initial_value")

    raw_channels = cfg.get("channels")
    if not raw_channels:
        raw_channels = [{"key": key} for key in available_keys]

    normalized: list[ChannelConfig] = []
    for entry in raw_channels:
        if "key" not in entry:
            raise ValueError("Each channel config must contain a 'key' entry.")
        key = entry["key"]
        required = entry.get("required", default_required)
        raw_diff = entry.get("max_time_diff_ms")
        if raw_diff is None:
            max_abs_diff_ns = default_diff_ns
        else:
            max_abs_diff_ns = ns_from_ms(raw_diff)
        strategy = entry.get("strategy", default_strategy)
        allow_future = entry.get("allow_future", default_allow_future)
        initial_value = entry.get("initial_value", default_initial_value)
        normalized.append(
            ChannelConfig(
                key=key,
                required=bool(required),
                max_abs_diff_ns=max_abs_diff_ns,
                strategy=str(strategy),
                allow_future=bool(allow_future),
                initial_value=initial_value,
            )
        )

    required_keys = [cfg.key for cfg in normalized if cfg.required]
    missing = [key for key in required_keys if key not in available_keys]
    if missing:
        raise KeyError(f"Required keys missing from dataset: {missing}")

    return normalized, defaults


def build_processing_config(config: dict, available_keys: list[str]) -> dict:
    """Combine channel-level and global settings into a single config dict."""
    channel_cfgs, channel_defaults = normalize_channel_configs(config, available_keys)

    target_frequency_hz = config.get("target_frequency_hz") or channel_defaults.get(
        "target_frequency_hz"
    )
    if target_frequency_hz is None:
        raise ValueError("Please set a single 'target_frequency_hz' in the config.")

    return {
        "channels": channel_cfgs,
        "target_frequency_hz": float(target_frequency_hz),
        "drop_incomplete_frames": config.get("drop_incomplete_frames", False),
    }


def build_time_grid(start_ns: int, end_ns: int, fps: float) -> list[int]:
    if fps <= 0:
        raise ValueError("target_frequency_hz must be positive.")
    dt = int(NANOSECONDS_IN_SECOND / fps)
    if dt <= 0:
        raise ValueError("Computed timestep must be positive.")
    timeline: list[int] = []
    current = start_ns
    while current <= end_ns:
        timeline.append(current)
        current += dt
    if timeline[-1] < end_ns:
        timeline.append(end_ns)
    return timeline


def align_stream(
    timeline: list[int],
    samples: list[Any],
    timestamps: list[int],
    channel_cfg: ChannelConfig,
) -> tuple[list[Any | None], list[int | None]]:
    # Special handling for gripper/command: construct a simple step sequence on the aligned grid.
    if channel_cfg.key == "gripper/command":

        def _cmd_value(msg: Any) -> float | None:
            if isinstance(msg, (int, float)):
                return float(msg)
            if isinstance(msg, dict):
                data = msg.get("data", [])
                if isinstance(data, list) and data:
                    try:
                        return float(data[0])
                    except (TypeError, ValueError):
                        return None
            return None

        parsed = []
        for s, ts in zip(samples, timestamps):
            if ts is None:
                continue
            val = _cmd_value(s)
            if val is None:
                continue
            parsed.append((ts, val))
        parsed.sort(key=lambda x: x[0])

        first_zero_ts = next((ts for ts, v in parsed if v <= 0.5), None)
        first_one_ts = next((ts for ts, v in parsed if v >= 0.5), None)

        init_val = (
            channel_cfg.initial_value
            if channel_cfg.initial_value is not None
            else {"data": [1.0]}
        )

        def _val_from_sample(sample: Any) -> float:
            if isinstance(sample, dict):
                data = sample.get("data", [])
                if isinstance(data, list) and data:
                    try:
                        return float(data[0])
                    except (TypeError, ValueError):
                        pass
            if isinstance(sample, (int, float)):
                return float(sample)
            return 1.0

        init_num = _val_from_sample(init_val)

        initial_ts = timeline[0] if timeline else 0

        def value_for_time(t: int) -> float:
            if first_zero_ts is None and first_one_ts is None:
                return init_num
            if first_zero_ts is None:
                return init_num
            if first_one_ts is None:
                return init_num if t < first_zero_ts else 0.0
            if t < first_zero_ts:
                return init_num
            if t < first_one_ts:
                return 0.0
            return 1.0

        def ts_marker_for_time(t: int) -> int:
            if first_zero_ts is None and first_one_ts is None:
                return initial_ts
            if first_zero_ts is None:
                return initial_ts
            if first_one_ts is None:
                return initial_ts if t < first_zero_ts else first_zero_ts
            if t < first_zero_ts:
                return initial_ts
            if t < first_one_ts:
                return first_zero_ts
            return first_one_ts

        aligned_vals = [value_for_time(t) for t in timeline]
        aligned_ts = [ts_marker_for_time(t) for t in timeline]
        aligned_samples = [{"data": [v]} for v in aligned_vals]
        return aligned_samples, aligned_ts

    # Drop entries with missing timestamps to avoid None arithmetic.
    filtered_pairs = [(s, ts) for s, ts in zip(samples, timestamps) if ts is not None]
    filtered_pairs.sort(key=lambda x: x[1])
    samples = [s for s, _ in filtered_pairs]
    timestamps = [ts for _, ts in filtered_pairs]

    result: list[Any | None] = []
    used_timestamps: list[int | None] = []

    idx = 0
    if channel_cfg.key == "gripper/command":
        last_sample = None
        last_timestamp = None
    else:
        last_sample = channel_cfg.initial_value
        last_timestamp = (
            timeline[0] if channel_cfg.initial_value is not None and timeline else None
        )
    total = len(timestamps)

    for target_time in timeline:
        while idx < total and timestamps[idx] <= target_time:
            last_sample = samples[idx]
            last_timestamp = timestamps[idx]
            idx += 1

        next_sample = samples[idx] if idx < total else None
        next_timestamp = timestamps[idx] if idx < total else None

        # Strategy selection: nearest (default), hold (prefer previous), future (prefer next).
        prev_delta = (
            abs(target_time - last_timestamp) if last_timestamp is not None else None
        )
        next_delta = (
            abs(next_timestamp - target_time) if next_timestamp is not None else None
        )

        candidate = None
        candidate_ts = None
        strategy = channel_cfg.strategy

        if strategy == "hold":
            candidate = last_sample
            candidate_ts = last_timestamp
            if candidate is None and channel_cfg.allow_future:
                candidate = next_sample
                candidate_ts = next_timestamp
        elif strategy == "future":
            candidate = next_sample
            candidate_ts = next_timestamp
            if candidate is None and last_sample is not None:
                candidate = last_sample
                candidate_ts = last_timestamp
        else:  # nearest
            if prev_delta is None and next_delta is None:
                candidate = None
                candidate_ts = None
            elif prev_delta is None:
                candidate = next_sample
                candidate_ts = next_timestamp
            elif next_delta is None or prev_delta <= next_delta:
                candidate = last_sample
                candidate_ts = last_timestamp
            else:
                candidate = next_sample
                candidate_ts = next_timestamp

        if candidate_ts is None:
            result.append(None)
            used_timestamps.append(None)
            continue

        if channel_cfg.max_abs_diff_ns is not None:
            if abs(candidate_ts - target_time) > channel_cfg.max_abs_diff_ns:
                result.append(None)
                used_timestamps.append(None)
                continue

        # Use shallow copies for complex objects to avoid accidental mutation across frames.
        if isinstance(candidate, (dict, list)):
            sample_value = copy.copy(candidate)
        else:
            sample_value = candidate
        result.append(sample_value)
        used_timestamps.append(candidate_ts)

    return result, used_timestamps


############
