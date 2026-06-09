from typing import Any

import cv2
import numpy as np


def _as_float_array(values: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr


def transform_pose(sample: Any, _: dict[str, Any]) -> np.ndarray | None:
    if sample is None:
        return None
    pose = sample.get("pose", {}) if isinstance(sample, dict) else {}
    position = pose.get("position", {})
    orientation = pose.get("orientation", {})
    return np.array(
        [
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
            float(orientation.get("x", 0.0)),
            float(orientation.get("y", 0.0)),
            float(orientation.get("z", 0.0)),
            float(orientation.get("w", 0.0)),
        ],
        dtype=np.float32,
    )


def transform_joint_position(sample: Any, _: dict[str, Any]) -> np.ndarray | None:
    if sample is None:
        return None
    positions = sample.get("position") if isinstance(sample, dict) else None
    if positions is None:
        return None
    return _as_float_array(positions)


def transform_joint_state(sample: Any, cfg: dict[str, Any]) -> np.ndarray | None:
    if sample is None:
        return None
    if not isinstance(sample, dict):
        return None
    include_velocity = bool(cfg.get("include_velocity", False))
    include_effort = bool(cfg.get("include_effort", False))
    pieces: list[np.ndarray] = []

    positions = sample.get("position", [])
    if positions:
        pieces.append(_as_float_array(positions))
    else:
        return None

    if include_velocity:
        velocities = sample.get("velocity", [])
        if velocities:
            pieces.append(_as_float_array(velocities))
        else:
            pieces.append(np.zeros_like(pieces[0]))

    if include_effort:
        effort = sample.get("effort", [])
        if effort:
            pieces.append(_as_float_array(effort))
        else:
            pieces.append(np.zeros_like(pieces[0]))

    return np.concatenate(pieces, axis=0) if pieces else None


def transform_float_array(sample: Any, _: dict[str, Any]) -> np.ndarray | None:
    if sample is None:
        return None
    if isinstance(sample, dict):
        values = sample.get("data", [])
    elif isinstance(sample, (list, tuple, np.ndarray)):
        values = sample
    else:
        values = [sample]
    arr = _as_float_array(values)
    if arr.size == 0:
        return None
    return arr


def transform_image_compressed(sample: Any, cfg: dict[str, Any]) -> np.ndarray | None:
    if sample is None:
        return None
    if not isinstance(sample, dict):
        return None
    raw = sample.get("data")
    if raw is None:
        return None
    img_array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    if cfg.get("convert_to_rgb", False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resize = cfg.get("resize")
    if resize and len(resize) == 2:
        height, width = int(resize[0]), int(resize[1])
        frame = cv2.resize(frame, (width, height))
    return frame


def transform_gripper_state(sample: Any, cfg: dict[str, Any]) -> np.ndarray | None:
    if sample is None or not isinstance(sample, dict):
        return None
    positions = sample.get("position")
    if positions is None:
        return None
    return _as_float_array(positions)
