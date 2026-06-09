#!/usr/bin/env python3
"""
Web viewer for CRISPControl dataset episodes.

Run with:
    python scripts/dataset_viewer/viewer_server.py --dataset datasets/test
"""
from __future__ import annotations

import argparse
import base64
import bisect
import json
import pickle
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
HTML_PATH = PACKAGE_DIR / "index.html"
CONFIG_PATH = PACKAGE_DIR / "config.json"
DEFAULT_EXPORTED_FPS = 30.0


def _ns_to_sec(ns: int) -> float:
    return ns / 1e9


def _encode_image(data_field: Any) -> str:
    if isinstance(data_field, bytes):
        raw = data_field
    elif isinstance(data_field, bytearray):
        raw = bytes(data_field)
    else:
        raw = bytes(data_field)
    return base64.b64encode(raw).decode("ascii")


def _encode_rgb_frame(frame: np.ndarray) -> str:
    if frame is None:
        return ""
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size == 0:
            return ""
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        return ""
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class CameraStream:
    idx: int
    name: str
    channel: str
    description: str = ""
    frames: List[dict] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)

    def frame_at(self, target_ns: int) -> Optional[Dict[str, Any]]:
        if not self.frames or not self.timestamps:
            return None
        pos = bisect.bisect_right(self.timestamps, target_ns) - 1
        if pos < 0:
            return None
        frame = self.frames[pos]
        encoded = _encode_image(frame.get("data", b""))
        fmt = frame.get("format", "jpeg")
        ts = self.timestamps[pos]
        return {
            "idx": self.idx,
            "name": self.name,
            "channel": self.channel,
            "timestamp_ns": ts,
            "timestamp_sec": _ns_to_sec(ts),
            "format": fmt,
            "data": encoded,
        }

    @property
    def available(self) -> bool:
        return bool(self.frames)


@dataclass
class ExportedCameraStream:
    idx: int
    name: str
    channel: str
    frames: List[np.ndarray] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)

    def frame_at(self, target_ns: int) -> Optional[Dict[str, Any]]:
        if not self.frames or not self.timestamps:
            return None
        pos = bisect.bisect_right(self.timestamps, target_ns) - 1
        if pos < 0:
            return None
        frame = self.frames[pos]
        encoded = _encode_rgb_frame(frame)
        if not encoded:
            return None
        ts = self.timestamps[pos]
        return {
            "idx": self.idx,
            "name": self.name,
            "channel": self.channel,
            "timestamp_ns": ts,
            "timestamp_sec": _ns_to_sec(ts),
            "format": "jpeg",
            "data": encoded,
        }

    @property
    def available(self) -> bool:
        return bool(self.frames)


@dataclass
class JointStream:
    names: List[str] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    positions: List[List[float]] = field(default_factory=list)

    def sample(self, target_ns: int) -> Optional[Dict[str, Any]]:
        if not self.timestamps:
            return None
        idx = bisect.bisect_right(self.timestamps, target_ns) - 1
        if idx < 0:
            return None
        ts = self.timestamps[idx]
        return {
            "timestamp_ns": ts,
            "timestamp_sec": _ns_to_sec(ts),
            "names": self.names,
            "positions": self.positions[idx],
        }


@dataclass
class PoseSeries:
    timestamps: List[int] = field(default_factory=list)
    positions: List[List[float]] = field(default_factory=list)

    def to_dict(self, origin_ns: int) -> Dict[str, Any]:
        clean = [
            (ts, pos)
            for ts, pos in zip(self.timestamps, self.positions)
            if ts is not None
        ]
        if not clean:
            return {"timestamps_sec": [], "positions": {"x": [], "y": [], "z": []}}
        clean_ts = [ts for ts, _ in clean]
        origin = origin_ns if origin_ns else clean_ts[0]
        rel_times = [(_ts - origin) / 1e9 for _ts in clean_ts]
        xs = [pos[0] for _, pos in clean]
        ys = [pos[1] for _, pos in clean]
        zs = [pos[2] for _, pos in clean]
        return {
            "timestamps_sec": rel_times,
            "positions": {"x": xs, "y": ys, "z": zs},
        }


def _joint_stream_to_series(stream: JointStream) -> Dict[str, Any]:
    clean = [
        (ts, pos)
        for ts, pos in zip(stream.timestamps, stream.positions)
        if ts is not None
    ]
    if not clean:
        return {
            "names": stream.names,
            "timestamps_sec": [],
            "origin_timestamp_ns": 0,
            "timestamps_ns": [],
            "positions": {name: [] for name in stream.names},
        }
    clean_ts = [ts for ts, _ in clean]
    origin = clean_ts[0]
    times = [(_ts - origin) / 1e9 for _ts in clean_ts]
    positions: Dict[str, List[float]] = {}
    for idx, name in enumerate(stream.names):
        positions[name] = [sample[idx] for _, sample in clean]
    return {
        "names": stream.names,
        "timestamps_sec": times,
        "origin_timestamp_ns": origin,
        "timestamps_ns": clean_ts,
        "positions": positions,
    }


class EpisodeData:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        with path.open("rb") as f:
            payload = pickle.load(f)
        self.metadata = payload.get("metadata", {})
        data_block = payload.get("data", {})
        ts_block = payload.get("timestamps", {})
        self.aligned_timeline = [
            ts for ts in (payload.get("all_timestamps") or []) if ts is not None
        ]
        self.channel_timestamps = {
            key: [ts for ts in (ts_block.get(key, []) or []) if ts is not None]
            for key in ts_block.keys()
        }
        all_ts_lists = [ts for ts in self.channel_timestamps.values() if ts]
        self.origin_ns = min((ts_list[0] for ts_list in all_ts_lists), default=0)
        self.channel_order = sorted(self.channel_timestamps.keys())
        self.robot_joints = self._build_joint_stream(
            data_block, ts_block, "robot/joint_states"
        )
        self.gripper_joints = self._build_joint_stream(
            data_block, ts_block, "gripper/joint_states"
        )
        self.gripper_command = self._build_gripper_command_series(
            data_block, ts_block, "gripper/command"
        )
        self.cameras = self._build_camera_streams(data_block, ts_block)
        self.pose_series = self._build_pose_series(data_block, ts_block)
        self.frame_times = self._build_frame_times()

    def _build_joint_stream(self, data_block, ts_block, key: str) -> JointStream:
        msgs = data_block.get(key, []) or []
        times = ts_block.get(key, []) or []
        valid_msgs = [msg for msg in msgs if msg]
        names = valid_msgs[0].get("name", []) if valid_msgs else []
        positions = [msg.get("position", []) if msg else [] for msg in msgs]
        return JointStream(names=names, timestamps=times, positions=positions)

    def _build_camera_streams(self, data_block, ts_block) -> Dict[int, CameraStream]:
        topic_meta = {}
        for entry in self.metadata.get("topics", []):
            if not isinstance(entry, dict):
                continue
            camera = entry.get("camera")
            if camera:
                key = (camera.get("camera_idx"), camera.get("channel"))
                topic_meta[key] = camera

        streams: Dict[int, CameraStream] = {}
        for key, frames in data_block.items():
            if not key.startswith("cameras/") or not key.endswith("color_image"):
                continue
            parts = key.split("/")
            if len(parts) != 3:
                continue
            cam_idx = int(parts[1])
            channel = parts[2]
            meta = topic_meta.get((cam_idx, channel), {})
            camera_name = meta.get("camera_name", f"camera_{cam_idx}")
            description = meta.get("description", "")
            timestamps = ts_block.get(key, [])
            streams[cam_idx] = CameraStream(
                idx=cam_idx,
                name=camera_name,
                channel=channel,
                description=description,
                frames=frames,
                timestamps=timestamps,
            )
        return streams

    def _build_pose_series(self, data_block, ts_block) -> Dict[str, PoseSeries]:
        alias_map = {
            "current": "robot/current_pose",
            "command": "command/target_pose",
        }
        series: Dict[str, PoseSeries] = {}
        for label, key in alias_map.items():
            msgs = data_block.get(key, []) or []
            times = ts_block.get(key, []) or []
            if not msgs or not times:
                continue
            positions: List[List[float]] = []
            for msg in msgs:
                if not msg:
                    positions.append([0.0, 0.0, 0.0])
                    continue
                pose = msg.get("pose", {})
                pos = pose.get("position", {}) if isinstance(pose, dict) else {}
                x = float(pos.get("x", 0.0))
                y = float(pos.get("y", 0.0))
                z = float(pos.get("z", 0.0))
                positions.append([x, y, z])
            series[label] = PoseSeries(timestamps=times, positions=positions)
        return series

    def _build_gripper_command_series(
        self, data_block, ts_block, key: str
    ) -> Dict[str, List[float]]:
        msgs = data_block.get(key, []) or []
        times = ts_block.get(key, []) or []
        values: List[float] = []
        for msg in msgs:
            if msg is None:
                values.append(0.0)
                continue
            if isinstance(msg, (int, float)):
                values.append(float(msg))
                continue
            data = msg.get("data", []) if isinstance(msg, dict) else []
            if isinstance(data, list) and data:
                try:
                    values.append(float(data[0]))
                    continue
                except (TypeError, ValueError):
                    pass
            values.append(0.0)
        return {"timestamps": times, "values": values}

    def _build_frame_times(self) -> List[int]:
        times = set()
        for stream in self.cameras.values():
            times.update(ts for ts in stream.timestamps if ts is not None)
        if not times:
            times.update(ts for ts in self.robot_joints.timestamps if ts is not None)
        return sorted(times)

    @property
    def frame_count(self) -> int:
        return len(self.frame_times)

    def frame_at(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self.frame_times:
            return None
        idx = max(0, min(idx, len(self.frame_times) - 1))
        ts = self.frame_times[idx]
        payload: Dict[str, Any] = {
            "episode": self.name,
            "index": idx,
            "timestamp_ns": ts,
            "timestamp_sec": _ns_to_sec(ts),
        }
        payload["joint_state"] = self.robot_joints.sample(ts)
        cam_payload = {}
        for cam_idx, stream in self.cameras.items():
            frame = stream.frame_at(ts)
            if frame:
                cam_payload[str(cam_idx)] = frame
        payload["cameras"] = cam_payload
        return payload

    def pose_series_dict(self) -> Dict[str, Any]:
        if not self.pose_series:
            return {"series": {}, "origin_timestamp_ns": 0}
        origins = [
            ts
            for series in self.pose_series.values()
            for ts in series.timestamps[:1]
            if ts is not None
        ]
        origin_ns = min(origins) if origins else 0
        timeline_ns = [ts for ts in self.aligned_timeline if ts is not None]
        return {
            "series": {
                label: series.to_dict(origin_ns)
                for label, series in self.pose_series.items()
            },
            "origin_timestamp_ns": origin_ns,
            "timeline_ns": timeline_ns,
            "timeline_sec": [_ns_to_sec(ts) for ts in timeline_ns],
        }

    def joint_series_dict(self) -> Dict[str, Any]:
        return {
            "robot": _joint_stream_to_series(self.robot_joints),
            "gripper": _joint_stream_to_series(self.gripper_joints),
            "gripper_command": self._gripper_command_series_dict(),
            "timeline_ns": [ts for ts in self.aligned_timeline if ts is not None],
            "timeline_sec": [
                _ns_to_sec(ts) for ts in self.aligned_timeline if ts is not None
            ],
        }

    def timestamp_series(
        self, start_ns: int | None = None, end_ns: int | None = None
    ) -> Dict[str, Any]:
        origin_ns = self.origin_ns
        window_start_ns = origin_ns if start_ns is None else start_ns
        window_end_ns = end_ns

        channels = []
        for key in self.channel_order:
            if "camera_info" in key:
                continue
            ts_list = self.channel_timestamps.get(key, [])
            filtered_ts = [
                ts
                for ts in ts_list
                if (window_start_ns is None or ts >= window_start_ns)
                and (window_end_ns is None or ts <= window_end_ns)
            ]
            channels.append({"key": key, "timestamps_ns": filtered_ts})
        return {
            "origin_timestamp_ns": window_start_ns or 0,
            "channels": channels,
        }

    def _gripper_command_series_dict(self) -> Dict[str, Any]:
        timestamps = [
            ts for ts in self.gripper_command.get("timestamps", []) if ts is not None
        ]
        values = self.gripper_command.get("values", [])
        if not timestamps or not values:
            return {
                "timestamps_sec": [],
                "timestamps_ns": [],
                "origin_timestamp_ns": 0,
                "values": [],
            }
        origin = timestamps[0]
        times = [(_ts - origin) / 1e9 for _ts in timestamps]
        return {
            "timestamps_sec": times,
            "origin_timestamp_ns": origin,
            "timestamps_ns": timestamps,
            "values": values,
        }

    def meta(self) -> Dict[str, Any]:
        start_ns = self.frame_times[0] if self.frame_times else 0
        end_ns = self.frame_times[-1] if self.frame_times else 0
        duration = _ns_to_sec(end_ns - start_ns) if end_ns >= start_ns else 0.0
        return {
            "episode": self.name,
            "path": str(self.path),
            "frame_count": self.frame_count,
            "start_timestamp_ns": start_ns,
            "end_timestamp_ns": end_ns,
            "duration_sec": duration,
            "cameras": [
                {
                    "idx": stream.idx,
                    "name": stream.name,
                    "channel": stream.channel,
                    "description": stream.description,
                    "frame_count": len(stream.frames),
                    "available": stream.available,
                }
                for stream in sorted(self.cameras.values(), key=lambda s: s.idx)
            ],
            "joint_names": self.robot_joints.names,
        }


class ExportedEpisodeData:
    def __init__(
        self,
        name: str,
        trajectory: Dict[str, Any],
        fps: float,
    ):
        self.name = name
        self.trajectory = trajectory
        self.fps = fps
        self.observations = trajectory.get("observations", {}) or {}
        self.actions = trajectory.get("actions", {}) or {}
        self.origin_ns = 0
        self._camera_keys = self._find_camera_keys()
        self.frame_count = self._infer_frame_count()
        self.frame_times = self._build_frame_times()
        self.cameras = self._build_camera_streams()
        self.robot_joints, self.gripper_joints = self._build_joint_streams()
        self.gripper_command = self._build_gripper_command_series()
        self.pose_series = self._build_pose_series()
        self.aligned_timeline = self.frame_times

    def _find_camera_keys(self) -> List[str]:
        camera_keys = []
        for key, value in self.observations.items():
            if (
                isinstance(value, np.ndarray)
                and value.ndim == 4
                and value.shape[-1] == 3
            ):
                camera_keys.append(key)
        return sorted(camera_keys)

    def _infer_frame_count(self) -> int:
        for value in self.observations.values():
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                return int(value.shape[0])
        return 0

    def _build_frame_times(self) -> List[int]:
        if self.frame_count <= 0:
            return []
        step_ns = int(1e9 / self.fps) if self.fps > 0 else 0
        return [idx * step_ns for idx in range(self.frame_count)]

    def _build_camera_streams(self) -> Dict[int, ExportedCameraStream]:
        streams: Dict[int, ExportedCameraStream] = {}
        for idx, key in enumerate(self._camera_keys):
            frames = self.observations.get(key)
            if not isinstance(frames, np.ndarray):
                continue
            timestamps = self.frame_times
            streams[idx] = ExportedCameraStream(
                idx=idx,
                name=key,
                channel="color_image",
                frames=list(frames),
                timestamps=timestamps,
            )
        return streams

    def _build_joint_streams(self) -> tuple[JointStream, JointStream]:
        proprio = self.observations.get("proprioception")
        if not isinstance(proprio, np.ndarray) or proprio.ndim != 2:
            return JointStream(), JointStream()
        pose_size = 7
        gripper_size = 1
        joint_size = proprio.shape[1] - pose_size - gripper_size
        if joint_size <= 0:
            return JointStream(), JointStream()
        joint_start = pose_size
        joint_end = joint_start + joint_size
        joint_names = [f"joint_{idx}" for idx in range(joint_size)]
        joint_positions = proprio[:, joint_start:joint_end].tolist()
        gripper_positions = proprio[:, joint_end : joint_end + gripper_size].tolist()
        return (
            JointStream(
                names=joint_names,
                timestamps=self.frame_times,
                positions=joint_positions,
            ),
            JointStream(
                names=["gripper"],
                timestamps=self.frame_times,
                positions=gripper_positions,
            ),
        )

    def _build_gripper_command_series(self) -> Dict[str, List[float]]:
        actions = self.actions.get("actions")
        if not isinstance(actions, np.ndarray) or actions.ndim != 2:
            return {"timestamps": [], "values": []}
        if actions.shape[1] < 8:
            return {"timestamps": [], "values": []}
        values = actions[:, 7].astype(float).tolist()
        return {"timestamps": self.frame_times, "values": values}

    def _build_pose_series(self) -> Dict[str, PoseSeries]:
        series: Dict[str, PoseSeries] = {}
        proprio = self.observations.get("proprioception")
        if (
            isinstance(proprio, np.ndarray)
            and proprio.ndim == 2
            and proprio.shape[1] >= 3
        ):
            positions = proprio[:, :3].astype(float).tolist()
            series["current"] = PoseSeries(
                timestamps=self.frame_times,
                positions=positions,
            )
        actions = self.actions.get("actions")
        if (
            isinstance(actions, np.ndarray)
            and actions.ndim == 2
            and actions.shape[1] >= 3
        ):
            cmd_positions = actions[:, :3].astype(float).tolist()
            series["command"] = PoseSeries(
                timestamps=self.frame_times,
                positions=cmd_positions,
            )
        return series

    def frame_at(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self.frame_times:
            return None
        idx = max(0, min(idx, len(self.frame_times) - 1))
        ts = self.frame_times[idx]
        payload: Dict[str, Any] = {
            "episode": self.name,
            "index": idx,
            "timestamp_ns": ts,
            "timestamp_sec": _ns_to_sec(ts),
        }
        cam_payload = {}
        for cam_idx, stream in self.cameras.items():
            frame = stream.frame_at(ts)
            if frame:
                cam_payload[str(cam_idx)] = frame
        payload["cameras"] = cam_payload
        payload["joint_state"] = self.robot_joints.sample(ts)
        return payload

    def pose_series_dict(self) -> Dict[str, Any]:
        if not self.pose_series:
            return {"series": {}, "origin_timestamp_ns": 0, "timeline_ns": []}
        origin_ns = self.frame_times[0] if self.frame_times else 0
        timeline_ns = self.frame_times
        return {
            "series": {
                label: series.to_dict(origin_ns)
                for label, series in self.pose_series.items()
            },
            "origin_timestamp_ns": origin_ns,
            "timeline_ns": timeline_ns,
            "timeline_sec": [_ns_to_sec(ts) for ts in timeline_ns],
        }

    def joint_series_dict(self) -> Dict[str, Any]:
        return {
            "robot": _joint_stream_to_series(self.robot_joints),
            "gripper": _joint_stream_to_series(self.gripper_joints),
            "gripper_command": self._gripper_command_series_dict(),
            "timeline_ns": [ts for ts in self.frame_times if ts is not None],
            "timeline_sec": [
                _ns_to_sec(ts) for ts in self.frame_times if ts is not None
            ],
        }

    def _gripper_command_series_dict(self) -> Dict[str, Any]:
        timestamps = [
            ts for ts in self.gripper_command.get("timestamps", []) if ts is not None
        ]
        values = self.gripper_command.get("values", [])
        if not timestamps or not values:
            return {
                "timestamps_sec": [],
                "timestamps_ns": [],
                "origin_timestamp_ns": 0,
                "values": [],
            }
        origin = timestamps[0]
        times = [(_ts - origin) / 1e9 for _ts in timestamps]
        return {
            "timestamps_sec": times,
            "origin_timestamp_ns": origin,
            "timestamps_ns": timestamps,
            "values": values,
        }

    def timestamp_series(
        self, start_ns: int | None = None, end_ns: int | None = None
    ) -> Dict[str, Any]:
        if not self.frame_times:
            return {"origin_timestamp_ns": 0, "channels": []}
        window_start_ns = 0 if start_ns is None else start_ns
        window_end_ns = end_ns
        filtered = [
            ts
            for ts in self.frame_times
            if (window_start_ns is None or ts >= window_start_ns)
            and (window_end_ns is None or ts <= window_end_ns)
        ]
        channels = []
        for idx, key in enumerate(self._camera_keys):
            channels.append({"key": f"camera/{idx}/{key}", "timestamps_ns": filtered})
        return {"origin_timestamp_ns": window_start_ns or 0, "channels": channels}

    def meta(self) -> Dict[str, Any]:
        start_ns = self.frame_times[0] if self.frame_times else 0
        end_ns = self.frame_times[-1] if self.frame_times else 0
        duration = _ns_to_sec(end_ns - start_ns) if end_ns >= start_ns else 0.0
        return {
            "episode": self.name,
            "path": "",
            "frame_count": self.frame_count,
            "start_timestamp_ns": start_ns,
            "end_timestamp_ns": end_ns,
            "duration_sec": duration,
            "cameras": [
                {
                    "idx": stream.idx,
                    "name": stream.name,
                    "channel": stream.channel,
                    "description": "",
                    "frame_count": len(stream.frames),
                    "available": stream.available,
                }
                for stream in sorted(self.cameras.values(), key=lambda s: s.idx)
            ],
            "joint_names": [],
        }


class EpisodeManager:
    def __init__(self, dataset_path: Path, exported_fps: float = DEFAULT_EXPORTED_FPS):
        self.dataset_path = dataset_path
        self.exported_fps = exported_fps
        self._cache: Dict[str, EpisodeData | ExportedEpisodeData] = {}
        self._lock = threading.Lock()
        self._exported_blob: Optional[Dict[str, Any]] = None
        self._exported_names: List[str] = []
        self._exported_trajectories: List[Dict[str, Any]] = []
        self._exported_metadata: Optional[Dict[str, Any]] = None
        self._exported_file: Optional[Path] = None
        self._exported_format: Optional[str] = None
        self._exported_num_episodes: Optional[int] = None

        if dataset_path.is_file():
            self._register_exported_file(dataset_path)
            self.episodes = {name: dataset_path for name in self._exported_names}
            return

        self.episodes = {
            path.stem: path for path in sorted(dataset_path.glob("ep_*.pkl"))
        }
        if self.episodes:
            return

        exported_file = self._find_exported_file(dataset_path)
        if exported_file is not None:
            self._register_exported_file(exported_file)
            self.episodes = {name: exported_file for name in self._exported_names}
        else:
            self.episodes = {}

    def list_episodes(self) -> List[Dict[str, Any]]:
        items = []
        for name, path in self.episodes.items():
            stats = path.stat() if path.exists() else None
            items.append(
                {
                    "name": name,
                    "path": str(path),
                    "size_bytes": stats.st_size if stats else 0,
                    "modified_time": stats.st_mtime if stats else 0.0,
                }
            )
        return items

    def get(self, name: str) -> EpisodeData:
        with self._lock:
            if name not in self.episodes:
                raise KeyError(f"Episode {name} not found")
            if name not in self._cache:
                if self._exported_file is not None:
                    index = self._exported_names.index(name)
                    if self._exported_format == "stream_v1":
                        traj = self._load_stream_episode(index)
                    else:
                        self._ensure_exported_loaded()
                        traj = self._exported_trajectories[index]
                    self._cache[name] = ExportedEpisodeData(
                        name=name, trajectory=traj, fps=self.exported_fps
                    )
                else:
                    self._cache[name] = EpisodeData(self.episodes[name])
            return self._cache[name]

    def _find_exported_file(self, dataset_dir: Path) -> Optional[Path]:
        candidates = sorted(dataset_dir.glob("*.pkl"))
        if len(candidates) == 1 and not candidates[0].name.startswith("ep_"):
            return candidates[0]
        return None

    def _register_exported_file(self, exported_path: Path) -> None:
        self._exported_file = exported_path
        metadata_path = exported_path.with_suffix(".json")
        header = None
        with exported_path.open("rb") as handle:
            header = pickle.load(handle)
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if isinstance(metadata, dict):
                self._exported_metadata = metadata
                names = metadata.get("episodes", [])
                if isinstance(names, list):
                    self._exported_names = names
        if isinstance(header, dict) and header.get("format") == "stream_v1":
            self._exported_format = "stream_v1"
            self._exported_num_episodes = header.get("num_episodes")
            if not self._exported_names and self._exported_num_episodes:
                self._exported_names = [
                    f"ep_{idx:05d}" for idx in range(self._exported_num_episodes)
                ]
        elif isinstance(header, dict) and "trajectories" in header:
            self._exported_format = "legacy"
            if not self._exported_names:
                self._exported_names = [
                    f"ep_{idx:05d}"
                    for idx in range(len(header.get("trajectories", [])))
                ]
        if not self._exported_names:
            self._exported_names = ["ep_00000"]

    def _load_stream_episode(self, index: int) -> Dict[str, Any]:
        if self._exported_file is None:
            raise ValueError("Exported dataset file not set.")
        with self._exported_file.open("rb") as f_stream:
            pickle.load(f_stream)
            for idx in range(index + 1):
                try:
                    traj = pickle.load(f_stream)
                except EOFError as exc:
                    raise ValueError(
                        f"Streamed dataset ended early at episode {idx}."
                    ) from exc
        return traj

    def _ensure_exported_loaded(self) -> None:
        if self._exported_blob is not None:
            return
        if self._exported_file is None:
            raise ValueError("Exported dataset file not set.")
        if self._exported_format == "stream_v1":
            raise ValueError("Streaming datasets should be loaded per-episode.")
        with self._exported_file.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "trajectories" in payload:
            trajectories = payload.get("trajectories", [])
        elif isinstance(payload, dict) and payload.get("format") == "stream_v1":
            trajectories = []
            num_episodes = payload.get("num_episodes")
            with self._exported_file.open("rb") as f_stream:
                pickle.load(f_stream)
                while True:
                    if num_episodes is not None and len(trajectories) >= num_episodes:
                        break
                    try:
                        traj = pickle.load(f_stream)
                    except EOFError:
                        break
                    trajectories.append(traj)
            if num_episodes is not None and len(trajectories) != num_episodes:
                raise ValueError(
                    f"Streamed dataset ended early: expected {num_episodes}, got {len(trajectories)}."
                )
        else:
            raise ValueError(
                f"Unsupported exported dataset format: {self._exported_file}"
            )
        if not self._exported_names or len(self._exported_names) != len(trajectories):
            self._exported_names = [f"ep_{idx:05d}" for idx in range(len(trajectories))]
        self._exported_blob = payload
        self._exported_trajectories = trajectories


class ViewerRequestHandler(BaseHTTPRequestHandler):
    manager: EpisodeManager = None  # type: ignore
    index_html: bytes = b""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_index()
            return
        if path == "/api/episodes":
            self._send_json({"episodes": self.manager.list_episodes()})
            return
        if path == "/api/meta":
            self._handle_meta(parsed.query)
            return
        if path == "/api/frame":
            self._handle_frame(parsed.query)
            return
        if path == "/api/poses":
            self._handle_pose_series(parsed.query)
            return
        if path == "/api/joints":
            self._handle_joint_series(parsed.query)
            return
        if path == "/api/timestamps":
            self._handle_timestamps(parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def log_message(self, format: str, *args: Any) -> None:
        message = "%s - - [%s] %s\n" % (
            self.client_address[0],
            time.strftime("%d/%b/%Y %H:%M:%S"),
            format % args,
        )
        print(message, end="")

    def _handle_meta(self, query: str) -> None:
        params = parse_qs(query)
        episode = params.get("episode", [None])[0]
        if not episode:
            self._send_json({"error": "missing episode parameter"}, status=400)
            return
        try:
            data = self.manager.get(episode).meta()
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=404)
            return
        self._send_json(data)

    def _handle_frame(self, query: str) -> None:
        params = parse_qs(query)
        episode = params.get("episode", [None])[0]
        idx_param = params.get("index", [None])[0]
        if not episode or idx_param is None:
            self._send_json({"error": "episode and index are required"}, status=400)
            return
        try:
            index = int(idx_param)
        except ValueError:
            self._send_json({"error": "index must be an integer"}, status=400)
            return
        try:
            frame = self.manager.get(episode).frame_at(index)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=404)
            return
        if frame is None:
            self._send_json({"error": "frame unavailable"}, status=404)
            return
        self._send_json(frame)

    def _handle_pose_series(self, query: str) -> None:
        params = parse_qs(query)
        episode = params.get("episode", [None])[0]
        if not episode:
            self._send_json({"error": "episode parameter required"}, status=400)
            return
        try:
            series = self.manager.get(episode).pose_series_dict()
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=404)
            return
        self._send_json(series)

    def _handle_joint_series(self, query: str) -> None:
        params = parse_qs(query)
        episode = params.get("episode", [None])[0]
        if not episode:
            self._send_json({"error": "episode parameter required"}, status=400)
            return
        try:
            series = self.manager.get(episode).joint_series_dict()
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=404)
            return
        self._send_json(series)

    def _handle_timestamps(self, query: str) -> None:
        params = parse_qs(query)
        episode = params.get("episode", [None])[0]
        if not episode:
            self._send_json({"error": "episode parameter required"}, status=400)
            return
        try:
            ep = self.manager.get(episode)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=404)
            return
        start_param = params.get("start", [None])[0]
        end_param = params.get("end", [None])[0]
        try:
            start_offset_ns = (
                int(float(start_param) * 1e9) if start_param is not None else None
            )
        except (TypeError, ValueError):
            self._send_json({"error": "start must be a number (seconds)"}, status=400)
            return
        try:
            end_offset_ns = (
                int(float(end_param) * 1e9) if end_param is not None else None
            )
        except (TypeError, ValueError):
            self._send_json({"error": "end must be a number (seconds)"}, status=400)
            return

        origin_ns = ep.origin_ns
        start_ns = origin_ns + start_offset_ns if start_offset_ns is not None else None
        end_ns = origin_ns + end_offset_ns if end_offset_ns is not None else None
        if start_ns is not None and end_ns is not None and end_ns < start_ns:
            self._send_json({"error": "end must be >= start"}, status=400)
            return
        series = ep.timestamp_series(start_ns=start_ns, end_ns=end_ns)
        self._send_json(series)

    def _serve_index(self) -> None:
        if not self.index_html:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "HTML template missing")
            return
        content = self.index_html
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)


def load_html() -> bytes:
    try:
        return HTML_PATH.read_text(encoding="utf-8").encode("utf-8")
    except FileNotFoundError:
        return b""


def load_config() -> Dict[str, Any]:
    defaults = {
        "dataset": "datasets/test",
        "host": "127.0.0.1",
        "port": 8766,
    }
    if not CONFIG_PATH.exists():
        return defaults
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return defaults
    if isinstance(data, dict):
        defaults.update({k: data[k] for k in defaults.keys() if k in data})
    return defaults


def parse_args(config: Dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web viewer for CRISPControl datasets."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(config["dataset"]),
        help="Directory containing episode *.pkl files.",
    )
    parser.add_argument("--host", default=config["host"], help="Address to bind.")
    parser.add_argument(
        "--port", type=int, default=config["port"], help="Port to bind."
    )
    return parser.parse_args()


def main() -> None:
    config = load_config()
    args = parse_args(config)
    dataset_dir = args.dataset
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory {dataset_dir} does not exist.")
    manager = EpisodeManager(dataset_dir)
    if not manager.episodes:
        raise SystemExit(f"No .pkl episodes found in {dataset_dir}")
    ViewerRequestHandler.manager = manager
    ViewerRequestHandler.index_html = load_html()
    if not ViewerRequestHandler.index_html:
        raise SystemExit(f"HTML template missing at {HTML_PATH}")
    server = ThreadingHTTPServer((args.host, args.port), ViewerRequestHandler)
    print(f"Serving dataset viewer at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
