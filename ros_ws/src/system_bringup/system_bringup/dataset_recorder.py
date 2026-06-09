#!/usr/bin/env python3
"""
Dataset recorder node for logging synchronized ROS demonstrations.
"""

from __future__ import annotations

import pickle
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
from pynput import keyboard
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.parameter import ParameterValue
from rclpy.qos import QoSProfile
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, Image
from std_srvs.srv import Trigger


@dataclass
class TopicSpec:
    name: str
    type_string: str
    alias: str
    msg_type: Any
    queue_size: int = 10


class DatasetRecorder(Node):
    """Configurable ROS 2 node for logging demonstrations."""

    def __init__(self) -> None:
        super().__init__("dataset_recorder")

        # Parameters -------------------------------------------------------------------------
        self.declare_parameter("dataset_name", "demo")
        self.declare_parameter("output_root", "~/franka_datasets")
        string_array_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING_ARRAY
        )

        def declare_string_array(name: str):
            self.declare_parameter(
                name,
                ParameterValue(
                    type=ParameterType.PARAMETER_STRING_ARRAY,
                    string_array_value=[],
                ),
                descriptor=string_array_descriptor,
            )

        declare_string_array("topics")
        declare_string_array("topic_types")
        declare_string_array("topic_aliases")
        self.declare_parameter("autostart", False)
        declare_string_array("metadata_tags")
        self.declare_parameter("enable_keyboard_shortcuts", True)
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("camera_config", "")

        self.dataset_name = self.get_parameter("dataset_name").value
        output_root = self.get_parameter("output_root").value
        self.output_dir = Path(output_root).expanduser().resolve()
        self.dataset_dir = self.output_dir / self.dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"Dataset directory: {self.dataset_dir}")

        self.metadata_tags = list(self.get_parameter("metadata_tags").value)
        self.autostart = bool(self.get_parameter("autostart").value)
        self.default_queue_size = int(self.get_parameter("queue_size").value)
        camera_config_param = str(
            self.get_parameter("camera_config").value or ""
        ).strip()
        self.camera_config_path = camera_config_param

        topic_names = list(self.get_parameter("topics").value)
        topic_types = list(self.get_parameter("topic_types").value)
        topic_aliases = list(self.get_parameter("topic_aliases").value)
        if not topic_names or not topic_types:
            raise RuntimeError("No topics configured for dataset recording.")
        if len(topic_names) != len(topic_types):
            raise RuntimeError("topics and topic_types must have the same length.")
        if topic_aliases and len(topic_aliases) != len(topic_names):
            raise RuntimeError(
                "topic_aliases must be empty or match the length of topics."
            )

        self.topic_specs: List[TopicSpec] = []
        for idx, topic_name in enumerate(topic_names):
            alias = (
                topic_aliases[idx]
                if topic_aliases
                else self._sanitize_alias(topic_name)
            )
            type_string = topic_types[idx]
            try:
                msg_type = get_message(type_string)
            except (AttributeError, ModuleNotFoundError, ValueError) as exc:
                raise RuntimeError(
                    f'Failed to import message type "{type_string}"'
                ) from exc

            self.topic_specs.append(
                TopicSpec(
                    name=topic_name,
                    type_string=type_string,
                    alias=alias,
                    msg_type=msg_type,
                    queue_size=self.default_queue_size,
                )
            )

        self.topic_metadata = {
            spec.alias: {
                "topic": spec.name,
                "type": spec.type_string,
            }
            for spec in self.topic_specs
        }
        self.camera_metadata: Dict[int, Dict[str, Any]] = {}
        if self.camera_config_path:
            self.camera_metadata = self._parse_camera_config(self.camera_config_path)
        self._annotate_camera_topics()

        self.recording = False
        self.current_episode: Optional[Dict[str, Any]] = None
        self.current_episode_start_wall: Optional[float] = None
        self.log_lock = threading.Lock()

        self.next_episode_idx = self._compute_next_index()
        self._write_manifest()

        # Subscriptions ---------------------------------------------------------------------
        subscriptions = []
        qos = QoSProfile(depth=self.default_queue_size)
        for spec in self.topic_specs:
            callback = self._create_callback(spec)
            sub = self.create_subscription(
                spec.msg_type,
                spec.name,
                callback,
                qos_profile=qos,
            )
            subscriptions.append(sub)
            self.get_logger().debug(f'Subscribed to {spec.name} → alias "{spec.alias}"')
        self._subscriptions = subscriptions

        # Services (optional remote control) ------------------------------------------------
        service_prefix = self.get_fully_qualified_name().rstrip("/")
        self._service_names = {
            "start": f"{service_prefix}/start_recording",
            "stop": f"{service_prefix}/stop_recording",
            "delete": f"{service_prefix}/delete_last_episode",
        }

        self.start_srv = self.create_service(
            Trigger, self._service_names["start"], self._handle_start_service
        )
        self.stop_srv = self.create_service(
            Trigger, self._service_names["stop"], self._handle_stop_service
        )
        self.delete_srv = self.create_service(
            Trigger, self._service_names["delete"], self._handle_delete_service
        )

        # Keyboard shortcuts ---------------------------------------------------------------
        enable_keyboard = bool(self.get_parameter("enable_keyboard_shortcuts").value)
        if enable_keyboard:
            if keyboard is None:
                self.get_logger().warn(
                    "Keyboard shortcuts requested but pynput is not installed. "
                    "Install python3-pynput to enable hotkeys."
                )
            else:
                self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
                self._keyboard_listener.start()
                self.get_logger().info(
                    "Keyboard controls: SPACE start/stop, DELETE remove last trajectory."
                )

        if self.autostart:
            success, message = self._start_recording()
            if success:
                self.get_logger().info("Auto recording enabled.")
            else:
                self.get_logger().warn(f"Failed to autostart recording: {message}")

    # ----------------------------------------------------------------------------------
    # Utility helpers
    # ----------------------------------------------------------------------------------
    def _sanitize_alias(self, topic_name: str) -> str:
        alias = topic_name.strip("/").replace("/", "_")
        return alias or "topic"

    def _compute_next_index(self) -> int:
        pattern = re.compile(r"ep_(\d+)\.pkl")
        indices = []
        for file in self.dataset_dir.glob("ep_*.pkl"):
            match = pattern.match(file.name)
            if match:
                indices.append(int(match.group(1)))
        next_idx = max(indices) + 1 if indices else 0
        self.get_logger().info(f"Next episode index: {next_idx}")
        return next_idx

    def _write_manifest(self) -> None:
        manifest = {
            "dataset_name": self.dataset_name,
            "output_dir": str(self.dataset_dir),
            "topics": list(self.topic_metadata.values()),
            "metadata_tags": self.metadata_tags,
        }
        if self.camera_metadata:
            manifest["cameras"] = list(self.camera_metadata.values())
        manifest_path = self.dataset_dir / "dataset_manifest.yaml"
        try:
            import yaml
        except ImportError:
            self.get_logger().warn("pyyaml is not installed; skipping manifest export.")
            return

        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )

    def _parse_camera_config(self, config_path: str) -> Dict[int, Dict[str, Any]]:
        resolved_path = Path(config_path).expanduser()
        if not resolved_path.exists():
            self.get_logger().warn(f"Camera config file not found: {resolved_path}")
            return {}
        try:
            import yaml
        except ImportError:
            self.get_logger().warn(
                "pyyaml is not installed; skipping camera metadata import."
            )
            return {}
        try:
            data = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - requires filesystem failures
            self.get_logger().warn(
                f"Failed to load camera config {resolved_path}: {exc}"
            )
            return {}

        cameras: Dict[int, Dict[str, Any]] = {}
        for entry in data.get("cameras", []):
            idx = entry.get("camera_idx")
            if idx is None:
                continue
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            cameras[idx_int] = {
                "camera_idx": idx_int,
                "camera_name": entry.get(
                    "camera_name", entry.get("name", f"camera_{idx_int}")
                ),
                "serial_no": entry.get("serial_no"),
                "camera_namespace": entry.get("camera_namespace", ""),
                "node_name": entry.get("node_name"),
                "camera_index": entry.get("camera_index"),
                "description": entry.get("description", ""),
                "config_path": str(resolved_path),
            }
        if not cameras:
            self.get_logger().warn(
                f"No valid camera_idx entries found in camera config: {resolved_path}"
            )
        return cameras

    def _annotate_camera_topics(self) -> None:
        if not self.camera_metadata:
            return
        for alias, metadata in self.topic_metadata.items():
            parts = alias.split("/")
            if len(parts) < 2 or parts[0] != "cameras":
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            camera_entry = self.camera_metadata.get(idx)
            if not camera_entry:
                continue
            channel = "/".join(parts[2:]) if len(parts) > 2 else ""
            metadata["camera"] = {
                **camera_entry,
                "channel": channel,
            }

    # ----------------------------------------------------------------------------------
    # Recording control
    # ----------------------------------------------------------------------------------
    def _start_recording(self) -> (bool, str):
        with self.log_lock:
            if self.recording:
                return False, "already recording"
            self.current_episode = {
                "data": {spec.alias: [] for spec in self.topic_specs},
                "timestamps": {spec.alias: [] for spec in self.topic_specs},
                "all_timestamps": [],
                "metadata": {
                    "dataset_name": self.dataset_name,
                    "topics": self.topic_metadata,
                    "tags": self.metadata_tags,
                },
            }
            self.current_episode_start_wall = time.time()
            self.recording = True
        self.get_logger().info("Recording started (press SPACE to stop).")
        return True, "recording started"

    def _stop_recording(self) -> (bool, str):
        with self.log_lock:
            if not self.recording or self.current_episode is None:
                return False, "not currently recording"
            if not any(
                len(samples) for samples in self.current_episode["data"].values()
            ):
                self.recording = False
                self.current_episode = None
                return False, "no messages recorded, skipping save"

            self.current_episode["metadata"][
                "start_wall_time"
            ] = self.current_episode_start_wall
            self.current_episode["metadata"]["end_wall_time"] = time.time()
            episode = self.current_episode
            self.recording = False
            self.current_episode = None

        output_path = self.dataset_dir / f"ep_{self.next_episode_idx:05d}.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(episode, handle, protocol=pickle.HIGHEST_PROTOCOL)

        duration = (
            episode["metadata"]["end_wall_time"]
            - episode["metadata"]["start_wall_time"]
            if episode["metadata"].get("end_wall_time")
            and episode["metadata"].get("start_wall_time")
            else 0.0
        )
        total_messages = len(episode["all_timestamps"])
        self.get_logger().info(
            f"Saved {output_path.name} ({duration:.2f}s, {total_messages} samples)."
        )
        self.next_episode_idx += 1
        return True, f"saved {output_path.name}"

    def _delete_last_episode(self) -> (bool, str):
        if self.recording:
            return False, "stop recording before deleting episodes"
        pattern = re.compile(r"ep_(\d+)\.pkl")
        episodes = sorted(self.dataset_dir.glob("ep_*.pkl"))
        if not episodes:
            return False, "no recorded episodes to delete"
        last_file = episodes[-1]
        match = pattern.match(last_file.name)
        last_file.unlink()
        self.get_logger().info(f"Deleted {last_file.name}")
        if match:
            self.next_episode_idx = int(match.group(1))
        return True, f"deleted {last_file.name}"

    # ----------------------------------------------------------------------------------
    # ROS interfaces
    # ----------------------------------------------------------------------------------
    def _create_callback(self, spec: TopicSpec):
        def callback(msg):
            if not self.recording or self.current_episode is None:
                return
            data_entry = self._process_message(msg, spec)
            if data_entry is None:
                return
            timestamp_ns = self.get_clock().now().nanoseconds
            with self.log_lock:
                if not self.recording or self.current_episode is None:
                    return
                self.current_episode["data"][spec.alias].append(data_entry)
                self.current_episode["timestamps"][spec.alias].append(timestamp_ns)
                self.current_episode["all_timestamps"].append(timestamp_ns)

        return callback

    def _process_message(self, msg, spec: TopicSpec) -> Optional[Any]:
        if isinstance(msg, CompressedImage):
            return {
                "header": message_to_ordereddict(msg.header),
                "format": msg.format,
                "data": bytes(msg.data),
            }
        if isinstance(msg, Image):
            return {
                "header": message_to_ordereddict(msg.header),
                "height": msg.height,
                "width": msg.width,
                "encoding": msg.encoding,
                "is_bigendian": msg.is_bigendian,
                "step": msg.step,
                "data": bytes(msg.data),
            }
        return message_to_ordereddict(msg)

    # ----------------------------------------------------------------------------------
    # Keyboard + service callbacks
    # ----------------------------------------------------------------------------------
    def _on_key_press(self, key):  # pragma: no cover - relies on pynput runtime
        try:
            if key == keyboard.Key.space:
                if self.recording:
                    self._stop_recording()
                else:
                    self._start_recording()
            elif key in (keyboard.Key.delete, keyboard.Key.backspace):
                self._delete_last_episode()
        except AttributeError:
            pass

    def _handle_start_service(self, _request, response):
        success, message = self._start_recording()
        response.success = success
        response.message = message
        return response

    def _handle_stop_service(self, _request, response):
        success, message = self._stop_recording()
        response.success = success
        response.message = message
        return response

    def _handle_delete_service(self, _request, response):
        success, message = self._delete_last_episode()
        response.success = success
        response.message = message
        return response


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording:
            node._stop_recording()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
