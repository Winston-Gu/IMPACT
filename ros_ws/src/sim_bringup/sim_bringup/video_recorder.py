#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32


class VideoRecorder(Node):
    """Record per-episode MP4s from a single RGB topic."""

    def __init__(self) -> None:
        super().__init__("video_recorder")

        self.declare_parameter(
            "image_topic", "/camera/third_person_view/color/image_rect_raw"
        )
        self.declare_parameter("episode_start_topic", "/episode/start")
        self.declare_parameter("episode_end_topic", "/episode/end")
        self.declare_parameter("log_name", "demo")
        self.declare_parameter("output_root", "logs")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("fourcc", "avc1")

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.episode_start_topic = str(self.get_parameter("episode_start_topic").value)
        self.episode_end_topic = str(self.get_parameter("episode_end_topic").value)
        self.log_name = str(self.get_parameter("log_name").value)
        self.output_root = str(self.get_parameter("output_root").value)
        self.fps = float(self.get_parameter("fps").value)
        self.fourcc = str(self.get_parameter("fourcc").value)

        if not self.log_name:
            self.log_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_episode: Optional[int] = None
        self._last_shape: Optional[tuple[int, int]] = None
        self._warned_encoding = False
        self._start_stamp_ns: Optional[int] = None

        self.create_subscription(Image, self.image_topic, self._on_image, 10)
        self.create_subscription(
            Int32, self.episode_start_topic, self._on_episode_start, 10
        )
        self.create_subscription(
            Int32, self.episode_end_topic, self._on_episode_end, 10
        )

        self.get_logger().info(
            f"Recording {self.image_topic} to logs/{self.log_name}/ep_*.mp4"
        )

    def _episode_path(self, idx: int) -> Path:
        root = Path(self.output_root).expanduser().resolve()
        target = root / self.log_name
        target.mkdir(parents=True, exist_ok=True)
        return target / f"ep_{idx}.mp4"

    def _open_writer(self, width: int, height: int, idx: int) -> None:
        if self._writer is not None:
            return
        output_path = self._episode_path(idx)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(
            os.fspath(output_path), fourcc, self.fps, (width, height)
        )
        if not self._writer.isOpened():
            self.get_logger().error(
                f"Failed to open video writer with codec {self.fourcc}: {output_path}"
            )
            self._writer = None
        else:
            self.get_logger().info(f"Recording episode {idx} to {output_path}")

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _on_episode_start(self, msg: Int32) -> None:
        if self._current_episode is not None:
            self.get_logger().warn(
                f"Episode {self._current_episode} already recording; closing."
            )
            self._close_writer()
        self._current_episode = int(msg.data)
        self._start_stamp_ns = self.get_clock().now().nanoseconds
        if self._last_shape is not None:
            self._open_writer(
                self._last_shape[1], self._last_shape[0], self._current_episode
            )

    def _on_episode_end(self, msg: Int32) -> None:
        if self._current_episode != int(msg.data):
            return
        self._close_writer()
        self.get_logger().info(f"Finished episode {self._current_episode}")
        self._current_episode = None
        self._start_stamp_ns = None

    def _on_image(self, msg: Image) -> None:
        if msg.encoding not in ("bgr8", "rgb8"):
            if not self._warned_encoding:
                self.get_logger().warn(
                    f"Unsupported image encoding '{msg.encoding}'; expected bgr8 or rgb8."
                )
                self._warned_encoding = True
            return
        if self._start_stamp_ns is not None:
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            if stamp_ns < self._start_stamp_ns:
                return

        height = int(msg.height)
        width = int(msg.width)
        expected = height * width * 3
        if len(msg.data) < expected:
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8)[:expected]
        frame = frame.reshape((height, width, 3))
        if msg.encoding == "rgb8":
            frame = frame[:, :, ::-1]

        self._last_shape = (height, width)
        if self._current_episode is None:
            return

        if self._writer is None:
            self._open_writer(width, height, self._current_episode)
        if self._writer is not None:
            self._writer.write(frame)


def main() -> None:
    rclpy.init()
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
