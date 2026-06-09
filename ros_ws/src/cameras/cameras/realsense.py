from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


@dataclass
class CameraParameters:
    serial: str
    camera_name: str
    camera_namespace: str
    base_frame_id: str
    color_frame_id: str
    depth_frame_id: str
    color_resolution: tuple[int, int, int]
    depth_resolution: tuple[int, int, int]
    publish_color: bool
    publish_depth: bool
    publish_compressed: bool
    align_depth: bool
    clip_distance: float
    initial_reset: bool
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int

    @property
    def topic_prefix(self) -> str:
        ns = self.camera_namespace.strip("/")
        prefix = f"/{ns}" if ns else ""
        return f"{prefix}/{self.camera_name}".replace("//", "/")

    @property
    def color_topic(self) -> str:
        return f"{self.topic_prefix}/color/image_rect_raw"

    @property
    def color_info_topic(self) -> str:
        return f"{self.topic_prefix}/color/camera_info"

    @property
    def color_compressed_topic(self) -> str:
        return f"{self.color_topic}/compressed"

    @property
    def depth_topic(self) -> str:
        return f"{self.topic_prefix}/depth/image_rect_raw"

    @property
    def depth_info_topic(self) -> str:
        return f"{self.topic_prefix}/depth/camera_info"


class RealSenseNode(Node):
    """Single RealSense camera bridge publishing ROS image streams."""

    def __init__(self) -> None:
        super().__init__("realsense_camera")
        self._declare_parameters()
        spec = self._resolve_camera_parameters()
        self.bridge = CvBridge()

        self.pipeline: rs.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(spec.serial)
        config.enable_stream(
            rs.stream.color,
            spec.color_resolution[0],
            spec.color_resolution[1],
            rs.format.bgr8,
            spec.color_resolution[2],
        )
        if spec.publish_depth:
            config.enable_stream(
                rs.stream.depth,
                spec.depth_resolution[0],
                spec.depth_resolution[1],
                rs.format.z16,
                spec.depth_resolution[2],
            )

        profile = self.pipeline.start(config)
        if spec.initial_reset:
            self.get_logger().info("Performing hardware reset...")
            device = profile.get_device()
            device.hardware_reset()
            self.pipeline.stop()
            profile = self.pipeline.start(config)

        self.depth_scale = 1.0
        if spec.publish_depth:
            self.depth_scale = (
                profile.get_device().first_depth_sensor().get_depth_scale()
            )
        self.align = (
            rs.align(rs.stream.color)
            if spec.align_depth and spec.publish_depth
            else None
        )

        self.spec = spec
        self.color_pub = self.create_publisher(Image, spec.color_topic, 10)
        self.color_info_pub = self.create_publisher(
            CameraInfo, spec.color_info_topic, 10
        )
        self.color_compressed_pub = None
        if spec.publish_color and spec.publish_compressed:
            self.color_compressed_pub = self.create_publisher(
                CompressedImage, spec.color_compressed_topic, 10
            )

        self.depth_pub = None
        self.depth_info_pub = None
        if spec.publish_depth:
            self.depth_pub = self.create_publisher(Image, spec.depth_topic, 10)
            self.depth_info_pub = self.create_publisher(
                CameraInfo, spec.depth_info_topic, 10
            )

        period = 1.0 / max(spec.color_resolution[2], 1)
        self.timer = self.create_timer(period, self._capture_callback)
        self.get_logger().info(
            f"Publishing RealSense serial {spec.serial} to {spec.color_topic} (color) "
            f"and {spec.depth_topic} (depth)."
        )

    # ----------------------------------------------------------------------------------
    # Parameter helpers
    # ----------------------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("serial", "")
        self.declare_parameter("camera_name", "realsense")
        self.declare_parameter("camera_namespace", "camera")
        self.declare_parameter("base_frame_id", "")
        self.declare_parameter("color_frame_id", "")
        self.declare_parameter("depth_frame_id", "")
        self.declare_parameter("color_width", 640)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("color_fps", 30)
        self.declare_parameter("depth_width", 640)
        self.declare_parameter("depth_height", 480)
        self.declare_parameter("depth_fps", 30)
        self.declare_parameter("publish_color", True)
        self.declare_parameter("publish_depth", True)
        self.declare_parameter("publish_compressed", True)
        self.declare_parameter("align_depth", True)
        self.declare_parameter("clip_distance", 3.0)
        self.declare_parameter("initial_reset", False)
        self.declare_parameter("crop_x", 0)
        self.declare_parameter("crop_y", 0)
        self.declare_parameter("crop_width", 0)
        self.declare_parameter("crop_height", 0)

    def _resolve_camera_parameters(self) -> CameraParameters:
        serial = str(self.get_parameter("serial").value or "").strip()
        if not serial:
            raise RuntimeError("RealSense serial parameter is required.")
        camera_name = str(self.get_parameter("camera_name").value or "realsense")
        namespace = str(self.get_parameter("camera_namespace").value or "camera")
        base_frame = str(
            self.get_parameter("base_frame_id").value or f"{camera_name}_link"
        )
        color_frame = str(
            self.get_parameter("color_frame_id").value
            or f"{camera_name}_color_optical_frame"
        )
        depth_frame = str(
            self.get_parameter("depth_frame_id").value
            or f"{camera_name}_depth_optical_frame"
        )
        color_res = (
            int(self.get_parameter("color_width").value or 640),
            int(self.get_parameter("color_height").value or 480),
            int(self.get_parameter("color_fps").value or 30),
        )
        depth_res = (
            int(self.get_parameter("depth_width").value or color_res[0]),
            int(self.get_parameter("depth_height").value or color_res[1]),
            int(self.get_parameter("depth_fps").value or color_res[2]),
        )
        return CameraParameters(
            serial=serial,
            camera_name=camera_name,
            camera_namespace=namespace,
            base_frame_id=base_frame,
            color_frame_id=color_frame,
            depth_frame_id=depth_frame,
            color_resolution=color_res,
            depth_resolution=depth_res,
            publish_color=bool(self.get_parameter("publish_color").value),
            publish_depth=bool(self.get_parameter("publish_depth").value),
            publish_compressed=bool(self.get_parameter("publish_compressed").value),
            align_depth=bool(self.get_parameter("align_depth").value),
            clip_distance=float(self.get_parameter("clip_distance").value or 0.0),
            initial_reset=bool(self.get_parameter("initial_reset").value),
            crop_x=int(self.get_parameter("crop_x").value or 0),
            crop_y=int(self.get_parameter("crop_y").value or 0),
            crop_width=int(self.get_parameter("crop_width").value or 0),
            crop_height=int(self.get_parameter("crop_height").value or 0),
        )

    # ----------------------------------------------------------------------------------
    # Publishing helpers
    # ----------------------------------------------------------------------------------
    def _capture_callback(self) -> None:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError as exc:
            self.get_logger().warn(f"Frame grab failed: {exc}")
            return
        if self.align:
            frames = self.align.process(frames)

        stamp = self.get_clock().now().to_msg()
        if self.spec.publish_color:
            color_frame = frames.get_color_frame()
            if color_frame:
                self._publish_color_frame(color_frame, stamp)
        if self.spec.publish_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                self._publish_depth_frame(depth_frame, stamp)

    def _publish_color_frame(self, color_frame: rs.video_frame, stamp) -> None:
        image = np.asanyarray(color_frame.get_data())
        image, crop_x, crop_y = self._apply_crop(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding="rgb8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.spec.color_frame_id
        self.color_pub.publish(image_msg)
        info = self._camera_info_from_frame(
            color_frame,
            image_msg.header,
            crop_x=crop_x,
            crop_y=crop_y,
            cropped_width=image.shape[1],
            cropped_height=image.shape[0],
        )
        self.color_info_pub.publish(info)
        if self.color_compressed_pub:
            compressed = CompressedImage()
            compressed.header = image_msg.header
            compressed.format = "jpeg"
            ok, encoded = cv2.imencode(".jpg", image)
            if ok:
                compressed.data = encoded.tobytes()
                self.color_compressed_pub.publish(compressed)
            else:
                self.get_logger().warn("Failed to JPEG encode color frame.")

    def _publish_depth_frame(self, depth_frame: rs.depth_frame, stamp) -> None:
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_meters = depth_image * self.depth_scale
        if self.spec.clip_distance > 0:
            depth_meters = np.where(
                depth_meters > self.spec.clip_distance, 0.0, depth_meters
            )
        depth_meters, crop_x, crop_y = self._apply_crop(depth_meters)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_meters, encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.spec.depth_frame_id
        if self.depth_pub:
            self.depth_pub.publish(depth_msg)
        if self.depth_info_pub:
            info = self._camera_info_from_frame(
                depth_frame,
                depth_msg.header,
                crop_x=crop_x,
                crop_y=crop_y,
                cropped_width=depth_meters.shape[1],
                cropped_height=depth_meters.shape[0],
            )
            self.depth_info_pub.publish(info)

    def _apply_crop(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        crop_w = self.spec.crop_width
        crop_h = self.spec.crop_height
        if crop_w <= 0 or crop_h <= 0:
            return image, 0, 0
        height, width = image.shape[:2]
        x0 = max(self.spec.crop_x, 0)
        y0 = max(self.spec.crop_y, 0)
        x1 = min(width, x0 + crop_w)
        y1 = min(height, y0 + crop_h)
        if x1 <= x0 or y1 <= y0:
            if not getattr(self, "_crop_warned", False):
                self.get_logger().warn("Invalid crop region; skipping crop.")
                self._crop_warned = True
            return image, 0, 0
        return image[y0:y1, x0:x1], x0, y0

    def _camera_info_from_frame(
        self,
        frame,
        header,
        crop_x: int = 0,
        crop_y: int = 0,
        cropped_width: int | None = None,
        cropped_height: int | None = None,
    ) -> CameraInfo:
        profile = frame.profile.as_video_stream_profile()
        intrinsics = profile.get_intrinsics()
        info = CameraInfo()
        info.header = header
        info.height = (
            cropped_height if cropped_height is not None else intrinsics.height
        )
        info.width = cropped_width if cropped_width is not None else intrinsics.width
        info.distortion_model = "plumb_bob"
        info.d = list(intrinsics.coeffs[:5])
        ppx = intrinsics.ppx - float(crop_x)
        ppy = intrinsics.ppy - float(crop_y)
        info.k = [
            intrinsics.fx,
            0.0,
            ppx,
            0.0,
            intrinsics.fy,
            ppy,
            0.0,
            0.0,
            1.0,
        ]
        info.p = [
            intrinsics.fx,
            0.0,
            ppx,
            0.0,
            0.0,
            intrinsics.fy,
            ppy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        return info

    def destroy_node(self) -> bool:
        try:
            self.pipeline.stop()
        except Exception:  # pragma: no cover - hardware dependent
            pass
        self.get_logger().info("RealSense pipeline stopped.")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
