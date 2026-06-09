import argparse
import csv
import os
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import cv2
import dill
import hydra
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray, Int32
from std_srvs.srv import Trigger

from impact.utils.pytorch_util import dict_apply


def load_policy_from_checkpoint(
    checkpoint_path: str,
    device,
    prefer_ema: bool = True,
    num_inference_steps: int | None = None,
):
    path = Path(checkpoint_path).expanduser()
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    if num_inference_steps is not None and num_inference_steps > 0:
        cfg.policy.num_inference_steps = int(num_inference_steps)
    state_dicts = payload["state_dicts"]

    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if prefer_ema and "ema_model" in state_dicts else "model"
    if key not in state_dicts:
        raise RuntimeError(f"Checkpoint missing policy weights for '{key}'.")
    policy.load_state_dict(state_dicts[key])
    policy.to(device)
    policy.eval()
    return policy


def _pose_to_array(msg: PoseStamped) -> np.ndarray:
    pose = msg.pose
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float32,
    )


class ImpactPolicyNode(Node):
    """ROS 2 node that runs a diffusion policy and publishes target commands."""

    def __init__(self) -> None:
        super().__init__("impact_policy_runner")

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("control_rate", 200.0)
        self.declare_parameter("n_obs_steps", 2)
        self.declare_parameter("resize_height", 256)
        self.declare_parameter("resize_width", 256)
        self.declare_parameter("use_compressed", True)
        self.declare_parameter("max_translation_speed", 0.1)
        self.declare_parameter("action_is_delta", False)
        self.declare_parameter("num_inference_steps", 0)
        self.declare_parameter("episode_start_topic", "/episode/start")
        self.declare_parameter("episode_end_topic", "/episode/end")
        self.declare_parameter("episode_duration", 15.0)
        self.declare_parameter("cube_pose_topic", "/mujoco_objects/pick_cube_pose")
        self.declare_parameter("basket_pose_topic", "/mujoco_objects/basket_pose")
        self.declare_parameter("episodes", 0)
        self.declare_parameter("enable_episode_timer", True)
        self.declare_parameter("benchmark_csv_path", "")
        self.declare_parameter("reset_service", "/mujoco_sim/reset_scene")
        self.declare_parameter("reset_timeout", 10.0)
        self.declare_parameter("reset_retry_delay", 0.5)
        self.declare_parameter("object_mass_service", "/mujoco_sim/get_object_mass")
        self.declare_parameter("post_reset_wait", 2.0)
        self.declare_parameter("reset_pose_timeout", 5.0)
        self.declare_parameter("reset_to_home", True)
        self.declare_parameter("open_gripper_on_reset", True)
        self.declare_parameter("gripper_open_value", 1.0)
        self.declare_parameter(
            "front_camera_topic", "/camera/front_camera/color/image_rect_raw"
        )
        self.declare_parameter("side_camera_topic", "")
        self.declare_parameter("current_pose_topic", "/current_pose")
        self.declare_parameter("joint_state_topic", "/franka/joint_states")
        self.declare_parameter("gripper_state_topic", "/gripper/joint_states")
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter(
            "gripper_command_topic", "/gripper/gripper_position_controller/commands"
        )
        self.declare_parameter("command_frame", "base")

        checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        if not checkpoint_path:
            raise RuntimeError("checkpoint_path parameter is required.")
        self._checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)

        self.device = torch.device(self.get_parameter("device").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.n_obs_steps = int(self.get_parameter("n_obs_steps").value)
        self.resize_height = int(self.get_parameter("resize_height").value)
        self.resize_width = int(self.get_parameter("resize_width").value)
        self.use_compressed = bool(self.get_parameter("use_compressed").value)
        self.max_translation_speed = float(
            self.get_parameter("max_translation_speed").value
        )
        self.action_is_delta = bool(self.get_parameter("action_is_delta").value)
        self.num_inference_steps = int(self.get_parameter("num_inference_steps").value)
        self.episode_start_topic = str(self.get_parameter("episode_start_topic").value)
        self.episode_end_topic = str(self.get_parameter("episode_end_topic").value)
        self.episode_duration = float(self.get_parameter("episode_duration").value)
        self.cube_pose_topic = str(self.get_parameter("cube_pose_topic").value)
        self.basket_pose_topic = str(self.get_parameter("basket_pose_topic").value)
        self.episodes = int(self.get_parameter("episodes").value)
        self.enable_episode_timer = bool(
            self.get_parameter("enable_episode_timer").value
        )
        self.benchmark_csv_path = str(self.get_parameter("benchmark_csv_path").value)
        self.reset_service = str(self.get_parameter("reset_service").value)
        self.reset_timeout = float(self.get_parameter("reset_timeout").value)
        self.reset_retry_delay = float(self.get_parameter("reset_retry_delay").value)
        self.object_mass_service = str(self.get_parameter("object_mass_service").value)
        self.post_reset_wait = float(self.get_parameter("post_reset_wait").value)
        self.reset_pose_timeout = float(self.get_parameter("reset_pose_timeout").value)
        self.reset_to_home = bool(self.get_parameter("reset_to_home").value)
        self.open_gripper_on_reset = bool(
            self.get_parameter("open_gripper_on_reset").value
        )
        self.gripper_open_value = float(self.get_parameter("gripper_open_value").value)

        self.front_topic = str(self.get_parameter("front_camera_topic").value)
        self.side_topic = str(self.get_parameter("side_camera_topic").value)
        self.pose_topic = str(self.get_parameter("current_pose_topic").value)
        self.joint_topic = str(self.get_parameter("joint_state_topic").value)
        self.gripper_topic = str(self.get_parameter("gripper_state_topic").value)
        self.target_topic = str(self.get_parameter("target_pose_topic").value)
        self.gripper_cmd_topic = str(self.get_parameter("gripper_command_topic").value)
        self.command_frame = str(self.get_parameter("command_frame").value)

        self.bridge = CvBridge()
        self.front_frames: Deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self.side_frames: Deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self.proprio_steps: Deque[np.ndarray] = deque(maxlen=self.n_obs_steps)

        self.latest_pose: Optional[PoseStamped] = None
        self.latest_joints: Optional[JointState] = None
        self.latest_gripper: Optional[JointState] = None
        self.home_pose: Optional[PoseStamped] = None
        self.cube_pose: Optional[PoseStamped] = None
        self.basket_pose: Optional[PoseStamped] = None
        self._initial_cube_pose: Optional[PoseStamped] = None
        self._initial_basket_pose: Optional[PoseStamped] = None
        self._initial_mass: Optional[float] = None
        self._await_initial_pose = False
        self._current_episode_idx: Optional[int] = None
        self._last_debug_time = 0.0
        self._reset_until: Optional[float] = None
        self._reset_hold_pose: Optional[PoseStamped] = None
        self._episode_active = not self.enable_episode_timer
        self._action_queue: Deque[np.ndarray] = deque()
        self._success_count = 0
        self._episode_count = 0
        self._last_ready_log = 0.0

        qos = qos_profile_sensor_data
        qos_depth = QoSProfile(depth=10)

        if self.use_compressed:
            self.create_subscription(
                CompressedImage, self.front_topic + "/compressed", self._front_cb, qos
            )
        else:
            self.create_subscription(Image, self.front_topic, self._front_cb, qos)
        if self.side_topic:
            if self.use_compressed:
                self.create_subscription(
                    CompressedImage,
                    self.side_topic + "/compressed",
                    self._side_cb,
                    qos,
                )
            else:
                self.create_subscription(Image, self.side_topic, self._side_cb, qos)

        self.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, qos_depth)
        self.create_subscription(
            JointState, self.joint_topic, self._joint_cb, qos_depth
        )
        self.create_subscription(
            JointState, self.gripper_topic, self._gripper_cb, qos_depth
        )
        self.create_subscription(
            PoseStamped, self.cube_pose_topic, self._on_cube_pose, qos_depth
        )
        self.create_subscription(
            PoseStamped, self.basket_pose_topic, self._on_basket_pose, qos_depth
        )

        self.target_pub = self.create_publisher(PoseStamped, self.target_topic, 10)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, self.gripper_cmd_topic, 10
        )
        self.episode_start_pub = self.create_publisher(
            Int32, self.episode_start_topic, 10
        )
        self.episode_end_pub = self.create_publisher(Int32, self.episode_end_topic, 10)

        self._load_policy(checkpoint_path)
        torch.backends.cudnn.benchmark = True

        self.reset_client = self.create_client(Trigger, self.reset_service)
        self.object_mass_client = self.create_client(Trigger, self.object_mass_service)

        self.timer = self.create_timer(1.0 / self.control_rate, self._step)
        self.get_logger().info("IMPACT policy runner started.")

    def _load_policy(self, checkpoint_path: str) -> None:
        self.policy = load_policy_from_checkpoint(
            str(Path(checkpoint_path).expanduser()),
            self.device,
            prefer_ema=True,
            num_inference_steps=self.num_inference_steps
            if self.num_inference_steps > 0
            else None,
        )

    def _clear_observation_buffers(self) -> None:
        self.front_frames.clear()
        self.side_frames.clear()
        self.proprio_steps.clear()
        self._action_queue.clear()

    def _clear_latest_reset_state(self) -> None:
        self.latest_pose = None
        self.latest_joints = None
        self.latest_gripper = None
        self.cube_pose = None
        self.basket_pose = None

    def _handle_episode_reset(self) -> None:
        self._clear_observation_buffers()
        self._reset_until = time.monotonic() + self.post_reset_wait
        self._publish_reset_hold_command()

    def _publish_reset_hold_command(self) -> None:
        if self.reset_to_home and self._reset_hold_pose is not None:
            home_msg = PoseStamped()
            home_msg.header.stamp = self.get_clock().now().to_msg()
            home_msg.header.frame_id = (
                self._reset_hold_pose.header.frame_id or self.command_frame
            )
            home_msg.pose = self._reset_hold_pose.pose
            self.target_pub.publish(home_msg)
        if self.open_gripper_on_reset:
            msg = Float64MultiArray()
            msg.data = [self.gripper_open_value]
            self.gripper_pub.publish(msg)

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.latest_pose = msg
        if self.home_pose is None:
            self.home_pose = msg
        self._maybe_update_proprio()

    def _on_cube_pose(self, msg: PoseStamped) -> None:
        self.cube_pose = msg
        if self._await_initial_pose and self._initial_cube_pose is None:
            self._initial_cube_pose = msg
            self._maybe_finalize_initial_pose()

    def _on_basket_pose(self, msg: PoseStamped) -> None:
        self.basket_pose = msg
        if self._await_initial_pose and self._initial_basket_pose is None:
            self._initial_basket_pose = msg
            self._maybe_finalize_initial_pose()

    def _joint_cb(self, msg: JointState) -> None:
        self.latest_joints = msg
        self._maybe_update_proprio()

    def _gripper_cb(self, msg: JointState) -> None:
        self.latest_gripper = msg
        self._maybe_update_proprio()

    def _maybe_update_proprio(self) -> None:
        if (
            self.latest_pose is None
            or self.latest_joints is None
            or self.latest_gripper is None
        ):
            return
        pose_arr = _pose_to_array(self.latest_pose)
        joint_arr = np.asarray(self.latest_joints.position, dtype=np.float32)
        gripper_arr = np.asarray(self.latest_gripper.position, dtype=np.float32)
        if gripper_arr.size > 1:
            gripper_arr = np.array([np.sum(gripper_arr[:2])], dtype=np.float32)
        proprio = np.concatenate([pose_arr, joint_arr, gripper_arr], axis=0)
        self.proprio_steps.append(proprio)

    def _decode_image(self, msg) -> Optional[np.ndarray]:
        if isinstance(msg, CompressedImage):
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if self.resize_height > 0 and self.resize_width > 0:
            frame = cv2.resize(
                frame,
                (self.resize_width, self.resize_height),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _front_cb(self, msg) -> None:
        frame = self._decode_image(msg)
        if frame is not None:
            self.front_frames.append(frame)

    def _side_cb(self, msg) -> None:
        frame = self._decode_image(msg)
        if frame is not None:
            self.side_frames.append(frame)

    def _ready(self) -> bool:
        ready = (
            len(self.front_frames) == self.n_obs_steps
            and len(self.proprio_steps) == self.n_obs_steps
        )
        if self.side_topic:
            ready = ready and len(self.side_frames) == self.n_obs_steps
        return ready

    def _build_obs(self) -> Dict[str, np.ndarray]:
        front = np.stack(list(self.front_frames), axis=0)
        proprio = np.stack(list(self.proprio_steps), axis=0)
        front = np.moveaxis(front, -1, 1).astype(np.float32) / 255.0
        obs = {
            "front_camera": front,
            "proprioception": proprio.astype(np.float32),
        }
        if self.side_topic:
            side = np.stack(list(self.side_frames), axis=0)
            side = np.moveaxis(side, -1, 1).astype(np.float32) / 255.0
            obs["side_camera"] = side
        return obs

    def _maybe_finalize_initial_pose(self) -> None:
        if self._initial_cube_pose is None or self._initial_basket_pose is None:
            return
        self._await_initial_pose = False

    def _query_object_mass(self) -> Optional[float]:
        if not self.object_mass_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Object mass service not available")
            return None
        future = self.object_mass_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            message = response.message if response else "No response"
            self.get_logger().warn(f"Failed to query object mass: {message}")
            return None
        try:
            return float(response.message)
        except ValueError:
            self.get_logger().warn(
                f"Invalid mass value from service: {response.message}"
            )
            return None

    def _write_benchmark_row(self, success: bool) -> None:
        if not self.benchmark_csv_path:
            return
        if self._initial_cube_pose is None and self.cube_pose is not None:
            self._initial_cube_pose = self.cube_pose
        if self._initial_basket_pose is None and self.basket_pose is not None:
            self._initial_basket_pose = self.basket_pose
        if self._initial_cube_pose is None or self._initial_basket_pose is None:
            self.get_logger().warn(
                "Missing initial cube/basket pose for benchmark logging."
            )
        csv_path = Path(self.benchmark_csv_path).expanduser()
        if not csv_path.is_absolute():
            csv_path = self._checkpoint_dir / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "episode",
            "cube_x",
            "cube_y",
            "cube_z",
            "basket_x",
            "basket_y",
            "basket_z",
            "cube_mass",
            "success",
        ]
        row = {
            "episode": self._current_episode_idx
            if self._current_episode_idx is not None
            else self._episode_count - 1,
            "cube_x": float(self._initial_cube_pose.pose.position.x)
            if self._initial_cube_pose is not None
            else float("nan"),
            "cube_y": float(self._initial_cube_pose.pose.position.y)
            if self._initial_cube_pose is not None
            else float("nan"),
            "cube_z": float(self._initial_cube_pose.pose.position.z)
            if self._initial_cube_pose is not None
            else float("nan"),
            "basket_x": float(self._initial_basket_pose.pose.position.x)
            if self._initial_basket_pose is not None
            else float("nan"),
            "basket_y": float(self._initial_basket_pose.pose.position.y)
            if self._initial_basket_pose is not None
            else float("nan"),
            "basket_z": float(self._initial_basket_pose.pose.position.z)
            if self._initial_basket_pose is not None
            else float("nan"),
            "cube_mass": self._initial_mass
            if self._initial_mass is not None
            else float("nan"),
            "success": int(success),
        }
        is_new = not csv_path.exists() or os.path.getsize(csv_path) == 0
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def is_cube_in_basket(self) -> bool:
        if self.cube_pose is None or self.basket_pose is None:
            return False
        basket_pos = self.basket_pose.pose.position
        cube_pos = self.cube_pose.pose.position

        inner_half_xy = 0.1875
        within_xy = (
            abs(cube_pos.x - basket_pos.x) <= inner_half_xy
            and abs(cube_pos.y - basket_pos.y) <= inner_half_xy
        )
        return within_xy

    def reset_scene(self) -> bool:
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Reset service not available")
            return False
        future = self.reset_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            message = response.message if response else "No response"
            self.get_logger().error(f"Reset scene failed: {message}")
            return False
        return True

    def reset_scene_with_retry(self) -> bool:
        start = time.time()
        while rclpy.ok():
            if self.reset_scene():
                return True
            if time.time() - start > self.reset_timeout:
                self.get_logger().error("Reset scene timed out")
                return False
            time.sleep(self.reset_retry_delay)
        return False

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def _message_after_reset(self, msg, reset_ns: int) -> bool:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return True
        stamp_ns = self._stamp_to_ns(stamp)
        if stamp_ns == 0:
            return True
        return stamp_ns > reset_ns

    def _wait_for_fresh_reset_state(self, reset_time) -> bool:
        start = time.monotonic()
        reset_ns = self._stamp_to_ns(reset_time)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            pose_ready = self.latest_pose is not None and self._message_after_reset(
                self.latest_pose, reset_ns
            )
            if pose_ready and self._reset_hold_pose is None:
                self.home_pose = self.latest_pose
                self._reset_hold_pose = self.latest_pose
                self._publish_reset_hold_command()
            joints_ready = self.latest_joints is not None and self._message_after_reset(
                self.latest_joints, reset_ns
            )
            gripper_ready = (
                self.latest_gripper is not None
                and self._message_after_reset(self.latest_gripper, reset_ns)
            )
            cube_ready = self.cube_pose is not None and self._message_after_reset(
                self.cube_pose, reset_ns
            )
            basket_ready = (
                self.basket_pose is not None
                and self._message_after_reset(self.basket_pose, reset_ns)
            )
            front_ready = len(self.front_frames) == self.n_obs_steps
            side_ready = not self.side_topic or len(self.side_frames) == self.n_obs_steps
            if (
                pose_ready
                and joints_ready
                and gripper_ready
                and cube_ready
                and basket_ready
                and front_ready
                and side_ready
                and len(self.proprio_steps) == self.n_obs_steps
            ):
                return True
            now = time.monotonic()
            if now - self._last_ready_log > 1.0:
                side_count = len(self.side_frames) if self.side_topic else 0
                side_status = f" side={side_count}" if self.side_topic else ""
                self.get_logger().info(
                    "Waiting for fresh reset state: "
                    f"pose={pose_ready} joints={joints_ready} "
                    f"gripper={gripper_ready} cube={cube_ready} "
                    f"basket={basket_ready} front={len(self.front_frames)}"
                    f"{side_status} proprio={len(self.proprio_steps)}"
                )
                self._last_ready_log = now
            if now - start > self.reset_pose_timeout:
                self.get_logger().error("Timed out waiting for fresh reset state")
                return False
        return False

    def _wait_for_policy_inputs(self) -> bool:
        start = time.monotonic()
        while rclpy.ok():
            if self._ready():
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.monotonic()
            if now - start > self.reset_pose_timeout:
                self.get_logger().error("Timed out waiting for policy inputs")
                return False
        return False

    def run_episodes(self) -> None:
        episodes = max(1, self.episodes)
        success_count = 0
        benchmark_dir = None
        if self.benchmark_csv_path:
            csv_path = Path(self.benchmark_csv_path).expanduser()
            if not csv_path.is_absolute():
                csv_path = self._checkpoint_dir / csv_path
            benchmark_dir = csv_path.parent
        for idx in range(episodes):
            self._episode_active = False
            self._reset_until = None
            self._reset_hold_pose = self.home_pose
            self._clear_observation_buffers()
            self._clear_latest_reset_state()
            self._publish_reset_hold_command()
            reset_time = self.get_clock().now().to_msg()
            if not self.reset_scene_with_retry():
                return
            self._clear_observation_buffers()
            if not self._wait_for_fresh_reset_state(reset_time):
                return
            if self.latest_pose is not None:
                self.home_pose = self.latest_pose
                self._reset_hold_pose = self.latest_pose
            self._handle_episode_reset()
            reset_start = time.monotonic()
            while rclpy.ok() and time.monotonic() - reset_start < self.post_reset_wait:
                rclpy.spin_once(self, timeout_sec=0.1)
            self._reset_until = None
            self._clear_observation_buffers()
            if not self._wait_for_policy_inputs():
                return
            self._current_episode_idx = idx
            self._initial_cube_pose = self.cube_pose
            self._initial_basket_pose = self.basket_pose
            self._initial_mass = self._query_object_mass()
            self._await_initial_pose = (
                self._initial_cube_pose is None or self._initial_basket_pose is None
            )
            self.get_logger().info(f"Starting episode {idx + 1}/{episodes}")
            self.episode_start_pub.publish(Int32(data=idx))
            self._episode_active = True
            start_time = time.monotonic()
            while rclpy.ok():
                elapsed = time.monotonic() - start_time
                if self.episode_duration > 0 and elapsed >= self.episode_duration:
                    break
                rclpy.spin_once(self, timeout_sec=0.1)
            self._episode_active = False
            self._action_queue.clear()
            self.episode_end_pub.publish(Int32(data=idx))
            success = self.is_cube_in_basket()
            self._episode_count += 1
            if success:
                self._success_count += 1
                success_count += 1
            success_rate = (
                self._success_count / self._episode_count
                if self._episode_count > 0
                else 0.0
            )
            self.get_logger().info(
                f"Episode {idx + 1}/{episodes} success={success} "
                f"(rate={success_rate:.3f})"
            )
            self._write_benchmark_row(success)
        success_rate = success_count / episodes if episodes > 0 else 0.0
        self.get_logger().info(
            f"Final success rate: {success_count}/{episodes} ({success_rate:.3f})"
        )
        if benchmark_dir is not None:
            video_dir = benchmark_dir / "videos"
            log_path = video_dir if video_dir.exists() else benchmark_dir
            self.get_logger().info(
                f"Benchmark result saved to {_format_benchmark_path(log_path)}"
            )

    def _step(self) -> None:
        now = time.monotonic()
        if self.enable_episode_timer and not self._episode_active:
            self._publish_reset_hold_command()
            return
        if self._reset_until is not None and now < self._reset_until:
            self._publish_reset_hold_command()
            if now - self._last_ready_log > 2.0:
                remaining = self._reset_until - now
                self.get_logger().info(f"Reset wait: {remaining:.2f}s remaining")
                self._last_ready_log = now
            return
        if not self._action_queue:
            if not self._ready():
                if now - self._last_ready_log > 2.0:
                    side_count = len(self.side_frames) if self.side_topic else 0
                    side_status = f" side={side_count}" if self.side_topic else ""
                    self.get_logger().info(
                        f"Waiting for inputs: front={len(self.front_frames)} "
                        f"proprio={len(self.proprio_steps)}{side_status} "
                        f"n_obs_steps={self.n_obs_steps}"
                    )
                    self._last_ready_log = now
                return
            obs = self._build_obs()
            obs = dict_apply(
                obs, lambda x: torch.as_tensor(x, device=self.device).unsqueeze(0)
            )
            with torch.inference_mode():
                infer_start = time.monotonic()
                result = self.policy.predict_action(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
            action_seq = result["action"][0].detach().cpu().numpy()
            if action_seq.ndim == 1:
                action_seq = action_seq[None, :]
            if now - self._last_debug_time > 2.0:
                self.get_logger().info(
                    f"Policy action chunk: {action_seq.shape} "
                    f"infer_ms={infer_ms:.2f}"
                )
                self._last_debug_time = now
            for action in action_seq:
                self._action_queue.append(action)
        action = self._action_queue.popleft()
        self._publish_action(action)

    def _publish_action(self, action: np.ndarray) -> None:
        if action.shape[0] < 8:
            self.get_logger().warn("Action dimension too small for pose+gripper.")
            return
        if self.action_is_delta:
            if self.latest_pose is None:
                self.get_logger().warn("Delta action requires current pose.")
                return
            current = _pose_to_array(self.latest_pose)
            action = action.copy()
            action[:3] = current[:3] + action[:3]
        if self.max_translation_speed > 0.0 and self.latest_pose is not None:
            max_step = self.max_translation_speed / max(self.control_rate, 1e-6)
            current = _pose_to_array(self.latest_pose)
            delta = action[:3] - current[:3]
            dist = float(np.linalg.norm(delta))
            if dist > max_step:
                action = action.copy()
                action[:3] = current[:3] + (delta / dist) * max_step
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.command_frame
        pose_msg.pose.position.x = float(action[0])
        pose_msg.pose.position.y = float(action[1])
        pose_msg.pose.position.z = float(action[2])
        pose_msg.pose.orientation.x = float(action[3])
        pose_msg.pose.orientation.y = float(action[4])
        pose_msg.pose.orientation.z = float(action[5])
        pose_msg.pose.orientation.w = float(action[6])
        self.target_pub.publish(pose_msg)

        gripper_msg = Float64MultiArray()
        gripper_msg.data = [float(action[7])]
        self.gripper_pub.publish(gripper_msg)


def _find_config_root() -> Optional[Path]:
    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        for parent in [start, *start.parents]:
            if (parent / "config" / "deploy").is_dir():
                return parent
    return None


def _resolve_checkpoint_dir(checkpoint_path: str) -> Path:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    base_dir = ckpt_path.parent
    if base_dir.name == "checkpoints":
        base_dir = base_dir.parent
    return base_dir


def _format_benchmark_path(path: Path) -> str:
    parts = path.parts
    if "logs" in parts:
        idx = parts.index("logs")
        return os.fspath(Path(*parts[idx:]))
    return os.fspath(path)


def _resolve_config_path(config_arg: str | None) -> Optional[str]:
    if not config_arg:
        return None
    config_path = Path(config_arg).expanduser()
    if not config_path.suffix:
        config_root = _find_config_root()
        if config_root is None:
            return None
        config_path = config_root / "config" / "deploy" / f"{config_arg}.yaml"
    return str(config_path.resolve())


def _parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMPACT ROS deployment runner.",
        add_help=True,
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config name or path. If a name is provided, it is resolved under "
            "config/deploy/<name>.yaml."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path to override checkpoint_path in the params file.",
    )
    parser.add_argument(
        "ros_args",
        nargs=argparse.REMAINDER,
        help="Additional ROS arguments (e.g. --ros-args ...).",
    )
    parsed, unknown = parser.parse_known_args(args=args)
    if unknown:
        parsed.ros_args = list(parsed.ros_args or []) + list(unknown)
    return parsed


def main(args=None) -> None:
    parsed = _parse_args(args=args)
    raw_overrides = [arg for arg in list(parsed.ros_args or []) if arg != "--ros-args"]
    overrides = []
    idx = 0
    while idx < len(raw_overrides):
        item = raw_overrides[idx]
        if item == "-p":
            if idx + 1 < len(raw_overrides):
                overrides.extend(["-p", raw_overrides[idx + 1]])
                idx += 2
                continue
            idx += 1
            continue
        overrides.extend(["-p", item])
        idx += 1
    config_path = _resolve_config_path(parsed.config)
    if parsed.checkpoint:
        ckpt_path = Path(parsed.checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (Path.cwd() / ckpt_path).resolve()
        overrides = [
            "-p",
            f"checkpoint_path:={ckpt_path}",
            *overrides,
        ]
    if config_path:
        ros_args = ["--params-file", config_path, *overrides]
    else:
        ros_args = list(overrides)
    if ros_args:
        ros_args = ["--ros-args", *ros_args]
        rclpy.logging.get_logger("policy_deployment").info(f"ros_args={ros_args}")
    rclpy.init(args=ros_args if ros_args else args)
    node = ImpactPolicyNode()
    try:
        if node.enable_episode_timer:
            node.run_episodes()
        else:
            rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
