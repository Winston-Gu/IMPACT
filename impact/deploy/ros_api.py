import argparse
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray

from impact.deploy.policy_loader import load_policy_from_checkpoint
from impact.utils.pytorch_util import dict_apply


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
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("n_obs_steps", 2)
        self.declare_parameter("resize_height", 256)
        self.declare_parameter("resize_width", 256)
        self.declare_parameter("use_compressed", True)
        self.declare_parameter("max_translation_speed", 0.1)
        self.declare_parameter("action_is_delta", False)
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
        self._last_debug_time = 0.0
        self._action_queue: Deque[np.ndarray] = deque()

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

        self.target_pub = self.create_publisher(PoseStamped, self.target_topic, 10)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, self.gripper_cmd_topic, 10
        )

        self._load_policy(checkpoint_path)
        torch.backends.cudnn.benchmark = True

        self.timer = self.create_timer(1.0 / self.control_rate, self._step)
        self.get_logger().info("IMPACT policy runner started.")

    def _load_policy(self, checkpoint_path: str) -> None:
        self.policy = load_policy_from_checkpoint(
            str(Path(checkpoint_path).expanduser()), self.device, prefer_ema=True
        )

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.latest_pose = msg
        self._maybe_update_proprio()

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

    def _step(self) -> None:
        now = time.monotonic()
        if not self._action_queue:
            if not self._ready():
                if now - self._last_debug_time > 2.0:
                    side_count = len(self.side_frames) if self.side_topic else 0
                    side_status = f" side={side_count}" if self.side_topic else ""
                    self.get_logger().info(
                        f"Waiting for inputs: front={len(self.front_frames)} "
                        f"proprio={len(self.proprio_steps)}{side_status}"
                    )
                    self._last_debug_time = now
                return
            obs = self._build_obs()
            obs = dict_apply(
                obs, lambda x: torch.as_tensor(x, device=self.device).unsqueeze(0)
            )
            with torch.inference_mode():
                result = self.policy.predict_action(obs)
            action_seq = result["action"][0].detach().cpu().numpy()
            if action_seq.ndim == 1:
                action_seq = action_seq[None, :]
            if now - self._last_debug_time > 2.0:
                self.get_logger().info(f"Policy action chunk: {action_seq.shape}")
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


def _resolve_config_path(config_arg: str | None) -> Optional[str]:
    if not config_arg:
        return None
    config_path = Path(config_arg).expanduser()
    if not config_path.suffix:
        config_path = (
            Path(__file__).resolve().parents[2] / "config/deploy" / f"{config_arg}.yaml"
        )
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
    return parser.parse_args(args=args)


def main(args=None) -> None:
    parsed = _parse_args(args=args)
    ros_args = list(parsed.ros_args or [])
    config_path = _resolve_config_path(parsed.config)
    if config_path:
        ros_args = ["--ros-args", "--params-file", config_path, *ros_args]
    if parsed.checkpoint:
        ckpt_path = Path(parsed.checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (Path.cwd() / ckpt_path).resolve()
        ros_args = [
            "--ros-args",
            "-p",
            f"checkpoint_path:={ckpt_path}",
            *ros_args,
        ]
    rclpy.init(args=ros_args if ros_args else args)
    node = ImpactPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
