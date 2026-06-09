#!/usr/bin/env python3
import time
from typing import Optional

import rclpy
from franka_msgs.action import Grasp, Move
from franka_msgs.msg import GraspEpsilon
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32
from std_srvs.srv import Trigger


class PickPlaceDemo(Node):
    def __init__(self):
        super().__init__("pick_place_demo")

        self.declare_parameter("frame_id", "fr3_link0")
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter("current_pose_topic", "/current_pose")
        self.declare_parameter("cube_pose_topic", "/mujoco_objects/pick_cube_pose")
        self.declare_parameter("basket_pose_topic", "/mujoco_objects/basket_pose")
        self.declare_parameter("grasp_action", "/franka_gripper/grasp")
        self.declare_parameter("move_action", "/franka_gripper/move")
        self.declare_parameter("table_surface_z", 0.53)
        self.declare_parameter("approach_height", 0.20)
        self.declare_parameter("grasp_offset_z", -0.05)
        self.declare_parameter("lift_height", 0.25)
        self.declare_parameter("basket_rim_offset", 0.1)
        self.declare_parameter("place_clearance", 0.02)
        self.declare_parameter("drop_depth", 0.0)
        self.declare_parameter("post_grasp_wait", 0.4)
        self.declare_parameter("pre_release_wait", 0.4)
        self.declare_parameter("move_wait", 1.0)
        self.declare_parameter("post_reset_wait", 2.0)
        self.declare_parameter("max_step", 0.04)
        self.declare_parameter("pose_tolerance", 0.01)
        self.declare_parameter("settle_timeout", 10.0)
        self.declare_parameter("step_rate", 50.0)
        self.declare_parameter("position_only", False)
        self.declare_parameter("grasp_speed", 0.2)
        self.declare_parameter("grasp_force", 200.0)
        self.declare_parameter("grasp_epsilon", 0.005)
        self.declare_parameter("open_width", 0.08)
        self.declare_parameter("open_speed", 0.2)
        self.declare_parameter("wait_for_gripper_actions", False)
        self.declare_parameter("post_lift_wait", 1.0)
        self.declare_parameter(
            "gripper_command_topic", "/gripper/gripper_position_controller/commands"
        )
        self.declare_parameter("gripper_min_width", 0.0)
        self.declare_parameter("gripper_max_width", 0.08)
        self.declare_parameter("record_dataset", False)
        self.declare_parameter(
            "dataset_start_service", "/dataset_recorder/start_recording"
        )
        self.declare_parameter(
            "dataset_stop_service", "/dataset_recorder/stop_recording"
        )
        self.declare_parameter(
            "dataset_delete_service", "/dataset_recorder/delete_last_episode"
        )
        self.declare_parameter("episodes", 5)
        self.declare_parameter("episode_start_topic", "/episode/start")
        self.declare_parameter("episode_end_topic", "/episode/end")
        self.declare_parameter("object_settle_duration", 0.8)
        self.declare_parameter("object_settle_timeout", 5.0)
        self.declare_parameter("object_settle_pos_threshold", 0.002)
        self.declare_parameter("reset_service", "/mujoco_sim/reset_scene")
        self.declare_parameter("home_tolerance", 0.02)
        self.declare_parameter("home_timeout", 10.0)
        self.declare_parameter("reset_timeout", 10.0)
        self.declare_parameter("reset_retry_delay", 0.5)
        self.declare_parameter("reset_pose_timeout", 5.0)
        self.declare_parameter("object_mass_service", "/mujoco_sim/get_object_mass")
        self.declare_parameter("mass_comp_kp", 400.0)
        self.declare_parameter("apply_z_offset", True)
        self.declare_parameter("use_reset_home_pose", True)

        self.frame_id = self.get_parameter("frame_id").value
        self.target_pose_topic = self.get_parameter("target_pose_topic").value
        self.current_pose_topic = self.get_parameter("current_pose_topic").value
        self.cube_pose_topic = self.get_parameter("cube_pose_topic").value
        self.basket_pose_topic = self.get_parameter("basket_pose_topic").value
        self.grasp_action = self.get_parameter("grasp_action").value
        self.move_action = self.get_parameter("move_action").value
        self.table_surface_z = float(self.get_parameter("table_surface_z").value)
        self.approach_height = float(self.get_parameter("approach_height").value)
        self.grasp_offset_z = float(self.get_parameter("grasp_offset_z").value)
        self.lift_height = float(self.get_parameter("lift_height").value)
        self.basket_rim_offset = float(self.get_parameter("basket_rim_offset").value)
        self.place_clearance = float(self.get_parameter("place_clearance").value)
        self.drop_depth = float(self.get_parameter("drop_depth").value)
        self.post_grasp_wait = float(self.get_parameter("post_grasp_wait").value)
        self.pre_release_wait = float(self.get_parameter("pre_release_wait").value)
        self.move_wait = float(self.get_parameter("move_wait").value)
        self.post_reset_wait = float(self.get_parameter("post_reset_wait").value)
        self.max_step = float(self.get_parameter("max_step").value)
        self.pose_tolerance = float(self.get_parameter("pose_tolerance").value)
        self.settle_timeout = float(self.get_parameter("settle_timeout").value)
        self.step_rate = float(self.get_parameter("step_rate").value)
        self.position_only = bool(self.get_parameter("position_only").value)
        self.grasp_speed = float(self.get_parameter("grasp_speed").value)
        self.grasp_force = float(self.get_parameter("grasp_force").value)
        self.grasp_epsilon = float(self.get_parameter("grasp_epsilon").value)
        self.open_width = float(self.get_parameter("open_width").value)
        self.open_speed = float(self.get_parameter("open_speed").value)
        self.wait_for_gripper_actions = bool(
            self.get_parameter("wait_for_gripper_actions").value
        )
        self.post_lift_wait = float(self.get_parameter("post_lift_wait").value)
        self.gripper_command_topic = str(
            self.get_parameter("gripper_command_topic").value
        )
        self.gripper_min_width = float(self.get_parameter("gripper_min_width").value)
        self.gripper_max_width = float(self.get_parameter("gripper_max_width").value)
        self.record_dataset = bool(self.get_parameter("record_dataset").value)
        self.dataset_start_service = str(
            self.get_parameter("dataset_start_service").value
        )
        self.dataset_stop_service = str(
            self.get_parameter("dataset_stop_service").value
        )
        self.dataset_delete_service = str(
            self.get_parameter("dataset_delete_service").value
        )
        self.episodes = int(self.get_parameter("episodes").value)
        self.episode_start_topic = self.get_parameter("episode_start_topic").value
        self.episode_end_topic = self.get_parameter("episode_end_topic").value
        self.object_settle_duration = float(
            self.get_parameter("object_settle_duration").value
        )
        self.object_settle_timeout = float(
            self.get_parameter("object_settle_timeout").value
        )
        self.object_settle_pos_threshold = float(
            self.get_parameter("object_settle_pos_threshold").value
        )
        self.reset_service = self.get_parameter("reset_service").value
        self.home_tolerance = float(self.get_parameter("home_tolerance").value)
        self.home_timeout = float(self.get_parameter("home_timeout").value)
        self.reset_timeout = float(self.get_parameter("reset_timeout").value)
        self.reset_retry_delay = float(self.get_parameter("reset_retry_delay").value)
        self.reset_pose_timeout = float(self.get_parameter("reset_pose_timeout").value)
        self.object_mass_service = str(self.get_parameter("object_mass_service").value)
        self.mass_comp_kp = float(self.get_parameter("mass_comp_kp").value)
        self.apply_z_offset = bool(self.get_parameter("apply_z_offset").value)
        self.use_reset_home_pose = bool(self.get_parameter("use_reset_home_pose").value)
        if self.mass_comp_kp <= 0.0:
            raise RuntimeError("mass_comp_kp must be > 0")

        self.pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 10)
        self.gripper_command_pub = self.create_publisher(
            Float64MultiArray, self.gripper_command_topic, 10
        )
        self.episode_start_pub = self.create_publisher(
            Int32, self.episode_start_topic, 10
        )
        self.episode_end_pub = self.create_publisher(Int32, self.episode_end_topic, 10)
        self.current_pose: Optional[PoseStamped] = None
        self.create_subscription(
            PoseStamped, self.current_pose_topic, self._on_current_pose, 10
        )
        self.cube_pose: Optional[PoseStamped] = None
        self.basket_pose: Optional[PoseStamped] = None
        self.create_subscription(
            PoseStamped, self.cube_pose_topic, self._on_cube_pose, 10
        )
        self.create_subscription(
            PoseStamped, self.basket_pose_topic, self._on_basket_pose, 10
        )

        self.grasp_client = ActionClient(self, Grasp, self.grasp_action)
        self.move_client = ActionClient(self, Move, self.move_action)
        self.reset_client = self.create_client(Trigger, self.reset_service)
        self.dataset_start_client = self.create_client(
            Trigger, self.dataset_start_service
        )
        self.dataset_stop_client = self.create_client(
            Trigger, self.dataset_stop_service
        )
        self.dataset_delete_client = self.create_client(
            Trigger, self.dataset_delete_service
        )
        self.object_mass_client = self.create_client(Trigger, self.object_mass_service)
        self.last_pose: Optional[Pose] = None
        self.home_pose: Optional[Pose] = None
        self.home_pose_override: Optional[Pose] = None
        self.mass_z_offset = 0.0

    def reset_scene(self) -> bool:
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Reset service not available")
            return False
        request = Trigger.Request()
        future = self.reset_client.call_async(request)
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

    def wait_until_home(self, timeout_sec: Optional[float] = None) -> bool:
        if self.home_pose is None:
            return True
        start = time.time()
        timeout = self.home_timeout if timeout_sec is None else timeout_sec
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_pose is None:
                continue
            dx = self.home_pose.position.x - self.current_pose.pose.position.x
            dy = self.home_pose.position.y - self.current_pose.pose.position.y
            dz = self.home_pose.position.z - self.current_pose.pose.position.z
            err = (dx * dx + dy * dy + dz * dz) ** 0.5
            if err <= self.home_tolerance:
                return True
            if time.time() - start > timeout:
                self.get_logger().warn(
                    f"Timed out waiting for home pose: err={err:.4f} tol={self.home_tolerance:.4f}"
                )
                return False
        return False

    def _on_current_pose(self, msg: PoseStamped):
        self.current_pose = msg

    def _on_cube_pose(self, msg: PoseStamped):
        self.cube_pose = msg

    def _on_basket_pose(self, msg: PoseStamped):
        self.basket_pose = msg

    def wait_for_current_pose(self, timeout_sec: float = 10.0) -> PoseStamped:
        start = time.time()
        while rclpy.ok() and self.current_pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout_sec:
                raise RuntimeError("Timed out waiting for /current_pose")
        return self.current_pose

    def wait_for_object_poses(
        self, timeout_sec: float = 10.0
    ) -> tuple[PoseStamped, PoseStamped]:
        start = time.time()
        while rclpy.ok() and (self.cube_pose is None or self.basket_pose is None):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout_sec:
                raise RuntimeError("Timed out waiting for object poses")
        return self.cube_pose, self.basket_pose

    def publish_pose(self, position, orientation, frame_id: Optional[str] = None):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id if frame_id else self.frame_id
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation = orientation
        self.pose_pub.publish(msg)
        self.last_pose = msg.pose

    def hold_position(self, duration: float, frame_id: Optional[str] = None):
        if duration <= 0.0:
            return
        rate_sleep = 1.0 / self.step_rate if self.step_rate > 0.0 else 0.02
        end_time = time.time() + duration
        while rclpy.ok() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.last_pose is not None:
                use_orientation = self.last_pose.orientation
                if self.position_only and self.current_pose is not None:
                    use_orientation = self.current_pose.pose.orientation
                self.publish_pose(
                    (
                        self.last_pose.position.x,
                        self.last_pose.position.y,
                        self.last_pose.position.z,
                    ),
                    use_orientation,
                    frame_id,
                )
            time.sleep(rate_sleep)

    def step_toward(
        self,
        target,
        orientation,
        frame_id: Optional[str] = None,
        pose_tolerance: Optional[float] = None,
        settle_timeout: Optional[float] = None,
        offset_z: float = 0.0,
    ):
        rate_sleep = 1.0 / self.step_rate if self.step_rate > 0.0 else 0.02
        start_time = time.time()
        effective_tol = (
            self.pose_tolerance if pose_tolerance is None else pose_tolerance
        )
        effective_timeout = (
            self.settle_timeout if settle_timeout is None else settle_timeout
        )
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_pose is None:
                continue
            dx = target[0] - self.current_pose.pose.position.x
            dy = target[1] - self.current_pose.pose.position.y
            dz = target[2] - self.current_pose.pose.position.z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist <= effective_tol:
                break
            if time.time() - start_time > effective_timeout:
                self.get_logger().warn(
                    f"Timed out stepping to target: err={dist:.4f}, current={self.current_pose.pose.position}, target={target}"
                )
                break
            scale = min(1.0, self.max_step / dist)
            next_pos = (
                self.current_pose.pose.position.x + dx * scale,
                self.current_pose.pose.position.y + dy * scale,
                self.current_pose.pose.position.z + dz * scale + offset_z,
            )
            use_orientation = orientation
            if self.position_only:
                use_orientation = self.current_pose.pose.orientation
            self.publish_pose(next_pos, use_orientation, frame_id)
            time.sleep(rate_sleep)

    def send_grasp(self, width: float) -> bool:
        if not self.grasp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Grasp action server not available")
            return False
        self._publish_gripper_command(width)
        goal = Grasp.Goal()
        goal.width = width
        goal.speed = self.grasp_speed
        goal.force = self.grasp_force
        goal.epsilon = GraspEpsilon(inner=self.grasp_epsilon, outer=self.grasp_epsilon)
        future = self.grasp_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Grasp goal rejected")
            return False
        if not self.wait_for_gripper_actions:
            return True
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if not result.success:
            self.get_logger().warn(f"Grasp failed: {result.error}")
        return result.success

    def send_open(self, width: float) -> bool:
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Move action server not available")
            return False
        self._publish_gripper_command(width)
        goal = Move.Goal()
        goal.width = width
        goal.speed = self.open_speed
        future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Move goal rejected")
            return False
        if not self.wait_for_gripper_actions:
            return True
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if not result.success:
            self.get_logger().warn(f"Move failed: {result.error}")
        return result.success

    def _call_trigger(self, client, label: str) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"{label} service not available")
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            message = response.message if response else "No response"
            self.get_logger().warn(f"{label} failed: {message}")
            return False
        return True

    def _publish_gripper_command(self, width: float) -> None:
        if self.gripper_max_width <= self.gripper_min_width:
            return
        normalized = (width - self.gripper_min_width) / (
            self.gripper_max_width - self.gripper_min_width
        )
        normalized = max(0.0, min(1.0, normalized))
        msg = Float64MultiArray()
        msg.data = [normalized]
        self.gripper_command_pub.publish(msg)

    def is_cube_in_basket(self) -> bool:
        if self.cube_pose is None or self.basket_pose is None:
            return False
        basket_pos = self.basket_pose.pose.position
        cube_pos = self.cube_pose.pose.position

        inner_half_xy = 0.2
        within_xy = (
            abs(cube_pos.x - basket_pos.x) <= inner_half_xy
            and abs(cube_pos.y - basket_pos.y) <= inner_half_xy
        )
        return within_xy

    def update_mass_offset(self) -> None:
        if not self.object_mass_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Object mass service not available")
            self.mass_z_offset = 0.0
            return
        future = self.object_mass_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            message = response.message if response else "No response"
            self.get_logger().warn(f"Failed to query object mass: {message}")
            self.mass_z_offset = 0.0
            return
        try:
            mass = float(response.message)
        except ValueError:
            self.get_logger().warn(
                f"Invalid mass value from service: {response.message}"
            )
            self.mass_z_offset = 0.0
            return
        self.mass_z_offset = mass * 9.8 / self.mass_comp_kp
        self.get_logger().info(
            f"Object mass={mass:.3f} kg, z_offset={self.mass_z_offset:.4f} m"
        )

    def run_sequence(self):
        if self.home_pose_override is not None:
            home_pose = self.home_pose_override
        else:
            home_pose = self.wait_for_current_pose().pose
        orientation = home_pose.orientation
        cube_pose_msg, basket_pose_msg = self.wait_for_object_poses()
        frame_id = cube_pose_msg.header.frame_id or self.frame_id
        self.last_pose = home_pose
        self.home_pose = home_pose

        cube_z = cube_pose_msg.pose.position.z + self.grasp_offset_z
        cube_approach_z = cube_z + self.approach_height

        basket_rim_z = basket_pose_msg.pose.position.z + self.basket_rim_offset
        # basket_approach_z = basket_rim_z + self.place_clearance
        basket_drop_z = basket_rim_z - self.drop_depth
        lift_z = cube_z + self.lift_height
        z_offset = self.mass_z_offset if self.apply_z_offset else 0.0
        self.get_logger().info(
            f"apply_z_offset={self.apply_z_offset}, mass_z_offset={self.mass_z_offset:.4f}, "
            f"z_offset={z_offset:.4f}"
        )

        tol_object = 0.02
        timeout_object = 3.0
        tol_transit = 0.03
        timeout_transit = 5.0
        timeout_lift = 10.0

        self.step_toward(
            (
                cube_pose_msg.pose.position.x,
                cube_pose_msg.pose.position.y,
                cube_approach_z,
            ),
            orientation,
            frame_id,
            pose_tolerance=tol_object,
            settle_timeout=timeout_object,
        )
        # self.hold_position(self.move_wait, frame_id)

        self.step_toward(
            (cube_pose_msg.pose.position.x, cube_pose_msg.pose.position.y, cube_z),
            orientation,
            frame_id,
            pose_tolerance=tol_object,
            settle_timeout=timeout_object,
        )
        # self.hold_position(self.move_wait, frame_id)

        self.send_grasp(0.0)
        self.hold_position(self.post_grasp_wait, frame_id)

        self.step_toward(
            (cube_pose_msg.pose.position.x, cube_pose_msg.pose.position.y, lift_z),
            orientation,
            frame_id,
            pose_tolerance=tol_transit,
            settle_timeout=timeout_lift,
            offset_z=z_offset,
        )

        self.hold_position(self.post_lift_wait, frame_id)

        self.step_toward(
            (
                basket_pose_msg.pose.position.x,
                basket_pose_msg.pose.position.y,
                lift_z,
            ),
            orientation,
            frame_id,
            pose_tolerance=tol_transit,
            settle_timeout=timeout_transit,
            offset_z=z_offset,
        )
        # self.hold_position(self.move_wait, frame_id)

        self.step_toward(
            (
                basket_pose_msg.pose.position.x,
                basket_pose_msg.pose.position.y,
                basket_drop_z,
            ),
            orientation,
            frame_id,
            pose_tolerance=tol_transit,
            settle_timeout=timeout_transit,
            offset_z=z_offset,
        )
        # self.hold_position(self.move_wait, frame_id)

        self.send_open(self.open_width)
        self.hold_position(self.move_wait, frame_id)

        self.step_toward(
            (
                basket_pose_msg.pose.position.x,
                basket_pose_msg.pose.position.y,
                home_pose.position.z,
            ),
            orientation,
            frame_id,
            pose_tolerance=tol_transit,
            settle_timeout=timeout_transit,
        )

        self.step_toward(
            (home_pose.position.x, home_pose.position.y, home_pose.position.z),
            orientation,
            frame_id,
            pose_tolerance=tol_transit,
            settle_timeout=timeout_transit,
        )
        self.hold_position(self.move_wait, frame_id)
        return home_pose

    def run_episodes(self):
        episodes = max(1, self.episodes)
        success_count = 0
        for idx in range(episodes):
            reset_time = self.get_clock().now().to_msg()
            if idx == 0:
                time.sleep(4.0)
            if not self.reset_scene_with_retry():
                return
            self.current_pose = None
            self.cube_pose = None
            self.basket_pose = None
            time.sleep(self.post_reset_wait)

            # First, wait for any pose messages after reset.
            if not self.wait_for_reset_poses(None, None, reset_time):
                return
            if self.use_reset_home_pose:
                reset_pose = self.wait_for_reset_current_pose(reset_time)
                if reset_pose is None:
                    return
                self.home_pose_override = reset_pose.pose
                self.publish_pose(
                    (
                        reset_pose.pose.position.x,
                        reset_pose.pose.position.y,
                        reset_pose.pose.position.z,
                    ),
                    reset_pose.pose.orientation,
                    reset_pose.header.frame_id,
                )
            else:
                self.home_pose_override = None
            self.update_mass_offset()
            self.get_logger().info(
                f"\033[0;32mStarting episode {idx + 1}/{episodes}\033[0m"
            )
            self.episode_start_pub.publish(Int32(data=idx))
            if self.record_dataset:
                self._call_trigger(self.dataset_start_client, "Dataset start")
            self.run_sequence()
            rclpy.spin_once(self, timeout_sec=0.1)
            success = self.is_cube_in_basket()
            if success:
                success_count += 1
            self.get_logger().info(f"Episode {idx + 1}/{episodes} success={success}")
            if self.record_dataset:
                self._call_trigger(self.dataset_stop_client, "Dataset stop")
                if not success:
                    self._call_trigger(self.dataset_delete_client, "Dataset delete")
            self.episode_end_pub.publish(Int32(data=idx))
            if idx < episodes - 1:
                if not self.wait_until_home():
                    self.get_logger().warn(
                        "Failed to return home before next episode; resetting scene."
                    )
                    if not self.reset_scene_with_retry():
                        return
        success_rate = success_count / episodes if episodes > 0 else 0.0
        self.get_logger().info(
            f"\033[0;32mEpisode success rate: {success_count}/{episodes} ({success_rate:.3f})\033[0m"
        )

    def wait_for_reset_poses(
        self, last_cube_stamp, last_basket_stamp, reset_time
    ) -> bool:
        start = time.time()
        reset_ns = reset_time.sec * 1_000_000_000 + reset_time.nanosec
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.cube_pose is None or self.basket_pose is None:
                if time.time() - start > self.reset_pose_timeout:
                    self.get_logger().error("Timed out waiting for reset poses")
                    return False
                continue
            cube_stamp = self.cube_pose.header.stamp
            basket_stamp = self.basket_pose.header.stamp
            cube_ns = cube_stamp.sec * 1_000_000_000 + cube_stamp.nanosec
            basket_ns = basket_stamp.sec * 1_000_000_000 + basket_stamp.nanosec
            if last_cube_stamp is None or last_basket_stamp is None:
                if cube_ns > reset_ns and basket_ns > reset_ns:
                    return True
            if (
                cube_stamp != last_cube_stamp
                and basket_stamp != last_basket_stamp
                and cube_ns > reset_ns
                and basket_ns > reset_ns
            ):
                return True
            if time.time() - start > self.reset_pose_timeout:
                self.get_logger().error("Timed out waiting for updated object poses")
                return False
        return False

    def wait_for_reset_current_pose(self, reset_time) -> Optional[PoseStamped]:
        start = time.time()
        reset_ns = reset_time.sec * 1_000_000_000 + reset_time.nanosec
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_pose is None:
                if time.time() - start > self.reset_pose_timeout:
                    self.get_logger().error("Timed out waiting for reset current_pose")
                    return None
                continue
            pose_stamp = self.current_pose.header.stamp
            pose_ns = pose_stamp.sec * 1_000_000_000 + pose_stamp.nanosec
            if pose_ns > reset_ns:
                return self.current_pose
            if time.time() - start > self.reset_pose_timeout:
                self.get_logger().error("Timed out waiting for updated current_pose")
                return None
        return None


def main():
    rclpy.init()
    node = PickPlaceDemo()
    try:
        node.run_episodes()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
