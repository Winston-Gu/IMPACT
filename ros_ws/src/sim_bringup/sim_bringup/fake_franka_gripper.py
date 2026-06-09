#!/usr/bin/env python3
"""Fake Franka gripper action server for simulation."""

import time
from dataclasses import dataclass

import rclpy
from franka_msgs.action import Grasp, Move
from rclpy import Parameter
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray


@dataclass
class _GripperState:
    current_width: float
    target_width: float
    speed: float
    active: bool


class FakeFrankaGripper(Node):
    """Serve /grasp and /move actions and publish joint states for a fake gripper."""

    def __init__(self) -> None:
        super().__init__("franka_gripper")

        self.declare_parameter("arm_id", "fr3")
        self.declare_parameter("joint_names", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("min_width", 0.0)
        self.declare_parameter("max_width", 0.08)
        self.declare_parameter("default_speed", 0.1)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("goal_tolerance", 0.001)
        self.declare_parameter("goal_timeout", 5.0)
        self.declare_parameter("mujoco_target_topic", "/mujoco_gripper/target_width")
        self.declare_parameter("mujoco_force_topic", "/mujoco_gripper/max_force")
        self.declare_parameter(
            "command_topic", "/gripper/gripper_position_controller/commands"
        )
        self.declare_parameter("normalized_joint_state_topic", "/gripper/joint_states")

        arm_id = self.get_parameter("arm_id").value
        joint_names_param = list(self.get_parameter("joint_names").value or [])
        if joint_names_param:
            self._joint_names = joint_names_param
        else:
            self._joint_names = [f"{arm_id}_finger_joint1", f"{arm_id}_finger_joint2"]

        self._min_width = float(self.get_parameter("min_width").value)
        self._max_width = float(self.get_parameter("max_width").value)
        self._default_speed = float(self.get_parameter("default_speed").value)
        self._publish_rate = float(self.get_parameter("publish_rate").value)
        self._goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self._goal_timeout = float(self.get_parameter("goal_timeout").value)
        self._mujoco_target_topic = str(self.get_parameter("mujoco_target_topic").value)
        self._mujoco_force_topic = str(self.get_parameter("mujoco_force_topic").value)
        self._command_topic = str(self.get_parameter("command_topic").value)
        self._normalized_joint_state_topic = str(
            self.get_parameter("normalized_joint_state_topic").value
        )

        initial_width = max(self._min_width, min(self._max_width, self._max_width))
        self._state = _GripperState(
            current_width=initial_width,
            target_width=initial_width,
            speed=self._default_speed,
            active=False,
        )

        self._joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self._normalized_joint_state_pub = self.create_publisher(
            JointState, self._normalized_joint_state_topic, 10
        )
        self._mujoco_target_pub = self.create_publisher(
            Float64, self._mujoco_target_topic, 10
        )
        self._mujoco_force_pub = self.create_publisher(
            Float64, self._mujoco_force_topic, 10
        )
        self.create_subscription(
            Float64MultiArray, self._command_topic, self._command_cb, 10
        )
        self._timer = self.create_timer(1.0 / max(self._publish_rate, 1.0), self._tick)

        self._grasp_server = ActionServer(
            self,
            Grasp,
            "grasp",
            goal_callback=self._accept_goal,
            cancel_callback=self._cancel_goal,
            execute_callback=self._execute_grasp,
        )
        self._move_server = ActionServer(
            self,
            Move,
            "move",
            goal_callback=self._accept_goal,
            cancel_callback=self._cancel_goal,
            execute_callback=self._execute_move,
        )

        self.get_logger().info(
            f"Fake gripper ready on {self.get_namespace()}/grasp and /move"
        )

    def _accept_goal(self, _goal) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _cancel_goal(self, _goal) -> CancelResponse:
        self._state.active = False
        return CancelResponse.ACCEPT

    def _tick(self) -> None:
        if self._state.active:
            self._advance_state()
        self._publish_joint_state()

    def _advance_state(self) -> None:
        delta = self._state.target_width - self._state.current_width
        step = self._state.speed / max(self._publish_rate, 1.0)
        if abs(delta) <= step:
            self._state.current_width = self._state.target_width
            self._state.active = False
            return
        self._state.current_width += step if delta > 0 else -step
        self._state.current_width = max(
            self._min_width, min(self._max_width, self._state.current_width)
        )

    def _publish_joint_state(self) -> None:
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = list(self._joint_names)
        finger_width = max(0.0, self._state.current_width * 0.5)
        joint_state.position = [finger_width, finger_width]
        joint_state.velocity = [0.0, 0.0]
        joint_state.effort = [0.0, 0.0]
        self._joint_pub.publish(joint_state)
        target_msg = Float64()
        target_msg.data = float(self._state.current_width)
        self._mujoco_target_pub.publish(target_msg)
        self._publish_normalized_state(joint_state.header)

    def _publish_normalized_state(self, header) -> None:
        width_range = self._max_width - self._min_width
        if width_range <= 0.0:
            return
        normalized = (self._state.current_width - self._min_width) / width_range
        normalized = max(0.0, min(1.0, normalized))
        normalized_msg = JointState()
        normalized_msg.header = header
        normalized_msg.name = ["gripper_width_normalized"]
        normalized_msg.position = [normalized]
        self._normalized_joint_state_pub.publish(normalized_msg)

    def _set_target(self, width: float, speed: float) -> None:
        width = max(self._min_width, min(self._max_width, width))
        speed = max(0.0, speed)
        if speed <= 0.0:
            speed = self._default_speed
        self._state.target_width = width
        self._state.speed = speed
        self._state.active = True

    def _command_cb(self, msg: Float64MultiArray) -> None:
        if not msg.data:
            return
        width_range = self._max_width - self._min_width
        if width_range <= 0.0:
            self.get_logger().warn(
                "Invalid gripper width range, expected max_width > min_width"
            )
            return
        command = max(0.0, min(1.0, float(msg.data[0])))
        width = self._min_width + command * width_range
        self._set_target(width, self._default_speed)

    def _run_goal(
        self, goal_handle, width: float, speed: float, tolerance: float, feedback_cls
    ):
        self._set_target(width, speed)
        start_time = time.monotonic()
        step_time = 1.0 / max(self._publish_rate, 1.0)

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self._state.active = False
                return False, "Canceled", True

            self._advance_state()
            self._publish_joint_state()
            current = self._state.current_width
            feedback = feedback_cls()
            feedback.current_width = current
            goal_handle.publish_feedback(feedback)

            if abs(self._state.target_width - current) <= tolerance:
                self._state.active = False
                return True, "", False

            if time.monotonic() - start_time > self._goal_timeout:
                self._state.active = False
                return False, "Timeout waiting for target width", False

            time.sleep(step_time)

        self._state.active = False
        return False, "ROS shutdown", False

    def _execute_grasp(self, goal_handle):
        goal = goal_handle.request
        if goal.force > 0.0:
            force_msg = Float64()
            force_msg.data = float(goal.force)
            self._mujoco_force_pub.publish(force_msg)
        tolerance = self._goal_tolerance
        success, error, canceled = self._run_goal(
            goal_handle, goal.width, goal.speed, tolerance, Grasp.Feedback
        )
        result = Grasp.Result()
        result.success = success
        result.error = error
        if canceled:
            return result
        goal_handle.succeed() if success else goal_handle.abort()
        return result

    def _execute_move(self, goal_handle):
        goal = goal_handle.request
        success, error, canceled = self._run_goal(
            goal_handle, goal.width, goal.speed, self._goal_tolerance, Move.Feedback
        )
        result = Move.Result()
        result.success = success
        result.error = error
        if canceled:
            return result
        goal_handle.succeed() if success else goal_handle.abort()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeFrankaGripper()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
