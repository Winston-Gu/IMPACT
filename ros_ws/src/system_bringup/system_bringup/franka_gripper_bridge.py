#!/usr/bin/env python3
"""Bridge normalized gripper commands to the Franka Hand Move/Grasp actions."""

from functools import partial
from typing import Optional

import rclpy
from franka_msgs.action import Grasp
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class FrankaGripperBridge(Node):
    """Forward normalized Float64MultiArray commands to the Franka Hand actions."""

    def __init__(self) -> None:
        super().__init__("franka_gripper_bridge")

        self.declare_parameter(
            "command_topic", "/gripper/gripper_position_controller/commands"
        )
        self.declare_parameter("gripper_namespace", "franka_gripper")
        self.declare_parameter("min_width", 0.0)
        self.declare_parameter("max_width", 0.08)
        self.declare_parameter("grasp_speed", 0.1)
        self.declare_parameter("grasp_force", 100.0)
        self.declare_parameter("grasp_epsilon_inner", 0.01)
        self.declare_parameter("grasp_epsilon_outer", 0.08)
        self.declare_parameter("hardware_joint_state_topic", "")
        self.declare_parameter("normalized_joint_state_topic", "/gripper/joint_states")

        command_topic = self.get_parameter("command_topic").value
        gripper_ns = self.get_parameter("gripper_namespace").value
        self._min_width = self.get_parameter("min_width").value
        self._max_width = self.get_parameter("max_width").value
        self._grasp_speed = self.get_parameter("grasp_speed").value
        self._grasp_force = self.get_parameter("grasp_force").value
        self._grasp_eps_inner = self.get_parameter("grasp_epsilon_inner").value
        self._grasp_eps_outer = self.get_parameter("grasp_epsilon_outer").value
        hardware_joint_state_topic = self.get_parameter(
            "hardware_joint_state_topic"
        ).value
        if not hardware_joint_state_topic:
            if gripper_ns.startswith("/"):
                hardware_joint_state_topic = f"{gripper_ns}/joint_states"
            else:
                hardware_joint_state_topic = f"/{gripper_ns}/joint_states"
        normalized_joint_state_topic = self.get_parameter(
            "normalized_joint_state_topic"
        ).value
        self._width_range = self._max_width - self._min_width

        self._grasp_client = ActionClient(self, Grasp, f"{gripper_ns}/grasp")
        self._active_goal: Optional[rclpy.task.Future] = None
        self._pending_width: Optional[float] = None
        self._retry_timer = self.create_timer(0.2, self._retry_pending)

        self.create_subscription(Float64MultiArray, command_topic, self._command_cb, 10)
        self._hardware_joint_state_sub = self.create_subscription(
            JointState, hardware_joint_state_topic, self._hardware_joint_state_cb, 10
        )
        self._normalized_joint_state_pub = self.create_publisher(
            JointState, normalized_joint_state_topic, 10
        )
        self.get_logger().info(
            f"Franka gripper bridge listening on {command_topic}, targeting {gripper_ns}/grasp "
            f"and mirroring {hardware_joint_state_topic} -> {normalized_joint_state_topic}"
        )

    def _command_cb(self, msg: Float64MultiArray) -> None:
        if not msg.data:
            return

        if self._width_range <= 0.0:
            self.get_logger().warn(
                "Invalid gripper width range, expected max_width > min_width"
            )
            return

        command = max(0.0, min(1.0, float(msg.data[0])))
        width = self._min_width + command * self._width_range

        if self._active_goal is not None and not self._active_goal.done():
            self._pending_width = width
            self.get_logger().debug(
                "Deferring gripper command while previous goal is pending"
            )
            return

        if not self._send_grasp_goal(width, command):
            self._pending_width = width

    def _goal_sent_cb(self, future: rclpy.task.Future, *, action_name: str) -> None:
        if future.cancelled():
            self.get_logger().warn(
                f"{action_name.capitalize()} goal cancelled before reaching server"
            )
            self._active_goal = None
            return
        if future.exception() is not None:
            self.get_logger().error(
                f"Failed to send {action_name} goal: {future.exception()}"
            )
            self._active_goal = None
            return

        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            partial(self._goal_result_cb, action_name=action_name)
        )
        self._active_goal = result_future

    def _goal_result_cb(self, future: rclpy.task.Future, *, action_name: str) -> None:
        self._active_goal = None
        if future.cancelled():
            self.get_logger().warn(
                f"{action_name.capitalize()} goal cancelled by server"
            )
            return
        if future.exception() is not None:
            self.get_logger().error(
                f"{action_name.capitalize()} goal failed: {future.exception()}"
            )
            return

        result = future.result()
        outcome = getattr(result, "result", result)
        success = getattr(outcome, "success", True)
        if success:
            self.get_logger().debug(f"{action_name.capitalize()} goal completed")
        else:
            self.get_logger().warn(f"{action_name.capitalize()} goal reported failure")

    def _send_grasp_goal(self, width: float, command: float) -> bool:
        goal = Grasp.Goal()
        goal.width = width
        goal.speed = self._grasp_speed
        goal.force = self._grasp_force
        goal.epsilon.inner = self._grasp_eps_inner
        goal.epsilon.outer = self._grasp_eps_outer

        if not self._grasp_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Grasp action server unavailable, deferring command")
            return False

        self.get_logger().debug(
            f"Sending gripper grasp goal: width={goal.width:.3f} m (cmd={command:.2f})"
        )
        send_future = self._grasp_client.send_goal_async(goal)
        send_future.add_done_callback(partial(self._goal_sent_cb, action_name="grasp"))
        self._active_goal = send_future
        return True

    def _retry_pending(self) -> None:
        if self._pending_width is None:
            return
        if self._active_goal is not None and not self._active_goal.done():
            return
        width = self._pending_width
        self._pending_width = None
        self._send_grasp_goal(width, command=width / self._width_range)

    def _hardware_joint_state_cb(self, msg: JointState) -> None:
        if self._width_range <= 0.0:
            return

        width = self._extract_width(msg)
        if width is None:
            return

        normalized = (width - self._min_width) / self._width_range
        normalized = max(0.0, min(1.0, normalized))

        normalized_msg = JointState()
        normalized_msg.header = msg.header
        normalized_msg.name = ["gripper_width_normalized"]
        normalized_msg.position = [normalized]
        self._normalized_joint_state_pub.publish(normalized_msg)

    @staticmethod
    def _extract_width(msg: JointState) -> Optional[float]:
        if msg.position:
            if len(msg.position) == 1:
                return float(msg.position[0])
            width = sum(float(value) for value in msg.position[:2])
            return max(0.0, width)
        return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrankaGripperBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
