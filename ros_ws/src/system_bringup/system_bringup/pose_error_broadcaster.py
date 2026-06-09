#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


class PoseErrorBroadcaster(Node):
    def __init__(self):
        super().__init__("pose_error_broadcaster")
        self.declare_parameter("current_pose_topic", "current_pose")
        self.declare_parameter("target_pose_topic", "target_pose")
        self.declare_parameter("output_topic", "pose_error")
        self.declare_parameter("publish_rate", 100.0)

        self._current_pose = None
        self._target_pose = None
        self._frame_mismatch_warned = False

        current_topic = self.get_parameter("current_pose_topic").value
        target_topic = self.get_parameter("target_pose_topic").value
        output_topic = self.get_parameter("output_topic").value
        publish_rate = float(self.get_parameter("publish_rate").value)
        if publish_rate <= 0.0:
            publish_rate = 100.0
            self.get_logger().warn("publish_rate <= 0; defaulting to 100.0 Hz.")

        self._current_sub = self.create_subscription(
            PoseStamped, current_topic, self._on_current_pose, 10
        )
        self._target_sub = self.create_subscription(
            PoseStamped, target_topic, self._on_target_pose, 10
        )
        self._publisher = self.create_publisher(TwistStamped, output_topic, 10)
        self._timer = self.create_timer(1.0 / publish_rate, self._publish_error)

    def _on_current_pose(self, msg):
        self._current_pose = msg

    def _on_target_pose(self, msg):
        self._target_pose = msg

    def _publish_error(self):
        if self._current_pose is None or self._target_pose is None:
            return

        current_frame = self._current_pose.header.frame_id
        target_frame = self._target_pose.header.frame_id
        if current_frame and target_frame and current_frame != target_frame:
            if not self._frame_mismatch_warned:
                self.get_logger().warn(
                    "Pose frames differ (current: '%s', target: '%s'); "
                    "skipping pose error publishing." % (current_frame, target_frame)
                )
                self._frame_mismatch_warned = True
            return

        if current_frame == target_frame:
            self._frame_mismatch_warned = False

        dx = self._target_pose.pose.position.x - self._current_pose.pose.position.x
        dy = self._target_pose.pose.position.y - self._current_pose.pose.position.y
        dz = self._target_pose.pose.position.z - self._current_pose.pose.position.z

        q_curr = (
            self._current_pose.pose.orientation.w,
            self._current_pose.pose.orientation.x,
            self._current_pose.pose.orientation.y,
            self._current_pose.pose.orientation.z,
        )
        q_tgt = (
            self._target_pose.pose.orientation.w,
            self._target_pose.pose.orientation.x,
            self._target_pose.pose.orientation.y,
            self._target_pose.pose.orientation.z,
        )
        q_curr_conj = (q_curr[0], -q_curr[1], -q_curr[2], -q_curr[3])
        q_err = _quat_multiply(q_tgt, q_curr_conj)

        w, x, y, z = q_err
        norm_q = math.sqrt(w * w + x * x + y * y + z * z)
        if norm_q > 0.0:
            w, x, y, z = w / norm_q, x / norm_q, y / norm_q, z / norm_q

        if w < 0.0:
            w, x, y, z = -w, -x, -y, -z

        v_norm = math.sqrt(x * x + y * y + z * z)
        if v_norm < 1e-9:
            rx = ry = rz = 0.0
        else:
            angle = 2.0 * math.atan2(v_norm, w)
            axis_x, axis_y, axis_z = x / v_norm, y / v_norm, z / v_norm
            rx = axis_x * angle
            ry = axis_y * angle
            rz = axis_z * angle

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = current_frame or target_frame
        msg.twist.linear.x = dx
        msg.twist.linear.y = dy
        msg.twist.linear.z = dz
        msg.twist.angular.x = rx
        msg.twist.angular.y = ry
        msg.twist.angular.z = rz
        self._publisher.publish(msg)


def main():
    rclpy.init()
    node = PoseErrorBroadcaster()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
