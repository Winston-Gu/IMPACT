#!/usr/bin/env python3

import argparse

import rclpy
from franka_msgs.srv import SetFullCollisionBehavior
from rclpy.node import Node


def _normalize_namespace(namespace):
    if not namespace:
        return ""
    return "/" + namespace.strip("/")


class CollisionBehaviorSetter(Node):
    def __init__(self, namespace):
        super().__init__("collision_behavior_setter")

        service_name = f"{_normalize_namespace(namespace)}/service_server/set_full_collision_behavior"
        self.cli = self.create_client(SetFullCollisionBehavior, service_name)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")
        self.send_request()

    def send_request(self):
        req = SetFullCollisionBehavior.Request()
        self.get_logger().info("Sending request.")

        req.lower_torque_thresholds_nominal = [40.0, 40.0, 40.0, 40.0, 19.0, 17.0, 14.0]
        req.upper_torque_thresholds_nominal = [60.0, 60.0, 60.0, 60.0, 29.0, 27.0, 24.0]
        req.lower_torque_thresholds_acceleration = [
            45.0,
            45.0,
            45.0,
            45.0,
            19.0,
            17.0,
            14.0,
        ]
        req.upper_torque_thresholds_acceleration = [
            65.0,
            65.0,
            65.0,
            65.0,
            29.0,
            27.0,
            24.0,
        ]
        req.lower_force_thresholds_nominal = [70.0, 70.0, 70.0, 60.0, 60.0, 60.0]
        req.upper_force_thresholds_nominal = [80.0, 80.0, 80.0, 70.0, 70.0, 70.0]
        req.lower_force_thresholds_acceleration = [70.0, 70.0, 70.0, 60.0, 60.0, 60.0]
        req.upper_force_thresholds_acceleration = [80.0, 80.0, 80.0, 70.0, 70.0, 70.0]

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Collision behavior set successfully")
        else:
            self.get_logger().error("Failed to set collision behavior")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Set Franka collision behavior via service call."
    )
    parser.add_argument(
        "--namespace",
        default="",
        help="Robot namespace (e.g., fr3_right). Leave empty for root namespace.",
    )
    parsed_args, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = CollisionBehaviorSetter(parsed_args.namespace)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
