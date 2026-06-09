#!/usr/bin/env python3
"""
Dual Joy-Con Teleoperation ROS 2 Node

Node for controlling dual robots with both Joy-Cons.
Left Joy-Con controls left robot, right Joy-Con controls right robot.
"""

import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose, PoseStamped

# Import embedded Joy-Con library
from joycon_teleop.joycon_lib import GyroTrackingJoyCon, get_L_id, get_R_id
from rclpy.node import Node
from visualization_msgs.msg import Marker

# Import GLM for quaternion math
try:
    from glm import conjugate, quat
except ImportError:
    print("ERROR: PyGLM is required. Install with: pip install PyGLM")
    exit(1)

# ANSI color codes for terminal output
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"  # No Color


class JoyConDualTeleopNode(Node):
    """ROS 2 Node for dual Joy-Con teleoperation"""

    def __init__(self):
        super().__init__("joycon_dual_teleop_node")

        # Declare parameters
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("world_frame", "base")
        self.declare_parameter("stick_velocity_scale", 0.3)
        self.declare_parameter("button_velocity_scale", 0.3)
        self.declare_parameter("position_limits", [2.0, 2.0, 2.0])

        # Connection parameters
        self.declare_parameter("wait_for_connection", True)
        self.declare_parameter("auto_reconnect", True)
        self.declare_parameter("connection_timeout", 30.0)  # Longer for dual connection
        self.declare_parameter("retry_interval", 2.0)

        # Get parameters
        self.publish_rate = self.get_parameter("publish_rate").value
        self.world_frame = self.get_parameter("world_frame").value
        self.stick_velocity_scale = self.get_parameter("stick_velocity_scale").value
        self.button_velocity_scale = self.get_parameter("button_velocity_scale").value
        self.position_limits = self.get_parameter("position_limits").value

        self.wait_for_connection = self.get_parameter("wait_for_connection").value
        self.auto_reconnect = self.get_parameter("auto_reconnect").value
        self.connection_timeout = self.get_parameter("connection_timeout").value
        self.retry_interval = self.get_parameter("retry_interval").value

        # Publishers for left Joy-Con
        self.left_pose_pub = self.create_publisher(PoseStamped, "joycon/left/pose", 10)
        self.left_marker_pub = self.create_publisher(Marker, "joycon/left/marker", 10)

        # Publishers for right Joy-Con
        self.right_pose_pub = self.create_publisher(
            PoseStamped, "joycon/right/pose", 10
        )
        self.right_marker_pub = self.create_publisher(Marker, "joycon/right/marker", 10)

        # Subscribers for robots' current end-effector poses
        self.left_robot_pose_sub = self.create_subscription(
            PoseStamped, "/left_arm/current_pose", self.left_robot_pose_callback, 10
        )
        self.right_robot_pose_sub = self.create_subscription(
            PoseStamped, "/right_arm/current_pose", self.right_robot_pose_callback, 10
        )
        self.left_robot_current_pose: Optional[Pose] = None
        self.right_robot_current_pose: Optional[Pose] = None

        # Joy-Con connections
        self.left_joycon: Optional[GyroTrackingJoyCon] = None
        self.right_joycon: Optional[GyroTrackingJoyCon] = None
        self.left_was_connected = False
        self.right_was_connected = False

        # Control states for left Joy-Con
        self.left_control_enabled = False
        self.left_orientation_control_enabled = False
        self.left_robot_orientation_at_enable = None
        self.left_reference_orientation = None

        # Control states for right Joy-Con
        self.right_control_enabled = False
        self.right_orientation_control_enabled = False
        self.right_robot_orientation_at_enable = None
        self.right_reference_orientation = None

        # Initial poses
        self.left_pose = Pose()
        self.left_pose.position.x = -0.3
        self.left_pose.position.y = 0.0
        self.left_pose.position.z = 0.3
        self.left_pose.orientation.x = 0.0
        self.left_pose.orientation.y = 0.0
        self.left_pose.orientation.z = 0.7071068
        self.left_pose.orientation.w = 0.7071068

        self.right_pose = Pose()
        self.right_pose.position.x = 0.3
        self.right_pose.position.y = 0.0
        self.right_pose.position.z = 0.3
        self.right_pose.orientation.x = 0.0
        self.right_pose.orientation.y = 0.0
        self.right_pose.orientation.z = 0.7071068
        self.right_pose.orientation.w = 0.7071068

        self.last_update_time = time.time()

        # Connect to Joy-Cons
        self.connect_joycons()

        # Create timer for publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        # Create monitoring timer
        if self.auto_reconnect:
            self.monitor_timer = self.create_timer(5.0, self.monitor_connections)

        self.get_logger().info(f"{GREEN}========================================{NC}")
        self.get_logger().info(f"{GREEN}Dual Joy-Con Teleoperation Started{NC}")
        self.get_logger().info(f"{GREEN}========================================{NC}")
        self.get_logger().info(f"Publishing at {self.publish_rate} Hz")
        self.get_logger().info("========================================")

    def connect_joycons(self):
        """Connect to both Joy-Con controllers"""
        self.get_logger().info("Connecting to both Joy-Cons...")

        max_retries = (
            int(self.connection_timeout / self.retry_interval)
            if self.wait_for_connection
            else 3
        )

        # Connect to left Joy-Con
        self.get_logger().info("Attempting to connect to Left Joy-Con...")
        for attempt in range(max_retries):
            try:
                left_id = get_L_id()
                if None not in left_id:
                    self.left_joycon = GyroTrackingJoyCon(*left_id)
                    self.get_logger().info(f"{GREEN}✓ Left Joy-Con connected{NC}")
                    self.get_logger().info(
                        "  Calibrating left (keep still for 2 seconds)..."
                    )
                    self.left_joycon.calibrate(seconds=2)
                    time.sleep(2.5)
                    self.left_was_connected = True
                    self.reset_reference("left")
                    break
                else:
                    if self.wait_for_connection:
                        self.get_logger().info(
                            f"Waiting for Left Joy-Con... (attempt {attempt + 1}/{max_retries})"
                        )
                    else:
                        self.get_logger().warn(
                            f"✗ Left Joy-Con not found (attempt {attempt + 1}/{max_retries})"
                        )
            except Exception as e:
                self.get_logger().error(
                    f"✗ Error connecting to Left Joy-Con (attempt {attempt + 1}/{max_retries}): {e}"
                )

            if attempt < max_retries - 1:
                time.sleep(self.retry_interval)

        # Connect to right Joy-Con
        self.get_logger().info("Attempting to connect to Right Joy-Con...")
        for attempt in range(max_retries):
            try:
                right_id = get_R_id()
                if None not in right_id:
                    self.right_joycon = GyroTrackingJoyCon(*right_id)
                    self.get_logger().info(f"{GREEN}✓ Right Joy-Con connected{NC}")
                    self.get_logger().info(
                        "  Calibrating right (keep still for 2 seconds)..."
                    )
                    self.right_joycon.calibrate(seconds=2)
                    time.sleep(2.5)
                    self.right_was_connected = True
                    self.reset_reference("right")
                    break
                else:
                    if self.wait_for_connection:
                        self.get_logger().info(
                            f"Waiting for Right Joy-Con... (attempt {attempt + 1}/{max_retries})"
                        )
                    else:
                        self.get_logger().warn(
                            f"✗ Right Joy-Con not found (attempt {attempt + 1}/{max_retries})"
                        )
            except Exception as e:
                self.get_logger().error(
                    f"✗ Error connecting to Right Joy-Con (attempt {attempt + 1}/{max_retries}): {e}"
                )

            if attempt < max_retries - 1:
                time.sleep(self.retry_interval)

        # Check if at least one connected
        if not self.left_joycon and not self.right_joycon:
            if self.wait_for_connection:
                self.get_logger().error("No Joy-Cons connected!")
                raise RuntimeError("No Joy-Cons connected after timeout!")
            else:
                self.get_logger().warn(
                    "No Joy-Cons connected - node will continue without input"
                )

    def left_robot_pose_callback(self, msg: PoseStamped):
        """Callback for left robot's current end-effector pose"""
        self.left_robot_current_pose = msg.pose

    def right_robot_pose_callback(self, msg: PoseStamped):
        """Callback for right robot's current end-effector pose"""
        self.right_robot_current_pose = msg.pose

    def reset_reference(self, side="both"):
        """Reset reference orientation"""
        if side in ["left", "both"] and self.left_joycon:
            q = self.left_joycon.direction_Q
            self.left_reference_orientation = quat(q.w, q.x, q.y, q.z)
            self.get_logger().info(
                f"{GREEN}Reset left Joy-Con reference orientation{NC}"
            )

        if side in ["right", "both"] and self.right_joycon:
            q = self.right_joycon.direction_Q
            self.right_reference_orientation = quat(q.w, q.x, q.y, q.z)
            self.get_logger().info(
                f"{GREEN}Reset right Joy-Con reference orientation{NC}"
            )

    def monitor_connections(self):
        """Monitor Joy-Con connections and log only when status changes"""
        # Check left Joy-Con
        if self.left_joycon:
            try:
                _ = self.left_joycon.get_status()
                left_connected = True
            except Exception:
                left_connected = False

            if left_connected != self.left_was_connected:
                if left_connected:
                    self.get_logger().info(f"{GREEN}✓ Left Joy-Con: Connected{NC}")
                else:
                    self.get_logger().warn(f"{RED}✗ Left Joy-Con: Disconnected{NC}")
                    if self.auto_reconnect:
                        self.get_logger().info(
                            "Attempting to reconnect Left Joy-Con..."
                        )
                        self.reconnect("left")
                self.left_was_connected = left_connected

        # Check right Joy-Con
        if self.right_joycon:
            try:
                _ = self.right_joycon.get_status()
                right_connected = True
            except Exception:
                right_connected = False

            if right_connected != self.right_was_connected:
                if right_connected:
                    self.get_logger().info(f"{GREEN}✓ Right Joy-Con: Connected{NC}")
                else:
                    self.get_logger().warn(f"{RED}✗ Right Joy-Con: Disconnected{NC}")
                    if self.auto_reconnect:
                        self.get_logger().info(
                            "Attempting to reconnect Right Joy-Con..."
                        )
                        self.reconnect("right")
                self.right_was_connected = right_connected

    def reconnect(self, side: str):
        """Attempt to reconnect a Joy-Con"""
        try:
            if side == "left":
                left_id = get_L_id()
                if None not in left_id:
                    self.left_joycon = GyroTrackingJoyCon(*left_id)
                    self.get_logger().info(f"{GREEN}✓ Left Joy-Con reconnected{NC}")
                    self.left_joycon.calibrate(seconds=2)
                    time.sleep(2.5)
                    self.reset_reference("left")
                else:
                    self.get_logger().warn("Left Joy-Con not found during reconnection")
            elif side == "right":
                right_id = get_R_id()
                if None not in right_id:
                    self.right_joycon = GyroTrackingJoyCon(*right_id)
                    self.get_logger().info(f"{GREEN}✓ Right Joy-Con reconnected{NC}")
                    self.right_joycon.calibrate(seconds=2)
                    time.sleep(2.5)
                    self.reset_reference("right")
                else:
                    self.get_logger().warn(
                        "Right Joy-Con not found during reconnection"
                    )
        except Exception as e:
            self.get_logger().error(f"Failed to reconnect {side} Joy-Con: {e}")

    def normalize_stick(self, raw_value, center=2048, deadzone=200):
        """
        Normalize Joy-Con analog stick from raw 12-bit value to -1.0 to 1.0

        The Joy-Con sticks output 12-bit values (0-4095) centered around 2048.
        This function applies a deadzone and normalizes to the -1 to 1 range.

        Args:
            raw_value: Raw stick value (0-4095)
            center: Center position (typically 2048)
            deadzone: Deadzone radius to ignore stick drift

        Returns:
            Normalized value from -1.0 to 1.0
        """
        offset = raw_value - center

        # Apply deadzone
        if abs(offset) < deadzone:
            return 0.0

        # Normalize to -1.0 to 1.0
        if offset > 0:
            return min(1.0, (offset - deadzone) / (4095 - center - deadzone))
        else:
            return max(-1.0, (offset + deadzone) / (center - deadzone))

    def get_relative_orientation(
        self, current_quat: quat, reference_quat: quat, robot_orientation_at_enable
    ) -> quat:
        """Calculate relative orientation from reference"""
        if reference_quat is None or robot_orientation_at_enable is None:
            return current_quat

        relative_joycon_rotation = current_quat * conjugate(reference_quat)
        new_robot_orientation = relative_joycon_rotation * robot_orientation_at_enable
        return new_robot_orientation

    def update_joycon_control(
        self,
        joycon,
        pose,
        control_enabled_attr,
        orientation_control_enabled_attr,
        reference_orientation_attr,
        robot_orientation_at_enable_attr,
        robot_current_pose,
        dt,
        side_name,
    ):
        """Update control for a single Joy-Con"""
        if not joycon:
            return

        try:
            status = joycon.get_status()

            # Access analog stick with correct nested structure
            stick_data = status.get("analog-sticks", {}).get(side_name, {})
            stick_x = self.normalize_stick(stick_data.get("horizontal", 2048))
            stick_y = self.normalize_stick(stick_data.get("vertical", 2048))

            # Access buttons with correct nested structure
            side_buttons = status.get("buttons", {}).get(side_name, {})

            # Get current control states
            control_enabled = getattr(self, control_enabled_attr)
            orientation_control_enabled = getattr(
                self, orientation_control_enabled_attr
            )

            # Toggle position control
            toggle_button = "x" if side_name == "right" else "up"
            if side_buttons.get(toggle_button) == 1:
                control_enabled = not control_enabled
                setattr(self, control_enabled_attr, control_enabled)
                msg = "enabled" if control_enabled else "disabled"
                self.get_logger().info(
                    f"{GREEN}{side_name.capitalize()} position control {msg}{NC}"
                )
                time.sleep(0.3)

            # Toggle orientation control
            orient_button = "b" if side_name == "right" else "down"
            if side_buttons.get(orient_button) == 1:
                orientation_control_enabled = not orientation_control_enabled
                setattr(
                    self, orientation_control_enabled_attr, orientation_control_enabled
                )
                if orientation_control_enabled:
                    setattr(
                        self,
                        robot_orientation_at_enable_attr,
                        robot_current_pose.orientation if robot_current_pose else None,
                    )
                    self.reset_reference(side_name)
                msg = "enabled" if orientation_control_enabled else "disabled"
                self.get_logger().info(
                    f"{GREEN}{side_name.capitalize()} orientation control {msg}{NC}"
                )
                time.sleep(0.3)

            # Update position if control enabled
            if control_enabled:
                velocity_x = stick_x * self.stick_velocity_scale
                velocity_y = stick_y * self.stick_velocity_scale

                pose.position.x += velocity_x * dt
                pose.position.y += velocity_y * dt

                # Z from buttons
                z_button = "zr" if side_name == "right" else "zl"
                z_up_button = "r" if side_name == "right" else "l"

                if side_buttons.get(z_button) == 1:
                    pose.position.z -= self.button_velocity_scale * dt
                if side_buttons.get(z_up_button) == 1:
                    pose.position.z += self.button_velocity_scale * dt

                # Apply limits
                pose.position.x = max(
                    -self.position_limits[0],
                    min(self.position_limits[0], pose.position.x),
                )
                pose.position.y = max(
                    -self.position_limits[1],
                    min(self.position_limits[1], pose.position.y),
                )
                pose.position.z = max(
                    0.0, min(self.position_limits[2], pose.position.z)
                )

            # Update orientation if control enabled
            reference_orientation = getattr(self, reference_orientation_attr)
            robot_orientation_at_enable = getattr(
                self, robot_orientation_at_enable_attr
            )

            if (
                orientation_control_enabled
                and reference_orientation
                and robot_orientation_at_enable
            ):
                q = joycon.direction_Q
                current_quat = quat(q.w, q.x, q.y, q.z)

                robot_orient_quat = quat(
                    robot_orientation_at_enable.w,
                    robot_orientation_at_enable.x,
                    robot_orientation_at_enable.y,
                    robot_orientation_at_enable.z,
                )

                new_orientation = self.get_relative_orientation(
                    current_quat, reference_orientation, robot_orient_quat
                )

                pose.orientation.w = new_orientation.w
                pose.orientation.x = new_orientation.x
                pose.orientation.y = new_orientation.y
                pose.orientation.z = new_orientation.z

        except Exception as e:
            self.get_logger().error(f"Error updating {side_name} Joy-Con control: {e}")

    def timer_callback(self):
        """Main control loop"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update left Joy-Con control
        self.update_joycon_control(
            self.left_joycon,
            self.left_pose,
            "left_control_enabled",
            "left_orientation_control_enabled",
            "left_reference_orientation",
            "left_robot_orientation_at_enable",
            self.left_robot_current_pose,
            dt,
            "left",
        )

        # Update right Joy-Con control
        self.update_joycon_control(
            self.right_joycon,
            self.right_pose,
            "right_control_enabled",
            "right_orientation_control_enabled",
            "right_reference_orientation",
            "right_robot_orientation_at_enable",
            self.right_robot_current_pose,
            dt,
            "right",
        )

        # Publish left pose
        if self.left_joycon:
            left_pose_msg = PoseStamped()
            left_pose_msg.header.stamp = self.get_clock().now().to_msg()
            left_pose_msg.header.frame_id = self.world_frame
            left_pose_msg.pose = self.left_pose
            self.left_pose_pub.publish(left_pose_msg)

        # Publish right pose
        if self.right_joycon:
            right_pose_msg = PoseStamped()
            right_pose_msg.header.stamp = self.get_clock().now().to_msg()
            right_pose_msg.header.frame_id = self.world_frame
            right_pose_msg.pose = self.right_pose
            self.right_pose_pub.publish(right_pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JoyConDualTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
