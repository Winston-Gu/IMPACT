#!/usr/bin/env python3
"""
Joy-Con Teleoperation ROS 2 Node - Analog Stick Control

Clean implementation using:
- Analog sticks for XYZ velocity control
- Gyroscope for orientation tracking
- No accelerometer (unreliable for position)
"""

import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion

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


class JoyConTeleopNode(Node):
    """ROS 2 Node for Joy-Con teleoperation using analog sticks"""

    def __init__(self):
        super().__init__("joycon_teleop_node")

        # Declare parameters
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter(
            "world_frame", "base"
        )  # Changed to 'base' to match controller expectations
        self.declare_parameter(
            "stick_velocity_scale", 0.3
        )  # m/s per full stick deflection
        self.declare_parameter(
            "button_velocity_scale", 0.3
        )  # m/s for button-based Z control
        self.declare_parameter(
            "position_limits", [2.0, 2.0, 2.0]
        )  # XYZ limits in meters

        # Declare joycon_connection parameters
        self.declare_parameter("joycon_connection.connect", "both")
        self.declare_parameter("joycon_connection.wait_for_connection", False)
        self.declare_parameter("joycon_connection.auto_reconnect", True)
        self.declare_parameter("joycon_connection.connection_timeout", 10.0)
        self.declare_parameter("joycon_connection.retry_interval", 2.0)

        # Get parameters
        self.publish_rate = self.get_parameter("publish_rate").value
        self.world_frame = self.get_parameter("world_frame").value
        self.stick_velocity_scale = self.get_parameter("stick_velocity_scale").value
        self.button_velocity_scale = self.get_parameter("button_velocity_scale").value
        self.position_limits = self.get_parameter("position_limits").value

        # Get joycon_connection parameters
        self.connect_which = self.get_parameter("joycon_connection.connect").value
        self.wait_for_connection = self.get_parameter(
            "joycon_connection.wait_for_connection"
        ).value
        self.auto_reconnect = self.get_parameter(
            "joycon_connection.auto_reconnect"
        ).value
        self.connection_timeout = self.get_parameter(
            "joycon_connection.connection_timeout"
        ).value
        self.retry_interval = self.get_parameter(
            "joycon_connection.retry_interval"
        ).value

        # Publishers
        self.left_pose_pub = self.create_publisher(PoseStamped, "joycon/left/pose", 10)
        self.right_pose_pub = self.create_publisher(
            PoseStamped, "joycon/right/pose", 10
        )
        self.left_marker_pub = self.create_publisher(Marker, "joycon/left/marker", 10)
        self.right_marker_pub = self.create_publisher(Marker, "joycon/right/marker", 10)

        # Subscriber for robot's current end-effector pose
        self.robot_pose_sub = self.create_subscription(
            PoseStamped, "/current_pose", self.robot_pose_callback, 10
        )
        self.robot_current_pose: Optional[Pose] = None

        # Joy-Con connections
        self.left_joycon: Optional[GyroTrackingJoyCon] = None
        self.right_joycon: Optional[GyroTrackingJoyCon] = None

        # Control enable states
        self.left_control_enabled = False
        self.right_control_enabled = False

        # Orientation control enable (separate from position control)
        self.left_orientation_control_enabled = False
        self.right_orientation_control_enabled = False

        # Stored robot orientation when orientation control is enabled
        self.left_robot_orientation_at_enable = None
        self.right_robot_orientation_at_enable = None

        # Poses - start at safe default positions
        # Pointing in +Y direction (90 degree rotation around Z axis)
        self.left_pose = Pose()
        self.left_pose.position.x = -0.3
        self.left_pose.position.y = 0.0
        self.left_pose.position.z = 0.3
        # Quaternion for 90° rotation around Z-axis: (0, 0, sin(45°), cos(45°))
        self.left_pose.orientation.x = 0.0
        self.left_pose.orientation.y = 0.0
        self.left_pose.orientation.z = 0.7071068
        self.left_pose.orientation.w = 0.7071068

        self.right_pose = Pose()
        self.right_pose.position.x = 0.3
        self.right_pose.position.y = 0.0
        self.right_pose.position.z = 0.3
        # Quaternion for 90° rotation around Z-axis
        self.right_pose.orientation.x = 0.0
        self.right_pose.orientation.y = 0.0
        self.right_pose.orientation.z = 0.7071068
        self.right_pose.orientation.w = 0.7071068

        # Reference orientations (set when ZL/ZR pressed)
        self.left_reference_orientation = None
        self.right_reference_orientation = None

        # Track previous connection status for monitoring
        self.left_was_connected = False
        self.right_was_connected = False

        self.last_update_time = time.time()

        # Connect to Joy-Cons
        self.connect_joycons()

        # Create timer for publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        # Create monitoring timer (check connection every 5 seconds)
        if self.auto_reconnect:
            self.monitor_timer = self.create_timer(5.0, self.monitor_connections)

        self.get_logger().info(f"{GREEN}========================================{NC}")
        self.get_logger().info(f"{GREEN}Joy-Con Teleoperation Node Started{NC}")
        self.get_logger().info(f"{GREEN}========================================{NC}")
        self.get_logger().info(f"Publishing at {self.publish_rate} Hz")
        # self.get_logger().info('Controls:')
        # self.get_logger().info('  Left Joy-Con:')
        # self.get_logger().info('    Up button    - Toggle position control on/off')
        # self.get_logger().info('    Down button  - Toggle orientation control on/off')
        # self.get_logger().info('    ZL           - Move down (-Z)')
        # self.get_logger().info('    L            - Move up (+Z)')
        # self.get_logger().info('  Right Joy-Con:')
        # self.get_logger().info('    X button     - Toggle position control on/off')
        # self.get_logger().info('    B button     - Toggle orientation control on/off')
        # self.get_logger().info('    ZR           - Move down (-Z)')
        # self.get_logger().info('    R            - Move up (+Z)')
        # self.get_logger().info('  Analog Stick   - XY position')
        # self.get_logger().info('  Gyro           - Orientation control (only when enabled)')
        self.get_logger().info("========================================")

    def connect_joycons(self):
        """Connect to Joy-Con controllers based on configuration parameters"""
        self.get_logger().info(
            f"Connecting to Joy-Cons (mode: {self.connect_which})..."
        )

        # Calculate max retries from timeout
        max_retries = (
            int(self.connection_timeout / self.retry_interval)
            if self.wait_for_connection
            else 3
        )

        # Connect to left Joy-Con (if requested)
        left_connected = False
        if self.connect_which in ["left", "both"]:
            self.get_logger().info("Attempting to connect to Left Joy-Con...")
            for attempt in range(max_retries):
                try:
                    left_id = get_L_id()
                    if None not in left_id:
                        self.left_joycon = GyroTrackingJoyCon(*left_id)
                        self.get_logger().info("✓ Left Joy-Con connected")
                        self.get_logger().info(
                            "  Calibrating (keep still for 2 seconds)..."
                        )
                        self.left_joycon.calibrate(seconds=2)
                        time.sleep(2.5)
                        left_connected = True
                        self.left_was_connected = (
                            True  # Track initial connection status
                        )
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

            if not left_connected:
                if self.wait_for_connection:
                    self.get_logger().error(
                        f"Left Joy-Con connection timeout after {self.connection_timeout}s"
                    )
                    if self.connect_which == "left":
                        raise RuntimeError("Left Joy-Con connection failed!")
                else:
                    self.get_logger().warn("Left Joy-Con connection failed")

        # Connect to right Joy-Con (if requested)
        right_connected = False
        if self.connect_which in ["right", "both"]:
            self.get_logger().info("Attempting to connect to Right Joy-Con...")
            for attempt in range(max_retries):
                try:
                    right_id = get_R_id()
                    if None not in right_id:
                        self.right_joycon = GyroTrackingJoyCon(*right_id)
                        self.get_logger().info("✓ Right Joy-Con connected")
                        self.get_logger().info(
                            "  Calibrating (keep still for 2 seconds)..."
                        )
                        self.right_joycon.calibrate(seconds=2)
                        time.sleep(2.5)
                        right_connected = True
                        self.right_was_connected = (
                            True  # Track initial connection status
                        )
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

            if not right_connected:
                if self.wait_for_connection:
                    self.get_logger().error(
                        f"Right Joy-Con connection timeout after {self.connection_timeout}s"
                    )
                    if self.connect_which == "right":
                        raise RuntimeError("Right Joy-Con connection failed!")
                else:
                    self.get_logger().warn("Right Joy-Con connection failed")

        # Final check
        if not self.left_joycon and not self.right_joycon:
            self.get_logger().error("No Joy-Cons connected!")
            if self.wait_for_connection:
                raise RuntimeError("No Joy-Cons connected after timeout!")
            else:
                self.get_logger().warn("Node will continue without Joy-Con input")

        # Reset reference orientations
        self.reset_reference("both")

    def robot_pose_callback(self, msg: PoseStamped):
        """Callback to receive robot's current end-effector pose"""
        self.robot_current_pose = msg.pose

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
        if self.connect_which in ["left", "both"]:
            if self.left_joycon:
                try:
                    # Try to get status to verify connection is alive
                    _ = self.left_joycon.get_status()
                    left_connected = True
                except Exception:
                    left_connected = False

                # Only log when status changes
                if left_connected != self.left_was_connected:
                    if left_connected:
                        self.get_logger().info(f"{GREEN}✓ Left Joy-Con: Connected{NC}")
                    else:
                        self.get_logger().warn(f"{RED}✗ Left Joy-Con: Disconnected{NC}")
                        if self.auto_reconnect:
                            self.get_logger().info(
                                "Attempting to reconnect Left Joy-Con..."
                            )
                            self._reconnect_joycon("left")
                    self.left_was_connected = left_connected

        # Check right Joy-Con
        if self.connect_which in ["right", "both"]:
            if self.right_joycon:
                try:
                    # Try to get status to verify connection is alive
                    _ = self.right_joycon.get_status()
                    right_connected = True
                except Exception:
                    right_connected = False

                # Only log when status changes
                if right_connected != self.right_was_connected:
                    if right_connected:
                        self.get_logger().info(f"{GREEN}✓ Right Joy-Con: Connected{NC}")
                    else:
                        self.get_logger().warn(
                            f"{RED}✗ Right Joy-Con: Disconnected{NC}"
                        )
                        if self.auto_reconnect:
                            self.get_logger().info(
                                "Attempting to reconnect Right Joy-Con..."
                            )
                            self._reconnect_joycon("right")
                    self.right_was_connected = right_connected

    def _reconnect_joycon(self, side: str):
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

    def get_relative_orientation(
        self, current_quat: quat, reference_quat: quat, robot_orientation_at_enable
    ) -> quat:
        """Calculate relative orientation from reference

        Args:
            current_quat: Current joycon orientation
            reference_quat: Joycon orientation when orientation control was enabled
            robot_orientation_at_enable: Robot's orientation when orientation control was enabled

        Returns:
            New robot orientation based on joycon movement
        """
        if reference_quat is None or robot_orientation_at_enable is None:
            return current_quat

        # Calculate the relative rotation of the joycon since enabling control
        relative_joycon_rotation = current_quat * conjugate(reference_quat)

        # Apply this relative rotation to the robot's orientation at enable time
        robot_quat_at_enable = quat(
            robot_orientation_at_enable.w,
            robot_orientation_at_enable.x,
            robot_orientation_at_enable.y,
            robot_orientation_at_enable.z,
        )

        # Flip roll by negating X component (joycon coordinate system correction)
        relative_flipped = quat(
            relative_joycon_rotation.w,
            -relative_joycon_rotation.x,
            relative_joycon_rotation.y,
            relative_joycon_rotation.z,
        )

        # Rotate 90° around Z to align joycon frame with robot frame
        z90 = quat(0.7071068, 0.0, 0.0, 0.7071068)  # 90° around Z
        relative_corrected = z90 * relative_flipped

        # Apply to robot's orientation at enable
        new_orientation = relative_corrected * robot_quat_at_enable

        return new_orientation

    def quaternion_to_msg(self, q: quat) -> Quaternion:
        """Convert GLM quaternion to ROS Quaternion message"""
        msg = Quaternion()
        msg.w = float(q.w)
        msg.x = float(q.x)
        msg.y = float(q.y)
        msg.z = float(q.z)
        return msg

    def create_joycon_marker(self, pose: Pose, marker_id: int, color: tuple) -> Marker:
        """Create a cube marker to represent a Joy-Con"""
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "joycon"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Set pose
        marker.pose = pose

        # Set size (Joy-Con dimensions)
        marker.scale.x = 0.03  # width
        marker.scale.y = 0.10  # height
        marker.scale.z = 0.02  # thickness

        # Set color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8

        marker.lifetime.sec = 0

        return marker

    def clamp(self, value, min_val, max_val):
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))

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

    def update_left_pose(self):
        """Update left Joy-Con pose using analog stick + gyro"""
        if not self.left_joycon:
            return

        status = self.left_joycon.get_status()
        dt = time.time() - self.last_update_time

        # Check for orientation control toggle (Down button)
        if status["buttons"]["left"].get("down", 0):
            if (
                not hasattr(self, "_left_ori_toggle_debounce")
                or time.time() - self._left_ori_toggle_debounce > 0.5
            ):
                self.left_orientation_control_enabled = (
                    not self.left_orientation_control_enabled
                )
                self._left_ori_toggle_debounce = time.time()

                # When enabling orientation control, reset reference to current joycon position
                if self.left_orientation_control_enabled:
                    self.reset_reference("left")
                    # Store the robot's current orientation
                    self.left_robot_orientation_at_enable = self.left_pose.orientation
                    self.get_logger().info("Left orientation control ENABLED")
                    self.get_logger().info(
                        f"  Locked robot orientation: quat=[{self.left_robot_orientation_at_enable.x:.3f}, "
                        f"{self.left_robot_orientation_at_enable.y:.3f}, "
                        f"{self.left_robot_orientation_at_enable.z:.3f}, "
                        f"{self.left_robot_orientation_at_enable.w:.3f}]"
                    )
                else:
                    self.get_logger().info("Left orientation control DISABLED")
                time.sleep(0.3)  # Debounce

        # Safety: Check if Up button is pressed to toggle control
        if status["buttons"]["left"].get("up", 0):
            if (
                not hasattr(self, "_left_toggle_debounce")
                or time.time() - self._left_toggle_debounce > 0.5
            ):
                was_enabled = self.left_control_enabled
                self.left_control_enabled = not self.left_control_enabled
                self._left_toggle_debounce = time.time()

                # When enabling control, initialize Joy-Con pose from robot's current pose
                if (
                    not was_enabled
                    and self.left_control_enabled
                    and self.robot_current_pose
                ):
                    self.left_pose.position.x = self.robot_current_pose.position.x
                    self.left_pose.position.y = self.robot_current_pose.position.y
                    self.left_pose.position.z = self.robot_current_pose.position.z
                    self.left_pose.orientation = self.robot_current_pose.orientation
                    # Don't reset reference here - only when orientation control is enabled
                    self.get_logger().info(
                        f"Left position control ENABLED - Initialized from robot pose: "
                        f"[{self.left_pose.position.x:.3f}, "
                        f"{self.left_pose.position.y:.3f}, "
                        f"{self.left_pose.position.z:.3f}]"
                    )
                    self.get_logger().info(
                        "  Orientation locked. Press Down to enable orientation control."
                    )
                elif not was_enabled and self.left_control_enabled:
                    self.get_logger().warn(
                        "Left position control ENABLED - No robot pose available, using default"
                    )
                    self.get_logger().info(
                        "  Orientation locked. Press Down to enable orientation control."
                    )
                else:
                    self.get_logger().info(
                        f"Left position control {'ENABLED' if self.left_control_enabled else 'DISABLED'}"
                    )

        # Get button states for Z control
        zl_pressed = status["buttons"]["left"].get("zl", 0)
        l_pressed = status["buttons"]["left"].get("l", 0)

        # Only update pose if control is enabled
        if self.left_control_enabled:
            # Update orientation from gyroscope ONLY if orientation control is enabled
            if self.left_orientation_control_enabled:
                current_quat = self.left_joycon.direction_Q
                relative_quat = self.get_relative_orientation(
                    current_quat,
                    self.left_reference_orientation,
                    self.left_robot_orientation_at_enable,
                )
                self.left_pose.orientation = self.quaternion_to_msg(relative_quat)
            # Otherwise, keep the current orientation (from robot's initial pose)

            # Get analog stick values (RAW 0-4095, need normalization)
            stick = status["analog-sticks"]["left"]
            stick_x_raw = stick["horizontal"]
            stick_y_raw = stick["vertical"]

            # Normalize from 12-bit (0-4095) to -1.0 to 1.0
            # Center is around 2048, deadzone to avoid drift
            stick_x = self.normalize_stick(stick_x_raw)
            stick_y = self.normalize_stick(stick_y_raw)

            # Map stick to velocity (m/s)
            vel_x = stick_x * self.stick_velocity_scale
            vel_y = stick_y * self.stick_velocity_scale

            # Update XY position from analog stick
            self.left_pose.position.x += vel_x * dt
            self.left_pose.position.y += vel_y * dt

            # Check buttons for Z control
            vel_z = 0.0
            if zl_pressed:  # ZL = move down
                vel_z = -self.button_velocity_scale
            if l_pressed:  # L = move up
                vel_z = self.button_velocity_scale

            # Update Z position from buttons
            self.left_pose.position.z += vel_z * dt

        # Clamp to limits
        self.left_pose.position.x = self.clamp(
            self.left_pose.position.x, -self.position_limits[0], self.position_limits[0]
        )
        self.left_pose.position.y = self.clamp(
            self.left_pose.position.y, -self.position_limits[1], self.position_limits[1]
        )
        self.left_pose.position.z = self.clamp(
            self.left_pose.position.z, 0.0, self.position_limits[2]
        )

    def update_right_pose(self):
        """Update right Joy-Con pose using analog stick + gyro"""
        if not self.right_joycon:
            return

        status = self.right_joycon.get_status()
        dt = time.time() - self.last_update_time

        # Check for orientation control toggle (B button)
        if status["buttons"]["right"].get("b", 0):
            if (
                not hasattr(self, "_right_ori_toggle_debounce")
                or time.time() - self._right_ori_toggle_debounce > 0.5
            ):
                self.right_orientation_control_enabled = (
                    not self.right_orientation_control_enabled
                )
                self._right_ori_toggle_debounce = time.time()

                # When enabling orientation control, reset reference to current joycon position
                if self.right_orientation_control_enabled:
                    self.reset_reference("right")
                    # Store the robot's current orientation
                    self.right_robot_orientation_at_enable = self.right_pose.orientation
                    self.get_logger().info("Right orientation control ENABLED")
                    self.get_logger().info(
                        f"  Locked robot orientation: quat=[{self.right_robot_orientation_at_enable.x:.3f}, "
                        f"{self.right_robot_orientation_at_enable.y:.3f}, "
                        f"{self.right_robot_orientation_at_enable.z:.3f}, "
                        f"{self.right_robot_orientation_at_enable.w:.3f}]"
                    )
                else:
                    self.get_logger().info("Right orientation control DISABLED")
                time.sleep(0.3)  # Debounce

        # Safety: Check if X button is pressed to toggle control
        if status["buttons"]["right"].get("x", 0):
            if (
                not hasattr(self, "_right_toggle_debounce")
                or time.time() - self._right_toggle_debounce > 0.5
            ):
                was_enabled = self.right_control_enabled
                self.right_control_enabled = not self.right_control_enabled
                self._right_toggle_debounce = time.time()

                # When enabling control, initialize Joy-Con pose from robot's current pose
                if (
                    not was_enabled
                    and self.right_control_enabled
                    and self.robot_current_pose
                ):
                    self.right_pose.position.x = self.robot_current_pose.position.x
                    self.right_pose.position.y = self.robot_current_pose.position.y
                    self.right_pose.position.z = self.robot_current_pose.position.z
                    self.right_pose.orientation = self.robot_current_pose.orientation
                    # Don't reset reference here - only when orientation control is enabled

                    # Log orientation for debugging
                    ori = self.robot_current_pose.orientation
                    import math

                    # Convert to Euler for readability
                    x, y, z, w = ori.x, ori.y, ori.z, ori.w
                    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
                    pitch = math.asin(2 * (w * y - z * x))
                    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

                    self.get_logger().info(
                        f"Right position control ENABLED - Initialized from robot pose: "
                        f"[{self.right_pose.position.x:.3f}, "
                        f"{self.right_pose.position.y:.3f}, "
                        f"{self.right_pose.position.z:.3f}]"
                    )
                    self.get_logger().info(
                        f"  Robot orientation (quat): x={ori.x:.3f}, y={ori.y:.3f}, z={ori.z:.3f}, w={ori.w:.3f}"
                    )
                    self.get_logger().info(
                        f"  Robot orientation (euler): roll={math.degrees(roll):.1f}°, pitch={math.degrees(pitch):.1f}°, yaw={math.degrees(yaw):.1f}°"
                    )
                    self.get_logger().info(
                        "  Orientation locked. Press B to enable orientation control."
                    )
                elif not was_enabled and self.right_control_enabled:
                    self.get_logger().warn(
                        "Right position control ENABLED - No robot pose available, using default"
                    )
                    self.get_logger().info(
                        "  Orientation locked. Press B to enable orientation control."
                    )
                else:
                    self.get_logger().info(
                        f"Right position control {'ENABLED' if self.right_control_enabled else 'DISABLED'}"
                    )

        # Get button states for Z control
        zr_pressed = status["buttons"]["right"].get("zr", 0)
        r_pressed = status["buttons"]["right"].get("r", 0)

        # Only update pose if control is enabled
        if self.right_control_enabled:
            # Update orientation from gyroscope ONLY if orientation control is enabled
            if self.right_orientation_control_enabled:
                current_quat = self.right_joycon.direction_Q
                relative_quat = self.get_relative_orientation(
                    current_quat,
                    self.right_reference_orientation,
                    self.right_robot_orientation_at_enable,
                )
                self.right_pose.orientation = self.quaternion_to_msg(relative_quat)
            # Otherwise, keep the current orientation (from robot's initial pose)

            # Get analog stick values (RAW 0-4095, need normalization)
            stick = status["analog-sticks"]["right"]
            stick_x_raw = stick["horizontal"]
            stick_y_raw = stick["vertical"]

            # Normalize from 12-bit (0-4095) to -1.0 to 1.0
            stick_x = self.normalize_stick(stick_x_raw)
            stick_y = self.normalize_stick(stick_y_raw)

            # Map stick to velocity (m/s)
            vel_x = stick_x * self.stick_velocity_scale
            vel_y = stick_y * self.stick_velocity_scale

            # Update XY position from analog stick
            self.right_pose.position.x += vel_x * dt
            self.right_pose.position.y += vel_y * dt

            # Check buttons for Z control
            vel_z = 0.0
            if zr_pressed:  # ZR = move down
                vel_z = -self.button_velocity_scale
            if r_pressed:  # R = move up
                vel_z = self.button_velocity_scale

            # Update Z position from buttons
            self.right_pose.position.z += vel_z * dt

        # Clamp to limits
        self.right_pose.position.x = self.clamp(
            self.right_pose.position.x,
            -self.position_limits[0],
            self.position_limits[0],
        )
        self.right_pose.position.y = self.clamp(
            self.right_pose.position.y,
            -self.position_limits[1],
            self.position_limits[1],
        )
        self.right_pose.position.z = self.clamp(
            self.right_pose.position.z, 0.0, self.position_limits[2]
        )

    def timer_callback(self):
        """Publish Joy-Con poses at specified rate"""
        try:
            self.update_left_pose()
            self.update_right_pose()
            self.last_update_time = time.time()

            now = self.get_clock().now().to_msg()

            # Publish left pose and marker
            if self.left_joycon:
                left_msg = PoseStamped()
                left_msg.header.stamp = now
                left_msg.header.frame_id = self.world_frame
                left_msg.pose = self.left_pose

                # Only publish pose when control is enabled
                if self.left_control_enabled:
                    self.left_pose_pub.publish(left_msg)

                # Always publish marker for visualization
                left_marker = self.create_joycon_marker(
                    self.left_pose, 0, (1.0, 0.0, 0.0)
                )
                self.left_marker_pub.publish(left_marker)

            # Publish right pose and marker
            if self.right_joycon:
                right_msg = PoseStamped()
                right_msg.header.stamp = now
                right_msg.header.frame_id = self.world_frame
                right_msg.pose = self.right_pose

                # Only publish pose when control is enabled
                if self.right_control_enabled:
                    self.right_pose_pub.publish(right_msg)

                # Always publish marker for visualization
                right_marker = self.create_joycon_marker(
                    self.right_pose, 1, (0.0, 0.0, 1.0)
                )
                self.right_marker_pub.publish(right_marker)

        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {e}")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = JoyConTeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
