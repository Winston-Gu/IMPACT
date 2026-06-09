#!/usr/bin/env python3
"""
Single Joy-Con Teleoperation ROS 2 Node

Simplified node for controlling a single robot with one Joy-Con.
Uses right Joy-Con by default.
"""

import math
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion

# Import embedded Joy-Con library
from joycon_teleop.joycon_lib import GyroTrackingJoyCon, get_L_id, get_R_id
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# ANSI color codes for terminal output
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"  # No Color


# Import GLM for quaternion math
try:
    from glm import conjugate, quat
except ImportError:
    print("ERROR: PyGLM is required. Install with: pip install PyGLM")
    exit(1)


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert roll-pitch-yaw (radians) to a geometry_msgs Quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class JoyConSingleTeleopNode(Node):
    """ROS 2 Node for single Joy-Con teleoperation"""

    def __init__(self):
        super().__init__("joycon_single_teleop_node")

        # Declare parameters
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("world_frame", "base")
        self.declare_parameter("stick_velocity_scale", 0.3)
        self.declare_parameter("button_velocity_scale", 0.3)
        self.declare_parameter("position_limits", [2.0, 2.0, 2.0])

        # Connection parameters
        self.declare_parameter("joycon_side", "right")  # 'right' or 'left'
        self.declare_parameter("wait_for_connection", True)
        self.declare_parameter("auto_reconnect", True)
        self.declare_parameter("connection_timeout", 10.0)
        self.declare_parameter("retry_interval", 2.0)
        self.declare_parameter("robot_pose_topic", "/current_pose")
        self.declare_parameter("target_pose_topic", "")
        self.declare_parameter(
            "gripper_command_topic", "/gripper/gripper_position_controller/commands"
        )
        self.declare_parameter("gripper_state_topic", "/gripper/joint_states")

        # Get parameters
        self.publish_rate = self.get_parameter("publish_rate").value
        self.world_frame = self.get_parameter("world_frame").value
        self.stick_velocity_scale = self.get_parameter("stick_velocity_scale").value
        self.button_velocity_scale = self.get_parameter("button_velocity_scale").value
        self.position_limits = self.get_parameter("position_limits").value

        self.joycon_side = self.get_parameter("joycon_side").value
        self.wait_for_connection = self.get_parameter("wait_for_connection").value
        self.auto_reconnect = self.get_parameter("auto_reconnect").value
        self.connection_timeout = self.get_parameter("connection_timeout").value
        self.retry_interval = self.get_parameter("retry_interval").value
        self.robot_pose_topic = self.get_parameter("robot_pose_topic").value
        self.target_pose_topic = self.get_parameter("target_pose_topic").value
        self.gripper_command_topic = self.get_parameter("gripper_command_topic").value
        self.gripper_state_topic = self.get_parameter("gripper_state_topic").value

        # Publishers
        if not self.target_pose_topic:
            self.target_pose_topic = f"joycon/{self.joycon_side}/pose"
        self.pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 10)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, self.gripper_command_topic, 10
        )
        # self.marker_pub = self.create_publisher(Marker, f'joycon/{self.joycon_side}/marker', 10)

        # Gripper state subscription (updates current opening)
        self.create_subscription(
            JointState,
            self.gripper_state_topic,
            self.gripper_state_callback,
            10,
        )

        # Subscriber for robot's current end-effector pose
        self.robot_pose_sub = self.create_subscription(
            PoseStamped, self.robot_pose_topic, self.robot_pose_callback, 10
        )
        self.robot_current_pose: Optional[Pose] = None

        # Joy-Con connection
        self.joycon: Optional[GyroTrackingJoyCon] = None
        self.was_connected = False

        # Control states
        self.control_enabled = False
        self.orientation_control_enabled = False
        self.robot_orientation_at_enable = None
        self.pose_initialized = False  # Track if pose has been initialized from robot

        # Initial pose (will be initialized from robot when control is enabled)
        # DO NOT use this until pose_initialized = True
        self.pose = None  # Explicitly None to catch bugs
        # self.pose.position.x = 0.3
        # self.pose.position.y = 0.0
        # self.pose.position.z = 0.3
        # self.pose.orientation.x = 0.0
        # self.pose.orientation.y = 0.0
        # self.pose.orientation.z = 0.7071068
        # self.pose.orientation.w = 0.7071068

        # Reference orientation
        self.reference_orientation = None
        self.last_update_time = time.time()

        # Smoothing for real robot (exponential moving average)
        self.smoothing_alpha = 0.3  # Lower = more smoothing (0.1-0.5 recommended)
        self.target_velocity = [0.0, 0.0, 0.0]  # Smoothed velocity commands
        self.gripper_command = (
            1.0  # Normalized desired gripper opening (0.0 closed, 1.0 open)
        )
        self.gripper_actual_width: Optional[float] = None
        self.gripper_synced = False
        self._prev_gripper_buttons = {"close": 0, "open": 0}

        # Default “point-down” orientation defined via roll/pitch/yaw
        self.default_down_orientation = rpy_to_quaternion(
            math.radians(-179.7909),
            math.radians(0.1166),
            math.radians(-44.4991),
        )

        # Connect to Joy-Con
        self.connect_joycon()

        # Create timer for publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        # Create monitoring timer (check connection every 10 seconds)
        self.monitor_timer = self.create_timer(10.0, self.monitor_connection)

        self.get_logger().info(f"{GREEN}Single Joy-Con Teleoperation Started{NC}")
        self.get_logger().info(f"Using {self.joycon_side} Joy-Con")
        self.get_logger().info(f"Publishing at {self.publish_rate} Hz")

    def connect_joycon(self):
        """Connect to Joy-Con controller"""
        self.get_logger().info(f"Connecting to {self.joycon_side} Joy-Con...")

        max_retries = (
            int(self.connection_timeout / self.retry_interval)
            if self.wait_for_connection
            else 3
        )

        for attempt in range(max_retries):
            try:
                # Get Joy-Con ID based on side
                if self.joycon_side == "left":
                    joycon_id = get_L_id()
                else:  # default to right
                    joycon_id = get_R_id()

                if None not in joycon_id:
                    self.joycon = GyroTrackingJoyCon(*joycon_id)
                    self.get_logger().info(
                        f"{GREEN}✓ {self.joycon_side.capitalize()} Joy-Con connected{NC}"
                    )
                    self.get_logger().info(
                        "  Calibrating (keep still for 2 seconds)..."
                    )
                    self.joycon.calibrate(seconds=2)
                    time.sleep(2.5)
                    self.was_connected = True
                    self.reset_reference()
                    return
                else:
                    if self.wait_for_connection:
                        self.get_logger().info(
                            f"Waiting for Joy-Con... (attempt {attempt + 1}/{max_retries})"
                        )
                    else:
                        self.get_logger().warn(
                            f"✗ Joy-Con not found (attempt {attempt + 1}/{max_retries})"
                        )
            except Exception as e:
                self.get_logger().error(
                    f"✗ Error connecting (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    self.get_logger().info(f"Retrying in {self.retry_interval}s...")

            if attempt < max_retries - 1:
                time.sleep(self.retry_interval)

        # Connection failed
        if self.wait_for_connection:
            self.get_logger().error(
                f"Joy-Con connection timeout after {self.connection_timeout}s"
            )
            raise RuntimeError("Joy-Con connection failed!")
        else:
            self.get_logger().warn(
                "Joy-Con connection failed - node will continue without input"
            )

    def robot_pose_callback(self, msg: PoseStamped):
        """Callback to receive robot's current end-effector pose"""
        self.robot_current_pose = msg.pose

    def normalize_stick(self, raw_value, center=2048, deadzone=400):
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

    def initialize_pose_from_robot(self):
        """Initialize Joy-Con pose from robot's current end-effector pose"""
        if self.robot_current_pose:
            self.pose = Pose()
            self.pose.position.x = self.robot_current_pose.position.x
            self.pose.position.y = self.robot_current_pose.position.y
            self.pose.position.z = self.robot_current_pose.position.z
            self.pose.orientation = self.robot_current_pose.orientation
            self.pose_initialized = True
            self.get_logger().info(
                f"{GREEN}Initialized Joy-Con pose from robot: "
                f"[{self.pose.position.x:.3f}, {self.pose.position.y:.3f}, {self.pose.position.z:.3f}]{NC}"
            )
            return True
        else:
            self.get_logger().warn(
                f"{YELLOW}Cannot initialize pose - robot pose not yet received{NC}"
            )
            self.control_enabled = False  # Disable control if initialization failed
            return False

    def reset_reference(self):
        """Reset reference orientation"""
        if self.joycon:
            q = self.joycon.direction_Q
            self.reference_orientation = quat(q.w, q.x, q.y, q.z)
            self.get_logger().info(f"{GREEN}Reset Joy-Con reference orientation{NC}")

    def recalibrate_orientation(self):
        """Snap EE orientation back to default pointing-down pose and re-sync Joy-Con."""
        if not self.pose_initialized or self.pose is None:
            self.get_logger().warn(
                f"{YELLOW}Cannot recalibrate - pose not initialized{NC}"
            )
            return

        # Apply default orientation
        self.pose.orientation.x = self.default_down_orientation.x
        self.pose.orientation.y = self.default_down_orientation.y
        self.pose.orientation.z = self.default_down_orientation.z
        self.pose.orientation.w = self.default_down_orientation.w

        # Update reference orientation so Joy-Con matches new pose
        self.reset_reference()
        self.robot_orientation_at_enable = self.pose.orientation

        self.get_logger().info(
            f"{GREEN}Recalibrated orientation to default pointing-down pose{NC}"
        )

    def monitor_connection(self):
        """Monitor Joy-Con connection and log status periodically"""
        if self.joycon:
            try:
                status = self.joycon.get_status()
                is_connected = True

                # Log detailed status using correct nested structure
                side_buttons = status.get("buttons", {}).get(self.joycon_side, {})
                stick = status.get("analog-sticks", {}).get(self.joycon_side, {})
                pressed_buttons = [k for k, v in side_buttons.items() if v == 1]

                status_msg = f"{GREEN}✓ Joy-Con: Connected{NC}"
                if pressed_buttons:
                    status_msg += f" | Buttons: {pressed_buttons}"
                # Check if stick is moved (center is 2048)
                stick_h = stick.get("horizontal", 2048) - 2048
                stick_v = stick.get("vertical", 2048) - 2048
                if abs(stick_h) > 100 or abs(stick_v) > 100:  # Deadzone threshold
                    status_msg += f" | Stick: ({stick_h}, {stick_v})"

                self.get_logger().info(status_msg)

            except Exception as e:
                is_connected = False
                self.get_logger().error(f"{RED}✗ Joy-Con: Disconnected - {e}{NC}")

            # Handle disconnection
            if not is_connected and self.was_connected:
                if self.auto_reconnect:
                    self.get_logger().info("Attempting to reconnect...")
                    self.reconnect()

            self.was_connected = is_connected
        else:
            self.get_logger().warn(f"{YELLOW}Joy-Con not initialized{NC}")

    def reconnect(self):
        """Attempt to reconnect Joy-Con"""
        try:
            if self.joycon_side == "left":
                joycon_id = get_L_id()
            else:
                joycon_id = get_R_id()

            if None not in joycon_id:
                self.joycon = GyroTrackingJoyCon(*joycon_id)
                self.get_logger().info(f"{GREEN}✓ Joy-Con reconnected{NC}")
                self.joycon.calibrate(seconds=2)
                time.sleep(2.5)
                self.reset_reference()
            else:
                self.get_logger().warn("Joy-Con not found during reconnection")
        except Exception as e:
            self.get_logger().error(f"Failed to reconnect: {e}")

    def gripper_state_callback(self, msg: JointState):
        if msg.position:
            normalized_width = max(0.0, min(1.0, float(msg.position[0])))
            self.gripper_actual_width = normalized_width
            if not self.gripper_synced:
                self.gripper_command = normalized_width
                self.gripper_synced = True

    def _publish_gripper_command(self, value: float):
        clamped = max(0.0, min(1.0, value))
        self.gripper_command = clamped
        gripper_cmd = Float64MultiArray()
        gripper_cmd.data = [clamped]
        self.gripper_pub.publish(gripper_cmd)

    def get_relative_orientation(
        self, current_quat: quat, reference_quat: quat, robot_orientation_at_enable
    ) -> quat:
        """Calculate relative orientation from reference"""
        if reference_quat is None or robot_orientation_at_enable is None:
            return current_quat

        relative_joycon_rotation = current_quat * conjugate(reference_quat)
        new_robot_orientation = relative_joycon_rotation * robot_orientation_at_enable
        return new_robot_orientation

    def timer_callback(self):
        """Main control loop"""
        if not self.joycon:
            return

        try:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time

            # Get Joy-Con state
            status = self.joycon.get_status()

            # Access analog stick based on Joy-Con side (note: plural 'analog-sticks')
            stick_data = status.get("analog-sticks", {}).get(self.joycon_side, {})
            raw_horizontal = stick_data.get("horizontal", 2048)
            raw_vertical = stick_data.get("vertical", 2048)
            # Flip axes so pushing forward/back maps to X and left/right to +Y
            stick_x = self.normalize_stick(raw_vertical)
            stick_y = -self.normalize_stick(raw_horizontal)

            # Access buttons based on Joy-Con side (buttons are nested by side)
            side_buttons = status.get("buttons", {}).get(self.joycon_side, {})
            shared_buttons = status.get("buttons", {}).get("shared", {})

            # Toggle position control (X button for right, Up for left)
            toggle_button = "x" if self.joycon_side == "right" else "up"
            if side_buttons.get(toggle_button) == 1:
                was_enabled = self.control_enabled
                self.control_enabled = not self.control_enabled

                # Initialize pose from robot when enabling control
                if not was_enabled and self.control_enabled:
                    self.initialize_pose_from_robot()

                msg = "enabled" if self.control_enabled else "disabled"
                self.get_logger().info(f"{GREEN}Position control {msg}{NC}")
                time.sleep(0.3)

            # Toggle orientation control (B button for right, Down for left)
            orient_button = "b" if self.joycon_side == "right" else "down"
            if side_buttons.get(orient_button) == 1:
                self.orientation_control_enabled = not self.orientation_control_enabled
                if self.orientation_control_enabled:
                    self.robot_orientation_at_enable = (
                        self.robot_current_pose.orientation
                        if self.robot_current_pose
                        else None
                    )
                    self.reset_reference()
                msg = "enabled" if self.orientation_control_enabled else "disabled"
                self.get_logger().info(f"{GREEN}Orientation control {msg}{NC}")
                time.sleep(0.3)

            # Home button triggers orientation recalibration
            if shared_buttons.get("home") == 1:
                self.recalibrate_orientation()
                time.sleep(0.3)

            # Update position if control enabled and pose is initialized
            if self.control_enabled and self.pose_initialized and self.pose is not None:
                # XY from stick
                velocity_x = stick_x * self.stick_velocity_scale
                velocity_y = stick_y * self.stick_velocity_scale

                self.pose.position.x += velocity_x * dt
                self.pose.position.y += velocity_y * dt

                # Z from buttons
                z_button = "zr" if self.joycon_side == "right" else "zl"
                z_up_button = "r" if self.joycon_side == "right" else "l"

                if side_buttons.get(z_button) == 1:
                    self.pose.position.z -= self.button_velocity_scale * dt
                if side_buttons.get(z_up_button) == 1:
                    self.pose.position.z += self.button_velocity_scale * dt

                # Apply limits
                self.pose.position.x = max(
                    -self.position_limits[0],
                    min(self.position_limits[0], self.pose.position.x),
                )
                self.pose.position.y = max(
                    -self.position_limits[1],
                    min(self.position_limits[1], self.pose.position.y),
                )
                self.pose.position.z = max(
                    0.0, min(self.position_limits[2], self.pose.position.z)
                )

            # Gripper control: Y button closes (grasp), A button opens (release)
            close_pressed = side_buttons.get("y", 0)
            open_pressed = side_buttons.get("a", 0)
            if close_pressed == 1 and self._prev_gripper_buttons["close"] == 0:
                self._publish_gripper_command(0.0)
            if open_pressed == 1 and self._prev_gripper_buttons["open"] == 0:
                self._publish_gripper_command(1.0)
            self._prev_gripper_buttons["close"] = close_pressed
            self._prev_gripper_buttons["open"] = open_pressed

            # Update orientation if control enabled
            if (
                self.orientation_control_enabled
                and self.reference_orientation
                and self.robot_orientation_at_enable
                and self.pose is not None
            ):
                q = self.joycon.direction_Q
                current_quat = quat(q.w, q.x, q.y, q.z)

                robot_orient_quat = quat(
                    self.robot_orientation_at_enable.w,
                    self.robot_orientation_at_enable.x,
                    self.robot_orientation_at_enable.y,
                    self.robot_orientation_at_enable.z,
                )

                new_orientation = self.get_relative_orientation(
                    current_quat, self.reference_orientation, robot_orient_quat
                )

                self.pose.orientation.w = new_orientation.w
                self.pose.orientation.x = new_orientation.x
                self.pose.orientation.y = new_orientation.y
                self.pose.orientation.z = new_orientation.z

            # Publish pose only when control is enabled AND pose has been initialized AND pose is not None
            if self.control_enabled and self.pose_initialized and self.pose is not None:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = self.world_frame
                pose_msg.pose = self.pose
                self.pose_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = JoyConSingleTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
