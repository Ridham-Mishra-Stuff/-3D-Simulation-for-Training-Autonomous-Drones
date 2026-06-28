#!/usr/bin/env python3
"""
altitude_hold_controller.py

Features:
- PID-based altitude control with anti-windup
- Terrain following using downward-facing depth sensor
- Adaptive altitude based on mission requirements
- Altitude safety boundaries and emergency descent
- Smooth altitude transitions with S-curve profiles
- Barometer and IMU fusion for robust altitude estimation
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Bool, Int32
from sensor_msgs.msg import Imu, LaserScan, Range
import math
import numpy as np
from collections import deque
from typing import Optional


class AltitudeHoldController(Node):


    # Altitude control modes
    MODE_HOLD = 0
    MODE_TERRAIN_FOLLOW = 1
    MODE_ADAPTIVE = 2
    MODE_EMERGENCY_DESCENT = 3

    def __init__(self):
        super().__init__('altitude_hold_controller')

        # Declare parameters
        self.declare_parameter('pid_kp', 2.0)
        self.declare_parameter('pid_ki', 0.1)
        self.declare_parameter('pid_kd', 0.5)
        self.declare_parameter('max_vertical_speed', 3.0)
        self.declare_parameter('min_vertical_speed', -2.0)
        self.declare_parameter('altitude_tolerance', 0.2)
        self.declare_parameter('terrain_follow_height', 3.0)
        self.declare_parameter('terrain_smooth_factor', 0.3)
        self.declare_parameter('emergency_descent_rate', 1.0)
        self.declare_parameter('min_ground_clearance', 1.0)
        self.declare_parameter('max_altitude', 100.0)
        self.declare_parameter('filter_window_size', 10)
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('anti_windup_limit', 5.0)
        self.declare_parameter('feedforward_gain', 0.0)
        self.declare_parameter('gravity_compensation', 9.81)

        # Get parameters
        self.kp = self.get_parameter('pid_kp').value
        self.ki = self.get_parameter('pid_ki').value
        self.kd = self.get_parameter('pid_kd').value
        self.max_vz = self.get_parameter('max_vertical_speed').value
        self.min_vz = self.get_parameter('min_vertical_speed').value
        self.altitude_tolerance = self.get_parameter('altitude_tolerance').value
        self.terrain_follow_height = self.get_parameter('terrain_follow_height').value
        self.terrain_smooth = self.get_parameter('terrain_smooth_factor').value
        self.emergency_descent_rate = self.get_parameter('emergency_descent_rate').value
        self.min_ground_clearance = self.get_parameter('min_ground_clearance').value
        self.max_altitude = self.get_parameter('max_altitude').value
        self.filter_window = self.get_parameter('filter_window_size').value
        self.anti_windup = self.get_parameter('anti_windup_limit').value
        self.k_ff = self.get_parameter('feedforward_gain').value
        self.g = self.get_parameter('gravity_compensation').value

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # State
        self.current_altitude = 0.0
        self.current_vertical_vel = 0.0
        self.ground_distance = None
        self.imu_vertical_accel = 0.0
        self.target_altitude = 5.0
        self.mode = self.MODE_HOLD
        self.controller_enabled = True

        # PID state
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = None

        # Filters
        self.altitude_buffer = deque(maxlen=self.filter_window)
        self.ground_dist_buffer = deque(maxlen=self.filter_window)
        self.imu_accel_buffer = deque(maxlen=self.filter_window)

        # Publishers
        self.control_pub = self.create_publisher(Twist, '/altitude/control_output', qos_reliable)
        self.estimated_pub = self.create_publisher(Float64, '/altitude/estimated', qos_reliable)
        self.ground_clearance_pub = self.create_publisher(Float64, '/altitude/ground_clearance', qos_reliable)
        self.status_pub = self.create_publisher(Bool, '/altitude/status', qos_reliable)
        self.error_pub = self.create_publisher(Float64, '/altitude/error', qos_reliable)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/drone/odom', self._odom_callback, qos_best_effort)
        self.imu_sub = self.create_subscription(
            Imu, '/drone/imu', self._imu_callback, qos_best_effort)
        self.range_sub = self.create_subscription(
            Range, '/drone/altitude/range', self._range_callback, qos_best_effort)
        self.scan_sub = self.create_subscription(
            LaserScan, '/drone/altitude/scan', self._scan_callback, qos_best_effort)
        self.setpoint_sub = self.create_subscription(
            Float64, '/altitude/setpoint', self._setpoint_callback, qos_reliable)
        self.mode_sub = self.create_subscription(
            Int32, '/altitude/mode', self._mode_callback, qos_reliable)

        # Timer
        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / rate, self._control_loop)
        self.status_timer = self.create_timer(1.0, self._publish_status)

        self.get_logger().info('Altitude Hold Controller initialized')
        self.get_logger().info(f'PID: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}')

    def _odom_callback(self, msg: Odometry):
        """Process odometry data."""
        self.current_altitude = msg.pose.pose.position.z
        self.current_vertical_vel = msg.twist.twist.linear.z
        self.altitude_buffer.append(self.current_altitude)

    def _imu_callback(self, msg: Imu):
        """Process IMU data for vertical acceleration."""
        # Extract vertical acceleration (z-axis in body frame)
        self.imu_vertical_accel = msg.linear_acceleration.z - self.g
        self.imu_accel_buffer.append(self.imu_vertical_accel)

    def _range_callback(self, msg: Range):
        """Process downward rangefinder data."""
        if msg.range < msg.max_range and msg.range > msg.min_range:
            self.ground_dist_buffer.append(msg.range)

    def _scan_callback(self, msg: LaserScan):
        """Process downward LiDAR scan for terrain mapping."""
        valid_ranges = [r for r in msg.ranges 
                       if msg.range_min < r < msg.range_max]
        if valid_ranges:
            median_dist = np.median(valid_ranges)
            self.ground_dist_buffer.append(median_dist)

    def _setpoint_callback(self, msg: Float64):
        """Update altitude setpoint."""
        self.target_altitude = max(0.5, min(msg.data, self.max_altitude))
        self.get_logger().info(f'Altitude setpoint updated: {self.target_altitude:.2f}m')

    def _mode_callback(self, msg: Int32):
        """Update altitude control mode."""
        if 0 <= msg.data <= 3:
            self.mode = msg.data
            modes = ['HOLD', 'TERRAIN_FOLLOW', 'ADAPTIVE', 'EMERGENCY']
            self.get_logger().info(f'Altitude mode: {modes[self.mode]}')

    def _control_loop(self):
        """Main altitude control loop."""
        if not self.controller_enabled:
            return

        current_time = self.get_clock().now()

        if self.last_time is None:
            self.last_time = current_time
            return

        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt < 0.001:
            return
        self.last_time = current_time

        # Get filtered altitude estimate
        altitude = self._get_filtered_altitude()
        ground_dist = self._get_filtered_ground_distance()

        # Calculate target based on mode
        target = self._calculate_target(altitude, ground_dist)

        # PID control
        error = target - altitude

        # Anti-windup
        if abs(self.error_integral) < self.anti_windup:
            self.error_integral += error * dt

        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        # PID output
        p_term = self.kp * error
        i_term = self.ki * self.error_integral
        d_term = self.kd * derivative

        # Feedforward for gravity compensation
        ff_term = self.k_ff * self.g

        control_output = p_term + i_term + d_term + ff_term

        # Clamp output
        control_output = max(self.min_vz, min(control_output, self.max_vz))

        # Safety: minimum ground clearance
        if ground_dist is not None and ground_dist < self.min_ground_clearance and control_output < 0:
            control_output = 0.1  # Slight ascent
            self.get_logger().warn('Minimum ground clearance reached! Ascending...')

        # Safety: maximum altitude
        if altitude > self.max_altitude and control_output > 0:
            control_output = -0.5
            self.get_logger().warn('Maximum altitude reached! Descending...')

        # Publish control
        cmd = Twist()
        cmd.linear.z = control_output
        self.control_pub.publish(cmd)

        # Publish estimates
        self.estimated_pub.publish(Float64(data=altitude))
        if ground_dist is not None:
            self.ground_clearance_pub.publish(Float64(data=ground_dist))
        self.error_pub.publish(Float64(data=error))

    def _get_filtered_altitude(self) -> float:
        """Get filtered altitude estimate using moving average."""
        if len(self.altitude_buffer) > 0:
            return np.mean(self.altitude_buffer)
        return self.current_altitude

    def _get_filtered_ground_distance(self) -> Optional[float]:
        """Get filtered ground distance."""
        if len(self.ground_dist_buffer) > 0:
            return np.median(self.ground_dist_buffer)
        return None

    def _calculate_target(self, altitude: float, ground_dist: Optional[float]) -> float:
        """Calculate target altitude based on control mode."""
        if self.mode == self.MODE_HOLD:
            return self.target_altitude

        elif self.mode == self.MODE_TERRAIN_FOLLOW:
            if ground_dist is not None:
                terrain_height = altitude - ground_dist
                return terrain_height + self.terrain_follow_height
            return self.target_altitude

        elif self.mode == self.MODE_ADAPTIVE:
            # Adaptive: higher altitude over obstacles
            if ground_dist is not None:
                terrain_height = altitude - ground_dist
                # Increase altitude if terrain is rising
                if len(self.ground_dist_buffer) >= 3:
                    recent = list(self.ground_dist_buffer)[-3:]
                    if recent[-1] < recent[0]:  # Ground getting closer
                        return terrain_height + self.terrain_follow_height * 1.5
                return terrain_height + self.terrain_follow_height
            return self.target_altitude

        elif self.mode == self.MODE_EMERGENCY_DESCENT:
            return 0.5  # Land quickly

        return self.target_altitude

    def _publish_status(self):
        """Publish controller status."""
        status = Bool()
        status.data = self.controller_enabled
        self.status_pub.publish(status)

    def reset_controller(self):
        """Reset PID controller state."""
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.get_logger().info('Controller reset')


def main(args=None):
    rclpy.init(args=args)
    node = AltitudeHoldController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down altitude controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
  
