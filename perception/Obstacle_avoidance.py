#!/usr/bin/env python3
"""
Features:
- Potential field obstacle avoidance
- Dynamic window approach for velocity selection
- Multi-sensor fusion (LiDAR, depth camera, ultrasonic)
- Emergency braking and recovery
- Obstacle tracking and prediction
- Configurable safety margins
- Smooth trajectory replanning

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, Point, PointStamped, Vector3
from sensor_msgs.msg import LaserScan, PointCloud2, Range
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64, Int32
import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class Obstacle:
  
    x: float
    y: float
    z: float
    distance: float
    angle: float
    confidence: float
    timestamp: float
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class ObstacleAvoidance(Node):
    
    # Avoidance modes
    MODE_POTENTIAL_FIELD = 0
    MODE_DYNAMIC_WINDOW = 1
    MODE_REACTIVE = 2
    MODE_HYBRID = 3

    def __init__(self):
        super().__init__('obstacle_avoidance')

        # Parameters
        self.declare_parameter('avoidance_mode', 3)
        self.declare_parameter('safety_radius', 2.0)
        self.declare_parameter('critical_radius', 1.0)
        self.declare_parameter('repulsive_gain', 5.0)
        self.declare_parameter('attractive_gain', 1.0)
        self.declare_parameter('max_avoidance_force', 3.0)
        self.declare_parameter('prediction_horizon', 2.0)
        self.declare_parameter('min_obstacle_size', 0.1)
        self.declare_parameter('sensor_fusion_weight_lidar', 0.5)
        self.declare_parameter('sensor_fusion_weight_depth', 0.3)
        self.declare_parameter('sensor_fusion_weight_sonar', 0.2)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('enable_avoidance', True)
        self.declare_parameter('emergency_stop_distance', 0.5)
        self.declare_parameter('recovery_time', 2.0)

        # Get parameters
        self.mode = self.get_parameter('avoidance_mode').value
        self.safety_radius = self.get_parameter('safety_radius').value
        self.critical_radius = self.get_parameter('critical_radius').value
        self.repulsive_gain = self.get_parameter('repulsive_gain').value
        self.attractive_gain = self.get_parameter('attractive_gain').value
        self.max_force = self.get_parameter('max_avoidance_force').value
        self.prediction_horizon = self.get_parameter('prediction_horizon').value
        self.emergency_dist = self.get_parameter('emergency_stop_distance').value
        self.recovery_time = self.get_parameter('recovery_time').value
        self.enabled = self.get_parameter('enable_avoidance').value

        # State
        self.current_pose = None
        self.current_velocity = None
        self.obstacles: List[Obstacle] = []
        self.lidar_data = None
        self.point_cloud = None
        self.sonar_data = None
        self.danger_level = 0.0
        self.emergency_active = False
        self.emergency_start_time = None
        self.target_velocity = Twist()

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/avoidance/cmd_vel', qos_reliable)
        self.obstacle_pub = self.create_publisher(PointStamped, '/avoidance/obstacles', qos_reliable)
        self.danger_pub = self.create_publisher(Float64, '/avoidance/danger_level', qos_reliable)
        self.status_pub = self.create_publisher(Bool, '/avoidance/status', qos_reliable)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/drone/odom', self._odom_callback, qos_best_effort)
        self.scan_sub = self.create_subscription(
            LaserScan, '/drone/scan', self._scan_callback, qos_best_effort)
        self.cloud_sub = self.create_subscription(
            PointCloud2, '/drone/points', self._cloud_callback, qos_best_effort)
        self.sonar_sub = self.create_subscription(
            Range, '/drone/sonar', self._sonar_callback, qos_best_effort)
        self.cmd_vel_in_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_in_callback, qos_reliable)
        self.enable_sub = self.create_subscription(
            Bool, '/avoidance/enable', self._enable_callback, qos_reliable)
        self.mode_sub = self.create_subscription(
            Int32, '/avoidance/mode', self._mode_callback, qos_reliable)

        # Timer
        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / rate, self._control_loop)

        self.get_logger().info('Obstacle Avoidance initialized')
        self.get_logger().info(f'Mode: {self.mode}, Safety radius: {self.safety_radius}m')

    def _odom_callback(self, msg: Odometry):
        """Process odometry."""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

    def _scan_callback(self, msg: LaserScan):
        """Process LiDAR scan."""
        self.lidar_data = msg
        self._process_lidar(msg)

    def _cloud_callback(self, msg: PointCloud2):
        """Process point cloud."""
        self.point_cloud = msg
        # Simplified: in production, use sensor_msgs_py.point_cloud2

    def _sonar_callback(self, msg: Range):
        """Process ultrasonic data."""
        self.sonar_data = msg
        if msg.range < msg.max_range and self.current_pose:
            # Sonar typically points downward or forward
            obs = Obstacle(
                x=self.current_pose.position.x + msg.range,
                y=self.current_pose.position.y,
                z=self.current_pose.position.z,
                distance=msg.range,
                angle=0.0,
                confidence=0.7,
                timestamp=self.get_clock().now().nanoseconds / 1e9
            )
            self._update_obstacle(obs)

    def _cmd_vel_in_callback(self, msg: Twist):
        """Receive target velocity command."""
        self.target_velocity = msg

    def _enable_callback(self, msg: Bool):
        """Enable/disable avoidance."""
        self.enabled = msg.data
        self.get_logger().info(f'Avoidance {"enabled" if self.enabled else "disabled"}')

    def _mode_callback(self, msg: Int32):
        """Update avoidance mode."""
        if 0 <= msg.data <= 3:
            self.mode = msg.data
            modes = ['POTENTIAL_FIELD', 'DYNAMIC_WINDOW', 'REACTIVE', 'HYBRID']
            self.get_logger().info(f'Avoidance mode: {modes[self.mode]}')

    def _process_lidar(self, scan: LaserScan):
        """Extract obstacles from LiDAR scan."""
        angle = scan.angle_min
        for range_val in scan.ranges:
            if scan.range_min < range_val < scan.range_max:
                if self.current_pose:
                    x = self.current_pose.position.x + range_val * math.cos(angle)
                    y = self.current_pose.position.y + range_val * math.sin(angle)
                    z = self.current_pose.position.z

                    obs = Obstacle(
                        x=x, y=y, z=z,
                        distance=range_val,
                        angle=angle,
                        confidence=1.0 - (range_val / scan.range_max),
                        timestamp=self.get_clock().now().nanoseconds / 1e9
                    )
                    self._update_obstacle(obs)
            angle += scan.angle_increment

    def _update_obstacle(self, new_obs: Obstacle):
        """Update obstacle list with new detection."""
        # Check if similar obstacle already exists
        for i, obs in enumerate(self.obstacles):
            dist = math.sqrt((obs.x - new_obs.x)**2 + (obs.y - new_obs.y)**2)
            if dist < 0.5:  # Same obstacle
                # Update with weighted average
                alpha = 0.3
                self.obstacles[i] = Obstacle(
                    x=alpha * new_obs.x + (1-alpha) * obs.x,
                    y=alpha * new_obs.y + (1-alpha) * obs.y,
                    z=alpha * new_obs.z + (1-alpha) * obs.z,
                    distance=new_obs.distance,
                    angle=new_obs.angle,
                    confidence=max(new_obs.confidence, obs.confidence),
                    timestamp=new_obs.timestamp,
                    velocity=((new_obs.x - obs.x) / 0.1, 
                             (new_obs.y - obs.y) / 0.1, 0.0)
                )
                return

        self.obstacles.append(new_obs)
        # Limit obstacle count
        if len(self.obstacles) > 100:
            self.obstacles.pop(0)

    def _control_loop(self):
        """Main avoidance control loop."""
        if not self.enabled or self.current_pose is None:
            self.cmd_vel_pub.publish(self.target_velocity)
            return

        # Check emergency state
        if self.emergency_active:
            elapsed = (self.get_clock().now().nanoseconds / 1e9 - 
                      self.emergency_start_time)
            if elapsed < self.recovery_time:
                # Emergency stop
                self.cmd_vel_pub.publish(Twist())
                self.danger_pub.publish(Float64(data=1.0))
                return
            else:
                self.emergency_active = False
                self.get_logger().info('Emergency recovery complete')

        # Calculate danger level
        self.danger_level = self._calculate_danger_level()
        self.danger_pub.publish(Float64(data=self.danger_level))

        # Check for emergency
        if self.danger_level > 0.9:
            self._trigger_emergency()
            return

        # Apply avoidance based on mode
        if self.mode == self.MODE_POTENTIAL_FIELD:
            corrected = self._potential_field_avoidance()
        elif self.mode == self.MODE_DYNAMIC_WINDOW:
            corrected = self._dynamic_window_avoidance()
        elif self.MODE_REACTIVE:
            corrected = self._reactive_avoidance()
        else:  # HYBRID
            corrected = self._hybrid_avoidance()

        self.cmd_vel_pub.publish(corrected)
        self.status_pub.publish(Bool(data=self.danger_level > 0.3))

    def _calculate_danger_level(self) -> float:
        """Calculate overall danger level from obstacles."""
        if not self.obstacles:
            return 0.0

        min_dist = min(obs.distance for obs in self.obstacles)

        if min_dist < self.emergency_dist:
            return 1.0
        elif min_dist < self.critical_radius:
            return 0.5 + 0.5 * (self.critical_radius - min_dist) / (self.critical_radius - self.emergency_dist)
        elif min_dist < self.safety_radius:
            return 0.5 * (self.safety_radius - min_dist) / (self.safety_radius - self.critical_radius)

        return 0.0

    def _trigger_emergency(self):
        """Trigger emergency stop."""
        self.emergency_active = True
        self.emergency_start_time = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().error('EMERGENCY: Obstacle too close! Stopping...')
        self.cmd_vel_pub.publish(Twist())

    def _potential_field_avoidance(self) -> Twist:
        """Potential field method for obstacle avoidance."""
        if not self.obstacles:
            return self.target_velocity

        # Calculate repulsive forces from obstacles
        fx, fy, fz = 0.0, 0.0, 0.0

        for obs in self.obstacles:
            if obs.distance < self.safety_radius:
                # Repulsive force
                force = self.repulsive_gain * (1.0 / obs.distance - 1.0 / self.safety_radius)
                force = min(force, self.max_force)

                dx = self.current_pose.position.x - obs.x
                dy = self.current_pose.position.y - obs.y
                dz = self.current_pose.position.z - obs.z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                if dist > 0.001:
                    fx += force * dx / dist
                    fy += force * dy / dist
                    fz += force * dz / dist

        # Apply to velocity command
        corrected = Twist()
        corrected.linear.x = self.target_velocity.linear.x + fx * 0.1
        corrected.linear.y = self.target_velocity.linear.y + fy * 0.1
        corrected.linear.z = self.target_velocity.linear.z + fz * 0.1
        corrected.angular.z = self.target_velocity.angular.z

        return corrected

    def _dynamic_window_avoidance(self) -> Twist:
        """Dynamic window approach for velocity selection."""
        # Simplified DWA: sample velocities and select best
        best_vel = self.target_velocity
        best_score = -float('inf')

        # Sample around target velocity
        for vx in np.linspace(-1, 1, 5):
            for vy in np.linspace(-1, 1, 5):
                sample = Twist()
                sample.linear.x = self.target_velocity.linear.x + vx * 0.5
                sample.linear.y = self.target_velocity.linear.y + vy * 0.5
                sample.linear.z = self.target_velocity.linear.z

                score = self._evaluate_velocity(sample)
                if score > best_score:
                    best_score = score
                    best_vel = sample

        return best_vel

    def _evaluate_velocity(self, vel: Twist) -> float:
        """Evaluate a velocity command for safety."""
        # Predict future position
        dt = self.prediction_horizon
        future_x = self.current_pose.position.x + vel.linear.x * dt
        future_y = self.current_pose.position.y + vel.linear.y * dt
        future_z = self.current_pose.position.z + vel.linear.z * dt

        # Check distance to all obstacles
        min_dist = float('inf')
        for obs in self.obstacles:
            # Predict obstacle position
            pred_obs_x = obs.x + obs.velocity[0] * dt
            pred_obs_y = obs.y + obs.velocity[1] * dt

            dist = math.sqrt((future_x - pred_obs_x)**2 + 
                           (future_y - pred_obs_y)**2 + 
                           (future_z - obs.z)**2)
            min_dist = min(min_dist, dist)

        # Score: higher is better (close to target, far from obstacles)
        target_score = -(vel.linear.x - self.target_velocity.linear.x)**2 -                        (vel.linear.y - self.target_velocity.linear.y)**2
        safety_score = min_dist if min_dist < self.safety_radius else self.safety_radius

        return target_score + safety_score * 10.0

    def _reactive_avoidance(self) -> Twist:
        """Simple reactive avoidance based on nearest obstacle."""
        if not self.obstacles:
            return self.target_velocity

        nearest = min(self.obstacles, key=lambda o: o.distance)

        if nearest.distance > self.safety_radius:
            return self.target_velocity

        corrected = Twist()

        # Steer away from nearest obstacle
        dx = self.current_pose.position.x - nearest.x
        dy = self.current_pose.position.y - nearest.y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist > 0.001:
            avoidance_strength = (self.safety_radius - nearest.distance) / self.safety_radius
            corrected.linear.x = self.target_velocity.linear.x + (dx / dist) * avoidance_strength
            corrected.linear.y = self.target_velocity.linear.y + (dy / dist) * avoidance_strength

        corrected.linear.z = self.target_velocity.linear.z
        corrected.angular.z = self.target_velocity.angular.z

        return corrected

    def _hybrid_avoidance(self) -> Twist:
        """Combine multiple avoidance strategies."""
        pf = self._potential_field_avoidance()
        reactive = self._reactive_avoidance()

        # Weighted combination
        corrected = Twist()
        w1 = 0.6  # Potential field weight
        w2 = 0.4  # Reactive weight

        corrected.linear.x = w1 * pf.linear.x + w2 * reactive.linear.x
        corrected.linear.y = w1 * pf.linear.y + w2 * reactive.linear.y
        corrected.linear.z = w1 * pf.linear.z + w2 * reactive.linear.z
        corrected.angular.z = self.target_velocity.angular.z

        return corrected


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down obstacle avoidance')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
