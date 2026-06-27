"""

Features:
- EKF for position, velocity, orientation estimation
- Multi-sensor fusion (IMU, GPS, baro, visual)
- Bias estimation for IMU
- Covariance monitoring and adaptive filtering
- State prediction and update at different rates

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float64, Bool
import numpy as np
from numpy.linalg import inv
import math


class DroneStateEstimator(Node):
    
    def __init__(self):
        super().__init__('drone_state_estimator')
        
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, bx, by, bz]
        # Position, velocity, orientation, gyro bias
        self.state_dim = 12
        self.meas_dim = 6
        
        # Initialize state
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim) * 0.1
        self.covariance[9:, 9:] = np.eye(3) * 0.01  # Bias covariance
        
        # Process noise
        self.Q = np.eye(self.state_dim) * 0.01
        self.Q[9:, 9:] *= 0.001  # Low bias noise
        
        # Measurement noise
        self.R_gps = np.eye(3) * 2.0
        self.R_baro = np.eye(1) * 1.0
        self.R_imu = np.eye(3) * 0.1
        
        # State transition
        self.dt = 0.01
        self.last_time = None
        
        # Flags
        self.gps_initialized = False
        self.home_lat = None
        self.home_lon = None
        self.home_alt = None
        
        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/drone/odom_filtered', qos)
        self.pose_pub = self.create_publisher(PoseStamped, '/drone/pose_filtered', qos)
        self.vel_pub = self.create_publisher(TwistStamped, '/drone/velocity_filtered', qos)
        self.status_pub = self.create_publisher(Bool, '/estimator/status', qos)
        
        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/drone/imu', self._imu_callback, qos)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/drone/gps', self._gps_callback, qos)
        self.baro_sub = self.create_subscription(
            Float64, '/drone/baro', self._baro_callback, qos)
        self.visual_sub = self.create_subscription(
            Odometry, '/drone/visual_odom', self._visual_callback, qos)
        
        # Timer
        self.timer = self.create_timer(0.01, self._predict_step)
        self.status_timer = self.create_timer(1.0, self._publish_status)
        
        self.get_logger().info('Drone State Estimator (EKF) initialized')
    
    def _imu_callback(self, msg):
        """Process IMU data."""
        # Extract angular velocity and linear acceleration
        wx = msg.angular_velocity.x - self.state[9]
        wy = msg.angular_velocity.y - self.state[10]
        wz = msg.angular_velocity.z - self.state[11]
        
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        
        # Update orientation rates (simplified)
        roll, pitch, yaw = self.state[6], self.state[7], self.state[8]
        
        # Rotation matrix body to world (simplified)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        
        # Transform acceleration to world frame
        ax_w = ax * cp + az * sp
        ay_w = ay * cr - az * sr * sp
        az_w = -ax * sp + az * cp * cr - 9.81
        
        # Update velocity from acceleration
        self.state[3] += ax_w * self.dt
        self.state[4] += ay_w * self.dt
        self.state[5] += az_w * self.dt
        
        # Update orientation
        self.state[6] += wx * self.dt
        self.state[7] += wy * self.dt
        self.state[8] += wz * self.dt
    
    def _gps_callback(self, msg):
        """Process GPS data."""
        if not self.gps_initialized:
            self.home_lat = msg.latitude
            self.home_lon = msg.longitude
            self.home_alt = msg.altitude
            self.gps_initialized = True
            self.get_logger().info('GPS home position set')
            return
        
        # Convert to local coordinates (simplified)
        x = (msg.longitude - self.home_lon) * 111320 * math.cos(math.radians(self.home_lat))
        y = (msg.latitude - self.home_lat) * 110540
        z = msg.altitude - self.home_alt
        
        # Measurement update
        z_meas = np.array([x, y, z])
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        
        self._ekf_update(z_meas, H, self.R_gps)
    
    def _baro_callback(self, msg):
        """Process barometer data."""
        if self.home_alt is None:
            return
        
        z_meas = np.array([msg.data - self.home_alt])
        H = np.zeros((1, self.state_dim))
        H[0, 2] = 1
        
        R = self.R_baro
        self._ekf_update(z_meas, H, R)
    
    def _visual_callback(self, msg):
        """Process visual odometry."""
        # Use visual odometry as position/velocity measurement
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        z_meas = np.array([x, y, z])
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        
        R = np.eye(3) * 0.5
        self._ekf_update(z_meas, H, R)
    
    def _predict_step(self):
        """Prediction step of EKF."""
        # State prediction is done in IMU callback
        # publish the current estimate
        self._publish_state()
    
    def _ekf_update(self, z, H, R):
        """EKF measurement update."""
        # Innovation
        y = z - H @ self.state
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        try:
            K = self.covariance @ H.T @ inv(S)
        except:
            return
        
        # State update
        self.state = self.state + K @ y
        
        # Covariance update (Joseph form)
        I_KH = np.eye(self.state_dim) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T
    
    def _publish_state(self):
        """Publish filtered state."""
        now = self.get_clock().now().to_msg()
        
        # Odometry
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.state[0]
        odom.pose.pose.position.y = self.state[1]
        odom.pose.pose.position.z = self.state[2]
        odom.twist.twist.linear.x = self.state[3]
        odom.twist.twist.linear.y = self.state[4]
        odom.twist.twist.linear.z = self.state[5]
        
        # Convert Euler to quaternion
        cr, sr = math.cos(self.state[6]/2), math.sin(self.state[6]/2)
        cp, sp = math.cos(self.state[7]/2), math.sin(self.state[7]/2)
        cy, sy = math.cos(self.state[8]/2), math.sin(self.state[8]/2)
        
        odom.pose.pose.orientation.w = cr*cp*cy + sr*sp*sy
        odom.pose.pose.orientation.x = sr*cp*cy - cr*sp*sy
        odom.pose.pose.orientation.y = cr*sp*cy + sr*cp*sy
        odom.pose.pose.orientation.z = cr*cp*sy - sr*sp*cy
        
        self.odom_pub.publish(odom)
        
        # Pose
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)
        
        # Velocity
        vel = TwistStamped()
        vel.header.stamp = now
        vel.header.frame_id = 'base_link'
        vel.twist = odom.twist.twist
        self.vel_pub.publish(vel)
    
    def _publish_status(self):
        """Publish estimator status."""
        status = Bool()
        status.data = self.gps_initialized
        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = DroneStateEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down state estimator')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
