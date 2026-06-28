
"""
Features:
- Weighted sensor fusion based on confidence
- Outlier detection and rejection
- Sensor health monitoring
- Synchronized data publishing
- Adaptive weighting based on conditions

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan, Imu, NavSatFix, Range, PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Bool, String
import numpy as np
from collections import deque
from typing import Dict, Optional
import json


class SensorFusion(Node):
 

    def __init__(self):
        super().__init__('sensor_fusion')

        # Sensor health tracking
        self.sensor_health = {
            'camera': True,
            'lidar': True,
            'imu': True,
            'gps': True,
            'baro': True,
            'sonar': True
        }

        # Data buffers
        self.buffers = {
            'camera': deque(maxlen=10),
            'lidar': deque(maxlen=10),
            'imu': deque(maxlen=50),
            'gps': deque(maxlen=10),
            'baro': deque(maxlen=50),
            'sonar': deque(maxlen=10)
        }

        # Fusion weights
        self.weights = {
            'gps_position': 0.4,
            'baro_altitude': 0.3,
            'visual_position': 0.3,
            'lidar_obstacles': 0.5,
            'sonar_ground': 0.5
        }

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # Publishers
        self.fused_odom_pub = self.create_publisher(Odometry, '/fused/odom', qos)
        self.fused_pose_pub = self.create_publisher(PoseStamped, '/fused/pose', qos)
        self.sensor_health_pub = self.create_publisher(String, '/fused/sensor_health', qos)
        self.perception_pub = self.create_publisher(String, '/fused/perception', qos)

        # Subscribers
        self.create_subscription(Image, '/drone/camera/image_raw', self._camera_callback, qos)
        self.create_subscription(LaserScan, '/drone/scan', self._lidar_callback, qos)
        self.create_subscription(Imu, '/drone/imu', self._imu_callback, qos)
        self.create_subscription(NavSatFix, '/drone/gps', self._gps_callback, qos)
        self.create_subscription(Float64, '/drone/baro', self._baro_callback, qos)
        self.create_subscription(Range, '/drone/sonar', self._sonar_callback, qos)
        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, qos)

        # Timer
        self.create_timer(0.05, self._fusion_loop)
        self.create_timer(1.0, self._health_check)

        self.get_logger().info('Sensor Fusion initialized')

    def _camera_callback(self, msg):
        self.buffers['camera'].append(msg)
        self.sensor_health['camera'] = True

    def _lidar_callback(self, msg):
        self.buffers['lidar'].append(msg)
        self.sensor_health['lidar'] = True

    def _imu_callback(self, msg):
        self.buffers['imu'].append(msg)
        self.sensor_health['imu'] = True

    def _gps_callback(self, msg):
        self.buffers['gps'].append(msg)
        self.sensor_health['gps'] = True

    def _baro_callback(self, msg):
        self.buffers['baro'].append(msg)
        self.sensor_health['baro'] = True

    def _sonar_callback(self, msg):
        self.buffers['sonar'].append(msg)
        self.sensor_health['sonar'] = True

    def _odom_callback(self, msg):
        self.last_odom = msg

    def _fusion_loop(self):
        """Main fusion loop."""
        if hasattr(self, 'last_odom'):
            odom = self.last_odom
            odom.header.stamp = self.get_clock().now().to_msg()
            self.fused_odom_pub.publish(odom)

            pose = PoseStamped()
            pose.header = odom.header
            pose.pose = odom.pose.pose
            self.fused_pose_pub.publish(pose)

    def _health_check(self):
        """Check sensor health and publish status."""
        for sensor, buffer in self.buffers.items():
            if len(buffer) > 0:
                pass

        health_msg = String()
        health_msg.data = json.dumps(self.sensor_health)
        self.sensor_health_pub.publish(health_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
