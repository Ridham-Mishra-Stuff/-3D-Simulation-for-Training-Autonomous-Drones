#!/usr/bin/env python3
"""

Features:
- Real-time metric collection
- CSV logging
- Statistical analysis
- Performance comparison (before/after training)
- Export to various formats

"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, String, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import csv
import json
import os
from datetime import datetime
from collections import deque
import numpy as np


class PerformanceLogger(Node):
  

    def __init__(self):
        super().__init__('performance_logger')

        self.declare_parameter('log_directory', '~/drone_logs')
        self.declare_parameter('log_rate', 10.0)
        self.declare_parameter('max_samples', 10000)
        self.declare_parameter('auto_save', True)
        self.declare_parameter('save_interval', 30.0)

        self.log_dir = os.path.expanduser(self.get_parameter('log_directory').value)
        os.makedirs(self.log_dir, exist_ok=True)

        self.samples = []
        self.mission_metrics = {
            'start_time': None,
            'end_time': None,
            'waypoints_reached': 0,
            'detections': 0,
            'obstacles_avoided': 0,
            'battery_start': 100.0,
            'battery_end': 100.0,
            'distance_traveled': 0.0,
            'max_altitude': 0.0,
            'min_altitude': 1000.0
        }

        self.current_position = None
        self.last_position = None

        # Subscribers
        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, 10)
        self.create_subscription(String, '/mission/status', self._status_callback, 10)
        self.create_subscription(Float64, '/detection/landing_pad/confidence', self._detection_callback, 10)
        self.create_subscription(Float64, '/battery/level', self._battery_callback, 10)
        self.create_subscription(Float64, '/avoidance/danger_level', self._danger_callback, 10)
        self.create_subscription(Bool, '/mission/waypoint_reached', self._wp_reached_callback, 10)

        # Timer
        self.create_timer(1.0 / self.get_parameter('log_rate').value, self._log_loop)
        self.create_timer(self.get_parameter('save_interval').value, self._auto_save)

        self.get_logger().info(f'Performance Logger initialized. Logs: {self.log_dir}')

    def _odom_callback(self, msg):
        self.current_position = msg.pose.pose.position
        alt = self.current_position.z

        self.mission_metrics['max_altitude'] = max(self.mission_metrics['max_altitude'], alt)
        self.mission_metrics['min_altitude'] = min(self.mission_metrics['min_altitude'], alt)

        if self.last_position:
            dx = self.current_position.x - self.last_position.x
            dy = self.current_position.y - self.last_position.y
            dz = self.current_position.z - self.last_position.z
            self.mission_metrics['distance_traveled'] += np.sqrt(dx**2 + dy**2 + dz**2)

        self.last_position = self.current_position

    def _status_callback(self, msg):
        try:
            status = json.loads(msg.data)
            if status.get('state') == 'takeoff' and self.mission_metrics['start_time'] is None:
                self.mission_metrics['start_time'] = datetime.now().isoformat()
            elif status.get('completed'):
                self.mission_metrics['end_time'] = datetime.now().isoformat()
                self._save_mission_summary()
        except:
            pass

    def _detection_callback(self, msg):
        if msg.data > 0.5:
            self.mission_metrics['detections'] += 1

    def _battery_callback(self, msg):
        if self.mission_metrics['start_time'] and self.mission_metrics['battery_start'] == 100.0:
            self.mission_metrics['battery_start'] = msg.data
        self.mission_metrics['battery_end'] = msg.data

    def _danger_callback(self, msg):
        if msg.data > 0.5:
            self.mission_metrics['obstacles_avoided'] += 1

    def _wp_reached_callback(self, msg):
        if msg.data:
            self.mission_metrics['waypoints_reached'] += 1

    def _log_loop(self):
        if self.current_position is None:
            return

        sample = {
            'timestamp': datetime.now().isoformat(),
            'x': self.current_position.x,
            'y': self.current_position.y,
            'z': self.current_position.z,
        }
        self.samples.append(sample)

        if len(self.samples) > self.get_parameter('max_samples').value:
            self.samples.pop(0)

    def _auto_save(self):
        if not self.get_parameter('auto_save').value:
            return

        filename = os.path.join(self.log_dir, f'drone_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        self._save_csv(filename)

    def _save_csv(self, filename):
        if not self.samples:
            return

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.samples[0].keys())
            writer.writeheader()
            writer.writerows(self.samples)

        self.get_logger().info(f'Saved {len(self.samples)} samples to {filename}')

    def _save_mission_summary(self):
        filename = os.path.join(self.log_dir, f'mission_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(filename, 'w') as f:
            json.dump(self.mission_metrics, f, indent=2)

        self.get_logger().info(f'Mission summary saved to {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = PerformanceLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
