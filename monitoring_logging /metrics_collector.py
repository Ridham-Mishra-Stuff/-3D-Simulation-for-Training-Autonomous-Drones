
import numpy as np
from typing import Dict, List, Deque
from collections import deque
from dataclasses import dataclass
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry


@dataclass
class FlightMetrics:
    total_distance: float = 0.0
    max_altitude: float = 0.0
    max_speed: float = 0.0
    flight_time: float = 0.0
    energy_consumed: float = 0.0
    waypoint_accuracy: float = 0.0


class MetricsCollector(Node):
    def __init__(self):
        super().__init__('metrics_collector')
        self.declare_parameter('window_size', 100)
        
        self.window_size = self.get_parameter('window_size').value
        self.positions = deque(maxlen=self.window_size)
        self.velocities = deque(maxlen=self.window_size)
        self.metrics = FlightMetrics()
        self.start_time = None
        
        self.create_subscription(Odometry, '/drone/odom', self.odom_callback, 10)
        self.create_timer(1.0, self.compute_metrics)

    def odom_callback(self, msg: Odometry):
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.positions.append(pos)
        self.velocities.append(vel)
        
        if self.start_time is None:
            self.start_time = self.get_clock().now()

    def compute_metrics(self):
        if len(self.positions) < 2: return
        
        distances = [np.linalg.norm(self.positions[i] - self.positions[i-1]) 
                    for i in range(1, len(self.positions))]
        self.metrics.total_distance = sum(distances)
        self.metrics.max_altitude = max(p[2] for p in self.positions)
        self.metrics.max_speed = max(np.linalg.norm(v) for v in self.velocities)
        
        if self.start_time:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.metrics.flight_time = elapsed
        
        self.get_logger().info(f'Distance: {self.metrics.total_distance:.1f}m, '
                              f'Max Alt: {self.metrics.max_altitude:.1f}m, '
                              f'Max Speed: {self.metrics.max_speed:.1f}m/s')

    def get_metrics(self) -> FlightMetrics:
        return self.metrics

    def reset(self):
        self.positions.clear()
        self.velocities.clear()
        self.metrics = FlightMetrics()
        self.start_time = None


def main(args=None):
    rclpy.init(args=args)
    node = MetricsCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
