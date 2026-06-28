#!/usr/bin/env python3
"""

Features:
- Occupancy grid update from sensor data
- 2D and 3D grid support
- Probabilistic occupancy
- Map saving/loading
- Coverage analysis

"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool, Float64
import numpy as np
import yaml
from typing import Tuple


class OccupancyMapper(Node):
  
    def __init__(self):
        super().__init__('occupancy_mapper')

        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('width', 100)
        self.declare_parameter('height', 100)
        self.declare_parameter('origin', [0.0, 0.0])
        self.declare_parameter('free_update', -0.4)
        self.declare_parameter('occupied_update', 0.6)
        self.declare_parameter('max_occupancy', 10.0)
        self.declare_parameter('min_occupancy', -10.0)

        self.resolution = self.get_parameter('resolution').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value

        # Initialize grid
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.visited = np.zeros((self.height, self.width), dtype=bool)

        self.origin_x = self.get_parameter('origin').value[0]
        self.origin_y = self.get_parameter('origin').value[1]

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.coverage_pub = self.create_publisher(Float64, '/map/coverage', 10)

        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, 10)
        self.create_subscription(LaserScan, '/drone/scan', self._scan_callback, 10)

        self.create_timer(1.0, self._publish_map)

        self.get_logger().info('Occupancy Mapper initialized')

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def _odom_callback(self, msg):
        """Mark current position as visited."""
        gx, gy = self._world_to_grid(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

        if 0 <= gx < self.width and 0 <= gy < self.height:
            self.visited[gy, gx] = True

    def _scan_callback(self, msg):
        """Update occupancy grid from laser scan."""
        if not hasattr(self, 'current_pose'):
            return

        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                ex = self.current_pose.position.x + r * np.cos(angle)
                ey = self.current_pose.position.y + r * np.sin(angle)

                self._update_ray(self.current_pose.position.x, self.current_pose.position.y, ex, ey)

            angle += msg.angle_increment

    def _update_ray(self, x0: float, y0: float, x1: float, y1: float):
        """Update grid along a ray."""
        gx0, gy0 = self._world_to_grid(x0, y0)
        gx1, gy1 = self._world_to_grid(x1, y1)

        if 0 <= gx0 < self.width and 0 <= gy0 < self.height:
            self.grid[gy0, gx0] += self.get_parameter('free_update').value

        if 0 <= gx1 < self.width and 0 <= gy1 < self.height:
            self.grid[gy1, gx1] += self.get_parameter('occupied_update').value

    def _publish_map(self):
        """Publish occupancy grid."""
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y

        # Convert log odds to occupancy values (0-100)
        occupancy = np.clip((1 - 1 / (1 + np.exp(self.grid))) * 100, 0, 100).astype(np.int8)
        grid_msg.data = occupancy.flatten().tolist()

        self.map_pub.publish(grid_msg)

        # Calculate coverage
        coverage = np.sum(self.visited) / (self.width * self.height) * 100
        self.coverage_pub.publish(Float64(data=coverage))


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
