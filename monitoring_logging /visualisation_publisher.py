#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Vector3
from std_msgs.msg import ColorRGBA


class VisualizationPublisher(Node):
    def __init__(self):
        super().__init__('visualization_publisher')
        self.marker_pub = self.create_publisher(MarkerArray, '/drone/visualization', 10)
        self.create_timer(0.1, self.publish_visualizations)
        self.markers = []

    def publish_visualizations(self):
        marker_array = MarkerArray()
        for i, marker in enumerate(self.markers):
            marker.id = i
            marker.header.stamp = self.get_clock().now().to_msg()
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def add_waypoint_marker(self, position: Tuple[float, float, float], 
                           color: Tuple[float, float, float, float] = (0, 1, 0, 1),
                           scale: float = 0.3):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.scale.x = marker.scale.y = marker.scale.z = scale
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        self.markers.append(marker)

    def add_path_line(self, points: List[Tuple[float, float, float]], 
                      color: Tuple[float, float, float, float] = (0, 0, 1, 1)):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        for pt in points:
            p = Point()
            p.x, p.y, p.z = pt
            marker.points.append(p)
        self.markers.append(marker)

    def add_obstacle_marker(self, position: Tuple[float, float, float], 
                            radius: float, color: Tuple[float, float, float, float] = (1, 0, 0, 0.5)):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.scale.x = marker.scale.y = marker.scale.z = radius * 2
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        self.markers.append(marker)

    def clear_markers(self):
        self.markers.clear()


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
