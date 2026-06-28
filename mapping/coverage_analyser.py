
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64, String
import numpy as np
import json


class CoverageAnalyzer(Node):
    
    def __init__(self):
        super().__init__('coverage_analyzer')

        self.declare_parameter('target_coverage', 90.0)

        self.report_pub = self.create_publisher(String, '/coverage/report', 10)
        self.score_pub = self.create_publisher(Float64, '/coverage/score', 10)

        self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)

        self.get_logger().info('Coverage Analyzer initialized')

    def _map_callback(self, msg):
        grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)

        total_cells = msg.info.width * msg.info.height
        explored = np.sum((grid >= 0) & (grid < 50))
        occupied = np.sum(grid > 50)
        unknown = np.sum(grid < 0)

        coverage = (explored + occupied) / total_cells * 100

        report = {
            'total_cells': int(total_cells),
            'explored': int(explored),
            'occupied': int(occupied),
            'unknown': int(unknown),
            'coverage_percent': round(coverage, 2),
            'target_coverage': self.get_parameter('target_coverage').value,
            'score': round(min(coverage / self.get_parameter('target_coverage').value * 100, 100), 2)
        }

        self.report_pub.publish(String(data=json.dumps(report)))
        self.score_pub.publish(Float64(data=report['score']))


def main(args=None):
    rclpy.init(args=args)
    node = CoverageAnalyzer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
