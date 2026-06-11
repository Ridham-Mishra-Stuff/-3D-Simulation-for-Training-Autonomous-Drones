import yaml
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class WaypointNavigator(Node):

    def __init__(self):

        super().__init__('waypoint_navigator')

        self.publisher = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        with open(
            '../config/waypoints.yaml',
            'r'
        ) as file:

            config = yaml.safe_load(file)

        self.waypoints = (
            config['mission']['waypoints']
        )

        self.index = 0

        self.timer = self.create_timer(
            8.0,
            self.send_waypoint
        )

    def send_waypoint(self):

        x, y, z = self.waypoints[self.index]

        goal = PoseStamped()

        goal.header.frame_id = "map"

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z

        self.publisher.publish(goal)

        self.get_logger().info(
            f"Waypoint {self.index+1}"
        )

        self.index = (
            self.index + 1
        ) % len(self.waypoints)


def main():

    rclpy.init()

    node = WaypointNavigator()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
