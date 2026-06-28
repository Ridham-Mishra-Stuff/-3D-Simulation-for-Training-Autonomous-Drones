
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math


class RewardCalculator(Node):
  
    def __init__(self):
        super().__init__('reward_calculator')

        self.declare_parameter('goal_position', [10.0, 10.0, 2.0])
        self.declare_parameter('collision_penalty', -100.0)
        self.declare_parameter('goal_reward', 100.0)
        self.declare_parameter('step_penalty', -1.0)
        self.declare_parameter('distance_weight', 1.0)

        self.goal = self.get_parameter('goal_position').value

        self.last_distance = None
        self.total_reward = 0.0

        self.reward_pub = self.create_publisher(Float64, '/reward', 10)
        self.components_pub = self.create_publisher(Float64, '/reward/components', 10)

        self.create_subscription(Odometry, '/drone/odom', self._odom_callback, 10)
        self.create_subscription(Bool, '/collision', self._collision_callback, 10)
        self.create_subscription(Bool, '/goal_reached', self._goal_callback, 10)

        self.create_timer(0.1, self._calculate_reward)

        self.get_logger().info('Reward Calculator initialized')

    def _odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def _collision_callback(self, msg):
        if msg.data:
            self._publish_reward(self.get_parameter('collision_penalty').value, 'collision')

    def _goal_callback(self, msg):
        if msg.data:
            self._publish_reward(self.get_parameter('goal_reward').value, 'goal')

    def _calculate_reward(self):
        if not hasattr(self, 'current_pose'):
            return

        dx = self.goal[0] - self.current_pose.position.x
        dy = self.goal[1] - self.current_pose.position.y
        dz = self.goal[2] - self.current_pose.position.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        if self.last_distance is not None:
            progress = self.last_distance - distance
            reward = progress * self.get_parameter('distance_weight').value
            reward += self.get_parameter('step_penalty').value
            self._publish_reward(reward, 'progress')

        self.last_distance = distance

    def _publish_reward(self, reward, component):
        self.total_reward += reward
        self.reward_pub.publish(Float64(data=reward))


def main(args=None):
    rclpy.init(args=args)
    node = RewardCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
