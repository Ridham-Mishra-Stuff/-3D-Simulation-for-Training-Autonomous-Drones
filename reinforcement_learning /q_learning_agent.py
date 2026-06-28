
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Bool
import numpy as np


class QLearningAgent(Node):
    def __init__(self):
        super().__init__('q_learning_agent')

        self.state_size = 100
        self.action_size = 9

        self.q_table = np.zeros((self.state_size, self.action_size))
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 1.0

        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/drone/odom', self._state_callback, 10)
        self.create_subscription(Float64, '/reward', self._reward_callback, 10)

        self.timer = self.create_timer(0.1, self._act)

        self.current_state = 0
        self.current_reward = 0.0

    def _state_callback(self, msg):
        dist = np.sqrt(msg.pose.pose.position.x**2 + msg.pose.pose.position.y**2)
        self.current_state = min(int(dist), self.state_size - 1)

    def _reward_callback(self, msg):
        self.current_reward = msg.data

    def _act(self):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            action = np.argmax(self.q_table[self.current_state])

        # Convert to twist
        twist = Twist()
        ax = action % 3 - 1
        ay = action // 3 - 1
        twist.linear.x = ax * 0.5
        twist.linear.y = ay * 0.5

        self.action_pub.publish(twist)

        # Update Q-table
        old_value = self.q_table[self.current_state, action]
        next_max = np.max(self.q_table[self.current_state])
        new_value = old_value + self.learning_rate * (self.current_reward + self.discount * next_max - old_value)
        self.q_table[self.current_state, action] = new_value

        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)


def main(args=None):
    rclpy.init(args=args)
    node = QLearningAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
