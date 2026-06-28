#!/usr/bin/env python3
"""

Features:
- Q-table based learning
- State discretization
- Reward shaping
- Epsilon-greedy exploration
- Model saving/loading
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import json
import os


class RLTrainer(Node):
 

    def __init__(self):
        super().__init__('rl_trainer')

        self.declare_parameter('learning_rate', 0.1)
        self.declare_parameter('discount_factor', 0.95)
        self.declare_parameter('epsilon_start', 1.0)
        self.declare_parameter('epsilon_end', 0.01)
        self.declare_parameter('epsilon_decay', 0.995)
        self.declare_parameter('state_bins', 10)
        self.declare_parameter('action_bins', 5)

        self.lr = self.get_parameter('learning_rate').value
        self.gamma = self.get_parameter('discount_factor').value
        self.epsilon = self.get_parameter('epsilon_start').value
        self.epsilon_end = self.get_parameter('epsilon_end').value
        self.epsilon_decay = self.get_parameter('epsilon_decay').value

        self.state_bins = self.get_parameter('state_bins').value
        self.action_bins = self.get_parameter('action_bins').value

        # Q-table
        self.q_table = {}
        self.last_state = None
        self.last_action = None

        self.training = False
        self.episode_count = 0
        self.total_reward = 0.0

        # Publishers
        self.action_pub = self.create_publisher(Twist, '/rl/action', 10)
        self.epsilon_pub = self.create_publisher(Float64, '/rl/epsilon', 10)
        self.status_pub = self.create_publisher(String, '/rl/status', 10)

        # Subscribers
        self.create_subscription(Odometry, '/drone/odom', self._state_callback, 10)
        self.create_subscription(Float64, '/mission/reward', self._reward_callback, 10)
        self.create_subscription(String, '/training/command', self._command_callback, 10)

        self.create_timer(0.1, self._training_loop)

        self.get_logger().info('RL Trainer initialized')

    def _discretize_state(self, odom):
        """Discretize continuous state."""
        x = int(odom.pose.pose.position.x) % self.state_bins
        y = int(odom.pose.pose.position.y) % self.state_bins
        z = int(odom.pose.pose.position.z) % self.state_bins
        return (x, y, z)

    def _get_action(self, state):
        """Epsilon-greedy action selection."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_bins * 3)

        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_bins * 3)
        else:
            return np.argmax(self.q_table[state])

    def _action_to_twist(self, action_idx):
        """Convert action index to Twist."""
        twist = Twist()
        bin_size = 2.0 / self.action_bins
        idx = action_idx % self.action_bins

        if action_idx < self.action_bins:
            twist.linear.x = (idx - self.action_bins//2) * bin_size
        elif action_idx < 2 * self.action_bins:
            twist.linear.y = (idx - self.action_bins//2) * bin_size
        else:
            twist.linear.z = (idx - self.action_bins//2) * bin_size

        return twist

    def _state_callback(self, msg):
        self.current_state = self._discretize_state(msg)

    def _reward_callback(self, msg):
        self.last_reward = msg.data
        self.total_reward += msg.data

        # Q-learning update
        if self.training and self.last_state is not None:
            if self.last_state not in self.q_table:
                self.q_table[self.last_state] = np.zeros(self.action_bins * 3)
            if self.current_state not in self.q_table:
                self.q_table[self.current_state] = np.zeros(self.action_bins * 3)

            current_q = self.q_table[self.last_state][self.last_action]
            next_max = np.max(self.q_table[self.current_state])
            new_q = current_q + self.lr * (self.last_reward + self.gamma * next_max - current_q)
            self.q_table[self.last_state][self.last_action] = new_q

    def _command_callback(self, msg):
        if msg.data == 'start':
            self.training = True
            self.get_logger().info('Training started')
        elif msg.data == 'stop':
            self.training = False
            self.get_logger().info('Training stopped')
        elif msg.data == 'save':
            self._save_model()
        elif msg.data == 'load':
            self._load_model()

    def _training_loop(self):
        if not self.training:
            return

        if hasattr(self, 'current_state'):
            action = self._get_action(self.current_state)
            twist = self._action_to_twist(action)
            self.action_pub.publish(twist)

            self.last_state = self.current_state
            self.last_action = action

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_pub.publish(Float64(data=self.epsilon))

    def _save_model(self):
        filename = 'q_table.json'
        with open(filename, 'w') as f:
            q_dict = {str(k): v.tolist() for k, v in self.q_table.items()}
            json.dump(q_dict, f)
        self.get_logger().info(f'Model saved to {filename}')

    def _load_model(self):
        filename = 'q_table.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                q_dict = json.load(f)
                self.q_table = {eval(k): np.array(v) for k, v in q_dict.items()}
            self.get_logger().info(f'Model loaded from {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = RLTrainer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
