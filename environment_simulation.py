
"""

Features:
- Configurable wind profiles (constant, gusty, turbulent)
- Direction and magnitude variation
- Altitude-dependent wind
- Publish wind force as Wrench message

"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Wrench
from std_msgs.msg import Float64, Bool
import numpy as np
import math


class WindDisturbance(Node):
  

    WIND_CONSTANT = 0
    WIND_GUSTY = 1
    WIND_TURBULENT = 2

    def __init__(self):
        super().__init__('wind_disturbance')

        self.declare_parameter('wind_mode', 1)
        self.declare_parameter('base_wind_speed', 2.0)
        self.declare_parameter('wind_direction', 0.0)
        self.declare_parameter('gust_frequency', 0.1)
        self.declare_parameter('gust_magnitude', 3.0)
        self.declare_parameter('turbulence_intensity', 0.3)
        self.declare_parameter('altitude_factor', 0.1)
        self.declare_parameter('enable_wind', True)

        self.mode = self.get_parameter('wind_mode').value
        self.base_speed = self.get_parameter('base_wind_speed').value
        self.direction = self.get_parameter('wind_direction').value
        self.gust_freq = self.get_parameter('gust_frequency').value
        self.gust_mag = self.get_parameter('gust_magnitude').value
        self.turbulence = self.get_parameter('turbulence_intensity').value
        self.alt_factor = self.get_parameter('altitude_factor').value
        self.enabled = self.get_parameter('enable_wind').value

        self.time = 0.0

        self.wind_pub = self.create_publisher(Wrench, '/wind/force', 10)
        self.vel_pub = self.create_publisher(Vector3, '/wind/velocity', 10)
        self.mag_pub = self.create_publisher(Float64, '/wind/magnitude', 10)

        self.create_subscription(Bool, '/wind/enable', self._enable_callback, 10)

        self.create_timer(0.1, self._wind_loop)

        self.get_logger().info('Wind Disturbance simulator initialized')

    def _enable_callback(self, msg):
        self.enabled = msg.data

    def _wind_loop(self):
        if not self.enabled:
            return

        self.time += 0.1

        # Calculate wind components
        if self.mode == self.WIND_CONSTANT:
            speed = self.base_speed
        elif self.mode == self.WIND_GUSTY:
            gust = self.gust_mag * abs(math.sin(2 * math.pi * self.gust_freq * self.time))
            speed = self.base_speed + gust
        else:  # TURBULENT
            noise = np.random.normal(0, self.turbulence * self.base_speed)
            speed = max(0, self.base_speed + noise)

        # Wind velocity components
        vx = speed * math.cos(self.direction)
        vy = speed * math.sin(self.direction)
        vz = np.random.normal(0, 0.1)

        # Publish wind velocity
        vel = Vector3()
        vel.x = vx
        vel.y = vy
        vel.z = vz
        self.vel_pub.publish(vel)

        # Publish wind force (simplified drag model)
        wrench = Wrench()
        wrench.force.x = 0.5 * 1.225 * vx * abs(vx) * 0.1
        wrench.force.y = 0.5 * 1.225 * vy * abs(vy) * 0.1
        wrench.force.z = 0.5 * 1.225 * vz * abs(vz) * 0.1
        self.wind_pub.publish(wrench)

        # Publish magnitude
        self.mag_pub.publish(Float64(data=speed))


def main(args=None):
    rclpy.init(args=args)
    node = WindDisturbance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
