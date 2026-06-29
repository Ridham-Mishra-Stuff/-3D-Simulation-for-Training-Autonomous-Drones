
import numpy as np
from typing import Optional
from dataclasses import dataclass
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist


@dataclass
class BatteryStatus:
    voltage: float
    current: float
    percentage: float
    temperature: float
    estimated_flight_time: float
    is_low: bool
    is_critical: bool


class BatteryMonitor(Node):
    def __init__(self):
        super().__init__('battery_monitor')
        self.declare_parameter('battery_capacity', 5000.0)
        self.declare_parameter('low_threshold', 20.0)
        self.declare_parameter('critical_threshold', 10.0)
        self.declare_parameter('nominal_voltage', 11.1)
        
        self.capacity = self.get_parameter('battery_capacity').value
        self.low_threshold = self.get_parameter('low_threshold').value
        self.critical_threshold = self.get_parameter('critical_threshold').value
        
        self.current_draw = 0.0
        self.status = BatteryStatus(11.1, 0.0, 100.0, 25.0, 0.0, False, False)
        
        self.create_subscription(Float64, '/drone/battery/voltage', self.voltage_callback, 10)
        self.create_subscription(Float64, '/drone/battery/current', self.current_callback, 10)
        self.create_subscription(Twist, '/drone/cmd_vel', self.cmd_callback, 10)
        
        self.status_pub = self.create_publisher(Float64, '/drone/battery/percentage', 10)
        self.warning_pub = self.create_publisher(Bool, '/drone/battery/low_warning', 10)
        self.critical_pub = self.create_publisher(Bool, '/drone/battery/critical_warning', 10)
        
        self.create_timer(1.0, self.update_battery)

    def voltage_callback(self, msg: Float64):
        self.status.voltage = msg.data
        self.status.percentage = self._voltage_to_percentage(msg.data)

    def current_callback(self, msg: Float64):
        self.status.current = msg.data

    def cmd_callback(self, msg: Twist):
        thrust = abs(msg.linear.z)
        self.current_draw = 2.0 + thrust * 8.0

    def _voltage_to_percentage(self, voltage: float) -> float:
        min_v = 9.0
        max_v = 12.6
        pct = (voltage - min_v) / (max_v - min_v) * 100
        return np.clip(pct, 0, 100)

    def update_battery(self):
        if self.current_draw > 0:
            remaining_mah = self.capacity * (self.status.percentage / 100)
            self.status.estimated_flight_time = (remaining_mah / self.current_draw) * 60
        
        self.status.is_low = self.status.percentage < self.low_threshold
        self.status.is_critical = self.status.percentage < self.critical_threshold
        
        self.status_pub.publish(Float64(data=self.status.percentage))
        self.warning_pub.publish(Bool(data=self.status.is_low))
        self.critical_pub.publish(Bool(data=self.status.is_critical))
        
        if self.status.is_critical:
            self.get_logger().warn('BATTERY CRITICAL - LAND IMMEDIATELY')

    def get_status(self) -> BatteryStatus:
        return self.status


def main(args=None):
    rclpy.init(args=args)
    node = BatteryMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
