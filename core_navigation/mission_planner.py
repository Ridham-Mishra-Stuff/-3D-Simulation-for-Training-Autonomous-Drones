#!/usr/bin/env python3
"""
Features:
- Mission sequence management with preconditions
- State machine for complex mission workflows
- Automatic mission recovery and retry logic
- Integration with waypoint trainer, detector, and avoidance
- Mission logging and telemetry
- Dynamic mission modification during execution
- Multi-mission queue support
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, String, Float64, Int32
from nav_msgs.msg import Odometry
import json
import yaml
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import time


class MissionPhase(Enum):
    """Mission execution phases."""
    INITIALIZING = auto()
    PRE_FLIGHT_CHECK = auto()
    TAKEOFF = auto()
    NAVIGATION = auto()
    INSPECTION = auto()
    DETECTION = auto()
    LANDING_APPROACH = auto()
    LANDING = auto()
    POST_FLIGHT = auto()
    COMPLETED = auto()
    ABORTED = auto()


@dataclass
class MissionStep:
    """Represents a single mission step."""
    name: str
    phase: MissionPhase
    action: str  # 'waypoint', 'hover', 'detect', 'inspect', 'land', 'wait'
    parameters: Dict = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    timeout: float = 60.0
    retry_count: int = 0
    max_retries: int = 3
    on_failure: str = 'abort'  # 'abort', 'retry', 'skip', 'continue'


@dataclass
class Mission:
    """Represents a complete mission."""
    name: str
    description: str
    steps: List[MissionStep]
    priority: int = 0
    requires_gps: bool = True
    max_duration: float = 600.0
    abort_on_low_battery: bool = True
    battery_threshold: float = 20.0


class MissionPlanner(Node):
   

    def __init__(self):
        super().__init__('mission_planner')

        # Parameters
        self.declare_parameter('default_mission_file', '')
        self.declare_parameter('auto_start', False)
        self.declare_parameter('pre_flight_timeout', 30.0)
        self.declare_parameter('takeoff_timeout', 60.0)
        self.declare_parameter('landing_timeout', 120.0)
        self.declare_parameter('step_check_interval', 0.5)
        self.declare_parameter('telemetry_rate', 1.0)

        # State
        self.current_mission: Optional[Mission] = None
        self.current_step_index = 0
        self.current_phase = MissionPhase.INITIALIZING
        self.mission_start_time = None
        self.step_start_time = None
        self.battery_level = 100.0
        self.is_executing = False
        self.is_paused = False
        self.mission_queue: List[Mission] = []
        self.mission_history: List[Dict] = []

        # Subsystem states
        self.drone_position = None
        self.landing_pad_detected = False
        self.danger_level = 0.0
        self.waypoint_reached = False

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # Publishers
        self.phase_pub = self.create_publisher(String, '/mission/current_phase', qos_reliable)
        self.step_progress_pub = self.create_publisher(Float64, '/mission/step_progress', qos_reliable)
        self.overall_progress_pub = self.create_publisher(Float64, '/mission/overall_progress', qos_reliable)
        self.telemetry_pub = self.create_publisher(String, '/mission/telemetry', qos_reliable)
        self.command_pub = self.create_publisher(String, '/mission/command_out', qos_reliable)
        self.alert_pub = self.create_publisher(String, '/mission/alert', qos_reliable)

        # Command publishers to subsystems
        self.wp_start_pub = self.create_publisher(Bool, '/mission/start', qos_reliable)
        self.wp_pause_pub = self.create_publisher(Bool, '/mission/pause', qos_reliable)
        self.detection_mode_pub = self.create_publisher(Int32, '/detection/mode', qos_reliable)
        self.avoidance_enable_pub = self.create_publisher(Bool, '/avoidance/enable', qos_reliable)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/drone/odom', self._odom_callback, qos_reliable)
        self.command_sub = self.create_subscription(
            String, '/mission/command', self._command_callback, qos_reliable)
        self.load_sub = self.create_subscription(
            String, '/mission/load_file', self._load_file_callback, qos_reliable)
        self.battery_sub = self.create_subscription(
            Float64, '/battery/level', self._battery_callback, qos_reliable)
        self.detection_sub = self.create_subscription(
            Bool, '/detection/landing_pad/detected', self._detection_callback, qos_reliable)
        self.danger_sub = self.create_subscription(
            Float64, '/avoidance/danger_level', self._danger_callback, qos_reliable)
        self.wp_reached_sub = self.create_subscription(
            Bool, '/mission/waypoint_reached', self._wp_reached_callback, qos_reliable)

        # Timers
        self.step_timer = self.create_timer(
            self.get_parameter('step_check_interval').value, self._step_loop)
        self.telemetry_timer = self.create_timer(
            self.get_parameter('telemetry_rate').value, self._publish_telemetry)

        # Load default mission
        default_file = self.get_parameter('default_mission_file').value
        if default_file:
            self._load_mission_file(default_file)

        if self.get_parameter('auto_start').value:
            self._start_mission()

        self.get_logger().info('Mission Planner initialized')

    def _odom_callback(self, msg: Odometry):
        """Track drone position."""
        self.drone_position = msg.pose.pose.position

    def _command_callback(self, msg: String):
        """Process mission commands."""
        command = msg.data.lower()

        if command == 'start':
            self._start_mission()
        elif command == 'pause':
            self._pause_mission()
        elif command == 'resume':
            self._resume_mission()
        elif command == 'abort':
            self._abort_mission()
        elif command == 'next':
            self._skip_to_next_step()
        elif command == 'reset':
            self._reset_mission()
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def _load_file_callback(self, msg: String):
        """Load mission from file."""
        self._load_mission_file(msg.data)

    def _battery_callback(self, msg: Float64):
        """Monitor battery level."""
        self.battery_level = msg.data

        if self.current_mission and self.current_mission.abort_on_low_battery:
            if self.battery_level < self.current_mission.battery_threshold:
                self._publish_alert('WARNING', f'Low battery: {self.battery_level:.1f}%. Initiating emergency landing.')
                self._abort_mission()

    def _detection_callback(self, msg: Bool):
        """Track landing pad detection."""
        self.landing_pad_detected = msg.data

    def _danger_callback(self, msg: Float64):
        """Track danger level."""
        self.danger_level = msg.data

    def _wp_reached_callback(self, msg: Bool):
        """Track waypoint reached events."""
        self.waypoint_reached = msg.data

    def _load_mission_file(self, filepath: str):
        """Load mission from YAML file."""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            steps = []
            for step_data in data.get('steps', []):
                phase_name = step_data.get('phase', 'NAVIGATION')
                phase = getattr(MissionPhase, phase_name, MissionPhase.NAVIGATION)

                step = MissionStep(
                    name=step_data.get('name', 'unnamed'),
                    phase=phase,
                    action=step_data.get('action', 'wait'),
                    parameters=step_data.get('parameters', {}),
                    preconditions=step_data.get('preconditions', []),
                    timeout=step_data.get('timeout', 60.0),
                    max_retries=step_data.get('max_retries', 3),
                    on_failure=step_data.get('on_failure', 'abort')
                )
                steps.append(step)

            mission = Mission(
                name=data.get('name', 'unnamed_mission'),
                description=data.get('description', ''),
                steps=steps,
                priority=data.get('priority', 0),
                requires_gps=data.get('requires_gps', True),
                max_duration=data.get('max_duration', 600.0),
                abort_on_low_battery=data.get('abort_on_low_battery', True),
                battery_threshold=data.get('battery_threshold', 20.0)
            )

            self.mission_queue.append(mission)
            self.get_logger().info(f'Loaded mission: {mission.name} with {len(steps)} steps')

        except Exception as e:
            self.get_logger().error(f'Failed to load mission file: {e}')

    def _start_mission(self):
        """Start mission execution."""
        if not self.mission_queue:
            self.get_logger().warn('No missions in queue')
            return

        self.current_mission = self.mission_queue.pop(0)
        self.current_step_index = 0
        self.current_phase = MissionPhase.INITIALIZING
        self.is_executing = True
        self.is_paused = False
        self.mission_start_time = time.time()
        self.step_start_time = time.time()

        self.get_logger().info(f'Starting mission: {self.current_mission.name}')
        self._publish_alert('INFO', f'Mission {self.current_mission.name} started')

        # Enable subsystems
        self.avoidance_enable_pub.publish(Bool(data=True))

    def _pause_mission(self):
        """Pause current mission."""
        if self.is_executing:
            self.is_paused = True
            self.wp_pause_pub.publish(Bool(data=True))
            self.get_logger().info('Mission paused')

    def _resume_mission(self):
        """Resume paused mission."""
        if self.is_executing and self.is_paused:
            self.is_paused = False
            self.wp_pause_pub.publish(Bool(data=False))
            self.get_logger().info('Mission resumed')

    def _abort_mission(self):
        """Abort current mission."""
        if self.current_mission:
            self.get_logger().warn(f'Aborting mission: {self.current_mission.name}')
            self.current_phase = MissionPhase.ABORTED
            self.is_executing = False

            # Emergency landing
            self._publish_alert('CRITICAL', 'Mission aborted. Initiating emergency landing.')

            # Log mission abort
            self.mission_history.append({
                'mission': self.current_mission.name,
                'status': 'aborted',
                'phase': self.current_phase.name,
                'step': self.current_step_index,
                'timestamp': time.time()
            })

    def _skip_to_next_step(self):
        """Skip to next mission step."""
        if self.current_mission and self.current_step_index < len(self.current_mission.steps) - 1:
            self.current_step_index += 1
            self.step_start_time = time.time()
            self.get_logger().info(f'Skipped to step: {self.current_mission.steps[self.current_step_index].name}')

    def _reset_mission(self):
        """Reset mission state."""
        self.is_executing = False
        self.is_paused = False
        self.current_step_index = 0
        self.current_phase = MissionPhase.INITIALIZING
        self.get_logger().info('Mission reset')

    def _step_loop(self):
        """Main step execution loop."""
        if not self.is_executing or self.is_paused:
            return

        if not self.current_mission:
            return

        # Check mission timeout
        elapsed = time.time() - self.mission_start_time
        if elapsed > self.current_mission.max_duration:
            self._publish_alert('WARNING', 'Mission timeout! Aborting.')
            self._abort_mission()
            return

        # Check if all steps complete
        if self.current_step_index >= len(self.current_mission.steps):
            self._complete_mission()
            return

        step = self.current_mission.steps[self.current_step_index]
        self.current_phase = step.phase

        # Check step timeout
        step_elapsed = time.time() - self.step_start_time
        if step_elapsed > step.timeout:
            self._handle_step_timeout(step)
            return

        # Execute step based on action type
        step_complete = self._execute_step(step)

        if step_complete:
            self.get_logger().info(f'Step completed: {step.name}')
            self.current_step_index += 1
            self.step_start_time = time.time()
            self.waypoint_reached = False

    def _execute_step(self, step: MissionStep) -> bool:
        """Execute a mission step and return completion status."""

        if step.action == 'waypoint':
            return self._execute_waypoint_step(step)
        elif step.action == 'hover':
            return self._execute_hover_step(step)
        elif step.action == 'detect':
            return self._execute_detect_step(step)
        elif step.action == 'inspect':
            return self._execute_inspect_step(step)
        elif step.action == 'land':
            return self._execute_land_step(step)
        elif step.action == 'wait':
            return self._execute_wait_step(step)
        elif step.action == 'takeoff':
            return self._execute_takeoff_step(step)
        else:
            self.get_logger().warn(f'Unknown action: {step.action}')
            return True

    def _execute_waypoint_step(self, step: MissionStep) -> bool:
        """Execute waypoint navigation step."""
        # Start waypoint navigation
        if not hasattr(self, '_wp_started'):
            self.wp_start_pub.publish(Bool(data=True))
            self._wp_started = True
            return False

        # Check if waypoint reached
        if self.waypoint_reached:
            self._wp_started = False
            return True

        return False

    def _execute_hover_step(self, step: MissionStep) -> bool:
        """Execute hover step."""
        duration = step.parameters.get('duration', 5.0)
        step_elapsed = time.time() - self.step_start_time

        # Publish hover command
        self.command_pub.publish(String(data=json.dumps({
            'command': 'hover',
            'duration': duration
        })))

        return step_elapsed >= duration

    def _execute_detect_step(self, step: MissionStep) -> bool:
        """Execute detection step."""
        target = step.parameters.get('target', 'landing_pad')

        # Enable detection
        if target == 'landing_pad':
            self.detection_mode_pub.publish(Int32(data=4))  # Hybrid mode

        # Check detection
        if self.landing_pad_detected:
            return True

        return False

    def _execute_inspect_step(self, step: MissionStep) -> bool:
        """Execute inspection step."""
        duration = step.parameters.get('duration', 10.0)
        step_elapsed = time.time() - self.step_start_time

        # Enable camera recording or specific inspection behavior
        self.command_pub.publish(String(data=json.dumps({
            'command': 'inspect',
            'duration': duration
        })))

        return step_elapsed >= duration

    def _execute_land_step(self, step: MissionStep) -> bool:
        """Execute landing step."""
        self.current_phase = MissionPhase.LANDING

        # Enable landing pad detection
        self.detection_mode_pub.publish(Int32(data=2))  # ArUco mode

        # Check if landed (altitude near zero)
        if self.drone_position and self.drone_position.z < 0.3:
            return True

        return False

    def _execute_wait_step(self, step: MissionStep) -> bool:
        """Execute wait step."""
        duration = step.parameters.get('duration', 1.0)
        step_elapsed = time.time() - self.step_start_time
        return step_elapsed >= duration

    def _execute_takeoff_step(self, step: MissionStep) -> bool:
        """Execute takeoff step."""
        self.current_phase = MissionPhase.TAKEOFF

        # Start waypoint trainer (which handles takeoff)
        self.wp_start_pub.publish(Bool(data=True))

        # Check if at target altitude
        target_alt = step.parameters.get('altitude', 5.0)
        if self.drone_position and abs(self.drone_position.z - target_alt) < 0.5:
            return True

        return False

    def _handle_step_timeout(self, step: MissionStep):
        """Handle step timeout."""
        self.get_logger().warn(f'Step timeout: {step.name}')

        if step.retry_count < step.max_retries:
            step.retry_count += 1
            self.step_start_time = time.time()
            self.get_logger().info(f'Retrying step {step.name} (attempt {step.retry_count}/{step.max_retries})')
        else:
            if step.on_failure == 'abort':
                self._abort_mission()
            elif step.on_failure == 'skip':
                self.current_step_index += 1
                self.step_start_time = time.time()
            elif step.on_failure == 'continue':
                self.current_step_index += 1
                self.step_start_time = time.time()

    def _complete_mission(self):
        """Handle mission completion."""
        self.current_phase = MissionPhase.COMPLETED
        self.is_executing = False

        duration = time.time() - self.mission_start_time
        self.get_logger().info(f'Mission completed in {duration:.1f}s')

        # Log completion
        self.mission_history.append({
            'mission': self.current_mission.name,
            'status': 'completed',
            'duration': duration,
            'timestamp': time.time()
        })

        self._publish_alert('INFO', f'Mission {self.current_mission.name} completed successfully!')

        # Start next mission if available
        if self.mission_queue:
            self.get_logger().info('Starting next mission in queue')
            self._start_mission()

    def _publish_telemetry(self):
        """Publish mission telemetry."""
        telemetry = {
            'executing': self.is_executing,
            'paused': self.is_paused,
            'phase': self.current_phase.name if self.current_phase else 'NONE',
            'step_index': self.current_step_index,
            'battery': round(self.battery_level, 1),
            'danger_level': round(self.danger_level, 2),
            'landing_pad_detected': self.landing_pad_detected,
        }

        if self.current_mission:
            total_steps = len(self.current_mission.steps)
            telemetry['mission'] = self.current_mission.name
            telemetry['total_steps'] = total_steps
            telemetry['progress'] = round(self.current_step_index / max(total_steps, 1) * 100, 1)

            if self.mission_start_time:
                telemetry['elapsed'] = round(time.time() - self.mission_start_time, 1)

        self.telemetry_pub.publish(String(data=json.dumps(telemetry)))
        self.phase_pub.publish(String(data=self.current_phase.name if self.current_phase else 'NONE'))

        if self.current_mission:
            progress = self.current_step_index / max(len(self.current_mission.steps), 1)
            self.overall_progress_pub.publish(Float64(data=progress * 100.0))

    def _publish_alert(self, level: str, message: str):
        """Publish mission alert."""
        alert = {
            'level': level,
            'message': message,
            'timestamp': time.time()
        }
        self.alert_pub.publish(String(data=json.dumps(alert)))
        self.get_logger().info(f'[{level}] {message}')


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down mission planner')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
