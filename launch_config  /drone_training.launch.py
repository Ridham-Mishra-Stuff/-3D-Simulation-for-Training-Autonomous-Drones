
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value='drone_inspection_arena'),
        
        # State Estimation
        Node(
            package='drone_simulation',
            executable='drone_state_estimator.py',
            name='state_estimator',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
        
        # Sensor Fusion
        Node(
            package='drone_simulation',
            executable='sensor_fusion.py',
            name='sensor_fusion',
            parameters=[{'filter_type': 'mahony'}]
        ),
        
        # Navigation
        Node(
            package='drone_simulation',
            executable='drone_waypoint_trainer.py',
            name='waypoint_trainer'
        ),
        
        Node(
            package='drone_simulation',
            executable='altitude_hold_controller.py',
            name='altitude_controller'
        ),
        
        # Perception
        Node(
            package='drone_simulation',
            executable='obstacle_avoidance.py',
            name='obstacle_avoidance'
        ),
        
        Node(
            package='drone_simulation',
            executable='yolo_detector.py',
            name='yolo_detector'
        ),
        
        # RL Training
        Node(
            package='drone_simulation',
            executable='rl_trainer.py',
            name='rl_trainer'
        ),
        
        Node(
            package='drone_simulation',
            executable='reward_calculator.py',
            name='reward_calculator'
        ),
        
        # Mapping
        Node(
            package='drone_simulation',
            executable='occupancy_mapper.py',
            name='occupancy_mapper'
        ),
        
        # Monitoring
        Node(
            package='drone_simulation',
            executable='performance_logger.py',
            name='performance_logger'
        ),
        
        Node(
            package='drone_simulation',
            executable='battery_monitor.py',
            name='battery_monitor'
        ),
        
        # Visualization
        Node(
            package='drone_simulation',
            executable='visualization_publisher.py',
            name='visualization'
        ),
        
        # Wind Simulation
        Node(
            package='drone_simulation',
            executable='wind_disturbance.py',
            name='wind_simulation',
            parameters=[{'wind_model': 'dryden', 'mean_wind_speed': 5.0}]
        )
    ])
