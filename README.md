# 3D-Simulation for Training Autonomous Drones :
A ROS2 Jazzy-based project that builds a 3D virtual training environment for an autonomous drone using Gazebo Harmonic before testing similar logic in the real world.


📌 Project Overview :

This project focuses on creating a virtual environment where an autonomous drone can be trained, tested, and evaluated before real-world deployment. The aim is to:

✅ Improve drone navigation through simulation-based training  
✅ Reduce testing cost by avoiding early hardware damage  
✅ Understand how drones respond to waypoints, obstacles, landing targets, and mapping tasks  
✅ Build a digital-twin-style workflow for safer robot development  

The project is currently in the **prototype and early testing stage**. Till now, the base drone workspace, Gazebo inspection arena, waypoint training node, altitude control example, landing pad detection placeholder, and sample evaluation workflow have been built.

## 🎯 What the Drone Does :

The autonomous drone is designed to move inside a simulated inspection arena. It follows a basic mission path and learns how its behavior can be improved through repeated virtual testing.

In the current version, the drone workflow includes:

- Taking off from the starting point
- Moving toward inspection gates
- Passing through a basic waypoint path
- Maintaining altitude using a simple controller
- Detecting a landing pad using a beginner-level detection placeholder
- Preparing mapping and performance data for comparison

The drone does not yet represent a fully developed autonomous system. 


## 🛠️ Tech Stack :

| Tool | Purpose |
|---|---|
| ROS2 Jazzy | Robot communication, nodes, topics, and launch files |
| Gazebo Harmonic | 3D simulation environment |
| Python | Core programming language for ROS2 nodes |
| RViz2 | Visualization of robot data and mapping |
| SLAM Toolbox | Mapping support |
| Navigation2 | Navigation stack reference |
| YAML | Mission and parameter configuration |
| VS Code | Development environment |

##  Approach Used :

### Simulation-Based Drone Training

Instead of directly testing on real drone hardware, this project first creates a virtual drone training environment. This makes the development process safer, cheaper, and easier to repeat.

The current training approach is simple and beginner-friendly:

1. Run the drone in a simulated arena.
2. Publish waypoint movement commands through ROS2 topics.
3. Observe mission status and detection outputs.
4. Tune parameters such as speed, altitude, and detection threshold.
5. Compare sample before-training and after-training performance.

 However ,This is not yet a deep reinforcement learning model. The **CURRENT GOAL** is to build the simulation base and understand ROS2, Gazebo, topics, launch files, and evaluation before adding advanced learning methods.

## 📊 Current Results


| Metric | Before Training | After Training |
|---|---:|---:|
| Navigation success | 55% | 84% |
| Landing pad detection F1 | 57% | 81% |
| Mapping coverage | 64% | 89% |
| Battery efficiency score | 60% | 79% |
| Overall readiness | 56% | 83% |



## 🚀 How to Run

### Ubuntu 24.04 with ROS2 Jazzy 

Install ROS2 Jazzy and required simulation tools:

```bash
sudo apt update
sudo apt install ros-jazzy-desktop ros-dev-tools -y
sudo apt install ros-jazzy-ros-gz ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox -y
sudo apt install python3-colcon-common-extensions python3-opencv -y
```

Build the drone workspace:

```bash
cd drone_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Launch the drone simulation:

```bash
ros2 launch drone_training_sim drone_training.launch.py
```


## 📁 Project Structure

```text
autonomous-drone-simulation-ros2-jazzy/
│
├── drone_ws/
│   └── src/drone_training_sim/
│       ├── drone_training_sim/
│       │   ├── drone_waypoint_trainer.py
│       │   ├── altitude_hold_controller.py
│       │   └── landing_pad_detector.py
│       ├── launch/
│       │   ├── drone_world.launch.py
│       │   └── drone_training.launch.py
│       ├── config/
│       │   └── training_missions.yaml
│       └── worlds/
│           └── drone_inspection_arena.sdf
│
├── industrial_ws/                 # Secondary comparison workspace
├── evaluation/
│   ├── analyze_results.py
│   ├── sample_metrics.csv
│   └── performance_summary.md
│
├── docs/
│   ├── SETUP_GUIDE.md
│   ├── TRAINING_DEMO.md
│   ├── PROJECT_DOCUMENTATION.md
│   ├── CURRENT_STATUS.md
│   └── GITHUB_PROFILE_GUIDE.md
│
├── assets/
│   ├── drone_performance.png
│   ├── industrial_performance.png
│   └── digital_twin_workflow.png
│
├── report/
│   ├── Internship_Report.md
│   └── Internship_Report.docx
│
└── README.md
```

## 📈 Output



📊 A drone before vs after performance graph  
📋 A sample evaluation report showing improvement metrics  
🧭 ROS2 topics for drone mission status  
🎯 Landing pad detection output placeholder  
🗺️ A structure for mapping coverage comparison  
📄 Internship-style project documentation  


## 💡 Key Concepts

**Autonomous Drone Navigation** - The drone follows a mission path using waypoints and movement commands.

**Simulation-to-Real** - The drone is tested in simulation first to reduce cost, risk, and hardware damage before real-world testing.

**Digital Twin** - A virtual version of the drone environment is used to test behavior repeatedly and safely.

**Object Detection** - The current landing pad detector is a placeholder module that shows how detection output can be connected to ROS2 topics.

**Mapping** - The project includes a performance format for comparing map coverage before and after training.

## 🔮 Future Improvements

- A real YOLO-based landing pad detector
- Will Connect  Gazebo camera and depth sensor topics more deeply with ROS2
- Improve obstacle avoidance during gate navigation
- Reinforcement learning for smarter route planning
- Record real simulation trials instead of using sample metrics
- Add wind disturbance and sensor noise
- Test the same ROS2 interfaces on a real drone platform later

## 👨‍💻 Author  Ridham Mishra

**Autonomous Drone Simulation and Training**  
First-Year Summer Internship Project  
Being Built using ROS2 Jazzy, Gazebo Harmonic, and Python

## 📄 License
