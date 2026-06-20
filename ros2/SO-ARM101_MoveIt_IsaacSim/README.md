# SO-ARM101 MoveIt Isaac Sim Integration

## Introduction

This repository demonstrates how to integrate MoveIt with Isaac Sim using ROS2 on the SO-ARM101 dual-arm robot. The SO-ARM101 (and older SO-ARM100) is an open-source dual-arm robot created by [TheRobotStudio](https://github.com/TheRobotStudio/SO-ARM100) in collaboration with [HuggingFace's LeRobot](https://github.com/huggingface/lerobot) for robotics research and education.

[MoveIt](https://moveit.ai/) provides robot motion planning, manipulation, and control capabilities in ROS2, enabling control of real robots from simulation. This tutorial focuses on simulation-only implementation, with Sim2Real capabilities planned for future releases.

**📺 Full video tutorial available on [LycheeAI YouTube Channel](https://www.youtube.com/@LycheeAI)**

## Get Your Own SO-ARM Robot 🤖

Want the real robot? Get your own SO-ARM101 (or SO-ARM100) from [WowRobo](https://shop.wowrobo.com/?sca_ref=8879221.Q7i7RSlTAVB) using discount code **`LYCHEEAI5`**

*Make sure to "Add to cart" or choose "More payment options" to apply the discount code!*

## Prerequisites

- **Isaac Sim** - [Installation Guide: Isaac Sim & Isaac Lab](https://www.notion.so/Installation-Guide-Isaac-Sim-Isaac-Lab-1c428763942b8089ad48cf88e01ad213?pvs=21)
- **Linux Ubuntu 22.04** with **ROS2 Humble**
- **ROS2 Humble Installation** - [Official Guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_ros.html)
- **ROS2 Basics Knowledge** - Complete the [TurtleBot Tutorial](https://www.youtube.com/watch?v=3cWQsvpwvQU&ab_channel=Nex-dynamicsrobot)

## Quick Start (Skip Tutorial)

If you want to skip the step-by-step tutorial or encounter issues, use this quick setup:

```bash
# Clone the repository
git clone https://github.com/MuammerBay/SO-ARM101_MoveIt_IsaacSim.git
cd SO-ARM101_MoveIt_IsaacSim

# Source ROS 2
source /opt/ros/humble/setup.bash

# Update rosdep database
rosdep update

# Install ALL dependencies automatically
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
colcon build

# Source the workspace
source install/setup.bash

# Launch MoveIt demo
ros2 launch so_arm_moveit_config demo.launch.py
```

### Start Isaac Sim (Separate Terminal)

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Run Isaac Sim with ROS2 enabled
[ISAAC-SIM-4.5-FOLDER]/isaac-sim.selector.sh
```

### Final Steps
1. Open the USD file (created from URDF) in Isaac Sim
2. Start the simulation (press Play)
3. Control the robot in RViz

## Step-by-Step Tutorial

*This tutorial is based on the excellent work by [robot mania](https://www.youtube.com/@robotmania8896) - highly recommended channel for robotics topics!*

### 1. ROS2 & MoveIt Setup

Source ROS2 Humble in every terminal:
```bash
source /opt/ros/humble/setup.bash
```

Create ROS workspace:
```bash
mkdir -p ~/so-arm_moveit_isaacsim_ws/src
cd ~/so-arm_moveit_isaacsim_ws
```

Install ROS2 dependencies:
```bash
sudo apt update
sudo apt install \
  ros-humble-moveit \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-gripper-controllers \
  ros-humble-topic-based-ros2-control
```

Create package directories:
```bash
cd ~/so-arm_moveit_isaacsim_ws/src
mkdir -p so_arm_moveit_config
mkdir -p so_arm_description
```

### 2. Robot Description Setup

Clone the SO-ARM URDF (adjusted for proper ROS2 format):
```bash
cd ~/so-arm_moveit_isaacsim_ws/src/so_arm_description
git clone https://github.com/MuammerBay/SO-ARM_ROS2_URDF.git .
```

Build workspace:
```bash
cd ~/so-arm_moveit_isaacsim_ws
colcon build
```

### 3. MoveIt Setup Assistant

Launch MoveIt Setup Assistant:
```bash
source ~/so-arm_moveit_isaacsim_ws/install/setup.bash
ros2 launch moveit_setup_assistant setup_assistant.launch.py
```

After setup, rebuild:
```bash
cd ~/so-arm_moveit_isaacsim_ws
colcon build
```

### 4. Configuration Adjustments

**Fix joint_limits.yaml**: Replace Integer values with Float values

**Update moveit_controllers.yaml**: Add to arm_controller:
```yaml
action_ns: follow_joint_trajectory
default: true
```

Rebuild after changes:
```bash
colcon build
```

Test MoveIt without Isaac Sim:
```bash
ros2 launch so_arm_moveit_config demo.launch.py
```

### 5. Isaac Sim Integration

Open Isaac Sim:
```bash
source /opt/ros/humble/setup.bash
[ISAAC-SIM-4.5-FOLDER]/isaac-sim.selector.sh
```

**Import URDF** with these settings:
- Stiffness: 17.8
- Damping: 0.60
- Set articulation root to base mesh (xform)

**Create ROS2 Action Graph**:
- Tools → Robotics → ROS2 Omnigraphs → Joint States
- Articulation: base mesh (xform)
- Joint States Topic: `/isaac_joint_states`
- Joint Commands Topic: `/isaac_joint_command`

### 6. Final Configuration

Update `src/so_arm_moveit_config/config/so101_new_calib.ros2_control.xacro`:

Replace:
```xml
<plugin>mock_components/GenericSystem</plugin>
```

With:
```xml
<!-- <plugin>mock_components/GenericSystem</plugin> -->
<plugin>topic_based_ros2_control/TopicBasedSystem</plugin>
<param name="joint_states_topic">/isaac_joint_states</param>
<param name="joint_commands_topic">/isaac_joint_command</param>
```

Final rebuild:
```bash
cd ~/so-arm_moveit_isaacsim_ws
colcon build
```

### 7. Launch and Test

Start MoveIt:
```bash
source install/setup.bash
ros2 launch so_arm_moveit_config demo.launch.py
```

1. Start Isaac Sim simulation (press Play)
2. Control robot in RViz
3. Enjoy your SO-ARM MoveIt integration!

## Docker Setup

[Quentin Deyna](https://www.linkedin.com/in/quentindeyna/) is working on a Docker setup to solve "works-on-my-machine" problems:
- [Docker Repository](https://github.com/qdeyna/SO-ARM_MoveIt_IsaacSim)

## Credits

- Tutorial based on [robot mania's](https://www.youtube.com/@robotmania8896) excellent Isaac Sim + MoveIt guide
- SO-ARM robot by [TheRobotStudio](https://github.com/TheRobotStudio/SO-ARM100)
- Integration with [HuggingFace LeRobot](https://github.com/huggingface/lerobot)
- Video tutorial on [LycheeAI YouTube Channel](https://www.youtube.com/@LycheeAI)

## Repository Structure

```
├── src/
│   ├── so_arm_description/     # Robot URDF and meshes
│   ├── so_arm_moveit_config/   # MoveIt configuration
│   └── isaac_sim_usd/          # Isaac Sim USD files
├── README.md
└── .gitignore
```

## License

This project follows the same license as the original SO-ARM project. Please refer to the original repositories for licensing information. 