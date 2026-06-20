

https://github.com/user-attachments/assets/c8a856b4-f56a-434c-9a9c-480f39db47bf



https://github.com/user-attachments/assets/ad5ae99a-e2fd-47b9-95ab-741c4ca46913

# SO101 Bimanual Handover for Isaac Lab

This repository is an overlay project for Isaac Lab. It contains the SO101 bimanual handover task, the SO-ARM101 Isaac Lab asset/config files, ACT inference script, and the ROS2/MoveIt bridge workspace used by the MoveIt-based data collection script.

The ROS2 workspace under `ros2/SO-ARM101_MoveIt_IsaacSim/` is based on a clone of `MuammerBay/SO-ARM101_MoveIt_IsaacSim`. The IsaacLab-to-MoveIt bridge server, `ros2/SO-ARM101_MoveIt_IsaacSim/scripts/isaaclab_ompl_bridge.py`, is a local bridge script written for this project.

## Contents

```text
scripts/environments/state_machine/
  bimanual_config/so101_bimanual_handover_env_cfg.py
  pick_and_place_so101_bimanual_handover_moveit_phase.py
  pick_and_place_so101_bimanual_handover_act_inference.py

source/isaaclab_assets/isaaclab_assets/robots/
  so_arm101.py

assets/robots/
  SO-ARM101-USD.usd

media/
  so101_bimanual_handover_success_01.mp4
  so101_bimanual_handover_success_02.mp4
  so101_bimanual_handover_success_01.gif
  so101_bimanual_handover_success_02.gif

ros2/SO-ARM101_MoveIt_IsaacSim/
  src/so_arm_description/
  src/so_arm_moveit_config/
  src/isaac_sim_usd/
  src/warehouse_ros_mongo/
  scripts/isaaclab_ompl_bridge.py
```

The repository does not include generated datasets, checkpoints, Isaac Lab logs, ROS2 `build/`, ROS2 `install/`, or ROS2 `log/` outputs.

## Demo Videos

Two successful ACT inference rollouts are included as scene-wide perspective videos:

### Success rollout 01

[![Success rollout 01](media/so101_bimanual_handover_success_01.gif)](media/so101_bimanual_handover_success_01.mp4)

[Open MP4](media/so101_bimanual_handover_success_01.mp4)

### Success rollout 02

[![Success rollout 02](media/so101_bimanual_handover_success_02.gif)](media/so101_bimanual_handover_success_02.mp4)

[Open MP4](media/so101_bimanual_handover_success_02.mp4)

These videos were recorded from an additional perspective camera, not from the policy input cameras. The ACT policy still uses the two gripper cameras and top camera as observations.

## Install Into Isaac Lab

Clone this repository outside your Isaac Lab checkout, then install the Isaac Lab overlay files into an existing Isaac Lab root:

```bash
git clone https://github.com/KeuntekLee/IsaacLab_so101_Bimanual_Handover.git
cd IsaacLab_so101_Bimanual_Handover
./install_overlay.sh /path/to/IsaacLab
```

Run Isaac Lab commands from the Isaac Lab root:

```bash
cd /path/to/IsaacLab
```

The ROS2 workspace is not copied into Isaac Lab by `install_overlay.sh`. Use it directly from this repository at `ros2/SO-ARM101_MoveIt_IsaacSim/`.

## Overall Flow

The data collection path has two independent processes:

1. ROS2/MoveIt side:
   - `ros2 launch so_arm_moveit_config demo.launch.py` starts MoveIt and provides the `/plan_kinematic_path` service.
   - `scripts/isaaclab_ompl_bridge.py` runs as a ROS2 node named `isaaclab_ompl_bridge`.
   - The bridge receives joint-space planning requests from Isaac Lab and forwards them to MoveIt OMPL.
   - When MoveIt returns a trajectory, the bridge publishes it back to Isaac Lab.

2. Isaac Lab side:
   - `pick_and_place_so101_bimanual_handover_moveit_phase.py` starts Isaac Sim/Isaac Lab and enables the Isaac Sim ROS2 bridge extension.
   - The script runs a phase-driven bimanual handover state machine.
   - For arm movement phases, Isaac Lab computes target joint goals and publishes them to ROS2.
   - While a MoveIt trajectory is active, the state machine holds the current phase and records frames at the dataset FPS.
   - Gripper-only phases are executed locally in Isaac Lab with a ramped jaw command.
   - If `--dataset_dir` is provided, successful episodes are saved as LeRobot data. Failed episodes are discarded from the saved episode count.

The main ROS2 interfaces are:

```text
Isaac Lab -> bridge
  /isaaclab/joint_goal          std_msgs/Float64MultiArray
                                either [goal] or [start, goal]
                                default joint order: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll

  /joint_states                 sensor_msgs/JointState
                                current Isaac Lab joint target/state for MoveIt start-state context

  /isaac_joint_states           sensor_msgs/JointState
                                optional mirror topic for MoveIt/topic-based ROS2 control compatibility

bridge -> Isaac Lab
  /isaaclab/planned_trajectory  trajectory_msgs/JointTrajectory
                                MoveIt OMPL result converted into executable arm waypoints

  /isaaclab/plan_status         std_msgs/String
                                READY, PLANNING, SUCCESS, FAILED, BUSY, INVALID_REQUEST logs

bridge -> MoveIt
  /plan_kinematic_path          moveit_msgs/srv/GetMotionPlan
                                default MoveIt planning service
```

Only the five arm joints are planned by MoveIt. The jaw is not sent to MoveIt; Isaac Lab appends and executes the jaw target locally. In Isaac Lab the full action order per robot is:

```text
Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw
```

The bimanual action/state order is `robot1[6] + robot2[6]`.

## ROS2/MoveIt Bridge Setup

Use a separate terminal for ROS2. The workspace was tested with Ubuntu 22.04 and ROS2 Humble.

```bash
cd /path/to/IsaacLab_so101_Bimanual_Handover/ros2/SO-ARM101_MoveIt_IsaacSim
source /opt/ros/humble/setup.bash

rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

Start MoveIt:

```bash
cd /path/to/IsaacLab_so101_Bimanual_Handover/ros2/SO-ARM101_MoveIt_IsaacSim
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch so_arm_moveit_config demo.launch.py
```

Start the IsaacLab-to-MoveIt bridge server in another ROS2 terminal:

```bash
cd /path/to/IsaacLab_so101_Bimanual_Handover/ros2/SO-ARM101_MoveIt_IsaacSim
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 scripts/isaaclab_ompl_bridge.py \
  --moveit-service /plan_kinematic_path \
  --joint-goal-topic /isaaclab/joint_goal \
  --planned-trajectory-topic /isaaclab/planned_trajectory \
  --plan-status-topic /isaaclab/plan_status \
  --joint-names Rotation Pitch Elbow Wrist_Pitch Wrist_Roll \
  --group-name arm \
  --pipeline-id ompl \
  --allowed-planning-time 5.0 \
  --num-planning-attempts 5 \
  --max-velocity-scaling-factor 0.1 \
  --max-acceleration-scaling-factor 0.1
```

The bridge prints status lines such as:

```text
READY joint_goal_topic=/isaaclab/joint_goal ...
PLANNING request_id=1 source=Float64MultiArray(start+goal) start=provided
SUCCESS request_id=1 source=Float64MultiArray(start+goal) points=...
FAILED request_id=... error_code=... error=...
```

## Data Collection

Use another terminal for Isaac Lab. The ROS2 bridge extension needs the ROS2 Humble bridge library path and the same DDS implementation as the ROS2 side.

```bash
cd /path/to/IsaacLab
source /home/keuntek/miniforge3/bin/activate isaacsim

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/keuntek/miniforge3/envs/isaacsim/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISTRO=humble

LIVESTREAM=1 ./isaaclab.sh -p scripts/environments/state_machine/pick_and_place_so101_bimanual_handover_moveit_phase.py \
  --num_envs 1 \
  --num_episodes 10 \
  --device cpu \
  --enable_cameras \
  --dataset_dir ./dataset/bimanual_handover
```

Useful data collection options:

```text
--trajectory-time-scale 0.4       speed up execution of returned MoveIt trajectories
--max-trajectory-control-steps N  cap dense trajectory length after interpolation
--plan-timeout 12.0               timeout while waiting for bridge trajectory
--no-wait-for-moveit-trajectories allow phase update without waiting for active trajectory completion
--vcodec libsvtav1                LeRobot video codec
```

When `--dataset_dir` is omitted, the simulation runs without writing LeRobot data. When `--dataset_dir` is provided, only successful episodes are saved, so the number of saved episodes targets `--num_episodes`.

## ACT Inference

Run ACT inference with a LeRobot `pretrained_model` checkpoint. This script does not require ROS2 or MoveIt.

```bash
cd /path/to/IsaacLab
source /home/keuntek/miniforge3/bin/activate isaacsim

./isaaclab.sh -p scripts/environments/state_machine/pick_and_place_so101_bimanual_handover_act_inference.py \
  --num_envs 1 \
  --num_episodes 1 \
  --device cpu \
  --enable_cameras \
  --policy-device cuda \
  --checkpoint /path/to/checkpoints/100000/pretrained_model \
  --n-action-steps 50 \
  --max_episode_steps 5000 \
  --success-hold-steps 60
```

To record successful scene-wide MP4 videos while running inference:

```bash
LIVESTREAM=1 ./isaaclab.sh -p scripts/environments/state_machine/pick_and_place_so101_bimanual_handover_act_inference.py \
  --num_envs 1 \
  --num_episodes 0 \
  --target-successes 2 \
  --device cpu \
  --enable_cameras \
  --policy-device cuda \
  --checkpoint /path/to/checkpoints/100000/pretrained_model \
  --n-action-steps 50 \
  --max_episode_steps 2500 \
  --success-hold-steps 60 \
  --perspective-video-dir ./media \
  --perspective-video-width 960 \
  --perspective-video-height 540 \
  --perspective-video-fps 30
```

`--target-successes` stops after the requested number of successful episodes. Failed episodes are discarded from the perspective video output.

You can also set the checkpoint path with:

```bash
export SO101_BIMANUAL_ACT_CHECKPOINT=/path/to/checkpoints/100000/pretrained_model
```

## Troubleshooting

- If Isaac Lab prints bridge import or `em`/ROS Python errors, confirm that the `isaacsim` conda environment is active and the `LD_LIBRARY_PATH`, `RMW_IMPLEMENTATION`, and `ROS_DISTRO` exports were set in the Isaac Lab terminal.
- If the bridge prints `moveit_service_unavailable`, confirm that `ros2 launch so_arm_moveit_config demo.launch.py` is running and that `/plan_kinematic_path` exists.
- If Isaac Lab times out waiting for a trajectory, check `/isaaclab/plan_status` and the bridge terminal for `FAILED`, `BUSY`, or MoveIt error codes.
- If planning frequently fails in a specific randomized grid region, reduce the grid range or increase `--allowed-planning-time`/`--num-planning-attempts` on the bridge server.
- If ACT inference produces nearly static arm actions from the camera policy, run with `LIVESTREAM=1` and `--enable_cameras`. This can change the camera/rendering path used by Isaac Sim.
- If recorded videos look faster than the live simulator, compare simulation time rather than wall-clock time. The dataset is written at 30 Hz simulation time; a slow live simulation will replay faster as a video.

## Notes

- Isaac Lab control rate is 120 Hz.
- LeRobot camera/video data is recorded at 30 Hz.
- ACT inference can run policy computation on CUDA while Isaac Lab simulation runs on CPU.
- Top/gripper cameras are included for LeRobot-compatible visual observations.
- The SO101 robot config resolves `assets/robots/SO-ARM101-USD.usd` relative to the Isaac Lab root after overlay installation.
