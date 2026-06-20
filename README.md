# SO101 Bimanual Handover for Isaac Lab

This repository is an overlay project for Isaac Lab. It contains only the SO101 bimanual handover task files, the SO-ARM101 robot config, and the USD asset needed by the task.

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
```

The overlay does not include generated datasets, checkpoints, Isaac Lab logs, or unrelated Isaac Lab source files.

## Install Into Isaac Lab

Clone this repository outside your Isaac Lab checkout, then install the overlay into an existing Isaac Lab root:

```bash
git clone https://github.com/KeuntekLee/IsaacLab_so101_Bimanual_Handover.git
cd IsaacLab_so101_Bimanual_Handover
./install_overlay.sh /path/to/IsaacLab
```

Then run commands from the Isaac Lab root:

```bash
cd /path/to/IsaacLab
```

## Data Collection

The MoveIt/ROS2 phase-driven script can generate LeRobot data. It expects the ROS2/MoveIt bridge to be running.

```bash
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

Failed episodes are discarded by the script; saved episode count targets `--num_episodes`.

## ACT Inference

Run ACT inference with a LeRobot `pretrained_model` checkpoint:

```bash
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

You can also set the checkpoint path with:

```bash
export SO101_BIMANUAL_ACT_CHECKPOINT=/path/to/checkpoints/100000/pretrained_model
```

## Notes

- The control rate is 120 Hz.
- Camera/video data is recorded at 30 Hz.
- ACT inference can run policy computation on CUDA while Isaac Lab simulation runs on CPU.
- The top/gripper cameras are included in the task scripts for LeRobot-compatible visual observations.
- The SO101 robot config resolves `assets/robots/SO-ARM101-USD.usd` relative to the Isaac Lab root after overlay installation.
