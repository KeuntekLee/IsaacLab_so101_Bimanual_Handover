# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-driven bi-manual SO101 handover with MoveIt/OMPL planning and LeRobot data collection.

Robot 1 (left/pick side) picks a cube → moves to handover zone → Robot 2 (right/place side)
grabs the cube from handover zone → places it in the bin.
"""

import argparse
import os
import queue
import signal
import threading
import time
from collections.abc import Sequence

from isaaclab.app import AppLauncher

_DIAG_RECEIVED_SIGNALS: list[str] = []


def _diag_signal_handler(signum, frame):
    signame = signal.Signals(signum).name
    msg = f"[DIAG] received signal {signame}({signum})"
    _DIAG_RECEIVED_SIGNALS.append(msg)
    print(msg, flush=True)
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


for _diag_sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    try:
        signal.signal(_diag_sig, _diag_signal_handler)
    except (AttributeError, OSError, ValueError):
        pass

parser = argparse.ArgumentParser(
    description="Phase-driven bi-manual SO101 handover with MoveIt/OMPL planning and LeRobot data collection."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument(
    "--dataset_dir", type=str, default=None,
    help="Directory to save LeRobot dataset (e.g., ./datasets/so101_bimanual_handover).",
)
parser.add_argument(
    "--num_episodes", type=int, default=10, help="Number of episodes to record (0 = infinite).",
)
parser.add_argument(
    "--vcodec", type=str, default="libsvtav1",
    help="Video codec: libsvtav1 (AV1), h264, hevc.",
)
parser.add_argument(
    "--save_failed_episodes", action="store_true", default=False,
    help="Deprecated: failed episodes are discarded from the training dataset.",
)
parser.add_argument(
    "--streaming_encoding", action="store_true", default=True,
    help="Use streaming video encoding (faster save_episode).",
)
parser.add_argument("--plan-timeout", type=float, default=12.0)
parser.add_argument("--ik-steps", type=int, default=160)
parser.add_argument("--ik-pos-tolerance", type=float, default=0.008)
parser.add_argument("--trajectory-point-steps", type=int, default=3)
parser.add_argument("--trajectory-time-scale", type=float, default=1.0)
parser.add_argument("--max-trajectory-control-steps", type=int, default=1200)
parser.add_argument("--wait-for-moveit-trajectories", dest="wait_for_moveit_trajectories", action="store_true", default=True)
parser.add_argument("--no-wait-for-moveit-trajectories", dest="wait_for_moveit_trajectories", action="store_false")
parser.add_argument("--moveit-goal-tolerance", type=float, default=1.0e-3)
parser.add_argument("--moveit-goal-step-dist", type=float, default=0.0)
parser.add_argument("--joint-goal-topic", default="/isaaclab/joint_goal")
parser.add_argument("--planned-trajectory-topic", default="/isaaclab/planned_trajectory")
parser.add_argument("--plan-status-topic", default="/isaaclab/plan_status")
parser.add_argument("--joint-state-topic", default="/joint_states")
parser.add_argument("--mirror-joint-state-topic", default="/isaac_joint_states")
parser.add_argument("--use-sim-time", dest="use_sim_time", action="store_true", default=True)
parser.add_argument("--no-use-sim-time", dest="use_sim_time", action="store_false")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Disabled for Windows isolation: isaacsim.ros2.bridge.check.exe crashes with
# an access violation on this machine before the control loop can run.
# import omni.kit.app
#
# ext_manager = omni.kit.app.get_app().get_extension_manager()
# ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
#
# print("[IsaacLab] Waiting for ROS2 bridge extension startup...")
# for _ in range(60):
#     simulation_app.update()
from isaacsim.core.utils.extensions import enable_extension

# (반드시 app_launcher.app 호출 아랫줄에서 실행되어야 합니다)
enable_extension("isaacsim.ros2.bridge")
    
import numpy as np
import torch
try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String
    from trajectory_msgs.msg import JointTrajectory
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

    # 🎯 FastDDS 통신을 위한 정석 QoS 프로필 정의
    moveit_qos = QoSProfile(
        depth=10,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE  # ⚠️ 이 줄이 없으면 윈도우에서 서브스크라이버가 통신을 거부합니다!
    )
    ROS_AVAILABLE = True
except ModuleNotFoundError:
    rclpy = None
    ExternalShutdownException = Exception
    Node = object
    Parameter = None
    JointState = object
    Float64MultiArray = object
    String = object
    JointTrajectory = object
    ROS_AVAILABLE = False
    print("[ROS] rclpy is not available; MoveIt planning is disabled for this run.")

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import TiledCameraCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab_tasks
from isaaclab.utils.math import quat_mul, quat_from_euler_xyz
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("[WARNING] LeRobot not installed. Data Collection disabled.")

# Import env config (must be after simulation_app init)
from bimanual_config.so101_bimanual_handover_env_cfg import (
    SO101BimanualHandoverEnvCfg,
    compute_handover_zone,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DATASET_FPS = 30
EXPECTED_CONTROL_HZ = 120
EXPECTED_RECORD_INTERVAL = EXPECTED_CONTROL_HZ // DATASET_FPS
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640

ARM_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
ISAAC_SO101_JOINT_NAMES = [*ARM_JOINT_NAMES, "Jaw"]

# LeRobot feature names for bi-manual (12D: 6 per robot)
LEROBOT_BIMANUAL_FEATURE_NAMES = [
    "r1.shoulder_pan.pos", "r1.shoulder_lift.pos", "r1.elbow_flex.pos",
    "r1.wrist_flex.pos", "r1.wrist_roll.pos", "r1.gripper.pos",
    "r2.shoulder_pan.pos", "r2.shoulder_lift.pos", "r2.elbow_flex.pos",
    "r2.wrist_flex.pos", "r2.wrist_roll.pos", "r2.gripper.pos",
]

STATE_MACHINE_EPISODE_STEP = 450
RECORD_START_STEP = 0
RECORD_END_STEP = 400
EPISODE_TIMEOUT_MARGIN_INTERVALS = 2
MAX_TARGET_STEP_DIST = 0.02
SUCCESS_THRESHOLD = 0.05
JOINT_LIMIT_SATURATION_TOL = 1.0e-3
PHASE_GRIPPER_HOLD_STEPS = 90
PHASE_ARM_SETTLE_STEPS = 12
GRIPPER_RAMP_STEPS = 45

LINK_OFFSET = (0.0, 0.01, 0.105)
HOME_EE_POS_W = (0.050, 0.021, 0.126)
HOME_EE_QUAT_WXYZ = (-0.693, -0.140, 0.140, 0.693)
TARGET_EE_QUAT_WXYZ = (-0.7071, 0.0, 0.0, 0.7071)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
JAW_OPEN_POS = 1.0
JAW_CLOSE_POS = 0.0

TOP_CAMERA_POS = (0.40, -0.06, 0.92)
TOP_CAMERA_FOCAL_LENGTH = 10.0
TOP_CAMERA_ROT_EULER_DEG = (0.0, 0.0, 0.0)

PHASE_COMMAND_STEPS = [0, 50, 90, 100, 150, 200, 250, 270, 290, 310, 360]
PHASE_HOLD_STEPS = [
    PHASE_ARM_SETTLE_STEPS,
    PHASE_ARM_SETTLE_STEPS,
    PHASE_GRIPPER_HOLD_STEPS,
    PHASE_ARM_SETTLE_STEPS,
    PHASE_ARM_SETTLE_STEPS,
    PHASE_ARM_SETTLE_STEPS,
    PHASE_GRIPPER_HOLD_STEPS,
    PHASE_GRIPPER_HOLD_STEPS,
    PHASE_ARM_SETTLE_STEPS,
    PHASE_ARM_SETTLE_STEPS,
    PHASE_GRIPPER_HOLD_STEPS,
]
R1_MOVEIT_COMMAND_STEPS = {0, 50, 100, 150, 290}
R2_MOVEIT_COMMAND_STEPS = {200, 310}
MOVEIT_COMMAND_STEPS = R1_MOVEIT_COMMAND_STEPS | R2_MOVEIT_COMMAND_STEPS
GRIPPER_ONLY_COMMAND_STEPS = {90, 250, 270, 360}

# --------------------------------------------------------------------------
# Phase names
# --------------------------------------------------------------------------

PHASE_NAMES = [
    "R1_APPROACH_CUBE",
    "R1_LOWER_TO_CUBE",
    "R1_CLOSE_GRIPPER",
    "R1_LIFT_CUBE",
    "R1_MOVE_HANDOVER",
    "R2_APPROACH_HANDOVER",
    "R2_CLOSE_GRIPPER",
    "R1_RELEASE_GRIPPER",
    "R1_HOME",
    "R2_PLACE_BIN",
    "R2_RELEASE_GRIPPER",
]


def phase_from_step(step: int) -> int:
    if step in PHASE_COMMAND_STEPS:
        return PHASE_COMMAND_STEPS.index(step)
    for idx, command_step in enumerate(PHASE_COMMAND_STEPS):
        if step < command_step:
            return max(0, idx - 1)
    return len(PHASE_COMMAND_STEPS) - 1


# --------------------------------------------------------------------------
# Bi-manual State Machine
# --------------------------------------------------------------------------

class BiManualHandoverStateMachine:
    """Torch-based phase machine for bi-manual handover.

    Legacy command-step labels are kept for logging and target reuse, but transitions are driven by
    completed arm/gripper actions instead of a fixed state-machine clock.
    """

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.phase_index = 0
        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # Robot 1 targets
        self.r1_des_ee_pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.r1_des_ee_quat = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        self.r1_des_gripper = torch.full((num_envs,), GRIPPER_CLOSE, dtype=torch.float32, device=device)

        # Robot 2 targets
        self.r2_des_ee_pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.r2_des_ee_quat = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        self.r2_des_gripper = torch.full((num_envs,), GRIPPER_CLOSE, dtype=torch.float32, device=device)

        # Constants
        self._home_pos = torch.tensor(HOME_EE_POS_W, dtype=torch.float32, device=device)
        self._home_quat = torch.tensor(HOME_EE_QUAT_WXYZ, dtype=torch.float32, device=device)
        self._target_quat = torch.tensor(TARGET_EE_QUAT_WXYZ, dtype=torch.float32, device=device)
        self._link_offset = torch.tensor(LINK_OFFSET, dtype=torch.float32, device=device)

        self.reset_idx()

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        self.phase_index = 0
        self.step_count[env_ids] = PHASE_COMMAND_STEPS[0]

        # Both robots → home
        '''
        for pos, quat, grip in [
            (self.r1_des_ee_pos, self.r1_des_ee_quat, self.r1_des_gripper),
            (self.r2_des_ee_pos, self.r2_des_ee_quat, self.r2_des_gripper),
        ]:
            pos[env_ids] = self._home_pos
            quat[env_ids] = self._home_quat
            grip[env_ids] = GRIPPER_CLOSE
        '''

    def advance(self):
        if self.phase_index < len(PHASE_COMMAND_STEPS) - 1:
            self.phase_index += 1
        self.step_count[:] = PHASE_COMMAND_STEPS[self.phase_index]

    def is_final_phase(self) -> bool:
        return self.phase_index >= len(PHASE_COMMAND_STEPS) - 1

    def current_command_step(self) -> int:
        return PHASE_COMMAND_STEPS[self.phase_index]

    def compute(
        self,
        cube_pos_w: torch.Tensor,
        box_pos_w: torch.Tensor,
        handover_zone: torch.Tensor,
        current_r2_ee_pos_w: torch.Tensor,
        current_r2_ee_quat_w: torch.Tensor,
    ) -> tuple:
        """Compute desired targets for both robots based on current phase.

        Returns:
            (r1_output, r2_output, step) where step is the legacy command-step label for logging.
        """
        command_step = self.current_command_step()
        step = torch.full((self.num_envs,), command_step, dtype=torch.int32, device=self.device)

        # ---- Robot 1: pick side ----
        m = step == 0
        # Approach above cube, gripper open 
        self.r1_des_ee_pos[m] = cube_pos_w[m] + self._link_offset + torch.tensor(
            (0.0, 0.0, 0.05), dtype=torch.float32, device=self.device
        )
        self.r1_des_ee_quat[m] = self._target_quat
        self.r1_des_gripper[m] = GRIPPER_OPEN
        
        self.r2_des_ee_pos[m] = torch.tensor([0.60, 0.0, 0.15], dtype=torch.float32, device=self.device)
        self.r2_des_ee_quat[m] = torch.tensor([0.7071, 0.0, 0.0, 0.7071], dtype=torch.float32, device=self.device)
        #self.r2_des_ee_quat[m] = torch.tensor(euler_angles_to_quat(np.array([90, 0, -90]), degrees=True),dtype=torch.float32, device=self.device)
        self.r2_des_gripper[m] = GRIPPER_OPEN

        m = step == 50
        # Grasp cube (lower to cube level)
        self.r1_des_ee_pos[m] = cube_pos_w[m] + self._link_offset
        self.r1_des_ee_quat[m] = self._target_quat
        self.r1_des_gripper[m] = GRIPPER_OPEN

        m = step == 90
        # Grasp cube (lower to cube level)
        self.r1_des_ee_pos[m] = cube_pos_w[m] + self._link_offset
        self.r1_des_ee_quat[m] = self._target_quat
        self.r1_des_gripper[m] = GRIPPER_CLOSE

        m = step == 100
        # Grasp cube (lower to cube level)
        self.r1_des_ee_pos[m] = cube_pos_w[m] + self._link_offset + torch.tensor([0.0,0.0,0.10])
        self.r1_des_ee_quat[m] = self._target_quat
        self.r1_des_gripper[m] = GRIPPER_CLOSE

        m = step == 150
        # Lift cube toward handover zone
        #self.r1_des_ee_pos[m] = handover_zone[m] + torch.tensor(
        #    (0.0, 0.0, 0.04), dtype=torch.float32, device=self.device
        #)
        #print(self._target_quat, euler_angles_to_quat(np.array([0, 90, 0]), degrees=True))
        #self.r1_des_ee_quat[m] = torch.tensor(math_utils.convert_quat(
        #    euler_angles_to_quat(np.array([0, 90, 0]), degrees=True),
        #    to="wxyz"
        #), dtype=torch.float32, device=self.device)
        #self.r1_des_ee_quat[m] = self._target_quat
        self.r1_des_ee_pos[m] = torch.tensor([0.30, 0.0, 0.25])
        #self.r1_des_ee_quat[m] = torch.tensor([-0.7071, 0.0, 0.7071, 0.0])
        self.r1_des_ee_quat[m] = torch.tensor([-0.5, -0.5, 0.5, 0.5])
        #self.r1_des_ee_quat[m] = torch.tensor(euler_angles_to_quat(np.array([0, 90, 0]), degrees=True),dtype=torch.float32, device=self.device)
        self.r1_des_gripper[m] = GRIPPER_CLOSE
        
        m = step == 200
        self.r2_des_ee_pos[m] = torch.tensor([0.50, 0.0, 0.23])
        self.r2_des_ee_quat[m] = torch.tensor([0.0, 0.7071, 0.0, 0.7071])
        #self.r2_des_ee_quat[m] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        #self.r1_des_ee_quat[m] = torch.tensor(euler_angles_to_quat(np.array([0, 90, 0]), degrees=True),dtype=torch.float32, device=self.device)
        self.r2_des_gripper[m] = GRIPPER_OPEN
        
        m = step == 250
        self.r2_des_ee_pos[m] = torch.tensor([0.50, 0.0, 0.23])
        self.r2_des_ee_quat[m] = torch.tensor([0.0, 0.7071, 0.0, 0.7071])
        #self.r2_des_ee_quat[m] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        #self.r1_des_ee_quat[m] = torch.tensor(euler_angles_to_quat(np.array([0, 90, 0]), degrees=True),dtype=torch.float32, device=self.device)
        self.r2_des_gripper[m] = GRIPPER_CLOSE
        
        m = step == 270
        self.r1_des_ee_pos[m] = torch.tensor([0.30, 0.0, 0.25])
        #self.r1_des_ee_quat[m] = torch.tensor([-0.7071, 0.0, 0.7071, 0.0])
        self.r1_des_ee_quat[m] = torch.tensor([-0.5, -0.5, 0.5, 0.5])
        #self.r1_des_ee_quat[m] = torch.tensor(euler_angles_to_quat(np.array([0, 90, 0]), degrees=True),dtype=torch.float32, device=self.device)
        self.r1_des_gripper[m] = GRIPPER_OPEN
        
        m = step == 290
        self.r1_des_ee_pos[m] = self._home_pos
        #self.r1_des_ee_quat[m] = torch.tensor([-0.7071, 0.0, 0.7071, 0.0])
        self.r1_des_ee_quat[m] = self._home_quat
        #self.r1_des_ee_quat[m] = torch.tensor(euler_angles_to_quat(np.array([0, 90, 0]), degrees=True),dtype=torch.float32, device=self.device)
        self.r1_des_gripper[m] = GRIPPER_OPEN
        
        self.r2_des_gripper[m] = GRIPPER_CLOSE
        
        m = step == 310
        self.r2_des_ee_pos[m] = box_pos_w[m] + torch.tensor(
            (0.0, 0.0, 0.15), dtype=torch.float32, device=self.device
        )
        self.r2_des_ee_quat[m] = torch.tensor([0.7071,0.0,0.0,0.7071])
        self.r2_des_gripper[m] = GRIPPER_CLOSE
        
        m = step == 360
        self.r2_des_ee_pos[m] = box_pos_w[m] + torch.tensor(
            (0.0, 0.0, 0.15), dtype=torch.float32, device=self.device
        )
        self.r2_des_ee_quat[m] = torch.tensor([0.7071,0.0,0.0,0.7071])
        self.r2_des_gripper[m] = GRIPPER_OPEN
        
        '''
        m = step == 290
        self.r1_des_ee_pos[m] = self._home_pos
        #self.r1_des_ee_quat[m] = torch.tensor([-0.7071, 0.0, 0.7071, 0.0])
        self.r1_des_ee_quat[m] = self._home_quat
        #self.r1_des_ee_quat[m] = torch.tensor(euler_angles_to_quat(np.array([0, 90, 0]), degrees=True),dtype=torch.float32, device=self.device)
        self.r1_des_gripper[m] = GRIPPER_CLOSE
        
        self.r2_des_ee_pos[m] = box_pos_w[m] + torch.tensor(
            (0.0, 0.0, 0.18), dtype=torch.float32, device=self.device
        )
        #self.r2_des_ee_quat[m] = torch.tensor([0.7071,0.0,0.0,0.7071])
        self.r2_des_gripper[m] = GRIPPER_CLOSE
        
        m = step == 340
        self.r2_des_ee_pos[m] = box_pos_w[m] + torch.tensor(
            (0.0, 0.0, 0.18), dtype=torch.float32, device=self.device
        )
        self.r2_des_ee_quat[m] = torch.tensor([0.7071,0.0,0.0,0.7071])
        self.r2_des_gripper[m] = GRIPPER_CLOSE
        '''
        
        '''
        m = step == 150
        # Hold at handover zone (waiting for R2 to grasp)
        self.r1_des_ee_pos[m] = handover_zone[m] + torch.tensor(
            (0.0, 0.0, 0.04), dtype=torch.float32, device=self.device
        )
        self.r1_des_ee_quat[m] = self._target_quat
        self.r1_des_gripper[m] = GRIPPER_CLOSE

        m = step == 200
        # Release cube and start going home
        self.r1_des_ee_pos[m] = handover_zone[m] + torch.tensor(
            (0.0, 0.0, 0.06), dtype=torch.float32, device=self.device
        )
        self.r1_des_ee_quat[m] = self._target_quat
        self.r1_des_gripper[m] = GRIPPER_OPEN

        m = step == 250
        # Go home
        self.r1_des_ee_pos[m] = self._home_pos
        self.r1_des_ee_quat[m] = self._home_quat
        self.r1_des_gripper[m] = GRIPPER_CLOSE
        '''
        
        '''
        # ---- Robot 2: place side ----
        m = step == 0
        # Idle at home
        self.r2_des_ee_pos[m] = self._home_pos
        self.r2_des_ee_quat[m] = self._home_quat
        self.r2_des_gripper[m] = GRIPPER_CLOSE

        m = step == 100
        # Approach handover zone, gripper open
        self.r2_des_ee_pos[m] = handover_zone[m] + torch.tensor(
            (0.0, 0.0, 0.06), dtype=torch.float32, device=self.device
        )
        self.r2_des_ee_quat[m] = self._target_quat
        self.r2_des_gripper[m] = GRIPPER_OPEN

        m = step == 150
        # Grasp cube at handover zone
        self.r2_des_ee_pos[m] = handover_zone[m] + torch.tensor(
            (0.0, 0.0, 0.04), dtype=torch.float32, device=self.device
        )
        self.r2_des_ee_quat[m] = self._target_quat
        self.r2_des_gripper[m] = GRIPPER_CLOSE

        m = step == 200
        # Lift from handover and move toward bin
        self.r2_des_ee_pos[m] = box_pos_w[m] + torch.tensor(
            (0.0, 0.0, 0.18), dtype=torch.float32, device=self.device
        )
        self.r2_des_ee_quat[m] = self._target_quat
        self.r2_des_gripper[m] = GRIPPER_CLOSE

        m = step == 250
        # Release in bin
        self.r2_des_ee_pos[m] = box_pos_w[m] + torch.tensor(
            (0.0, 0.0, 0.18), dtype=torch.float32, device=self.device
        )
        self.r2_des_ee_quat[m] = self._target_quat
        self.r2_des_gripper[m] = GRIPPER_OPEN

        m = step == 300
        # Go home
        self.r2_des_ee_pos[m] = self._home_pos
        self.r2_des_ee_quat[m] = self._home_quat
        self.r2_des_gripper[m] = GRIPPER_CLOSE
        '''
        # Build outputs: [ee_pos(3), ee_quat(4), gripper(1)] = 8D per robot
        r1_output = torch.cat(
            (self.r1_des_ee_pos, self.r1_des_ee_quat, self.r1_des_gripper.unsqueeze(-1)), dim=-1
        )
        r2_output = torch.cat(
            (self.r2_des_ee_pos, self.r2_des_ee_quat, self.r2_des_gripper.unsqueeze(-1)), dim=-1
        )
        
        return r1_output, r2_output, step


# --------------------------------------------------------------------------
# IK Teacher (per robot)
# --------------------------------------------------------------------------

class SO101DiffIKGoalGenerator:
    """Differential IK teacher for a single SO101 robot."""

    def __init__(self, robot, num_envs: int, device: str):
        self.robot = robot
        self.device = device
        missing = [joint_name for joint_name in ARM_JOINT_NAMES if joint_name not in robot.joint_names]
        if missing:
            raise ValueError(f"Robot missing arm joints: {missing}")
        self.joint_ids = [robot.joint_names.index(joint_name) for joint_name in ARM_JOINT_NAMES]

        body_ids, body_names = robot.find_bodies("gripper")
        if len(body_ids) != 1:
            raise ValueError(f"Expected one 'gripper' body, found {len(body_ids)}: {body_names}")
        self.body_idx = body_ids[0]

        if robot.is_fixed_base:
            self.jacobi_body_idx = self.body_idx - 1
            self.jacobi_joint_ids = self.joint_ids
        else:
            self.jacobi_body_idx = self.body_idx
            self.jacobi_joint_ids = [j + 6 for j in self.joint_ids]

        self.controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=num_envs,
            device=device,
        )

    def reset(self):
        self.controller.reset()

    def compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        return math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )

    def compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.jacobi_body_idx, :, self.jacobi_joint_ids
        ].clone()
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(self.robot.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def compute(self, target_pose_b: torch.Tensor) -> torch.Tensor:
        ee_pos_b, ee_quat_b = self.compute_frame_pose()
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        if torch.any(torch.linalg.norm(ee_quat_b, dim=-1) <= 1.0e-6):
            return joint_pos.clone()
        self.controller.set_command(target_pose_b, ee_pos_b, ee_quat_b)
        return self.controller.compute(ee_pos_b, ee_quat_b, self.compute_frame_jacobian(), joint_pos)

    def position_error(self, target_pos_w: torch.Tensor) -> float:
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        errors = torch.linalg.norm(ee_pos_w[:, :3] - target_pos_w[:, :3], dim=-1)
        return float(torch.max(errors).item())

# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------

def get_box_pos_w(box) -> torch.Tensor:
    if hasattr(box, "data") and hasattr(box.data, "root_pos_w"):
        return box.data.root_pos_w[:, :3].clone()
    pos, _ = box.get_world_poses()
    return pos[:, :3].clone()

def compute_success_xy_distance(cube_pos_w: torch.Tensor, box_pos_w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(cube_pos_w[:, :2] - box_pos_w[:, :2], dim=-1)

def check_success_mask(
    cube_pos_w: torch.Tensor, 
    box_pos_w: torch.Tensor, 
    threshold: float = SUCCESS_THRESHOLD
) -> torch.Tensor:
    return compute_success_xy_distance(cube_pos_w, box_pos_w) < threshold


def get_ordered_joint_ids(robot, joint_names: Sequence[str] = ISAAC_SO101_JOINT_NAMES) -> list[int]:
    missing = [j for j in joint_names if j not in robot.joint_names]
    if missing:
        raise ValueError(f"Robot missing joints: {missing}")
    return [robot.joint_names.index(j) for j in joint_names]


def compute_gripper_joint_target(gripper_cmd: torch.Tensor) -> torch.Tensor:
    open_t = torch.full((gripper_cmd.shape[0], 1), JAW_OPEN_POS, dtype=gripper_cmd.dtype, device=gripper_cmd.device)
    close_t = torch.full((gripper_cmd.shape[0], 1), JAW_CLOSE_POS, dtype=gripper_cmd.dtype, device=gripper_cmd.device)
    return torch.where(gripper_cmd.unsqueeze(-1) < 0.0, close_t, open_t)


def clamp_joint_targets(robot, joint_ids: Sequence[int], joint_targets: torch.Tensor) -> torch.Tensor:
    limits = robot.data.soft_joint_pos_limits[:, joint_ids, :]
    return torch.maximum(torch.minimum(joint_targets, limits[..., 1]), limits[..., 0])

def target_pose_in_robot_base(robot, raw_actions_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        raw_actions_w[:, :3],
        raw_actions_w[:, 3:7],
    )
    target_quat_b = target_quat_b / torch.clamp(torch.linalg.norm(target_quat_b, dim=-1, keepdim=True), min=1.0e-6)
    return target_pos_b, target_quat_b

def make_joint_goal_action(
    robot,
    ordered_joint_ids: Sequence[int],
    arm_joint_target: torch.Tensor,
    gripper_cmd: torch.Tensor,
    return_limit_saturation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    jaw_target = compute_gripper_joint_target(gripper_cmd)
    joint_action = torch.cat((arm_joint_target, jaw_target), dim=-1)

    if joint_action.shape[-1] != len(ISAAC_SO101_JOINT_NAMES):
        raise RuntimeError(f"Expected joint action shape (*,6) got {tuple(joint_action.shape)}")

    clamped = clamp_joint_targets(robot, ordered_joint_ids, joint_action)
    if return_limit_saturation:
        saturated = torch.any(torch.abs(joint_action - clamped) > JOINT_LIMIT_SATURATION_TOL, dim=-1)
        return clamped, saturated
    return clamped

def restore_scene_state(env: ManagerBasedEnv, scene_state: dict) -> None:
    for asset_name, articulation in env.scene.articulations.items():
        asset_state = scene_state["articulations"][asset_name]
        articulation.write_joint_state_to_sim(
            asset_state["joint_position"].clone(), asset_state["joint_velocity"].clone()
        )
        
    for asset_name, rigid_object in env.scene.rigid_objects.items():
        asset_state = scene_state["rigid_objects"][asset_name]
        rigid_object.write_root_pose_to_sim(asset_state["root_pos"].clone())
        if asset_name != "box":
            rigid_object.write_root_velocity_to_sim(asset_state["root_velocity"].clone())
    
    env.sim.forward()
    env.scene.update(dt=0.0)
    
def set_robot_joint_target(robot, ordered_joint_ids: Sequence[int], joint_targets: torch.Tensor) -> None:
    robot.set_joint_position_targets(joint_targets, joint_ids=ordered_joint_ids)


def write_robot_probe_joint_state(robot, ordered_joint_ids: Sequence[int], joint_targets: torch.Tensor) -> None:
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    joint_pos[:, ordered_joint_ids] = joint_targets
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def solve_moveit_joint_goal(
    env: ManagerBasedEnv,
    robot,
    generator: SO101DiffIKGoalGenerator,
    ordered_joint_ids: Sequence[int],
    raw_actions_w: torch.Tensor,
    current_actions: torch.Tensor,
    hold_targets: Sequence[tuple],
    robot_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    saved_joint_pos = robot.data.joint_pos.clone()
    saved_joint_vel = robot.data.joint_vel.clone()
    target_pos_w = raw_actions_w[:, :3]
    target_pos_b, target_quat_b = target_pose_in_robot_base(robot, raw_actions_w)
    target_pose_b = torch.cat((target_pos_b, target_quat_b), dim=-1)
    
    working_target = current_actions.detach().clone()
    final_arm_target = working_target[:, : len(ARM_JOINT_NAMES)].clone()
    final_saturated = torch.zeros(raw_actions_w.shape[0], dtype=torch.bool, device=raw_actions_w.device)
    
    try:
        generator.reset()
        for step_idx in range(args_cli.ik_steps):
            raw_arm_target = generator.compute(target_pose_b)
            working_target[:, : len(ARM_JOINT_NAMES)] = raw_arm_target
            clamped_target = clamp_joint_targets(robot, ordered_joint_ids, working_target)
            final_saturated = torch.any(torch.abs(clamped_target - working_target) > JOINT_LIMIT_SATURATION_TOL, dim=-1)
            working_target = clamped_target
            
            write_robot_probe_joint_state(robot, ordered_joint_ids, working_target)
            env.sim.forward()
            env.scene.update(dt=0.0)
            
            final_arm_target = working_target[:, : len(ARM_JOINT_NAMES)].clone()
            if generator.position_error(target_pos_w) < args_cli.ik_pos_tolerance:
                break
            if not simulation_app.is_running():
                break
        
        goal_action, jaw_saturated = make_joint_goal_action(
            robot, ordered_joint_ids, final_arm_target, raw_actions_w[:, -1], return_limit_saturation=True
        )
        final_saturated |= jaw_saturated
        error = generator.position_error(target_pos_w)
        print(
            f"[IK] {robot_name}: ee_position_error={error:.4f} m, "
            f"arm_goal={[round(value, 4) for value in final_arm_target[0].detach().cpu().tolist()]}"
        )
        return goal_action, final_saturated
    finally:
        robot.write_joint_state_to_sim(saved_joint_pos, saved_joint_vel)
        env.sim.forward()
        env.scene.update(dt=0.0)


class BridgeClient(Node):
    
    def __init__(self) -> None:
        super().__init__(
            "so101_bimanual_handover_moveit_client",
            parameter_overrides=[Parameter(name="use_sim_time", value=args_cli.use_sim_time)],
            automatically_declare_parameters_from_overrides=True,
        )
        self._trajectory_queue: queue.Queue[JointTrajectory] = queue.Queue(maxsize=1)
        self._last_status = ""
        self._status_lock = threading.Lock()
        
        self.joint_goal_pub = self.create_publisher(Float64MultiArray, args_cli.joint_goal_topic, 10)
        self.joint_state_pub = self.create_publisher(JointState, args_cli.joint_state_topic, 10)
        self.mirror_joint_state_pub = None
        if args_cli.mirror_joint_state_topic:
            self.mirror_joint_state_pub = self.create_publisher(JointState, args_cli.mirror_joint_state_topic, 10)
            
        self.create_subscription(JointTrajectory, args_cli.planned_trajectory_topic, self._on_trajectory, qos_profile=moveit_qos)
        self.create_subscription(String, args_cli.plan_status_topic, self._on_status, qos_profile=moveit_qos)
        
    @property
    def last_status(self) -> str:
        with self._status_lock:
            return self._last_status
        
    def clear_trajectory_queue(self) -> None:
        while True:
            try:
                self._trajectory_queue.get_nowait()
            except queue.Empty:
                break
            
    def publish_joint_goal(self, start_positions: Sequence[float], goal_positions: Sequence[float]) -> None:
        self.clear_trajectory_queue()
        msg = Float64MultiArray()
        msg.data = [float(value) for value in [*start_positions, *goal_positions]]
        self.joint_goal_pub.publish(msg)
        
    def take_trajectory(self) -> JointTrajectory | None:
        try:
            return self._trajectory_queue.get_nowait()
        except queue.Empty:
            return None
        
    def publish_joint_state(self, joint_names: Sequence[str], joint_positions: Sequence[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_names)
        msg.position = [float(value) for value in joint_positions]
        self.joint_state_pub.publish(msg)
        if self.mirror_joint_state_pub is not None:
            self.mirror_joint_state_pub.publish(msg)
            
    def _on_trajectory(self, msg: JointTrajectory) -> None:
        self.clear_trajectory_queue()
        self._trajectory_queue.put_nowait(msg)
        
    def _on_status(self, msg: String) -> None:
        with self._status_lock:
            self._last_status = msg.data
        print(f"[MoveIt bridge] {msg.data}")
        
class MoveItArmTrajectoryFollower:
    
    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device
        self.reset()
        
    def reset(self) -> None:
        self.path: torch.Tensor | None = None
        self.path_index = 0
        self.last_goal_arm: torch.Tensor | None = None
        self.jaw_target: torch.Tensor | None = None
        self.jaw_ramp_start: torch.Tensor | None = None
        self.jaw_ramp_steps = 0
        self.jaw_ramp_index = 0
        
    @property
    def is_active(self) -> bool:
        return self.path is not None and self.path_index < self.path.shape[0]
    
    def set_hold_jaw(
        self,
        jaw_target: torch.Tensor,
        ramp_steps: int = 0,
        current_jaw: torch.Tensor | None = None,
    ) -> None:
        self.jaw_target = jaw_target.detach().clone()
        if ramp_steps > 0:
            if current_jaw is None:
                current_jaw = self.jaw_target
            self.jaw_ramp_start = current_jaw.detach().clone()
            self.jaw_ramp_steps = max(1, int(ramp_steps))
            self.jaw_ramp_index = 0
        else:
            self.jaw_ramp_start = None
            self.jaw_ramp_steps = 0
            self.jaw_ramp_index = 0
        
    def set_trajectory(self, path: torch.Tensor, jaw_target: torch.Tensor) -> None:
        self.path = path.to(device=self.device)
        self.path_index = 0
        self.last_goal_arm = self.path[-1:].detach().clone()
        self.jaw_target = jaw_target.detach().clone()
        self.jaw_ramp_start = None
        self.jaw_ramp_steps = 0
        self.jaw_ramp_index = 0
        print(f"[MoveIt] {self.name}: loaded {self.path.shape[0]} control targets")
        
    def step(self, current_action: torch.Tensor) -> torch.Tensor:
        next_action = current_action.clone()
        if self.path is not None and self.path_index < self.path.shape[0]:
            next_action[:, : len(ARM_JOINT_NAMES)] = self.path[self.path_index].unsqueeze(0)
            self.path_index += 1
            if self.path_index >= self.path.shape[0]:
                self.path = None
        if self.jaw_target is not None:
            if self.jaw_ramp_start is not None and self.jaw_ramp_index < self.jaw_ramp_steps:
                alpha = float(self.jaw_ramp_index + 1) / float(self.jaw_ramp_steps)
                next_action[:, len(ARM_JOINT_NAMES) :] = (
                    self.jaw_ramp_start + alpha * (self.jaw_target - self.jaw_ramp_start)
                )
                self.jaw_ramp_index += 1
                if self.jaw_ramp_index >= self.jaw_ramp_steps:
                    self.jaw_ramp_start = None
            else:
                next_action[:, len(ARM_JOINT_NAMES) :] = self.jaw_target
        return next_action
    
def ros_spin(node: Node) -> None:
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    
def duration_to_sec(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9

def publish_moveit_joint_state(ros_node: BridgeClient, joint_action: torch.Tensor) -> None:
    ros_node.publish_joint_state(ISAAC_SO101_JOINT_NAMES, joint_action[0].detach().cpu().tolist())
    
def trajectory_to_arm_path(
    trajectory: JointTrajectory,
    current_arm_target: torch.Tensor,
    control_dt: float,
    device: str,
    fallback_steps: int,
    time_scale: float,
    max_control_steps: int,
) -> torch.Tensor:
    missing = [joint_name for joint_name in ARM_JOINT_NAMES if joint_name not in trajectory.joint_names]
    if missing:
        raise RuntimeError(f"MoveIt trajectory missing expected SO101 joints: {missing}")
    
    name_to_arm_id = {name: index for index, name in enumerate(ARM_JOINT_NAMES)}
    current = current_arm_target.detach().clone().to(device=device)
    previous_time = 0.0
    segment_steps: list[int] = []
    scaled_duration = 0.0
    
    for point in trajectory.points:
        point_time = duration_to_sec(point.time_from_start) * time_scale
        if point_time > previous_time:
            steps = max(1, int(round((point_time - previous_time) / control_dt)))
        else:
            steps = max(1, int(round(fallback_steps * time_scale)))
        segment_steps.append(steps)
        previous_time = point_time
        scaled_duration = max(scaled_duration, point_time)
    
    dense_step_count = sum(segment_steps)
    if max_control_steps > 0 and dense_step_count > max_control_steps:
        effective_cap = max(max_control_steps, len(segment_steps))
        compression = effective_cap / float(dense_step_count)
        segment_steps = [max(1, int(round(steps * compression))) for steps in segment_steps]
        while sum(segment_steps) > effective_cap:
            largest_idx = max(range(len(segment_steps)), key=segment_steps.__getitem__)
            if segment_steps[largest_idx] <= 1:
                break
            segment_steps[largest_idx] -= 1
        print(
            f"[MoveIt] trajectory dense target capped: original={dense_step_count}, "
            f"capped={sum(segment_steps)}, duratoin={scaled_duration:.3f}s"
        )
        
    path: list[torch.Tensor] = []
    
    for point, steps in zip(trajectory.points, segment_steps):
        next_target = current.clone()
        for joint_name, position in zip(trajectory.joint_names, point.positions):
            arm_id = name_to_arm_id.get(joint_name)
            if arm_id is not None:
                next_target[:, arm_id] = float(position)
        
        start = current.clone()
        for step_idx in range(steps):
            alpha = float(step_idx + 1) / float(steps)
            path.append(start + alpha * (next_target - start))
            
        current = next_target
    
    if not path:
        raise RuntimeError("MoveIt trajectory contained no executable points.")
    return torch.cat(path, dim=0)


def request_moveit_plan(
    ros_node: BridgeClient,
    robot_name: str,
    current_action: torch.Tensor,
    goal_action: torch.Tensor,
) -> JointTrajectory:
    start_positions = current_action[0, : len(ARM_JOINT_NAMES)].detach().cpu().tolist()
    goal_positions = goal_action[0, : len(ARM_JOINT_NAMES)].detach().cpu().tolist()
    print(
        f"[Plan] {robot_name}: start={[round(value, 4) for value in start_positions]}, "
        f"goal={[round(value, 4) for value in goal_positions]}"
    )

    publish_moveit_joint_state(ros_node, current_action)
    ros_node.publish_joint_goal(start_positions, goal_positions)
    
    deadline = time.monotonic() + args_cli.plan_timeout
    while simulation_app.is_running() and time.monotonic() < deadline:
        trajectory = ros_node.take_trajectory()
        if trajectory is not None:
            if not trajectory.points:
                raise RuntimeError(f"Received empty MoveIt trajectory for {robot_name}.")
            print(f"[Plan] {robot_name}: received trajectory with {len(trajectory.points)} points")
            return trajectory
        time.sleep(0.02)
        
    raise TimeoutError(
        f"Timed out waiting for MoveIt bridge trajectory for {robot_name}; last_status={ros_node.last_status!r}"
    )
    
def maybe_start_moveit_plan(
    ros_node: BridgeClient,
    follower: MoveItArmTrajectoryFollower,
    robot_name: str,
    command_step: int,
    plan_due: bool,
    current_action: torch.Tensor,
    goal_action: torch.Tensor,
    control_dt: float,
) -> bool:
    follower.set_hold_jaw(goal_action[:, len(ARM_JOINT_NAMES) :])
    if command_step not in MOVEIT_COMMAND_STEPS or not plan_due:
        return False
    
    current_delta = torch.max(torch.abs(goal_action[:, : len(ARM_JOINT_NAMES)] - current_action[:, : len(ARM_JOINT_NAMES)]))
    if float(current_delta.item()) <= args_cli.moveit_goal_tolerance:
        return False
    
    trajectory = request_moveit_plan(ros_node, robot_name, current_action, goal_action)
    arm_path = trajectory_to_arm_path(
        trajectory,
        current_action[:, : len(ARM_JOINT_NAMES)],
        control_dt,
        current_action.device,
        args_cli.trajectory_point_steps,
        args_cli.trajectory_time_scale,
        args_cli.max_trajectory_control_steps,
    )
    follower.set_trajectory(arm_path, goal_action[:, len(ARM_JOINT_NAMES) :])
    return True

def moveit_joint_goal_changed(goal_action: torch.Tensor, previous_goal: torch.Tensor | None) -> bool:
    if previous_goal is None:
        return True
    joint_delta = torch.max(torch.abs(goal_action[:, : len(ARM_JOINT_NAMES)] - previous_goal[:, : len(ARM_JOINT_NAMES)]))
    return float(joint_delta.item()) > args_cli.moveit_goal_tolerance

# --------------------------------------------------------------------------
# LeRobot helpers
# --------------------------------------------------------------------------

def create_lerobot_dataset(dataset_dir: str, vcodec: str = "libsvtav1", streaming_encoding: bool = True):
    """Create a LeRobot dataset for bi-manual SO101 (12D state/action)."""
    if not LEROBOT_AVAILABLE:
        return None

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),  # 6 per robot × 2
            "names": LEROBOT_BIMANUAL_FEATURE_NAMES,
            "fps": DATASET_FPS,
        },
        "action": {
            "dtype": "float32",
            "shape": (12,),
            "names": LEROBOT_BIMANUAL_FEATURE_NAMES,
            "fps": DATASET_FPS,
        },
    }

    video_keys = [
        "observation.images.gripper_r1",
        "observation.images.gripper_r2",
        "observation.images.top",
    ]
    for key in video_keys:
        features[key] = {
            "dtype": "video",
            "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": CAMERA_HEIGHT,
                "video.width": CAMERA_WIDTH,
                "video.codec": vcodec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": DATASET_FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    kwargs = {
        "repo_id": "so101_bimanual_handover",
        "root": dataset_dir,
        "fps": DATASET_FPS,
        "features": features,
    }
    import inspect
    sig = inspect.signature(LeRobotDataset.create)
    if "vcodec" in sig.parameters:
        kwargs["vcodec"] = vcodec
    if "streaming_encoding" in sig.parameters:
        kwargs["streaming_encoding"] = streaming_encoding

    print(f"[LeRobot] Creating dataset with kwargs: {list(kwargs.keys())}")
    return LeRobotDataset.create(**kwargs)


def camera_rgb_to_uint8(camera_data: torch.Tensor, env_idx: int = 0) -> np.ndarray:
    rgb_frame = camera_data[env_idx].detach().cpu().numpy().copy()
    if rgb_frame.shape[-1] > 3:
        rgb_frame = rgb_frame[..., :3]
    if rgb_frame.dtype != np.uint8:
        if rgb_frame.max() > 1.0:
            rgb_frame = rgb_frame / 255.0
        rgb_frame = np.clip(rgb_frame * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb_frame).copy()


def build_bimanual_lerobot_frame(
    robot1, robot2,
    ordered_joint_ids_1: Sequence[int],
    ordered_joint_ids_2: Sequence[int],
    gripper_cam_r1_data: torch.Tensor,
    gripper_cam_r2_data: torch.Tensor,
    top_cam_data: torch.Tensor | None = None,
    action: torch.Tensor | np.ndarray | None = None,
    task: str = "Robot1 picks cube → handover to Robot2 → Robot2 places in bin",
    env_idx: int = 0,
) -> dict:
    r1_state = robot1.data.joint_pos[env_idx, ordered_joint_ids_1].detach().cpu().numpy().astype(np.float32, copy=True)
    r2_state = robot2.data.joint_pos[env_idx, ordered_joint_ids_2].detach().cpu().numpy().astype(np.float32, copy=True)
    combined_state = np.concatenate([r1_state, r2_state])

    frame = {
        "observation.state": combined_state,
        "observation.images.gripper_r1": camera_rgb_to_uint8(gripper_cam_r1_data, env_idx),
        "observation.images.gripper_r2": camera_rgb_to_uint8(gripper_cam_r2_data, env_idx),
        "task": task,
    }
    if top_cam_data is not None:
        frame["observation.images.top"] = camera_rgb_to_uint8(top_cam_data, env_idx)
    if action is not None:
        if isinstance(action, torch.Tensor):
            action = action[env_idx].detach().cpu().numpy()
        frame["action"] = np.asarray(action, dtype=np.float32).copy()
    return frame


def validate_episode_frames(episode_frames: Sequence[dict]) -> list[dict]:
    completed = []
    for i, frame in enumerate(episode_frames):
        if "action" not in frame:
            raise KeyError(f"Frame {i} missing 'action'.")
        completed.append(dict(frame))
    return completed


def get_record_interval(control_dt: float) -> tuple[int, float]:
    control_hz = 1.0 / control_dt
    record_interval = int(round(control_hz / DATASET_FPS))
    if record_interval <= 0:
        raise ValueError(f"Invalid record interval {record_interval} from control_hz={control_hz:.3f}")
    if abs(control_hz - record_interval * DATASET_FPS) > 1e-3:
        raise ValueError(f"Control rate {control_hz:.3f} Hz is not an integer multiple of {DATASET_FPS} Hz.")
    if record_interval != EXPECTED_RECORD_INTERVAL:
        print(
            f"[Timing] WARNING: expected {EXPECTED_CONTROL_HZ}/{DATASET_FPS}={EXPECTED_RECORD_INTERVAL}, "
            f"got control_hz={control_hz:.3f}, interval={record_interval}."
        )
    return record_interval, control_hz


def add_top_camera_to_scene_cfg(scene_cfg):
    scene_cfg.camera_top = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraTop",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            f_stop=1000.0,
            focal_length=TOP_CAMERA_FOCAL_LENGTH,
            focus_distance=TOP_CAMERA_POS[2],
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=TOP_CAMERA_POS,
            rot=euler_angles_to_quat(np.array(TOP_CAMERA_ROT_EULER_DEG), degrees=True),
            convention="opengl",
        ),
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    env_cfg = SO101BimanualHandoverEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    if args_cli.trajectory_time_scale <= 0.0:
        raise ValueError("--trajectory-time-scale must be freater than 0.0.")
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        env_cfg.scene.num_envs = 1

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    record_interval, control_hz = get_record_interval(control_dt)
    add_top_camera_to_scene_cfg(env_cfg.scene)
    if args_cli.dataset_dir is not None:
        env_cfg.num_rerenders_on_reset = max(env_cfg.num_rerenders_on_reset, 20)

    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    if env.num_envs != 1:
        raise RuntimeError("MoveIt handover script currently supprots exactly one IsaacLab environments.")
    
    ros_node = None
    if ROS_AVAILABLE:
        print("[ROS] initializing rclpy node")
        rclpy.init()
        print("[ROS] rclpy initialized")
        ros_node = BridgeClient()
        print("[ROS] BridgeClient created")
        spin_thread = threading.Thread(target=ros_spin, args=(ros_node,), daemon=True)
        spin_thread.start()
        print("[ROS] spin thread started")

    # State machine
    handover_sm = BiManualHandoverStateMachine(num_envs=env.num_envs, device=env.device)

    # Scene entities
    robot1 = env.scene["robot1"]
    robot2 = env.scene["robot2"]
    cube = env.scene["cube"]
    box = env.scene["box"]
    gripper_cam_r1 = env.scene["camera_ego_1"]
    gripper_cam_r2 = env.scene["camera_ego_2"]
    top_camera = env.scene["camera_top"]

    ee_idx_1 = robot1.body_names.index("gripper")
    ee_idx_2 = robot2.body_names.index("gripper")
    ordered_jids_1 = get_ordered_joint_ids(robot1)
    ordered_jids_2 = get_ordered_joint_ids(robot2)

    # IK teachers
    goal_generator1 = SO101DiffIKGoalGenerator(robot=robot1, num_envs=env.num_envs, device=env.device)
    goal_generator2 = SO101DiffIKGoalGenerator(robot=robot2, num_envs=env.num_envs, device=env.device)
    follower1 = MoveItArmTrajectoryFollower("robot1", env.device)
    follower2 = MoveItArmTrajectoryFollower("robot2", env.device)

    # LeRobot dataset
    lerobot_dataset = None
    if args_cli.dataset_dir is not None and LEROBOT_AVAILABLE:
        print(f"[LeRobot] Creating dataset at: {args_cli.dataset_dir}")
        lerobot_dataset = create_lerobot_dataset(
            dataset_dir=args_cli.dataset_dir,
            vcodec=args_cli.vcodec,
            streaming_encoding=args_cli.streaming_encoding,
        )
        print("[LeRobot] Dataset created successfully")
    elif args_cli.dataset_dir is not None and not LEROBOT_AVAILABLE:
        print("[LeRobot] WARNING: LeRobot not installed. Data collection disabled.")

    # Action buffer (12D: 6 per robot)
    actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim), dtype=torch.float32, device=env.device
    )
    r1_home_joint_target = robot1.data.joint_pos[:, ordered_jids_1].clone()
    r2_home_joint_target = robot2.data.joint_pos[:, ordered_jids_2].clone()
    current_joint_target_1 = robot1.data.joint_pos[:, ordered_jids_1].clone()
    current_joint_target_2 = robot2.data.joint_pos[:, ordered_jids_2].clone()
    actions[:] = torch.cat([current_joint_target_1, current_joint_target_2], dim=-1)

    is_recording = lerobot_dataset is not None

    # Tracking state
    saved_episode_count = 0
    attempted_episode_count = 0
    discarded_episode_count = 0
    episode_control_step = 0
    total_control_step = 0
    pending_episode_save = False
    force_policy_update = True
    phase_command_active = False
    phase_hold_counter = 0
    last_moveit_pose_1 = None
    last_moveit_pose_2 = None
    episode_record_frames = [[] for _ in range(env.num_envs)]
    bad_episode = [False for _ in range(env.num_envs)]
    bad_episode_reason = ["" for _ in range(env.num_envs)]
    last_phase = -1
    last_record_command_step = -1

    def target_reached() -> bool:
        if args_cli.num_episodes <= 0:
            return False
        if is_recording:
            return saved_episode_count >= args_cli.num_episodes
        return attempted_episode_count >= args_cli.num_episodes

    def finish_episode_batch(
        reason: str, 
        success_mask: torch.Tensor, 
        ctrl_steps: int,
        success_distance: torch.Tensor | None = None,
    ):
        nonlocal attempted_episode_count, saved_episode_count, discarded_episode_count
        nonlocal episode_record_frames, bad_episode, bad_episode_reason

        success_vals = success_mask.detach().cpu().tolist()
        if success_distance is not None:
            success_distance_vals = success_distance.detach().cpu().tolist()
        else:
            success_distance_vals = [None for _ in range(env.num_envs)]
        saved_this_batch = 0
        for env_idx in range(env.num_envs):
            attempted_episode_count += 1
            success = bool(success_vals[env_idx])
            distance_xy = success_distance_vals[env_idx]
            distance_msg = "n/a" if distance_xy is None else f"{distance_xy:.4f}"
            frame_count = len(episode_record_frames[env_idx])
            should_save = frame_count > 0 and not bad_episode[env_idx] and success
            skip_reason = ""
            if frame_count <= 0:
                skip_reason = "no_frames"
            elif bad_episode[env_idx]:
                skip_reason = f"bad_episode:{bad_episode_reason[env_idx] or 'unknown'}"
            elif not success:
                skip_reason = f"success_false:distance_xy={distance_msg},threshold={SUCCESS_THRESHOLD:.4f}"
            elif target_reached():
                skip_reason = "target_count_reached"

            if lerobot_dataset is not None and should_save and not target_reached():
                try:
                    for frame in validate_episode_frames(episode_record_frames[env_idx]):
                        lerobot_dataset.add_frame(frame)
                    lerobot_dataset.save_episode()
                    saved_episode_count += 1
                    saved_this_batch += 1
                    print(
                        f"[Episode {saved_episode_count}] Saved "
                        f"(env={env_idx}, attempt={attempted_episode_count}, reason={reason}, "
                        f"success={success}, distance_xy={distance_msg}, frames={frame_count}, "
                        f"ctrl_steps={ctrl_steps})"
                    )
                except Exception as e:
                    bad_episode[env_idx] = True
                    bad_episode_reason[env_idx] = f"save_failed:{e}"
                    if hasattr(lerobot_dataset, "clear_episode_buffer"):
                        lerobot_dataset.clear_episode_buffer()
                    print(f"[WARN] env={env_idx} save failed; discarding: {e}")
            else:
                if not should_save:
                    discarded_episode_count += 1
                if not is_recording:
                    print(
                        f"[Episode attempt {attempted_episode_count}] Skipped "
                        f"(env={env_idx}, reason={reason}, skip={skip_reason or 'not_saved'}, "
                        f"success={success}, distance_xy={distance_msg}, frames={frame_count}, "
                        f"bad_episode={bad_episode[env_idx]}, ctrl_steps={ctrl_steps})"
                    )
                else:
                    print(
                        f"[Attempt {attempted_episode_count}] Discarded "
                        f"(env={env_idx}, reason={reason}, skip={skip_reason or 'not_saved'}, "
                        f"success={success}, distance_xy={distance_msg}, frames={frame_count}, "
                        f"saved={saved_episode_count}/{args_cli.num_episodes})"
                    )
        print(
            f"[Batch] attempts={attempted_episode_count}, saved={saved_episode_count}, "
            f"discarded={discarded_episode_count}, saved_this_batch={saved_this_batch}"
        )
        episode_record_frames = [[] for _ in range(env.num_envs)]
        bad_episode = [False for _ in range(env.num_envs)]
        bad_episode_reason = ["" for _ in range(env.num_envs)]

    def reset_episode_state():
        nonlocal r1_home_joint_target, r2_home_joint_target
        nonlocal current_joint_target_1, current_joint_target_2
        nonlocal last_moveit_pose_1, last_moveit_pose_2
        nonlocal episode_control_step, force_policy_update, pending_episode_save
        nonlocal phase_command_active, phase_hold_counter
        nonlocal episode_record_frames, bad_episode, bad_episode_reason, last_phase, last_record_command_step

        env.reset()
        handover_sm.reset_idx()
        goal_generator1.reset()
        goal_generator2.reset()
        follower1.reset()
        follower2.reset()
        r1_home_joint_target = robot1.data.joint_pos[:, ordered_jids_1].clone()
        r2_home_joint_target = robot2.data.joint_pos[:, ordered_jids_2].clone()
        current_joint_target_1 = robot1.data.joint_pos[:, ordered_jids_1].clone()
        current_joint_target_2 = robot2.data.joint_pos[:, ordered_jids_2].clone()
        actions[:] = torch.cat([current_joint_target_1, current_joint_target_2], dim=-1)
        last_moveit_pose_1 = None
        last_moveit_pose_2 = None
        episode_record_frames = [[] for _ in range(env.num_envs)]
        bad_episode = [False for _ in range(env.num_envs)]
        bad_episode_reason = ["" for _ in range(env.num_envs)]
        episode_control_step = 0
        pending_episode_save = False
        force_policy_update = True
        phase_command_active = False
        phase_hold_counter = 0
        last_phase = -1
        last_record_command_step = -1
    show_markers = not is_recording
    ee_marker1 = goal_marker1 = ee_marker2 = goal_marker2 = None
    if show_markers:
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker1 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current1"))
        goal_marker1 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal1"))
        ee_marker2 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current2"))
        goal_marker2 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal2"))
    # Print config summary
    print(f"\n{'=' * 60}")
    print("Phase-Driven Bi-Manual SO101 Handover State Machine")
    print(f"{'=' * 60}")
    print(f"Environments: {env.num_envs}")
    print(f"Device: {env.device}")
    print(f"Control rate: {control_hz:.1f} Hz")
    print(f"Dataset rate: {DATASET_FPS} Hz (record_interval={record_interval})")
    print(f"Action space: {env.action_manager.total_action_dim}D "
          f"(Robot1[6] + Robot2[6])")
    print(f"Joint order per robot: {ISAAC_SO101_JOINT_NAMES}")
    print(f"MoveIt joint goal topic: {args_cli.joint_goal_topic}")
    print(f"MoveIt trajectory topic: {args_cli.planned_trajectory_topic}")
    print(f"MoveIt plan timeout: {args_cli.plan_timeout:.1f}s")
    print(f"MoveIt trajectory time scale: {args_cli.trajectory_time_scale:.3f}")
    print(f"MoveIt max trajectory control steps: {args_cli.max_trajectory_control_steps}")
    print(f"Wait for MoveIt trajectories: {args_cli.wait_for_moveit_trajectories}")
    print(f"Recording: {'YES' if is_recording else 'NO'}")
    print(f"Markers: {'ON' if show_markers else 'OFF'}")
    if is_recording:
        print(f"Dataset: {args_cli.dataset_dir}")
        print(f"Max episodes: {args_cli.num_episodes}")
        print(f"Video codec: {args_cli.vcodec}")
        print(f"State-machine phases: {len(PHASE_COMMAND_STEPS)}")
        print(f"Recorded steps: [{RECORD_START_STEP}, {RECORD_END_STEP}]")
    print(f"{'=' * 60}\n")

    loop_exit_reason = "not_entered"
    try:
        while True:
            if not simulation_app.is_running():
                loop_exit_reason = "simulation_app_not_running"
                print(
                    "[DIAG] main loop exit: simulation_app.is_running() returned False "
                    f"(attempts={attempted_episode_count}, episode_control_step={episode_control_step}, "
                    f"total_control_step={total_control_step}, pending_episode_save={pending_episode_save})",
                    flush=True,
                )
                break
            with torch.inference_mode():
                if target_reached():
                    count = saved_episode_count if is_recording else attempted_episode_count
                    label = "Saved" if is_recording else "Completed"
                    loop_exit_reason = "target_reached"
                    print(f"\n[Done] {label} {count} episodes. Exiting.")
                    break

                # Episode save check
                if pending_episode_save and episode_control_step % record_interval == 0:
                    cube_pos_w = cube.data.root_pos_w[:, :3].clone()
                    box_pos_w = get_box_pos_w(box)
                    success_distance = compute_success_xy_distance(cube_pos_w, box_pos_w)
                    success_mask = success_distance < SUCCESS_THRESHOLD
                    finish_episode_batch("state_machine_cycle", success_mask, episode_control_step, success_distance)
                    reset_episode_state()
                    continue

                # Current observations
                ee_pos_w_1 = robot1.data.body_pos_w[:, ee_idx_1, :].clone()
                ee_pos_w_2 = robot2.data.body_pos_w[:, ee_idx_2, :].clone()
                ee_quat_w_2 = robot2.data.body_quat_w[:, ee_idx_2, :].clone()
                cube_pos_w = cube.data.root_pos_w[:, :3].clone()
                box_pos_w = get_box_pos_w(box)
                handover_zone = compute_handover_zone(env)

                # Phase commands are generated only when the previous phase has completed.
                # LeRobot recording remains tied to dataset FPS and continues through MoveIt trajectories.
                should_record = False
                cam_r1_data = cam_r2_data = top_data = None
                is_record_tick = episode_control_step % record_interval == 0
                waiting_for_moveit = args_cli.wait_for_moveit_trajectories and (
                    follower1.is_active or follower2.is_active
                )
                policy_update_due = force_policy_update and not waiting_for_moveit
                if force_policy_update and waiting_for_moveit and total_control_step % 100 == 0:
                    print("[MoveIt] holding phase until active trajectory completes")
                if policy_update_due:
                    raw_r1, raw_r2, step_at_command = handover_sm.compute(
                        cube_pos_w, box_pos_w, handover_zone, ee_pos_w_2, ee_quat_w_2
                    )

                    command_step = int(step_at_command[0].item())
                    last_record_command_step = command_step
                    current_phase = phase_from_step(command_step)
                    command_is_record_step = RECORD_START_STEP <= command_step < RECORD_END_STEP

                    # Compute joint targets for both robots
                    r1_action = make_joint_goal_action(
                        robot1,
                        ordered_jids_1,
                        current_joint_target_1[:, : len(ARM_JOINT_NAMES)],
                        raw_r1[:, -1],
                    )
                    r2_action = make_joint_goal_action(
                        robot2,
                        ordered_jids_2,
                        current_joint_target_2[:, : len(ARM_JOINT_NAMES)],
                        raw_r2[:, -1],
                    )
                    r1_saturated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                    r2_saturated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                    planned_1 = planned_2 = False
                    r1_arm_plan_step = ROS_AVAILABLE and command_step in R1_MOVEIT_COMMAND_STEPS
                    r2_arm_plan_step = ROS_AVAILABLE and command_step in R2_MOVEIT_COMMAND_STEPS

                    if r1_arm_plan_step and command_step == 290:
                        r1_action = r1_home_joint_target.detach().clone()
                        r1_action[:, len(ARM_JOINT_NAMES) :] = compute_gripper_joint_target(raw_r1[:, -1])
                        print("[MoveIt] robot1 step290 uses saved IsaacLab home joint goal")
                    elif r1_arm_plan_step:
                        r1_action, r1_saturated = solve_moveit_joint_goal(
                            env,
                            robot1,
                            goal_generator1,
                            ordered_jids_1,
                            raw_r1,
                            current_joint_target_1,
                            [(robot2, ordered_jids_2, current_joint_target_2.detach().clone())],
                            "robot1",
                        )
                    if r2_arm_plan_step:
                        r2_action, r2_saturated = solve_moveit_joint_goal(
                            env,
                            robot2,
                            goal_generator2,
                            ordered_jids_2,
                            raw_r2,
                            current_joint_target_2,
                            [(robot1, ordered_jids_1, current_joint_target_1.detach().clone())],
                            "robot2",
                        )

                    gripper_ramp_steps = GRIPPER_RAMP_STEPS if command_step in GRIPPER_ONLY_COMMAND_STEPS else 0
                    follower1.set_hold_jaw(
                        r1_action[:, len(ARM_JOINT_NAMES) :],
                        ramp_steps=gripper_ramp_steps,
                        current_jaw=current_joint_target_1[:, len(ARM_JOINT_NAMES) :],
                    )
                    follower2.set_hold_jaw(
                        r2_action[:, len(ARM_JOINT_NAMES) :],
                        ramp_steps=gripper_ramp_steps,
                        current_jaw=current_joint_target_2[:, len(ARM_JOINT_NAMES) :],
                    )
                    if gripper_ramp_steps > 0:
                        print(
                            f"[Gripper] ramp step={command_step} "
                            f"duration={gripper_ramp_steps} control steps"
                        )

                    if command_is_record_step and (r1_arm_plan_step or r2_arm_plan_step):
                        sat1_vals = r1_saturated.detach().cpu().tolist()
                        sat2_vals = r2_saturated.detach().cpu().tolist()
                        for idx in range(env.num_envs):
                            if sat1_vals[idx] or sat2_vals[idx]:
                                print(
                                    f"[WARN] env={idx} joint limit saturation at step={command_step}; "
                                    "keeping episode for success/save evaluation"
                                )
                    
                    if r1_arm_plan_step:
                        r1_plan_due = moveit_joint_goal_changed(r1_action, last_moveit_pose_1)
                        planned_1 = maybe_start_moveit_plan(
                            ros_node,
                            follower1,
                            "robot1",
                            command_step,
                            r1_plan_due,
                            current_joint_target_1,
                            r1_action,
                            control_dt,
                        )
                        if r1_plan_due:
                            last_moveit_pose_1 = r1_action.detach().clone()
                    
                    if r2_arm_plan_step:
                        r2_plan_due = moveit_joint_goal_changed(r2_action, last_moveit_pose_2)
                        planned_2 = maybe_start_moveit_plan(
                            ros_node,
                            follower2,
                            "robot2",
                            command_step,
                            r2_plan_due,
                            current_joint_target_2,
                            r2_action,
                            control_dt,
                        )
                        if r2_plan_due:
                            last_moveit_pose_2 = r2_action.detach().clone()
                       
                    if command_step == 0 and r2_arm_plan_step:
                        print(f"\n[DEBUG R2 PIPELINE]")
                        print(f" state_machine target: pos={handover_sm.r2_des_ee_pos[0].tolist()}, quat={handover_sm.r2_des_ee_quat[0].tolist()}")
                        print(f" current EE pos (world): {ee_pos_w_2[0].tolist()}")
                        print(f" MoveIt joint goal r2_action: {[round(x,4) for x in r2_action[0].tolist()]}")
                        print(f" current_joint_target_2: {[round(x,4) for x in current_joint_target_2[0].tolist()]}")
                        print(f" planned_1={planned_1} planned_2={planned_2}")
                        print(f" r2 saturated: {r2_saturated[0].item()}")
                        print(f" R2 root pos: {robot2.data.root_pos_w[0].tolist()}")
                        print(f" R2 root quat: {robot2.data.root_quat_w[0].tolist()}")
                    force_policy_update = False
                    
                    if current_phase != last_phase:
                        r1_tgt = handover_sm.r1_des_ee_pos[0].cpu().numpy()
                        r2_tgt = handover_sm.r2_des_ee_pos[0].cpu().numpy()
                        print(
                            f" STATE->{PHASE_NAMES[current_phase]} "
                            f"step={command_step} "
                            f"r1_ee=({r1_tgt[0]:.3f},{r1_tgt[1]:.3f},{r1_tgt[2]:.3f}) "
                            f"r2_ee=({r2_tgt[0]:.3f},{r2_tgt[1]:.3f},{r2_tgt[2]:.3f})"
                        )
                        last_phase = current_phase
                    force_policy_update = False
                    phase_command_active = True
                    phase_hold_counter = 0
                should_record = (
                    is_recording and lerobot_dataset is not None
                    and is_record_tick
                    and RECORD_START_STEP <= last_record_command_step < RECORD_END_STEP
                )

                # Camera capture is tied to dataset FPS, not to state-machine command updates.
                if should_record:
                    try:
                        env.sim.render()
                        cam_r1_data = gripper_cam_r1.data.output["rgb"]
                        cam_r2_data = gripper_cam_r2.data.output["rgb"]
                        if top_camera is not None:
                            top_data = top_camera.data.output["rgb"]
                    except Exception as e:
                        print(f"[ERROR] Camera capture failed: {e}")
                # Combine 12D action
                current_joint_target_1 = clamp_joint_targets(
                    robot1, ordered_jids_1, follower1.step(current_joint_target_1)
                )
                current_joint_target_2 = clamp_joint_targets(
                    robot2, ordered_jids_2, follower2.step(current_joint_target_2)
                )
                actions[:] = torch.cat([current_joint_target_1, current_joint_target_2], dim=-1)

                # Record frames
                has_cam_data = cam_r1_data is not None and cam_r2_data is not None
                if should_record and has_cam_data:
                    for idx in range(env.num_envs):
                        if bad_episode[idx]:
                            continue
                        try:
                            episode_record_frames[idx].append(
                                build_bimanual_lerobot_frame(
                                    robot1, robot2,
                                    ordered_jids_1, ordered_jids_2,
                                    cam_r1_data, cam_r2_data, top_data,
                                    action=actions, env_idx=idx,
                                )
                            )
                            fc = len(episode_record_frames[idx])
                            if fc % 100 == 0:
                                print(f"[LeRobot] env={idx} buffered {fc} frames")
                        except Exception as e:
                            bad_episode[idx] = True
                            bad_episode_reason[idx] = f"buffer_failed:{e}"
                            episode_record_frames[idx] = []
                            print(f"[ERROR] env={idx} buffer failed; discarding: {e}")
                elif should_record:
                    for idx in range(env.num_envs):
                        if not bad_episode[idx]:
                            bad_episode[idx] = True
                            bad_episode_reason[idx] = "camera_data_missing"
                            episode_record_frames[idx] = []
                    print("[ERROR] Camera data missing; discarding batch")

                if total_control_step <= 3:
                    print(f"\n[DEBUG ACTION STEP={total_control_step}]")
                    print(f"  actions[0]: {[round(x,4) for x in actions[0].tolist()]}")
                    print(f"  r1 joints: {[round(x,4) for x in current_joint_target_1[0].tolist()]}")
                    print(f"  r2 joints: {[round(x,4) for x in current_joint_target_2[0].tolist()]}")
                
                _, _ = env.step(actions)
                total_control_step += 1
                episode_control_step += 1

                if phase_command_active:
                    if follower1.is_active or follower2.is_active:
                        phase_hold_counter = 0
                    else:
                        phase_hold_counter += 1
                        command_step = handover_sm.current_command_step()
                        hold_steps = PHASE_HOLD_STEPS[handover_sm.phase_index]
                        if phase_hold_counter >= hold_steps:
                            if handover_sm.is_final_phase():
                                pending_episode_save = True
                                phase_command_active = False
                            else:
                                handover_sm.advance()
                                force_policy_update = True
                                phase_command_active = False
                                phase_hold_counter = 0
                                last_record_command_step = handover_sm.current_command_step()
                                if command_step in GRIPPER_ONLY_COMMAND_STEPS:
                                    print(f"[Phase] gripper hold complete at step={command_step}")
                
                if total_control_step % 100 == 0 or total_control_step <= 3:
                    r1_ee = ee_pos_w_1[0].cpu().numpy()
                    r2_ee = ee_pos_w_2[0].cpu().numpy()
                    cube_np = cube_pos_w[0].cpu().numpy()
                    r1_jaw = robot1.data.joint_pos[:, ordered_jids_1[-1]][0].item()
                    r2_jaw = robot2.data.joint_pos[:, ordered_jids_2[-1]][0].item()
                    print(
                        f" step={total_control_step}"
                        f"r1_ee=({r1_ee[0]:.3f},{r1_ee[1]:.3f},{r1_ee[2]:.3f}) "
                        f"r2_ee=({r2_ee[0]:.3f},{r2_ee[1]:.3f},{r2_ee[2]:.3f}) "
                        f"cube=({cube_np[0]:.3f},{cube_np[1]:.3f},{cube_np[2]:.3f}) "
                        f"r1_jaw={r1_jaw:.3f} r2_jaw={r2_jaw:.3f}"
                    )
                
                if show_markers:
                    ee_quat_w1 = robot1.data.body_quat_w[:, ee_idx_1, :].clone()
                    ee_marker1.visualize(robot1.data.body_pos_w[:, ee_idx_1, :].clone(), ee_quat_w1)
                    goal_marker1.visualize(handover_sm.r1_des_ee_pos, handover_sm.r1_des_ee_quat)

                    ee_quat_w2 = robot2.data.body_quat_w[:, ee_idx_2, :].clone()
                    ee_marker2.visualize(robot2.data.body_pos_w[:, ee_idx_2, :].clone(), ee_quat_w2)
                    goal_marker2.visualize(handover_sm.r2_des_ee_pos, handover_sm.r2_des_ee_quat)
    except KeyboardInterrupt:
        loop_exit_reason = "keyboard_interrupt"
        print("\n[Interrupted] Saving current episode...")
    except SystemExit as e:
        loop_exit_reason = f"system_exit:{e.code}"
        print(f"\n[DIAG] SystemExit caught in main loop: code={e.code}", flush=True)
    finally:
        try:
            app_running_now = simulation_app.is_running()
        except Exception as e:
            app_running_now = f"error:{e}"
        print(
            "[DIAG] entering finally "
            f"(reason={loop_exit_reason}, app_running={app_running_now}, "
            f"attempts={attempted_episode_count}, saved={saved_episode_count}, "
            f"discarded={discarded_episode_count}, "
            f"episode_control_step={episode_control_step}, total_control_step={total_control_step}, "
            f"pending_episode_save={pending_episode_save}, signals={_DIAG_RECEIVED_SIGNALS})",
            flush=True,
        )
        if lerobot_dataset is not None:
            print("[LeRobot] Finalizing dataset...")
            try:
                if episode_control_step > 0:
                    cube_pos_w = cube.data.root_pos_w[:, :3].clone()
                    box_pos_w = get_box_pos_w(box)
                    success_distance = compute_success_xy_distance(cube_pos_w, box_pos_w)
                    success_mask = success_distance < SUCCESS_THRESHOLD
                    finish_episode_batch(
                        "interrupted_or_shutdown", success_mask, episode_control_step, success_distance
                    )
            except Exception as e:
                print(f"[WARN] Failed to save final episode: {e}")
            try:
                lerobot_dataset.finalize()
                print("[LeRobot] Dataset finalized successfully")
            except Exception as e:
                print(f"[WARN] Failed to finalize dataset: {e}")
        if ros_node is not None:
            ros_node.destroy_node()
        if ROS_AVAILABLE and rclpy.ok():
            rclpy.shutdown()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
