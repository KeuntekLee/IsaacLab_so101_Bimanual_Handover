# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO-ARM101 robot.

The following configurations are available:

* :obj:`SO_ARM101_CFG`: SO-ARM101 robot with jaw gripper

Reference: SO-ARM101 USD asset
"""

from pathlib import Path

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat


##
# Configuration
##

STIFFNESS = 50.0
DAMPING = 50.0
EFFORT_LIMIT = 12.0
SO101_USD_PATH = Path(__file__).resolve().parents[4] / "assets/robots/SO-ARM101-USD.usd"
SO_ARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.7071, 0.0, 0.0, 0.7071),
        joint_pos={
            "Rotation": 0.0,
            "Pitch": -1.57,
            "Elbow":  1.57,
            "Wrist_Pitch": 1.17,
            "Wrist_Roll": -1.57,
            "Jaw": 0,
        },
        #pos=(0, 0, 0),
        #rot=euler_angles_to_quat(np.array([0, 0, 90]), degrees=True),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            effort_limit_sim=13.23,
            #velocity_limit_sim=5.5,
            stiffness={
                "Rotation": 800.0,  # Highest - moves all mass
                "Pitch": 800.0,  # Slightly less than rotation
                "Elbow": 800.0,  # Reduced based on less mass
                "Wrist_Pitch": 800.0,  # Reduced for less mass
                "Wrist_Roll": 800.0,  # Low mass to move
            },
            damping={
                "Rotation": 40.0,
                "Pitch": 40.0,
                "Elbow": 40.0,
                "Wrist_Pitch": 40.0,
                "Wrist_Roll": 40.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=5.0,  # Increased from 1.9 to 2.5 for stronger grip
            #velocity_limit_sim=1.5,
            stiffness=4.0,  # Increased from 25.0 to 60.0 for more reliable closing
            damping=0.3,  # Increased from 10.0 to 20.0 for stability
        ),
       
    },
)
'''
actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=5.0,
            velocity_limit_sim=3.0,
            stiffness=55.0,
            damping=0.7,   
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=5.0,
            velocity_limit_sim=6.28,
            stiffness=30.0,
            damping=0.8,   
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=5.0,
            velocity_limit_sim=6.28,
            stiffness=25.0,
            damping=0.7,   
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=5.0,
            velocity_limit_sim=6.0,
            stiffness=12.0,
            damping=0.5,   
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=5.0,
            velocity_limit_sim=6.0,
            stiffness=7.0,
            damping=0.5,   
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=5.0,
            velocity_limit_sim=6.0,
            stiffness=4.0,
            damping=0.30,   
        ),
'''
'''
actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,   
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,   
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,   
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,   
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,   
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=30.0,
            velocity_limit_sim=30.0,
            stiffness=4,
            damping=0.30,   
        ),
    },
'''
'''
    actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30,
            stiffness=55,
            damping=0.7,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30,
            stiffness=30,
            damping=0.8,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30,
            stiffness=25,
            damping=0.7,
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30,
            stiffness=12,
            damping=0.5,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30,
            stiffness=7,
            damping=0.5,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=30,
            stiffness=4,
            damping=0.3,
        ),
    },
'''
SO101_CONTACT_GRASP_CFG = SO_ARM101_CFG.copy()
SO101_CONTACT_GRASP_CFG.spawn.activate_contact_sensors = True
"""Configuration of SO-ARM101 robot."""
