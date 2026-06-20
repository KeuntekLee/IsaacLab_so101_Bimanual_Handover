# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bi-manual SO101 handover environment configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaacsim.core.utils.rotations import euler_angles_to_quat

from isaaclab_assets.robots.so_arm101 import SO_ARM101_CFG


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Minimal pick-place scene used by the SO101 bimanual overlay."""

    robot: ArticulationCfg = MISSING
    cube: RigidObjectCfg = MISSING
    box: AssetBaseCfg = MISSING
    camera_ego: TiledCameraCfg = MISSING

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )


@configclass
class ObservationsCfg:
    """Placeholder base config; bimanual observations are defined from scratch below."""

    pass
# Bin: use a simple cuboid instead of remote USD to avoid asset dependency
BIN_SIZE = (0.20, 0.20, 0.05)
BIN_COLOR = (0.0, 0.3, 0.6)

#pos=(0.0, 0.0, 0.0),
#rot=(0.7071, 0.0, 0.0, 0.7071),

# Robot placement (relative to env origin)
ROBOT1_ROOT_POS = (0.0, 0.0, 0.0)   # left side (pick side)
ROBOT2_ROOT_POS = (0.8, 0.02, 0.0)   # right side (place side)

ROBOT1_ROOT_ROT = (0.7071, 0.0, 0.0, 0.7071)   # left side (pick side)
ROBOT2_ROOT_ROT = (0.7071, 0.0, 0.0, -0.7071)   # right side (place side)

# Cube/bin grid points are relative to each environment origin. They are kept inside
# conservative reachable regions for the scripted MoveIt handover policy.
CUBE_GRID_POINTS = (
    (0.17, -0.13),
    (0.17, -0.10),
    (0.17, -0.07),
    (0.17, -0.04),
    (0.17, -0.01),
    (0.19, -0.13),
    (0.19, -0.10),
    (0.19, -0.07),
    (0.19, -0.04),
    (0.19, -0.01),
    (0.21, -0.13),
    (0.21, -0.10),
    (0.21, -0.07),
    (0.21, -0.04),
    (0.21, -0.01),
    (0.23, -0.13),
    (0.23, -0.10),
    (0.23, -0.07),
    (0.23, -0.04),
    (0.23, -0.01),
    (0.25, -0.13),
    (0.25, -0.10),
    (0.25, -0.07),
    (0.25, -0.04),
    (0.25, -0.01),
)
BIN_GRID_POINTS = (
    (0.52, -0.13),
    (0.52, -0.10),
    (0.52, -0.07),
    (0.52, -0.04),
    (0.52, -0.01),
    (0.55, -0.13),
    (0.55, -0.10),
    (0.55, -0.07),
    (0.55, -0.04),
    (0.55, -0.01),
    (0.58, -0.13),
    (0.58, -0.10),
    (0.58, -0.07),
    (0.58, -0.04),
    (0.58, -0.01),
    (0.61, -0.13),
    (0.61, -0.10),
    (0.61, -0.07),
    (0.61, -0.04),
    (0.61, -0.01),
    (0.64, -0.13),
    (0.64, -0.10),
    (0.64, -0.07),
    (0.64, -0.04),
    (0.64, -0.01),
)
CUBE_Z = 0.015
BIN_Z = 0.0
CUBE_BIN_GRID_XY_JITTER = 0.01
CUBE_BIN_GRID_YAW_JITTER = 0.25
CUBE_R1_REACH_DISTANCE = (0.16, 0.31)
BIN_R2_REACH_DISTANCE = (0.14, 0.33)

MIN_CUBE_BIN_DISTANCE = 0.12
MAX_LAYOUT_SAMPLE_TRIES = 50

BIN_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd"
BIN_SCALE = (0.35, 0.35, 0.35)

# --------------------------------------------------------------------------
# Helper MDP functions
# --------------------------------------------------------------------------

def _get_root_pos_w(asset) -> torch.Tensor:
    if hasattr(asset, "data") and hasattr(asset.data, "root_pos_w"):
        return asset.data.root_pos_w[:, :3]
    positions, _ = asset.get_world_poses()
    return positions[:, :3]


def cube_position_in_robot1_root_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot1"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]
    return _get_root_pos_w(cube) - robot.data.root_pos_w[:, :3]


def box_position_in_robot2_root_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot2"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]
    return _get_root_pos_w(box) - robot.data.root_pos_w[:, :3]


def ee1_to_cube_distance(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot1", body_names="gripper"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :].clone()
    cube_pos_w = _get_root_pos_w(cube)
    return torch.norm(ee_pos_w - cube_pos_w, dim=1, keepdim=True)


def ee2_to_box_distance(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot2", body_names="gripper"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :].clone()
    box_pos_w = _get_root_pos_w(box)
    return torch.norm(ee_pos_w - box_pos_w, dim=1, keepdim=True)


def compute_handover_zone(env) -> torch.Tensor:
    """Compute handover zone as midpoint between two robot roots, slightly above table."""
    robot1 = env.scene["robot1"]
    robot2 = env.scene["robot2"]
    mid_xy = (robot1.data.root_pos_w[:, :2] + robot2.data.root_pos_w[:, :2]) / 2.0
    handover_z = torch.full((mid_xy.shape[0], 1), 0.14, device=mid_xy.device, dtype=mid_xy.dtype)
    return torch.cat([mid_xy, handover_z], dim=1)


def _sample_pose_values(
    num_samples: int,
    pose_range: dict[str, tuple[float, float]] | None,
    device,
) -> torch.Tensor:
    pose_range = pose_range or {}
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_samples, 6), device=device)


def _sample_grid_pose_values(
    num_samples: int,
    grid_points: Sequence[tuple[float, float]],
    z: float,
    xy_jitter: float,
    yaw_jitter: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.tensor(grid_points, dtype=torch.float32, device=device)
    grid_ids = torch.randint(0, grid.shape[0], (num_samples,), device=device)
    xy = grid[grid_ids]
    if xy_jitter > 0.0:
        xy += math_utils.sample_uniform(-xy_jitter, xy_jitter, (num_samples, 2), device=device)
    samples = torch.zeros((num_samples, 6), dtype=torch.float32, device=device)
    samples[:, :2] = xy
    samples[:, 2] = z
    if yaw_jitter > 0.0:
        samples[:, 5] = math_utils.sample_uniform(-yaw_jitter, yaw_jitter, (num_samples,), device=device)
    return samples, grid_ids


def _write_dynamic_root_pose(asset, positions, orientations, env_ids):
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=positions.device), env_ids=env_ids)


def _write_kinematic_root_pose(asset, positions, orientations, env_ids):
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)


def reset_bimanual_scene(env, env_ids):
    """Reset all articulations and rigid objects to defaults."""
    for name, obj in env.scene.rigid_objects.items():
        default_root = obj.data.default_root_state[env_ids].clone()
        default_root[:, 0:3] += env.scene.env_origins[env_ids]
        obj.write_root_pose_to_sim(default_root[:, :7], env_ids=env_ids)
        if name != "box":
            obj.write_root_velocity_to_sim(default_root[:, 7:], env_ids=env_ids)

    for art in env.scene.articulations.values():
        default_root = art.data.default_root_state[env_ids].clone()
        default_root[:, 0:3] += env.scene.env_origins[env_ids]
        art.write_root_pose_to_sim(default_root[:, :7], env_ids=env_ids)
        art.write_root_velocity_to_sim(default_root[:, 7:], env_ids=env_ids)
        art.write_joint_state_to_sim(
            art.data.default_joint_pos[env_ids].clone(),
            art.data.default_joint_vel[env_ids].clone(),
            env_ids=env_ids,
        )


def randomize_cube_and_bin_bimanual(
    env,
    env_ids,
    cube_grid_points=CUBE_GRID_POINTS,
    box_grid_points=BIN_GRID_POINTS,
    cube_z=CUBE_Z,
    box_z=BIN_Z,
    xy_jitter=CUBE_BIN_GRID_XY_JITTER,
    yaw_jitter=CUBE_BIN_GRID_YAW_JITTER,
    cube_reach_distance=CUBE_R1_REACH_DISTANCE,
    box_reach_distance=BIN_R2_REACH_DISTANCE,
    min_distance=MIN_CUBE_BIN_DISTANCE,
    max_tries=MAX_LAYOUT_SAMPLE_TRIES,
):
    """Randomize cube and bin on reachable grid cells with small jitter."""
    cube_obj = env.scene["cube"]
    box_obj = env.scene["box"]
    robot1 = env.scene["robot1"]
    robot2 = env.scene["robot2"]
    num_envs = len(env_ids)
    device = cube_obj.device

    cube_samples, cube_grid_ids = _sample_grid_pose_values(
        num_envs, cube_grid_points, cube_z, xy_jitter, yaw_jitter, device
    )
    box_samples, box_grid_ids = _sample_grid_pose_values(
        num_envs, box_grid_points, box_z, xy_jitter, yaw_jitter, device
    )

    env_origins = env.scene.env_origins[env_ids]
    robot1_xy = robot1.data.root_pos_w[env_ids, :2]
    robot2_xy = robot2.data.root_pos_w[env_ids, :2]
    cube_reach_min, cube_reach_max = cube_reach_distance
    box_reach_min, box_reach_max = box_reach_distance

    for _ in range(max_tries):
        cube_xy_w = cube_samples[:, :2] + env_origins[:, :2]
        box_xy_w = box_samples[:, :2] + env_origins[:, :2]
        cube_r1_dist = torch.linalg.norm(cube_xy_w - robot1_xy, dim=1)
        box_r2_dist = torch.linalg.norm(box_xy_w - robot2_xy, dim=1)
        too_close = torch.linalg.norm(cube_xy_w - box_xy_w, dim=1) < min_distance
        invalid = (
            too_close
            | (cube_r1_dist < cube_reach_min)
            | (cube_r1_dist > cube_reach_max)
            | (box_r2_dist < box_reach_min)
            | (box_r2_dist > box_reach_max)
        )
        if not bool(torch.any(invalid).item()):
            break
        num_invalid = int(invalid.sum().item())
        cube_samples[invalid], cube_grid_ids[invalid] = _sample_grid_pose_values(
            num_invalid, cube_grid_points, cube_z, xy_jitter, yaw_jitter, device
        )
        box_samples[invalid], box_grid_ids[invalid] = _sample_grid_pose_values(
            num_invalid, box_grid_points, box_z, xy_jitter, yaw_jitter, device
        )

    cube_positions = cube_samples[:, :3] + env_origins
    box_positions = box_samples[:, :3] + env_origins
    cube_quats = math_utils.quat_from_euler_xyz(
        cube_samples[:, 3], cube_samples[:, 4], cube_samples[:, 5]
    )
    box_quats = math_utils.quat_from_euler_xyz(box_samples[:, 3], box_samples[:, 4], box_samples[:, 5])

    _write_dynamic_root_pose(cube_obj, cube_positions, cube_quats, env_ids)
    _write_kinematic_root_pose(box_obj, box_positions, box_quats, env_ids)

    if num_envs > 0:
        cube_r1_dist = torch.linalg.norm(cube_positions[:, :2] - robot1_xy, dim=1)
        box_r2_dist = torch.linalg.norm(box_positions[:, :2] - robot2_xy, dim=1)
        print(
            "[Randomize] "
            f"env={int(env_ids[0].item())} "
            f"cube_grid={int(cube_grid_ids[0].item())} "
            f"cube=({cube_positions[0, 0].item():.3f},{cube_positions[0, 1].item():.3f},{cube_positions[0, 2].item():.3f}) "
            f"r1_dist={cube_r1_dist[0].item():.3f} "
            f"bin_grid={int(box_grid_ids[0].item())} "
            f"bin=({box_positions[0, 0].item():.3f},{box_positions[0, 1].item():.3f},{box_positions[0, 2].item():.3f}) "
            f"r2_dist={box_r2_dist[0].item():.3f}"
        )


# --------------------------------------------------------------------------
# Config classes
# --------------------------------------------------------------------------

@configclass
class BimanualEventCfg:
    reset_all = EventTerm(func=reset_bimanual_scene, mode="reset")
    randomize_cube_and_box = EventTerm(
        func=randomize_cube_and_bin_bimanual,
        mode="reset",
    )


@configclass
class BimanualJointActionCfg:
    """12D joint position action: Robot1[6] + Robot2[6]."""
    robot1_arm = mdp.JointPositionActionCfg(
        asset_name="robot1",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        scale=1.0, offset=0.0, use_default_offset=False, preserve_order=True,
    )
    robot1_gripper = mdp.JointPositionActionCfg(
        asset_name="robot1",
        joint_names=["Jaw"],
        scale=1.0, offset=0.0, use_default_offset=False, preserve_order=True,
    )
    robot2_arm = mdp.JointPositionActionCfg(
        asset_name="robot2",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        scale=1.0, offset=0.0, use_default_offset=False, preserve_order=True,
    )
    robot2_gripper = mdp.JointPositionActionCfg(
        asset_name="robot2",
        joint_names=["Jaw"],
        scale=1.0, offset=0.0, use_default_offset=False, preserve_order=True,
    )


@configclass
class BimanualObservationsCfg(ObservationsCfg):
    def __post_init__(self):
        # Do NOT call super().__post_init__() — parent's PolicyCfg references "robot" (singular)
        # which does not exist in the bimanual scene (only "robot1" and "robot2").
        # Build policy group from scratch with correct entity names.
        self.policy = BimanualObservationsCfg.BimanualPolicyCfg()

    @configclass
    class BimanualPolicyCfg(ObsGroup):
        joint_pos_r1 = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot1")},
        )
        joint_vel_r1 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot1")},
        )
        joint_pos_r2 = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot2")},
        )
        joint_vel_r2 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot2")},
        )
        cube_position = ObsTerm(func=cube_position_in_robot1_root_frame)
        box_position = ObsTerm(func=box_position_in_robot2_root_frame)
        ee1_to_cube_dist = ObsTerm(
            func=ee1_to_cube_distance,
            params={"robot_cfg": SceneEntityCfg("robot1", body_names="gripper")},
        )
        ee2_to_box_dist = ObsTerm(
            func=ee2_to_box_distance,
            params={"robot_cfg": SceneEntityCfg("robot2", body_names="gripper")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


@configclass
class SO101BimanualHandoverEnvCfg(ManagerBasedEnvCfg):
    """Bi-manual SO101 handover environment: Robot1 picks cube → handover → Robot2 places in bin."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BimanualObservationsCfg = BimanualObservationsCfg()
    actions: BimanualJointActionCfg = BimanualJointActionCfg()
    events: BimanualEventCfg = BimanualEventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.sim.sync_real_time = True

        # Robot 1 (pick side, left) — use SO_ARM101_CFG.replace() to preserve init_state
        self.scene.robot1 = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot1",
            init_state=SO_ARM101_CFG.init_state.replace(pos=ROBOT1_ROOT_POS, rot=ROBOT1_ROOT_ROT),
        )

        # Robot 2 (place side, right)
        self.scene.robot2 = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot2",
            init_state=SO_ARM101_CFG.init_state.replace(pos=ROBOT2_ROOT_POS, rot=ROBOT2_ROOT_ROT),
        )

        # Cube (pick target)
        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, -0.12, 0.015], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            ),
        )

        # Box / Bin (place target) — cuboid instead of remote USD
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.15, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=BIN_USD_PATH,
                scale=BIN_SCALE,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
        )
        '''
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.32, 0.14, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=sim_utils.CuboidCfg(
                size=BIN_SIZE,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=PreviewSurfaceCfg(diffuse_color=BIN_COLOR),
            ),
        )
        '''
        # Disable base PickPlaceSceneCfg fields (we use robot1/robot2 and camera_ego_1/camera_ego_2)
        self.scene.robot = None
        self.scene.camera_ego = None

        # Gripper cameras (one per robot) — spawn new PinholeCamera prim at gripper path
        for idx in (1, 2):
            name = f"camera_ego_{idx}"
            setattr(self.scene, name, TiledCameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot{idx}/gripper/gripper_cam",
                update_period=0.0,
                height=480,
                width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    projection_type="pinhole",
                    f_stop=100.0,
                    focal_length=13.5,
                    focus_distance=0.05,
                ),
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(-0.005, 0.06, -0.062),
                    rot=euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True),
                    convention="opengl",
                ),
            ))


@configclass
class SO101BimanualHandoverEnvCfg_PLAY(SO101BimanualHandoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
