# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bi-manual SO101 handover inference with a LeRobot ACT checkpoint.

This script reuses the bi-manual handover scene and executes the learned policy directly as 12D joint
position targets: Robot1[6] + Robot2[6].
"""

import argparse
import os
import signal
import time
from collections.abc import Sequence
from pathlib import Path

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


DEFAULT_CHECKPOINT = os.environ.get("SO101_BIMANUAL_ACT_CHECKPOINT", "")

parser = argparse.ArgumentParser(description="Run ACT inference for bi-manual SO101 handover.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments. Only 1 is supported.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of inference episodes. 0 = infinite.")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="LeRobot pretrained_model directory.")
parser.add_argument("--policy-device", type=str, default="auto", help="Torch device for ACT inference: auto, cpu, cuda.")
parser.add_argument("--policy-fps", type=int, default=30, help="Policy action rate used by the dataset.")
parser.add_argument(
    "--n-action-steps",
    type=int,
    default=50,
    help="How many predicted ACT chunk actions to consume before querying the model again.",
)
parser.add_argument("--max_episode_steps", type=int, default=2000, help="Maximum control steps per episode.")
parser.add_argument("--success-threshold", type=float, default=0.05, help="Cube-bin XY success threshold in meters.")
parser.add_argument(
    "--success-hold-steps",
    type=int,
    default=60,
    help="End an episode after success is held for this many control steps. 0 disables early success termination.",
)
parser.add_argument(
    "--progress-log-interval",
    type=int,
    default=120,
    help="Control-step interval for inference progress logs. 0 disables periodic logs.",
)
parser.add_argument("--task", type=str, default="Robot1 picks cube -> handover to Robot2 -> Robot2 places in bin")
parser.add_argument("--debug-actions", action="store_true", default=False, help="Print first policy actions.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaacsim.core.utils.rotations import euler_angles_to_quat

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import TiledCameraCfg

try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.control_utils import predict_action

    LEROBOT_AVAILABLE = True
except ImportError as exc:
    PreTrainedConfig = None
    ACTPolicy = None
    make_pre_post_processors = None
    predict_action = None
    LEROBOT_AVAILABLE = False
    print(f"[ERROR] LeRobot policy dependencies are not available: {exc}")

from bimanual_config.so101_bimanual_handover_env_cfg import SO101BimanualHandoverEnvCfg


CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
EXPECTED_CONTROL_HZ = 120
ARM_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
ISAAC_SO101_JOINT_NAMES = [*ARM_JOINT_NAMES, "Jaw"]
JOINT_LIMIT_SATURATION_TOL = 1.0e-3

TOP_CAMERA_POS = (0.40, -0.06, 0.92)
TOP_CAMERA_FOCAL_LENGTH = 10.0
TOP_CAMERA_ROT_EULER_DEG = (0.0, 0.0, 0.0)


def get_ordered_joint_ids(robot, joint_names: Sequence[str] = ISAAC_SO101_JOINT_NAMES) -> list[int]:
    missing = [joint_name for joint_name in joint_names if joint_name not in robot.joint_names]
    if missing:
        raise ValueError(f"Robot missing joints: {missing}")
    return [robot.joint_names.index(joint_name) for joint_name in joint_names]


def clamp_joint_targets(robot, joint_ids: Sequence[int], joint_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    limits = robot.data.soft_joint_pos_limits[:, joint_ids, :]
    clamped = torch.maximum(torch.minimum(joint_targets, limits[..., 1]), limits[..., 0])
    saturated = torch.any(torch.abs(joint_targets - clamped) > JOINT_LIMIT_SATURATION_TOL, dim=-1)
    return clamped, saturated


def get_box_pos_w(box) -> torch.Tensor:
    if hasattr(box, "data") and hasattr(box.data, "root_pos_w"):
        return box.data.root_pos_w[:, :3].clone()
    pos, _ = box.get_world_poses()
    return pos[:, :3].clone()


def compute_success_xy_distance(cube_pos_w: torch.Tensor, box_pos_w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(cube_pos_w[:, :2] - box_pos_w[:, :2], dim=-1)


def compute_point_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a[:, :3] - b[:, :3], dim=-1)


def format_xyz(pos_w: torch.Tensor) -> str:
    values = pos_w[0].detach().cpu().tolist()
    return f"({values[0]:.3f},{values[1]:.3f},{values[2]:.3f})"


def camera_rgb_to_uint8(camera_data: torch.Tensor, env_idx: int = 0) -> np.ndarray:
    rgb_frame = camera_data[env_idx].detach().cpu().numpy().copy()
    if rgb_frame.shape[-1] > 3:
        rgb_frame = rgb_frame[..., :3]
    if rgb_frame.dtype != np.uint8:
        if rgb_frame.max() > 1.0:
            rgb_frame = rgb_frame / 255.0
        rgb_frame = np.clip(rgb_frame * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb_frame).copy()


def get_record_interval(control_dt: float, policy_fps: int) -> tuple[int, float]:
    control_hz = 1.0 / control_dt
    record_interval = int(round(control_hz / policy_fps))
    if record_interval <= 0:
        raise ValueError(f"Invalid policy interval {record_interval} from control_hz={control_hz:.3f}")
    if abs(control_hz - record_interval * policy_fps) > 1e-3:
        raise ValueError(f"Control rate {control_hz:.3f} Hz is not an integer multiple of policy_fps={policy_fps}.")
    if abs(control_hz - EXPECTED_CONTROL_HZ) > 1e-3:
        print(f"[Timing] WARNING: expected control_hz={EXPECTED_CONTROL_HZ}, got {control_hz:.3f}")
    return record_interval, control_hz


def add_top_camera_to_scene_cfg(scene_cfg) -> None:
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


def build_policy_observation(
    robot1,
    robot2,
    ordered_joint_ids_1: Sequence[int],
    ordered_joint_ids_2: Sequence[int],
    gripper_cam_r1_data: torch.Tensor,
    gripper_cam_r2_data: torch.Tensor,
    top_cam_data: torch.Tensor,
    env_idx: int = 0,
) -> dict[str, np.ndarray]:
    r1_state = robot1.data.joint_pos[env_idx, ordered_joint_ids_1].detach().cpu().numpy().astype(np.float32, copy=True)
    r2_state = robot2.data.joint_pos[env_idx, ordered_joint_ids_2].detach().cpu().numpy().astype(np.float32, copy=True)
    return {
        "observation.state": np.concatenate([r1_state, r2_state]).astype(np.float32, copy=False),
        "observation.images.gripper_r1": camera_rgb_to_uint8(gripper_cam_r1_data, env_idx),
        "observation.images.gripper_r2": camera_rgb_to_uint8(gripper_cam_r2_data, env_idx),
        "observation.images.top": camera_rgb_to_uint8(top_cam_data, env_idx),
    }


def load_act_policy(checkpoint_path: str, policy_device: str, n_action_steps: int):
    if not LEROBOT_AVAILABLE:
        raise RuntimeError("LeRobot is required for ACT inference.")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if getattr(config, "type", None) != "act":
        raise ValueError(f"Expected ACT checkpoint, got type={getattr(config, 'type', None)!r}")
    if n_action_steps <= 0 or n_action_steps > config.chunk_size:
        raise ValueError(f"--n-action-steps must be in [1, {config.chunk_size}], got {n_action_steps}")

    config.device = policy_device
    config.n_action_steps = n_action_steps
    policy = ACTPolicy.from_pretrained(checkpoint, config=config)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": policy_device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    return policy, preprocessor, postprocessor, torch.device(policy_device)


def resolve_policy_device(policy_device: str) -> str:
    if policy_device != "auto":
        return policy_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("[DIAG] main() entered", flush=True)
    if args_cli.success_hold_steps < 0:
        raise ValueError("--success-hold-steps must be non-negative.")
    if not args_cli.checkpoint:
        raise ValueError("Provide --checkpoint or set SO101_BIMANUAL_ACT_CHECKPOINT.")
    env_cfg = SO101BimanualHandoverEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    if env_cfg.scene.num_envs != 1:
        raise RuntimeError("ACT inference script currently supports exactly one Isaac Lab environment.")
    env_cfg.num_rerenders_on_reset = max(env_cfg.num_rerenders_on_reset, 20)
    add_top_camera_to_scene_cfg(env_cfg.scene)

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    policy_interval, control_hz = get_record_interval(control_dt, args_cli.policy_fps)

    resolved_policy_device = resolve_policy_device(args_cli.policy_device)
    print("[DIAG] loading ACT policy", flush=True)
    policy, preprocessor, postprocessor, policy_device = load_act_policy(
        args_cli.checkpoint, resolved_policy_device, args_cli.n_action_steps
    )
    print("[DIAG] ACT policy loaded", flush=True)

    print("[DIAG] creating ManagerBasedEnv", flush=True)
    env = ManagerBasedEnv(cfg=env_cfg)
    print("[DIAG] ManagerBasedEnv created", flush=True)
    env.reset()
    print("[DIAG] env.reset() completed", flush=True)

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

    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), dtype=torch.float32, device=env.device)
    current_joint_target_1 = robot1.data.joint_pos[:, ordered_jids_1].clone()
    current_joint_target_2 = robot2.data.joint_pos[:, ordered_jids_2].clone()
    actions[:] = torch.cat([current_joint_target_1, current_joint_target_2], dim=-1)

    print(f"\n{'=' * 60}", flush=True)
    print("Bi-Manual SO101 ACT Inference")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Policy device: {resolved_policy_device} (requested={args_cli.policy_device})")
    print(f"Control rate: {control_hz:.1f} Hz")
    print(f"Policy rate: {args_cli.policy_fps} Hz (interval={policy_interval} control steps)")
    print(f"n_action_steps: {args_cli.n_action_steps}")
    print(f"Max episode steps: {args_cli.max_episode_steps}")
    print(f"Success hold steps: {args_cli.success_hold_steps}")
    print(f"Action space: {env.action_manager.total_action_dim}D (Robot1[6] + Robot2[6])")
    print(f"Joint order per robot: {ISAAC_SO101_JOINT_NAMES}")
    print(f"{'=' * 60}\n", flush=True)

    saved_successes = 0
    attempted_episodes = 0
    episode_step = 0
    total_control_step = 0
    policy_query_count = 0
    saturation_count = 0
    success_hold_count = 0
    last_action_wall_time = 0.0
    loop_exit_reason = "not_entered"
    initial_cube_z = cube.data.root_pos_w[:, 2].clone()
    max_cube_z = initial_cube_z.clone()
    min_r1_cube_dist = torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device)
    min_r2_cube_dist = torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device)
    min_cube_bin_xy = torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device)
    milestone_flags = {
        "r1_jaw_opened": False,
        "r2_jaw_opened": False,
        "r1_near_cube": False,
        "r1_jaw_closing": False,
        "cube_lifted": False,
        "r2_near_cube": False,
        "r2_jaw_closing": False,
        "cube_near_bin": False,
        "success_zone": False,
    }

    def reset_episode_state() -> None:
        nonlocal episode_step, policy_query_count, saturation_count, success_hold_count
        nonlocal current_joint_target_1, current_joint_target_2, actions
        nonlocal initial_cube_z, max_cube_z, min_r1_cube_dist, min_r2_cube_dist, min_cube_bin_xy, milestone_flags
        env.reset()
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        current_joint_target_1 = robot1.data.joint_pos[:, ordered_jids_1].clone()
        current_joint_target_2 = robot2.data.joint_pos[:, ordered_jids_2].clone()
        actions[:] = torch.cat([current_joint_target_1, current_joint_target_2], dim=-1)
        episode_step = 0
        policy_query_count = 0
        saturation_count = 0
        success_hold_count = 0
        initial_cube_z = cube.data.root_pos_w[:, 2].clone()
        max_cube_z = initial_cube_z.clone()
        min_r1_cube_dist = torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device)
        min_r2_cube_dist = torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device)
        min_cube_bin_xy = torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device)
        milestone_flags = {key: False for key in milestone_flags}

    def collect_episode_metrics() -> dict[str, torch.Tensor]:
        cube_pos_w = cube.data.root_pos_w[:, :3].clone()
        box_pos_w = get_box_pos_w(box)
        ee_pos_w_1 = robot1.data.body_pos_w[:, ee_idx_1, :].clone()
        ee_pos_w_2 = robot2.data.body_pos_w[:, ee_idx_2, :].clone()
        cube_bin_xy = compute_success_xy_distance(cube_pos_w, box_pos_w)
        r1_cube_dist = compute_point_distance(ee_pos_w_1, cube_pos_w)
        r2_cube_dist = compute_point_distance(ee_pos_w_2, cube_pos_w)
        r2_bin_dist = compute_point_distance(ee_pos_w_2, box_pos_w)
        return {
            "cube_pos_w": cube_pos_w,
            "box_pos_w": box_pos_w,
            "ee_pos_w_1": ee_pos_w_1,
            "ee_pos_w_2": ee_pos_w_2,
            "cube_bin_xy": cube_bin_xy,
            "r1_cube_dist": r1_cube_dist,
            "r2_cube_dist": r2_cube_dist,
            "r2_bin_dist": r2_bin_dist,
        }

    def update_episode_metrics(metrics: dict[str, torch.Tensor]) -> None:
        nonlocal max_cube_z, min_r1_cube_dist, min_r2_cube_dist, min_cube_bin_xy
        max_cube_z = torch.maximum(max_cube_z, metrics["cube_pos_w"][:, 2])
        min_r1_cube_dist = torch.minimum(min_r1_cube_dist, metrics["r1_cube_dist"])
        min_r2_cube_dist = torch.minimum(min_r2_cube_dist, metrics["r2_cube_dist"])
        min_cube_bin_xy = torch.minimum(min_cube_bin_xy, metrics["cube_bin_xy"])

    def print_progress(metrics: dict[str, torch.Tensor], sat_this_step: bool = False) -> None:
        r1_jaw = robot1.data.joint_pos[:, ordered_jids_1[-1]][0].item()
        r2_jaw = robot2.data.joint_pos[:, ordered_jids_2[-1]][0].item()
        r1_tgt_jaw = current_joint_target_1[:, -1][0].item()
        r2_tgt_jaw = current_joint_target_2[:, -1][0].item()
        cube_z = metrics["cube_pos_w"][0, 2].item()
        lifted = cube_z - initial_cube_z[0].item()
        print(
            f"[Progress] step={episode_step:04d} t={episode_step / control_hz:05.2f}s "
            f"query={policy_query_count:03d} inf_ms={last_action_wall_time * 1000.0:05.1f} "
            f"cube={format_xyz(metrics['cube_pos_w'])} dz={lifted:+.3f} "
            f"bin={format_xyz(metrics['box_pos_w'])} cube_bin_xy={metrics['cube_bin_xy'][0].item():.3f} "
            f"r1_cube={metrics['r1_cube_dist'][0].item():.3f} r2_cube={metrics['r2_cube_dist'][0].item():.3f} "
            f"r2_bin={metrics['r2_bin_dist'][0].item():.3f} "
            f"jaw=({r1_jaw:.3f}->{r1_tgt_jaw:.3f},{r2_jaw:.3f}->{r2_tgt_jaw:.3f}) "
            f"success_hold={success_hold_count}/{args_cli.success_hold_steps} "
            f"sat_total={saturation_count} sat_now={sat_this_step}",
            flush=True,
        )

    def finish_episode(success: bool, reason: str, metrics: dict[str, torch.Tensor]) -> None:
        nonlocal attempted_episodes, saved_successes
        attempted_episodes += 1
        saved_successes += int(success)
        success_distance = metrics["cube_bin_xy"]
        print(
            f"[Episode {attempted_episodes}] "
            f"success={success} reason={reason} distance_xy={success_distance[0].item():.4f} "
            f"min_distance_xy={min_cube_bin_xy[0].item():.4f} "
            f"max_cube_z={max_cube_z[0].item():.4f} "
            f"min_r1_cube={min_r1_cube_dist[0].item():.4f} "
            f"min_r2_cube={min_r2_cube_dist[0].item():.4f} "
            f"policy_queries={policy_query_count} saturations={saturation_count} "
            f"steps={episode_step}",
            flush=True,
        )

    def maybe_print_milestones(metrics: dict[str, torch.Tensor]) -> None:
        nonlocal milestone_flags
        cube_lift = metrics["cube_pos_w"][0, 2].item() - initial_cube_z[0].item()
        r1_jaw = robot1.data.joint_pos[:, ordered_jids_1[-1]][0].item()
        r2_jaw = robot2.data.joint_pos[:, ordered_jids_2[-1]][0].item()
        if r1_jaw > 0.75:
            milestone_flags["r1_jaw_opened"] = True
        if r2_jaw > 0.75:
            milestone_flags["r2_jaw_opened"] = True
        checks = [
            ("r1_near_cube", metrics["r1_cube_dist"][0].item() < 0.055, "R1 gripper is near cube"),
            (
                "r1_jaw_closing",
                milestone_flags["r1_jaw_opened"] and r1_jaw < 0.45,
                "R1 jaw is closing/closed after opening",
            ),
            ("cube_lifted", cube_lift > 0.035, "Cube is lifted"),
            ("r2_near_cube", metrics["r2_cube_dist"][0].item() < 0.070, "R2 gripper is near cube"),
            (
                "r2_jaw_closing",
                milestone_flags["r2_jaw_opened"] and r2_jaw < 0.45,
                "R2 jaw is closing/closed after opening",
            ),
            ("cube_near_bin", metrics["cube_bin_xy"][0].item() < 0.100, "Cube is near bin"),
            ("success_zone", metrics["cube_bin_xy"][0].item() < args_cli.success_threshold, "Cube is in success zone"),
        ]
        for key, passed, message in checks:
            if passed and not milestone_flags[key]:
                milestone_flags[key] = True
                print(
                    f"[Milestone] step={episode_step:04d} t={episode_step / control_hz:05.2f}s "
                    f"{message}: cube={format_xyz(metrics['cube_pos_w'])} "
                    f"cube_bin_xy={metrics['cube_bin_xy'][0].item():.3f} "
                    f"r1_cube={metrics['r1_cube_dist'][0].item():.3f} "
                    f"r2_cube={metrics['r2_cube_dist'][0].item():.3f} "
                    f"jaw=({r1_jaw:.3f},{r2_jaw:.3f})",
                    flush=True,
                )

    try:
        print("[DIAG] entering inference loop", flush=True)
        while True:
            if not simulation_app.is_running():
                loop_exit_reason = "simulation_app_not_running"
                print("[DIAG] main loop exit: simulation_app.is_running() returned False", flush=True)
                break
            if args_cli.num_episodes > 0 and attempted_episodes >= args_cli.num_episodes:
                loop_exit_reason = "target_reached"
                print(f"\n[Done] Completed {attempted_episodes} episodes. successes={saved_successes}")
                break

            with torch.inference_mode():
                if episode_step >= args_cli.max_episode_steps:
                    metrics = collect_episode_metrics()
                    update_episode_metrics(metrics)
                    success = bool((metrics["cube_bin_xy"] < args_cli.success_threshold)[0].item())
                    finish_episode(success, "max_steps", metrics)
                    reset_episode_state()
                    continue

                if episode_step % policy_interval == 0:
                    env.sim.render()
                    observation = build_policy_observation(
                        robot1,
                        robot2,
                        ordered_jids_1,
                        ordered_jids_2,
                        gripper_cam_r1.data.output["rgb"],
                        gripper_cam_r2.data.output["rgb"],
                        top_camera.data.output["rgb"],
                    )
                    action_start_time = time.perf_counter()
                    action_cpu = predict_action(
                        observation=observation,
                        policy=policy,
                        device=policy_device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=args_cli.task,
                        robot_type="so101_bimanual",
                    )
                    last_action_wall_time = time.perf_counter() - action_start_time
                    policy_query_count += 1

                    action = torch.as_tensor(action_cpu, dtype=torch.float32, device=env.device).reshape(1, -1)
                    if action.shape[-1] != env.action_manager.total_action_dim:
                        raise RuntimeError(
                            f"Policy action shape {tuple(action.shape)} does not match env action dim "
                            f"{env.action_manager.total_action_dim}."
                        )
                    current_joint_target_1, sat1 = clamp_joint_targets(robot1, ordered_jids_1, action[:, :6])
                    current_joint_target_2, sat2 = clamp_joint_targets(robot2, ordered_jids_2, action[:, 6:])
                    sat_this_step = bool(sat1[0].item() or sat2[0].item())
                    saturation_count += int(sat1[0].item()) + int(sat2[0].item())
                    actions[:] = torch.cat([current_joint_target_1, current_joint_target_2], dim=-1)

                    if args_cli.debug_actions and policy_query_count <= 5:
                        print(
                            f"[Policy] ep_step={episode_step} query={policy_query_count} "
                            f"inference_ms={last_action_wall_time * 1000.0:.2f} "
                            f"action={[round(x, 4) for x in actions[0].detach().cpu().tolist()]}",
                            flush=True,
                        )
                else:
                    sat_this_step = False

                metrics = collect_episode_metrics()
                update_episode_metrics(metrics)
                maybe_print_milestones(metrics)
                current_success = bool((metrics["cube_bin_xy"] < args_cli.success_threshold)[0].item())
                if current_success:
                    success_hold_count += 1
                else:
                    success_hold_count = 0
                if (
                    args_cli.progress_log_interval > 0
                    and episode_step % args_cli.progress_log_interval == 0
                ):
                    print_progress(metrics, sat_this_step=sat_this_step)

                if args_cli.success_hold_steps > 0 and success_hold_count >= args_cli.success_hold_steps:
                    finish_episode(True, f"success_hold:{success_hold_count}", metrics)
                    reset_episode_state()
                    continue

                _, _ = env.step(actions)
                total_control_step += 1
                episode_step += 1

    except KeyboardInterrupt:
        loop_exit_reason = "keyboard_interrupt"
        print("\n[INFO] Interrupted by user.", flush=True)
    finally:
        print(
            f"[Summary] exit={loop_exit_reason} attempts={attempted_episodes} "
            f"successes={saved_successes} total_control_steps={total_control_step}",
            flush=True,
        )
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", flush=True)
        raise
