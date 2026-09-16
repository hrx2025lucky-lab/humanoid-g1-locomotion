# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--compact-terrain",
    action="store_true",
    default=False,
    help="1x1 terrain grid, tight env_spacing, and (unless overridden) small patch size + border so the mesh "
    "is not a 8m cell with 20m apron. Combine with --terrain-patch-size / --terrain-border-width if needed.",
)
parser.add_argument("--terrain-rows", type=int, default=None, help="Override terrain generator rows for play.")
parser.add_argument("--terrain-cols", type=int, default=None, help="Override terrain generator cols for play.")
parser.add_argument(
    "--terrain-patch-size",
    nargs=2,
    type=float,
    default=None,
    metavar=("W", "L"),
    help="Sub-terrain patch size in meters (width, length). Warmup default is 8x8; large values plus border_width "
    "dominate the viewport. Example for local play: --terrain-patch-size 3 3",
)
parser.add_argument(
    "--terrain-border-width",
    type=float,
    default=None,
    help="TerrainGenerator border_width in meters. Warmup default 20 adds a huge flat apron (bad for 1-robot play). "
    "Try 1.0--3.0 with a small --terrain-patch-size.",
)
parser.add_argument(
    "--terrain-play-row",
    type=int,
    default=None,
    help="Pin all env origins to this terrain grid row (0 .. rows-1). Use with --terrain-play-col. "
    "Matches TerrainImporter.terrain_origins[row, col] (row = difficulty / level axis).",
)
parser.add_argument(
    "--terrain-play-col",
    type=int,
    default=None,
    help="Pin all env origins to this terrain grid column (0 .. cols-1). Use with --terrain-play-row.",
)
parser.add_argument(
    "--terrain-play-index",
    type=int,
    default=None,
    help="Pin all env origins to one patch: row = index // cols, col = index %% cols (after any row/col overrides). "
    "Alternative to --terrain-play-row/--terrain-play-col.",
)
parser.add_argument("--cmd-lin-x", type=float, default=None, help="Override commanded linear velocity x (m/s).")
parser.add_argument("--cmd-lin-y", type=float, default=None, help="Override commanded linear velocity y (m/s).")
parser.add_argument("--cmd-yaw-rate", type=float, default=None, help="Override commanded yaw rate (rad/s).")
parser.add_argument(
    "--cmd-raw-rate",
    type=float,
    default=None,
    help="Alias of --cmd-yaw-rate (kept for convenience).",
)
parser.add_argument(
    "--footstep-markers",
    action="store_true",
    default=False,
    help="Visualize persistent touchdown foot placement markers during evaluation.",
)

parser.add_argument(
    "--footstep-history",
    type=int,
    default=120,
    help="Per-foot marker history length to keep in the simulator.",
)

parser.add_argument(
    "--footstep-scale",
    type=float,
    default=0.05,
    help="Scale of each touchdown marker frame.",
)
parser.add_argument(
    "--footstep-force-threshold",
    type=float,
    default=2.0,
    help="Force magnitude threshold (N) for declaring foot contact touchdown.",
)
parser.add_argument(
    "--height-scan-print",
    action="store_true",
    default=False,
    help="Print height-scan ray heights (env 0) to the terminal at a fixed rate (matches policy height_scan).",
)
parser.add_argument(
    "--height-scan-print-hz",
    type=float,
    default=5.0,
    help="Print rate (Hz) for --height-scan-print.",
)
parser.add_argument(
    "--com-prediction-markers",
    action="store_true",
    default=False,
    help="Visualize future COM XY prediction markers using command integration.",
)
parser.add_argument(
    "--com-prediction-horizon-steps",
    type=int,
    default=15,
    help="Total rollout horizon in steps for COM prediction.",
)
parser.add_argument(
    "--com-prediction-marker-interval",
    type=int,
    default=5,
    help="Step interval for showing COM prediction markers.",
)
parser.add_argument(
    "--com-prediction-scale",
    type=float,
    default=0.06,
    help="Marker scale for COM prediction visualization.",
)
parser.add_argument(
    "--height-scan-vis",
    action="store_true",
    default=False,
    help="Enable RayCaster height_scanner debug visualization (rays / hit debug) when the scene defines height_scanner.",
)
parser.add_argument(
    "--hoi-height-scan-vis",
    action="store_true",
    default=False,
    help="Enable analytical HOI height-scan hit marker visualization (env 0) in viewer.",
)
parser.add_argument(
    "--hoi-height-scan-print",
    action="store_true",
    default=False,
    help="Print analytical HOI height-scan values (env 0) to terminal.",
)
parser.add_argument(
    "--hoi-height-scan-print-hz",
    type=float,
    default=5.0,
    help="Print rate (Hz) for --hoi-height-scan-print.",
)
# HOI terrain mimic: optional play-time overrides (must run before env cfg is instantiated).
parser.add_argument(
    "--hoi-play-no-curriculum",
    action="store_true",
    default=False,
    help="Set HOI_CURRICULUM_ENABLE=0 before loading play env cfg (no curriculum level-up during play).",
)
parser.add_argument(
    "--hoi-play-no-dr",
    action="store_true",
    default=False,
    help="Set HOI_CURRICULUM_DR_ENABLE=0 before loading play env cfg (freeze startup DR in tracking_env_cfg).",
)
parser.add_argument(
    "--hoi-play-no-obs-noise",
    action="store_true",
    default=False,
    help="Disable policy observation corruption (enable_corruption=False) after env cfg load. "
    "If the policy was trained with corruption on, this shifts the observation distribution and can look worse.",
)
parser.add_argument(
    "--hoi-play-exact-motion-joints",
    action="store_true",
    default=False,
    help="After env cfg load, set motion command joint_position_range to (0,0). "
    "HOI startup DR freeze does not remove MotionCommand joint resample noise; use this for exact reference pose at reset.",
)
parser.add_argument(
    "--hoi-play-deterministic",
    action="store_true",
    default=False,
    help="Shorthand for --hoi-play-no-curriculum --hoi-play-no-dr --hoi-play-no-obs-noise. "
    "For train-matched observations, prefer only the first two flags (keep obs corruption on).",
)
parser.add_argument(
    "--hoi-play-curriculum-level",
    type=int,
    default=None,
    metavar="N",
    help="Pin HOI curriculum level at play start (0..3): sets push strength/interval, DR ranges, and "
    "ee termination threshold for that level. Play mode does not level up during long rollouts, so use "
    "this to test large-DR / max-push behavior (e.g. --hoi-play-curriculum-level 3).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# 上游把这个模块从 isaaclab.utils 挪到了 isaaclab_rl.utils，
# 旧路径会让 play.py 直接 ModuleNotFoundError 起不来。
# train.py 不引用它，所以训练正常、只有回放路径会崩。
try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from helpers import (
    ComPredictionMarkerTracker,
    FootstepMarkerTracker,
    HeightScanLivePrint,
    HoiHeightScanLivePrint,
    apply_fixed_velocity_command,
    convert_legacy_policy_cfg,
)

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _pin_play_terrain_patch(scene_env, row: int, col: int) -> None:
    """Place every env origin on ``terrain_origins[row, col]`` (generator terrains only)."""
    terrain = scene_env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("Pinned terrain patch requested but the scene has no generator terrain origins.")
    nrows, ncols = terrain.terrain_origins.shape[0], terrain.terrain_origins.shape[1]
    if not (0 <= row < nrows and 0 <= col < ncols):
        raise ValueError(f"Terrain patch ({row}, {col}) out of bounds for grid {nrows}x{ncols}.")
    origin = terrain.terrain_origins[row, col]
    n = scene_env.num_envs
    terrain.env_origins[:n] = origin.unsqueeze(0).expand(n, -1)
    if hasattr(terrain, "terrain_levels"):
        terrain.terrain_levels[:n] = row
    if hasattr(terrain, "terrain_types"):
        terrain.terrain_types[:n] = col
    print(f"[INFO] Pinned env origins to terrain patch row={row}, col={col} (grid {nrows}x{ncols}).")


def _hoi_play_apply_env_and_obs_flags(args: argparse.Namespace) -> bool:
    """Apply HOI play overrides. Returns True if policy obs corruption should be disabled.

    Curriculum/DR use os.environ read during env cfg ``__post_init__`` / curriculum state init;
    those variables must be set before ``parse_env_cfg`` instantiates the config class.
    CLI flags override any pre-existing env values for reproducible one-liners.
    """
    use_deterministic = bool(getattr(args, "hoi_play_deterministic", False))
    no_curriculum = use_deterministic or bool(getattr(args, "hoi_play_no_curriculum", False))
    no_dr = use_deterministic or bool(getattr(args, "hoi_play_no_dr", False))
    no_obs_noise = use_deterministic or bool(getattr(args, "hoi_play_no_obs_noise", False))

    if no_curriculum:
        os.environ["HOI_CURRICULUM_ENABLE"] = "0"
        print("[INFO] HOI play: HOI_CURRICULUM_ENABLE=0 (curriculum progression off)")
    if no_dr:
        os.environ["HOI_CURRICULUM_DR_ENABLE"] = "0"
        print("[INFO] HOI play: HOI_CURRICULUM_DR_ENABLE=0 (startup DR frozen in env cfg)")
    if use_deterministic:
        print(
            "[INFO] HOI play: --hoi-play-deterministic (no curriculum, no DR, no policy obs noise; "
            "last item can mismatch training if corruption was enabled)"
        )
    if bool(getattr(args, "hoi_height_scan_vis", False)):
        os.environ["HOI_HEIGHT_SCAN_DEBUG_VIS"] = "1"
        print("[INFO] HOI play: HOI_HEIGHT_SCAN_DEBUG_VIS=1 (analytical scandot markers on)")

    return no_obs_noise


def main():
    """Play with RSL-RL agent."""
    disable_policy_obs_noise = _hoi_play_apply_env_and_obs_flags(args_cli)
    play_terrain_patch: tuple[int, int] | None = None
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if disable_policy_obs_noise and hasattr(env_cfg, "observations"):
        policy_group = getattr(env_cfg.observations, "policy", None)
        if policy_group is not None and hasattr(policy_group, "enable_corruption"):
            policy_group.enable_corruption = False
            print("[INFO] HOI play: policy observation corruption disabled (enable_corruption=False)")
    if getattr(args_cli, "hoi_play_exact_motion_joints", False):
        motion_cmd = getattr(getattr(env_cfg, "commands", None), "motion", None)
        if motion_cmd is not None and hasattr(motion_cmd, "joint_position_range"):
            motion_cmd.joint_position_range = (0.0, 0.0)
            print("[INFO] HOI play: motion joint_position_range set to (0,0) (--hoi-play-exact-motion-joints)")
    apply_fixed_velocity_command(env_cfg, args_cli)
    if args_cli.height_scan_vis and hasattr(env_cfg.scene, "height_scanner"):
        env_cfg.scene.height_scanner.debug_vis = True
        print("[INFO] Height scan raycaster debug visualization enabled.")
    if hasattr(env_cfg.scene, "terrain") and getattr(env_cfg.scene.terrain, "terrain_generator", None) is not None:
        if args_cli.compact_terrain:
            env_cfg.scene.terrain.terrain_generator.num_rows = 1
            env_cfg.scene.terrain.terrain_generator.num_cols = 1
            if hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
                env_cfg.scene.terrain.max_init_terrain_level = 0
            # Keep env compact in the viewport for one-robot inspection.
            if hasattr(env_cfg.scene, "env_spacing"):
                env_cfg.scene.env_spacing = 1.5
            tg_compact = env_cfg.scene.terrain.terrain_generator
            # rows/cols=1 still leaves a huge mesh if size=(8,8) and border_width=20 (training defaults).
            if args_cli.terrain_patch_size is None:
                tg_compact.size = (4.0, 4.0)
            if args_cli.terrain_border_width is None:
                tg_compact.border_width = 2.0
            print(
                "[INFO] Compact terrain: rows=1, cols=1, level=0, "
                f"patch_m={tg_compact.size}, border_width_m={tg_compact.border_width}"
            )
        if args_cli.terrain_rows is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = max(1, int(args_cli.terrain_rows))
        if args_cli.terrain_cols is not None:
            env_cfg.scene.terrain.terrain_generator.num_cols = max(1, int(args_cli.terrain_cols))
        if args_cli.terrain_rows is not None or args_cli.terrain_cols is not None:
            print(
                "[INFO] Terrain override:"
                f" rows={env_cfg.scene.terrain.terrain_generator.num_rows},"
                f" cols={env_cfg.scene.terrain.terrain_generator.num_cols}"
            )

        tg = env_cfg.scene.terrain.terrain_generator
        if args_cli.terrain_patch_size is not None:
            tg.size = (float(args_cli.terrain_patch_size[0]), float(args_cli.terrain_patch_size[1]))
            print(f"[INFO] Terrain patch size override: {tg.size} m")
        if args_cli.terrain_border_width is not None:
            tg.border_width = float(args_cli.terrain_border_width)
            print(f"[INFO] Terrain border_width override: {tg.border_width} m")

        terrain_patch_pinned = False
        play_row, play_col = None, None
        if args_cli.terrain_play_index is not None:
            idx = int(args_cli.terrain_play_index)
            nr, nc = int(tg.num_rows), int(tg.num_cols)
            if not (0 <= idx < nr * nc):
                raise ValueError(f"--terrain-play-index {idx} out of range for {nr}x{nc} grid (need 0..{nr * nc - 1}).")
            play_row, play_col = idx // nc, idx % nc
            terrain_patch_pinned = True
        elif args_cli.terrain_play_row is not None or args_cli.terrain_play_col is not None:
            if args_cli.terrain_play_row is None or args_cli.terrain_play_col is None:
                raise ValueError("Use both --terrain-play-row and --terrain-play-col, or pass --terrain-play-index.")
            play_row, play_col = int(args_cli.terrain_play_row), int(args_cli.terrain_play_col)
            terrain_patch_pinned = True

        if terrain_patch_pinned:
            if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
                env_cfg.curriculum.terrain_levels = None
                print("[INFO] Terrain level curriculum disabled (pinned patch would otherwise move with success).")
            tg.curriculum = False
            play_terrain_patch = (play_row, play_col)

    if (
        args_cli.terrain_play_index is not None
        or args_cli.terrain_play_row is not None
        or args_cli.terrain_play_col is not None
    ) and play_terrain_patch is None:
        raise ValueError(
            "Terrain patch pinning requires a task whose scene uses terrain_type='generator' with terrain_generator."
        )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    scene_env = env.unwrapped
    if play_terrain_patch is not None:
        _pin_play_terrain_patch(scene_env, play_terrain_patch[0], play_terrain_patch[1])
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if getattr(args_cli, "hoi_play_curriculum_level", None) is not None:
        from unitree_rl_lab.tasks.mimic.robots.g1_29dof.hoi_mimic_terrain import curriculum_HOI as hoi_curriculum

        level = int(args_cli.hoi_play_curriculum_level)
        state = hoi_curriculum._get_state(scene_env)
        max_level = int(state["num_levels"]) - 1
        level = max(0, min(level, max_level))
        state["level"] = level
        hoi_curriculum._apply_level(scene_env, level, resample_dr=True)
        print(
            f"[INFO] HOI play: pinned curriculum level to {level} "
            f"(push + DR + ee threshold for this level; DR resampled)"
        )

    footstep_markers = None
    if args_cli.footstep_markers:
        footstep_markers = FootstepMarkerTracker(
            scene_env,
            history=args_cli.footstep_history,
            marker_scale=args_cli.footstep_scale,
            contact_force_threshold=args_cli.footstep_force_threshold,
        )

    height_scan_print = None
    if args_cli.height_scan_print:
        if hasattr(scene_env.scene, "sensors") and "height_scanner" in scene_env.scene.sensors:
            height_scan_print = HeightScanLivePrint(scene_env, print_hz=args_cli.height_scan_print_hz)
        else:
            print("[WARN] --height-scan-print ignored: scene has no 'height_scanner' sensor.")

    hoi_height_scan_print = None
    if args_cli.hoi_height_scan_print:
        try:
            hoi_height_scan_print = HoiHeightScanLivePrint(scene_env, print_hz=args_cli.hoi_height_scan_print_hz)
        except Exception as exc:
            print(f"[WARN] --hoi-height-scan-print ignored: {exc}")

    com_prediction_markers = None
    if args_cli.com_prediction_markers:
        com_prediction_markers = ComPredictionMarkerTracker(
            scene_env,
            horizon_steps=args_cli.com_prediction_horizon_steps,
            marker_interval=args_cli.com_prediction_marker_interval,
            marker_scale=args_cli.com_prediction_scale,
        )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner_cfg = convert_legacy_policy_cfg(agent_cfg.to_dict())
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Extract the neural network module with compatibility across rsl-rl versions.
    if hasattr(runner.alg, "policy"):
        policy_nn = runner.alg.policy
    elif hasattr(runner.alg, "actor_critic"):
        policy_nn = runner.alg.actor_critic
    elif hasattr(runner.alg, "get_policy"):
        policy_nn = runner.alg.get_policy()
    elif hasattr(runner.alg, "actor"):
        policy_nn = runner.alg.actor
    else:
        raise AttributeError("Unable to resolve policy module from runner.alg")

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        # Newer rsl-rl API
        runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
    except AttributeError:
        # Backward compatibility fallback
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            if footstep_markers is not None:
                footstep_markers.update()
            if height_scan_print is not None:
                height_scan_print.update()
            if hoi_height_scan_print is not None:
                hoi_height_scan_print.update()
            if com_prediction_markers is not None:
                com_prediction_markers.update()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
