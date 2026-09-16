"""Convert HOI ``qpos`` trajectories into mimic training ``.npz``.

This script replays HOI robot states on the same articulation used by mimic training
(``UNITREE_G1_29DOF_MIMIC_CFG``), reads per-body states from Isaac Lab, and writes the
canonical mimic schema consumed by ``MotionLoader``.

Examples:
    python scripts/mimic/hoi_to_mimic_npz.py -f motion_dataset/HOI/robot-terrain/climb_15_z_scale_1.0.npz \\
        --output-dir /home/sustech/unitree_project/logs/hoi_mimic_data

    python scripts/mimic/hoi_to_mimic_npz.py --task terrain --filter climb_15 \\
        --output-dir /home/sustech/unitree_project/logs/hoi_mimic_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import os

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert HOI motion to mimic npz format.")
parser.add_argument(
    "--file",
    "-f",
    type=str,
    default=None,
    help="Path to one HOI .npz file (robot-object or robot-terrain).",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    choices=["object", "terrain"],
    help="Batch mode over HOI subset.",
)
parser.add_argument("--hoi-root", type=str, default=None, help="HOI dataset root. Inferred when possible.")
parser.add_argument("--filter", type=str, default="", help="Batch mode filename substring filter.")
parser.add_argument("--output-dir", type=str, default=None, help="Where converted files are written.")
parser.add_argument("--suffix", type=str, default="_mimic", help="Suffix before .npz, default: _mimic")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.file is None and args_cli.task is None:
    parser.error("Provide either --file/-f or --task {object|terrain}.")
if args_cli.file is not None and args_cli.task is not None:
    parser.error("Use either --file or --task, not both.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG
from unitree_rl_lab.utils.hoi.motion_loader import load_hoi_npz
from unitree_rl_lab.utils.hoi.path_resolver import list_npz_for_task, resolve_assets


@configclass
class ConvertSceneCfg(InteractiveSceneCfg):
    """Scene for kinematic replay and body-state extraction."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=500.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = UNITREE_G1_29DOF_MIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _default_hoi_root() -> Path:
    if args_cli.hoi_root:
        return Path(args_cli.hoi_root).resolve()
    if args_cli.file:
        p = Path(args_cli.file).resolve()
        if p.parent.name in ("robot-object", "robot-terrain", "robot-object-terrain"):
            return p.parent.parent
    here = Path(__file__).resolve().parent
    guess = (here / "../../datasets").resolve()
    if guess.is_dir():
        return guess
    return Path.cwd()


def _iter_npz_paths() -> list[Path]:
    root = _default_hoi_root()
    if args_cli.file:
        return [Path(args_cli.file).resolve()]
    assert args_cli.task in ("object", "terrain")
    return list_npz_for_task(root, args_cli.task, args_cli.filter)


def _output_path(src: Path) -> Path:
    if args_cli.output_dir:
        out_dir = Path(args_cli.output_dir).resolve()
    else:
        out_dir = src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{src.stem}{args_cli.suffix}.npz"


def _write_metadata(out_npz: Path, src_npz: Path, ap_subset: str, ap_terrain_urdf: str | None, motion) -> Path:
    if ap_subset == "robot-object":
        meta_path = out_npz.with_suffix(".box.json")
        payload = {
            "source_npz": str(src_npz),
            "object_init_pos": motion.object_pos[0].detach().cpu().tolist(),
            "object_init_quat": motion.object_quat[0].detach().cpu().tolist(),
        }
    elif ap_subset == "robot-terrain":
        meta_path = out_npz.with_suffix(".terrain.json")
        payload = {
            "source_npz": str(src_npz),
            "terrain_urdf": ap_terrain_urdf,
            # HOI terrain is world-aligned by construction; we randomize around this.
            "terrain_init_pos": [0.0, 0.0, 0.0],
            "terrain_init_quat": [1.0, 0.0, 0.0, 0.0],
        }
    else:
        raise ValueError(f"Unsupported subset for metadata writing: {ap_subset}")
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def _convert_one(npz_path: Path) -> tuple[Path, Path]:
    ap = resolve_assets(npz_path, _default_hoi_root())
    if ap.subset not in ("robot-object", "robot-terrain"):
        raise ValueError(
            f"Only robot-object and robot-terrain are supported in this converter. Got subset={ap.subset} for {npz_path}"
        )

    motion = load_hoi_npz(str(npz_path), device=args_cli.device)
    if ap.subset == "robot-object" and not motion.has_object:
        raise ValueError(f"{npz_path} does not include object channels; expected robot-object subset.")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / max(float(motion.fps), 1.0)
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ConvertSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot: Articulation = scene["robot"]
    joint_names = list(UNITREE_G1_29DOF_MIMIC_CFG.joint_sdk_names)
    joint_indices, matched = robot.find_joints(joint_names, preserve_order=True)
    if len(joint_indices) != len(joint_names):
        missing = sorted(set(joint_names) - set(matched))
        raise ValueError(f"Joint names missing on mimic robot: {missing}")
    joint_indices_t = torch.tensor(joint_indices, dtype=torch.long, device=sim.device)

    log = {
        "fps": np.asarray([float(motion.fps)], dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    joint_pos_seq = motion.joint_pos.to(device=sim.device)
    joint_vel_seq = motion.joint_vel.to(device=sim.device)
    root_pos_seq = motion.root_pos.to(device=sim.device)
    root_quat_seq = motion.root_quat.to(device=sim.device)
    root_lin_vel_seq = motion.root_lin_vel.to(device=sim.device)
    root_ang_vel_seq = motion.root_ang_vel.to(device=sim.device)

    for t in range(motion.num_frames):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos_seq[t].unsqueeze(0)
        root_state[:, 3:7] = root_quat_seq[t].unsqueeze(0)
        root_state[:, 7:10] = root_lin_vel_seq[t].unsqueeze(0)
        root_state[:, 10:13] = root_ang_vel_seq[t].unsqueeze(0)
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indices_t] = joint_pos_seq[t].unsqueeze(0)
        joint_vel[:, joint_indices_t] = joint_vel_seq[t].unsqueeze(0)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Kinematic replay for extraction: no physics stepping.
        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].detach().cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].detach().cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].detach().cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].detach().cpu().numpy().copy())

    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[key] = np.stack(log[key], axis=0).astype(np.float32)

    out_npz = _output_path(npz_path)
    np.savez(out_npz, **log)
    meta_path = _write_metadata(out_npz, npz_path, ap.subset, ap.terrain_urdf, motion)
    return out_npz, meta_path


def main() -> None:
    paths = _iter_npz_paths()
    if not paths:
        print("No HOI files matched.", file=sys.stderr)
        return

    for i, p in tqdm(enumerate(paths), total=len(paths), desc="Converting HOI to mimic npz"):
        print(f"[{i + 1}/{len(paths)}] converting: {p}")
        out_npz, meta_path = _convert_one(p)
        print(f"  -> mimic npz: {out_npz}")
        print(f"  -> meta file: {meta_path}")


if __name__ == "__main__":
    main()
    # simulation_app.close()
    os._exit(0)  # Force exit to avoid hanging Isaac Lab threads.
