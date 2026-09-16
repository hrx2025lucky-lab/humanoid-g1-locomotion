"""Isaac Lab viewer for OmniRetarget HOI (Humanoid-Object Interaction) ``.npz`` trajectories.

Single file:
    python scripts/mimic/view_hoi_npz.py -f /path/to/motion_dataset/HOI/robot-object/foo.npz

Batch (subset under ``--hoi-root``, optional filename filter, Enter advances):
    python scripts/mimic/view_hoi_npz.py --task object-terrain --filter scene_00 \\
        --hoi-root /path/to/motion_dataset/HOI
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay HOI OmniRetarget motion in Isaac Lab.")
parser.add_argument("--file", "-f", type=str, default=None, help="Path to a single HOI .npz file.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    choices=["object", "terrain", "object-terrain"],
    help="Batch mode: which subset under --hoi-root (requires --file omitted).",
)
parser.add_argument(
    "--hoi-root",
    type=str,
    default=None,
    help="Root of HOI dataset (contains models/, robot-object/, ...). Default: infer from -f or cwd.",
)
parser.add_argument("--filter", type=str, default="", help="Batch mode: substring filter on .npz basename.")
parser.add_argument(
    "--robot-source",
    type=str,
    default="urdf",
    choices=["urdf", "usd"],
    help="Robot asset: dataset URDF (default) or UNITREE_G1_29DOF USD (visual may differ from retargeting).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.file is None and args_cli.task is None:
    parser.error("Provide either --file/-f or --task {object|terrain|object-terrain}.")
if args_cli.file is not None and args_cli.task is not None:
    parser.error("Use either --file or --task, not both.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.utils.hoi.assets import make_object_cfg, make_robot_cfg, make_terrain_cfg
from unitree_rl_lab.utils.hoi.motion_loader import load_hoi_npz
from unitree_rl_lab.utils.hoi.path_resolver import list_npz_for_task, resolve_assets


def _try_remove_envs_prim() -> None:
    """Remove ``/World/envs`` so a new :class:`InteractiveScene` can be built (batch mode)."""
    try:
        import omni.usd  # type: ignore

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath("/World/envs")
        if prim.IsValid():
            stage.RemovePrim("/World/envs")
    except Exception:
        pass


def _default_hoi_root() -> Path:
    if args_cli.hoi_root:
        return Path(args_cli.hoi_root).resolve()
    if args_cli.file:
        p = Path(args_cli.file).resolve()
        if p.parent.name in ("robot-object", "robot-terrain", "robot-object-terrain"):
            return p.parent.parent
    # repo layout: unitree_rl_lab/scripts/mimic -> ../../../motion_dataset/HOI
    here = Path(__file__).resolve().parent
    guess = (here / "../../datasets").resolve()
    if guess.is_dir():
        return guess
    return Path.cwd()


def _iter_npz_paths() -> list[Path]:
    root = _default_hoi_root()
    if args_cli.file:
        return [Path(args_cli.file).resolve()]
    assert args_cli.task is not None
    return list_npz_for_task(root, args_cli.task, args_cli.filter)


@configclass
class HOISceneRobotObjectCfg(InteractiveSceneCfg):
    """Robot + manipulated object (largebox)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = MISSING
    hoi_object: AssetBaseCfg = MISSING


@configclass
class HOISceneRobotTerrainCfg(InteractiveSceneCfg):
    """Robot + static terrain."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = MISSING
    hoi_terrain: AssetBaseCfg = MISSING


@configclass
class HOISceneRobotObjectTerrainCfg(InteractiveSceneCfg):
    """Robot + chair (kinematic) + static terrain."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = MISSING
    hoi_object: AssetBaseCfg = MISSING
    hoi_terrain: AssetBaseCfg = MISSING


def _build_scene_cfg(ap: AssetPaths, use_usd: bool, asset_tag: str = "") -> InteractiveSceneCfg:
    suffix = f"_{asset_tag}" if asset_tag else ""
    robot_cfg = make_robot_cfg(ap.robot_urdf, use_usd=use_usd, prim_path=f"{{ENV_REGEX_NS}}/Robot{suffix}")
    if ap.subset == "robot-object":
        assert ap.object_urdf is not None
        return HOISceneRobotObjectCfg(
            num_envs=1,
            env_spacing=2.0,
            robot=robot_cfg,
            hoi_object=make_object_cfg(ap.object_urdf, prim_path=f"{{ENV_REGEX_NS}}/HOI_Object{suffix}"),
        )
    if ap.subset == "robot-terrain":
        assert ap.terrain_urdf is not None
        return HOISceneRobotTerrainCfg(
            num_envs=1,
            env_spacing=2.0,
            robot=robot_cfg,
            hoi_terrain=make_terrain_cfg(ap.terrain_urdf, prim_path=f"{{ENV_REGEX_NS}}/HOI_Terrain{suffix}"),
        )
    if ap.subset == "robot-object-terrain":
        assert ap.object_urdf is not None and ap.terrain_urdf is not None
        return HOISceneRobotObjectTerrainCfg(
            num_envs=1,
            env_spacing=2.0,
            robot=robot_cfg,
            hoi_object=make_object_cfg(ap.object_urdf, prim_path=f"{{ENV_REGEX_NS}}/HOI_Object{suffix}"),
            hoi_terrain=make_terrain_cfg(ap.terrain_urdf, prim_path=f"{{ENV_REGEX_NS}}/HOI_Terrain{suffix}"),
        )
    raise AssertionError(ap.subset)


def _replay_one(
    npz_path: Path,
    *,
    use_usd: bool,
    hoi_root: Path | None,
    loop_forever: bool,
    asset_tag: str = "",
    clear_envs_before_spawn: bool = False,
) -> None:
    if clear_envs_before_spawn:
        _try_remove_envs_prim()

    ap = resolve_assets(npz_path, hoi_root)
    motion = load_hoi_npz(str(npz_path), device=args_cli.device)

    preview = np.load(str(npz_path), allow_pickle=True)
    fps = float(np.asarray(preview["fps"]).reshape(-1)[0])

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / max(fps, 1.0)
    sim = SimulationContext(sim_cfg)

    scene_cfg = _build_scene_cfg(ap, use_usd=use_usd, asset_tag=asset_tag)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    try:
        hoi_object: Articulation | RigidObject | None = scene["hoi_object"]
    except KeyError:
        hoi_object = None

    joint_names = list(UNITREE_G1_29DOF_CFG.joint_sdk_names)
    joint_indices, matched = robot.find_joints(joint_names, preserve_order=True)
    if len(joint_indices) != len(joint_names):
        missing = sorted(set(joint_names) - set(matched))
        raise ValueError(f"Joint names missing on robot: {missing}")

    device = sim.device
    joint_indices_t = torch.tensor(joint_indices, dtype=torch.long, device=device)

    motion_joint_pos = motion.joint_pos.to(device=device)
    motion_joint_vel = motion.joint_vel.to(device=device)
    motion_root_pos = motion.root_pos.to(device=device)
    motion_root_quat = motion.root_quat.to(device=device)
    motion_root_lin_vel = motion.root_lin_vel.to(device=device)
    motion_root_ang_vel = motion.root_ang_vel.to(device=device)

    has_obj = motion.has_object and hoi_object is not None
    if has_obj:
        motion_obj_pos = motion.object_pos.to(device=device)
        motion_obj_quat = motion.object_quat.to(device=device)
        motion_obj_lin_vel = motion.object_lin_vel.to(device=device)
        motion_obj_ang_vel = motion.object_ang_vel.to(device=device)

    frame_id = torch.zeros(scene.num_envs, dtype=torch.long, device=device)

    while simulation_app.is_running():
        current = frame_id[0].item()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_root_pos[current].unsqueeze(0)
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_root_quat[current].unsqueeze(0)
        root_states[:, 7:10] = motion_root_lin_vel[current].unsqueeze(0)
        root_states[:, 10:13] = motion_root_ang_vel[current].unsqueeze(0)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indices_t] = motion_joint_pos[current].unsqueeze(0)
        joint_vel[:, joint_indices_t] = motion_joint_vel[current].unsqueeze(0)

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        if has_obj:
            obj_root = hoi_object.data.default_root_state.clone()
            obj_root[:, :3] = motion_obj_pos[current].unsqueeze(0)
            obj_root[:, :2] += scene.env_origins[:, :2]
            obj_root[:, 3:7] = motion_obj_quat[current].unsqueeze(0)
            obj_root[:, 7:10] = motion_obj_lin_vel[current].unsqueeze(0)
            obj_root[:, 10:13] = motion_obj_ang_vel[current].unsqueeze(0)
            hoi_object.write_root_state_to_sim(obj_root)

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].detach().cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.7]), pos_lookat)

        frame_id += 1
        if frame_id[0] >= motion.num_frames:
            if loop_forever:
                frame_id[:] = 0
            else:
                break


def main() -> None:
    use_usd = args_cli.robot_source == "usd"
    hoi_root: Path | None = Path(args_cli.hoi_root).resolve() if args_cli.hoi_root else None

    paths = _iter_npz_paths()
    if not paths:
        print("No .npz files matched.", file=sys.stderr)
        return

    loop_forever = args_cli.file is not None
    for i, npz_path in enumerate(paths):
        print(f"HOI replay [{i + 1}/{len(paths)}]: {npz_path}")
        _replay_one(
            npz_path,
            use_usd=use_usd,
            hoi_root=hoi_root,
            loop_forever=loop_forever,
            asset_tag=str(i),
            clear_envs_before_spawn=(not loop_forever) and i > 0,
        )
        if i + 1 < len(paths) and args_cli.file is None:
            try:
                input("Press Enter for next trajectory (or close Isaac window to quit)...")
            except EOFError:
                break


if __name__ == "__main__":
    main()
    simulation_app.close()
