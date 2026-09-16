"""Standalone Isaac Lab viewer for OmniRetarget HOI ``.npz`` trajectories.

This script is self-contained: it does not depend on ``unitree_rl_lab.utils.hoi``.

Single file:
    python scripts/mimic/standalone_view_hoi_npz.py -f /path/to/HOI/robot-terrain/foo.npz

Batch (subset under ``--hoi-root``):
    python scripts/mimic/standalone_view_hoi_npz.py --task terrain --hoi-root /path/to/HOI
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import MISSING, dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


DEFAULT_G1_29_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

parser = argparse.ArgumentParser(description="Standalone HOI replay in Isaac Lab.")
parser.add_argument("--file", "-f", type=str, default=None, help="Path to a single HOI .npz file.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    choices=["object", "terrain", "object-terrain"],
    help="Batch mode: which subset under --hoi-root (requires --file omitted).",
)
parser.add_argument("--hoi-root", type=str, default=None, help="Root folder containing HOI models/ and subsets.")
parser.add_argument("--filter", type=str, default="", help="Batch mode: substring filter on .npz basename.")
parser.add_argument(
    "--robot-source",
    type=str,
    default="urdf",
    choices=["urdf", "usd"],
    help="Robot asset source.",
)
parser.add_argument(
    "--robot-usd",
    type=str,
    default="/home/sustech/unitree_project/unitree_rl_lab/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
    help="Robot USD path when --robot-source usd.",
)
parser.add_argument(
    "--joint-names",
    type=str,
    default=",".join(DEFAULT_G1_29_JOINTS),
    help="Comma-separated robot joint names matching HOI qpos channels.",
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
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@dataclass
class HOIMotionData:
    fps: float
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    has_object: bool
    object_pos: torch.Tensor
    object_quat: torch.Tensor
    object_lin_vel: torch.Tensor
    object_ang_vel: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


@dataclass
class AssetPaths:
    robot_urdf: str
    object_urdf: str | None
    terrain_urdf: str | None
    subset: str


def _finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, dt, axis=0).astype(np.float32)


def load_hoi_npz(file_path: str, device: str | torch.device = "cpu") -> HOIMotionData:
    data = np.load(file_path, allow_pickle=True)
    if "qpos" not in data.files or "fps" not in data.files:
        raise ValueError(f"HOI npz must contain 'qpos' and 'fps'. Got keys: {sorted(data.files)}")

    qpos = np.asarray(data["qpos"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] not in (36, 43):
        raise ValueError(f"Expected qpos shape (T, 36) or (T, 43); got {qpos.shape}")

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    dt = 1.0 / max(fps, 1e-6)
    has_object = qpos.shape[1] == 43

    quat_wxyz = qpos[:, 0:4]
    root_pos = qpos[:, 4:7]
    joint_pos = qpos[:, 7:36]

    root_lin_vel = _finite_diff(root_pos, dt)
    joint_vel = _finite_diff(joint_pos, dt)
    root_ang_vel = np.zeros((qpos.shape[0], 3), dtype=np.float32)

    t = qpos.shape[0]
    if has_object:
        obj_quat = qpos[:, 36:40]
        obj_pos = qpos[:, 40:43]
        object_lin_vel = _finite_diff(obj_pos, dt)
        object_ang_vel = np.zeros((t, 3), dtype=np.float32)
    else:
        obj_quat = np.zeros((t, 4), dtype=np.float32)
        obj_quat[:, 0] = 1.0
        obj_pos = np.zeros((t, 3), dtype=np.float32)
        object_lin_vel = np.zeros((t, 3), dtype=np.float32)
        object_ang_vel = np.zeros((t, 3), dtype=np.float32)

    dev = torch.device(device)
    return HOIMotionData(
        fps=fps,
        root_pos=torch.from_numpy(root_pos).to(dev),
        root_quat=torch.from_numpy(quat_wxyz).to(dev),
        joint_pos=torch.from_numpy(joint_pos).to(dev),
        joint_vel=torch.from_numpy(joint_vel).to(dev),
        root_lin_vel=torch.from_numpy(root_lin_vel).to(dev),
        root_ang_vel=torch.from_numpy(root_ang_vel).to(dev),
        has_object=has_object,
        object_pos=torch.from_numpy(obj_pos).to(dev),
        object_quat=torch.from_numpy(obj_quat).to(dev),
        object_lin_vel=torch.from_numpy(object_lin_vel).to(dev),
        object_ang_vel=torch.from_numpy(object_ang_vel).to(dev),
    )


def _natural_sort_key(text: str) -> tuple:
    parts = re.findall(r"\d+|\D+", text)
    key: list = []
    for part in parts:
        key.append((0, int(part)) if part.isdigit() else (1, part.lower()))
    return tuple(key)


def list_npz_for_task(hoi_root: Path, task: str, filter_substr: str = "") -> list[Path]:
    subset_dir = {"object": "robot-object", "terrain": "robot-terrain", "object-terrain": "robot-object-terrain"}[task]
    base = hoi_root / subset_dir
    if not base.is_dir():
        raise FileNotFoundError(f"HOI subset directory not found: {base}")
    paths = sorted(base.glob("*.npz"), key=lambda p: _natural_sort_key(p.name))
    if filter_substr:
        paths = [p for p in paths if filter_substr in p.name]
    return paths


def resolve_assets(npz_path: str | Path, hoi_root: str | Path | None = None) -> AssetPaths:
    npz_path = Path(npz_path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    subset = npz_path.parent.name
    if subset not in ("robot-object", "robot-terrain", "robot-object-terrain"):
        raise ValueError(f"Unexpected HOI subset folder: {subset!r}")

    if hoi_root is None:
        hoi_root = npz_path.parent.parent
    hoi_root = Path(hoi_root).resolve()
    models = hoi_root / "models"
    file_name = npz_path.stem

    if subset == "robot-object":
        robot_urdf = str(models / "g1" / "g1_29dof.urdf")
        object_urdf = str(models / "largebox" / "largebox.urdf")
        terrain_urdf = None
    elif subset == "robot-terrain":
        robot_urdf = str(models / "g1" / "g1_29dof_spherehand.urdf")
        object_urdf = None
        terrain_urdf = str(models / "terrain" / file_name[:8] / f"multi_boxes{file_name[8:]}.urdf")
    else:
        robot_urdf = str(models / "g1" / "g1_29dof_spherehand.urdf")
        if "z_scale" in file_name:
            z_scale = file_name[-12:]
        else:
            z_scale = "_z_scale_1.0"
        terrain_urdf = str(models / "terrain" / file_name[:8] / f"multi_boxes{z_scale}.urdf")
        if "original" in file_name:
            object_urdf = str(models / "chair" / "chair.urdf")
        else:
            object_urdf = str(models / "chair" / f"{file_name[9:25]}.urdf")

    if not os.path.isfile(robot_urdf):
        raise FileNotFoundError(f"Robot URDF missing: {robot_urdf}")
    if object_urdf is not None and not os.path.isfile(object_urdf):
        raise FileNotFoundError(f"Object URDF missing: {object_urdf}")
    if terrain_urdf is not None and not os.path.isfile(terrain_urdf):
        raise FileNotFoundError(f"Terrain URDF missing: {terrain_urdf}")

    return AssetPaths(robot_urdf=robot_urdf, object_urdf=object_urdf, terrain_urdf=terrain_urdf, subset=subset)


def ensure_world_link_urdf(urdf_path: str) -> str:
    path = Path(urdf_path).resolve()
    text = path.read_text(encoding="utf-8")
    if re.search(r'<link\s+name\s*=\s*["\']world["\']', text):
        return str(path)
    robot_open = re.search(r"<robot\b[^>]*>", text)
    if not robot_open:
        raise ValueError(f"No <robot> tag in {urdf_path}")
    patched = text[: robot_open.end()] + '\n  <link name="world"/>' + text[robot_open.end() :]
    patched_path = path.with_name(f"{path.stem}_isaac_world{path.suffix}")
    patched_path.write_text(patched, encoding="utf-8")
    return str(patched_path)


def make_robot_cfg(urdf_path: str, *, prim_path: str, use_usd: bool) -> ArticulationCfg:
    if use_usd:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=os.path.abspath(args_cli.robot_usd)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8), joint_vel={".*": 0.0}),
            actuators={
                "replay": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
            },
        )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=os.path.abspath(urdf_path),
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8), joint_vel={".*": 0.0}),
        actuators={
            "replay": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
        },
    )


def make_object_cfg(urdf_path: str, *, prim_path: str) -> AssetBaseCfg:
    abs_urdf = os.path.abspath(urdf_path)
    has_joint = re.search(r"<joint\b", Path(abs_urdf).read_text(encoding="utf-8")) is not None
    if has_joint:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=abs_urdf,
                fix_base=False,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            ),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={}),
            actuators={},
        )
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=abs_urdf,
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
    )


def make_terrain_cfg(urdf_path: str, *, prim_path: str) -> AssetBaseCfg:
    fixed_urdf = ensure_world_link_urdf(urdf_path)
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=os.path.abspath(fixed_urdf),
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
    )


def _try_remove_envs_prim() -> None:
    try:
        import omni.usd  # type: ignore

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        if stage.GetPrimAtPath("/World/envs").IsValid():
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
    here = Path(__file__).resolve().parent
    guess = (here / "../../../motion_dataset/HOI").resolve()
    return guess if guess.is_dir() else Path.cwd()


def _iter_npz_paths() -> list[Path]:
    root = _default_hoi_root()
    if args_cli.file:
        return [Path(args_cli.file).resolve()]
    assert args_cli.task is not None
    return list_npz_for_task(root, args_cli.task, args_cli.filter)


@configclass
class HOISceneRobotObjectCfg(InteractiveSceneCfg):
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
    robot_cfg = make_robot_cfg(ap.robot_urdf, prim_path=f"{{ENV_REGEX_NS}}/Robot{suffix}", use_usd=use_usd)
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
    assert ap.object_urdf is not None and ap.terrain_urdf is not None
    return HOISceneRobotObjectTerrainCfg(
        num_envs=1,
        env_spacing=2.0,
        robot=robot_cfg,
        hoi_object=make_object_cfg(ap.object_urdf, prim_path=f"{{ENV_REGEX_NS}}/HOI_Object{suffix}"),
        hoi_terrain=make_terrain_cfg(ap.terrain_urdf, prim_path=f"{{ENV_REGEX_NS}}/HOI_Terrain{suffix}"),
    )


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

    joint_names = [j.strip() for j in args_cli.joint_names.split(",") if j.strip()]
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

