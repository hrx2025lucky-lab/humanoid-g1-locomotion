"""IsaacLab viewer for retargeted G1 motion datasets in NPZ format.

Example:
    python scripts/mimic/view_retargeted_npz.py \
      -f /home/sustech/unitree_project/motion_dataset/AMASS_Retargeted_for_G1/g1/KIT/3/walking_run02_poses_100_jpos.npz
"""


# unitree_rl_lab/scripts/mimic/view_retargeted_npz.py \
#   -f "/home/sustech/unitree_project/motion_dataset/AMASS_Retargeted_for_G1/g1/KIT/3/walking_run02_poses_100_jpos.npz"

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay a retargeted G1 motion NPZ in IsaacLab.")
parser.add_argument("--file", "-f", type=str, required=True, help="Path to retargeted motion .npz file.")
parser.add_argument(
    "--root-body-name",
    type=str,
    default="pelvis",
    help="Body name used as root in the motion file. Falls back to first body if not found.",
)
parser.add_argument(
    "--quat-order",
    type=str,
    default="wxyz",
    choices=["wxyz", "xyzw"],
    help="Quaternion order in the loaded motion file.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG


@dataclass
class MotionData:
    fps: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _to_wxyz(quat: np.ndarray, quat_order: str) -> np.ndarray:
    if quat_order == "wxyz":
        return quat
    # xyzw -> wxyz
    return quat[..., [3, 0, 1, 2]]


def _finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, dt, axis=0).astype(np.float32)


def _quat_to_angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    # Minimal fallback for schemas without angular velocity.
    # Keeps playback stable; angular velocity is mainly used for root state completeness.
    if quat_wxyz.shape[0] <= 1:
        return np.zeros((quat_wxyz.shape[0], 3), dtype=np.float32)
    return np.zeros((quat_wxyz.shape[0], 3), dtype=np.float32)


def load_retargeted_npz(file_path: str, root_body_name: str, quat_order: str) -> MotionData:
    data = np.load(file_path, allow_pickle=True)
    keys = set(data.files)

    # Retargeted schema
    if {"dof_names", "dof_positions", "body_names", "body_positions", "body_rotations"}.issubset(keys):
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        joint_pos = np.asarray(data["dof_positions"], dtype=np.float32)
        joint_vel = np.asarray(data["dof_velocities"], dtype=np.float32)

        body_names = [str(x) for x in data["body_names"]]
        root_body_index = body_names.index(root_body_name) if root_body_name in body_names else 0

        body_pos = np.asarray(data["body_positions"], dtype=np.float32)
        body_quat = _to_wxyz(np.asarray(data["body_rotations"], dtype=np.float32), quat_order)
        body_lin_vel = np.asarray(data["body_linear_velocities"], dtype=np.float32)
        body_ang_vel = np.asarray(data["body_angular_velocities"], dtype=np.float32)

        return MotionData(
            fps=fps,
            joint_pos=torch.from_numpy(joint_pos),
            joint_vel=torch.from_numpy(joint_vel),
            root_pos=torch.from_numpy(body_pos[:, root_body_index, :]),
            root_quat=torch.from_numpy(body_quat[:, root_body_index, :]),
            root_lin_vel=torch.from_numpy(body_lin_vel[:, root_body_index, :]),
            root_ang_vel=torch.from_numpy(body_ang_vel[:, root_body_index, :]),
        )

    # Existing mimic schema
    if {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}.issubset(keys):
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        return MotionData(
            fps=fps,
            joint_pos=torch.from_numpy(np.asarray(data["joint_pos"], dtype=np.float32)),
            joint_vel=torch.from_numpy(np.asarray(data["joint_vel"], dtype=np.float32)),
            root_pos=torch.from_numpy(body_pos[:, 0, :]),
            root_quat=torch.from_numpy(body_quat[:, 0, :]),
            root_lin_vel=torch.from_numpy(body_lin_vel[:, 0, :]),
            root_ang_vel=torch.from_numpy(body_ang_vel[:, 0, :]),
        )

    # Lightweight retargeted schema: joint + base only
    if {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}.issubset(keys):
        fps = float(np.asarray(data["framerate"]).reshape(-1)[0])
        dt = 1.0 / max(fps, 1.0)
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = _finite_diff(joint_pos, dt)
        root_pos = np.asarray(data["base_pos_w"], dtype=np.float32)
        root_quat = _to_wxyz(np.asarray(data["base_quat_w"], dtype=np.float32), quat_order)
        root_lin_vel = _finite_diff(root_pos, dt)
        root_ang_vel = _quat_to_angular_velocity(root_quat, dt)
        return MotionData(
            fps=fps,
            joint_pos=torch.from_numpy(joint_pos),
            joint_vel=torch.from_numpy(joint_vel),
            root_pos=torch.from_numpy(root_pos),
            root_quat=torch.from_numpy(root_quat),
            root_lin_vel=torch.from_numpy(root_lin_vel),
            root_ang_vel=torch.from_numpy(root_ang_vel),
        )

    raise ValueError(f"Unsupported npz schema in {file_path}. Keys: {sorted(keys)}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion = load_retargeted_npz(args_cli.file, args_cli.root_body_name, args_cli.quat_order)
    data = np.load(args_cli.file, allow_pickle=True)
    file_keys = set(data.files)

    if "dof_names" in file_keys:
        dof_names = [str(x) for x in data["dof_names"]]
        joint_indices, matched_names = robot.find_joints(dof_names, preserve_order=True)
        if len(joint_indices) != len(dof_names):
            missing = sorted(set(dof_names) - set(matched_names))
            raise ValueError(f"Some dof_names are not found on G1 articulation: {missing}")
    elif "joint_names" in file_keys:
        dof_names = [str(x) for x in data["joint_names"]]
        joint_indices, matched_names = robot.find_joints(dof_names, preserve_order=True)
        if len(joint_indices) != len(dof_names):
            missing = sorted(set(dof_names) - set(matched_names))
            raise ValueError(f"Some joint_names are not found on G1 articulation: {missing}")
    else:
        # mimic schema already uses robot joint order
        joint_indices = list(range(motion.joint_pos.shape[1]))

    device = sim.device
    joint_indices_t = torch.tensor(joint_indices, dtype=torch.long, device=device)

    motion_joint_pos = motion.joint_pos.to(device=device)
    motion_joint_vel = motion.joint_vel.to(device=device)
    motion_root_pos = motion.root_pos.to(device=device)
    motion_root_quat = motion.root_quat.to(device=device)
    motion_root_lin_vel = motion.root_lin_vel.to(device=device)
    motion_root_ang_vel = motion.root_ang_vel.to(device=device)

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
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].detach().cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.7]), pos_lookat)

        frame_id += 1
        if frame_id[0] >= motion.num_frames:
            frame_id[:] = 0


def main():
    preview = np.load(args_cli.file, allow_pickle=True)
    if "fps" in preview.files:
        fps = float(np.asarray(preview["fps"]).reshape(-1)[0])
    elif "framerate" in preview.files:
        fps = float(np.asarray(preview["framerate"]).reshape(-1)[0])
    else:
        fps = 50.0

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / max(fps, 1.0)
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
