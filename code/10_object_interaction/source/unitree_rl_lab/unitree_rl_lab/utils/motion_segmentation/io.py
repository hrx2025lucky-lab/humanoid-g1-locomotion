from __future__ import annotations

from pathlib import Path

import numpy as np

from unitree_rl_lab.utils.hoi.path_resolver import resolve_assets
from unitree_rl_lab.utils.motion_segmentation.types import NormalizedMotion


def _finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, dt, axis=0).astype(np.float32)


def _safe_array_dict(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k]).copy()
    return out


def _load_hoi_qpos(path: Path, data: np.lib.npyio.NpzFile) -> NormalizedMotion:
    qpos = np.asarray(data["qpos"], dtype=np.float32)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    dt = 1.0 / max(fps, 1e-6)
    if qpos.ndim != 2 or qpos.shape[1] not in (36, 43):
        raise ValueError(f"Expected HOI qpos shape (T, 36|43), got {qpos.shape} from {path}")

    root_quat = qpos[:, 0:4]
    root_pos = qpos[:, 4:7]
    joint_pos = qpos[:, 7:36]
    joint_vel = _finite_diff(joint_pos, dt)
    root_lin_vel = _finite_diff(root_pos, dt)
    root_ang_vel = np.zeros((qpos.shape[0], 3), dtype=np.float32)

    object_quat = None
    object_pos = None
    if qpos.shape[1] == 43:
        object_quat = qpos[:, 36:40].copy()
        object_pos = qpos[:, 40:43].copy()

    extra = {"has_object": qpos.shape[1] == 43}
    # Best-effort: do not fail on non-standard path layouts.
    try:
        ap = resolve_assets(path)
        extra["assets"] = {
            "subset": ap.subset,
            "robot_urdf": ap.robot_urdf,
            "object_urdf": ap.object_urdf,
            "terrain_urdf": ap.terrain_urdf,
        }
    except Exception:
        pass

    return NormalizedMotion(
        source_path=path,
        schema="hoi_qpos",
        fps=fps,
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        object_pos=object_pos,
        object_quat_wxyz=object_quat,
        raw_arrays=_safe_array_dict(data),
        extra_metadata=extra,
    )


def _load_mimic(path: Path, data: np.lib.npyio.NpzFile) -> NormalizedMotion:
    required = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "fps"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Mimic npz missing keys: {missing} in {path}")

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
    if body_pos_w.ndim != 3 or body_pos_w.shape[1] < 1:
        raise ValueError(f"Invalid mimic body_pos_w shape: {body_pos_w.shape}")

    return NormalizedMotion(
        source_path=path,
        schema="mimic",
        fps=fps,
        root_pos=body_pos_w[:, 0, :].copy(),
        root_quat_wxyz=body_quat_w[:, 0, :].copy(),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_lin_vel=body_lin_vel_w[:, 0, :].copy(),
        root_ang_vel=body_ang_vel_w[:, 0, :].copy(),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        raw_arrays=_safe_array_dict(data),
        extra_metadata={"has_object": False},
    )


def load_motion_npz(path: str | Path) -> NormalizedMotion:
    """Load HOI qpos or mimic npz into a common motion structure."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Motion file not found: {p}")

    data = np.load(str(p), allow_pickle=True)
    keys = set(data.files)
    if {"qpos", "fps"}.issubset(keys):
        return _load_hoi_qpos(p, data)
    if {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}.issubset(keys):
        return _load_mimic(p, data)
    raise ValueError(f"Unsupported motion npz schema in {p}. Keys={sorted(keys)}")

