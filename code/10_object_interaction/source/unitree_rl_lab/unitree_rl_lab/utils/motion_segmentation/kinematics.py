from __future__ import annotations

import numpy as np

from unitree_rl_lab.utils.motion_segmentation.types import NormalizedMotion


def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Extract yaw from quaternions in wxyz convention."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)


def compute_motion_features(motion: NormalizedMotion) -> dict[str, np.ndarray]:
    """Compute timeline features useful for manual segmentation."""
    dt = 1.0 / max(motion.fps, 1e-6)
    root_speed = np.linalg.norm(motion.root_lin_vel, axis=1).astype(np.float32)
    yaw = yaw_from_quat_wxyz(motion.root_quat_wxyz)
    yaw_rate = np.gradient(yaw, dt).astype(np.float32) if motion.num_frames > 1 else np.zeros_like(yaw)
    joint_speed = np.linalg.norm(motion.joint_vel, axis=1).astype(np.float32)
    base_height = motion.root_pos[:, 2].astype(np.float32)
    return {
        "time_s": (np.arange(motion.num_frames, dtype=np.float32) / max(motion.fps, 1e-6)),
        "root_speed": root_speed,
        "yaw_rate": yaw_rate,
        "joint_speed": joint_speed,
        "base_height": base_height,
    }

