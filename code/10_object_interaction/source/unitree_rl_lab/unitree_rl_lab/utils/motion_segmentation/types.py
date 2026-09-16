from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


MotionSchema = Literal["hoi_qpos", "mimic"]


@dataclass
class SegmentLabel:
    """A manually labeled segment over a frame interval (inclusive bounds)."""

    segment_id: str
    label: str
    start_frame: int
    end_frame: int
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "label": self.label,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "notes": self.notes,
        }


@dataclass
class NormalizedMotion:
    """Unified representation for HOI qpos and mimic npz motions."""

    source_path: Path
    schema: MotionSchema
    fps: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_lin_vel: np.ndarray
    root_ang_vel: np.ndarray
    object_pos: np.ndarray | None = None
    object_quat_wxyz: np.ndarray | None = None
    body_pos_w: np.ndarray | None = None
    body_quat_w: np.ndarray | None = None
    body_lin_vel_w: np.ndarray | None = None
    body_ang_vel_w: np.ndarray | None = None
    raw_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    extra_metadata: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    def with_slice(self, start_frame: int, end_frame: int) -> "NormalizedMotion":
        """Return a sliced copy using inclusive frame bounds."""
        lo = int(start_frame)
        hi = int(end_frame)
        if lo < 0 or hi < lo or hi >= self.num_frames:
            raise ValueError(f"Invalid slice [{lo}, {hi}] for num_frames={self.num_frames}")
        s = slice(lo, hi + 1)
        return NormalizedMotion(
            source_path=self.source_path,
            schema=self.schema,
            fps=self.fps,
            root_pos=self.root_pos[s].copy(),
            root_quat_wxyz=self.root_quat_wxyz[s].copy(),
            joint_pos=self.joint_pos[s].copy(),
            joint_vel=self.joint_vel[s].copy(),
            root_lin_vel=self.root_lin_vel[s].copy(),
            root_ang_vel=self.root_ang_vel[s].copy(),
            object_pos=None if self.object_pos is None else self.object_pos[s].copy(),
            object_quat_wxyz=None if self.object_quat_wxyz is None else self.object_quat_wxyz[s].copy(),
            body_pos_w=None if self.body_pos_w is None else self.body_pos_w[s].copy(),
            body_quat_w=None if self.body_quat_w is None else self.body_quat_w[s].copy(),
            body_lin_vel_w=None if self.body_lin_vel_w is None else self.body_lin_vel_w[s].copy(),
            body_ang_vel_w=None if self.body_ang_vel_w is None else self.body_ang_vel_w[s].copy(),
            raw_arrays={k: v[s].copy() if getattr(v, "ndim", 0) > 0 and v.shape[0] == self.num_frames else v for k, v in self.raw_arrays.items()},
            extra_metadata=dict(self.extra_metadata),
        )

