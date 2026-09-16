"""Portable expert-motion loading and sampling for AMP.

This module intentionally has no Isaac Lab imports so motion files can be
validated and unit-tested without launching the simulator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_FEATURES = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "base_height",
    "joint_pos",
    "joint_vel",
    "key_links_pos_b",
)


@dataclass
class MotionDatasetCfg:
    """Description of expert clips and the AMP frame assembled from them."""

    motion_dir: str
    joint_names: tuple[str, ...]
    profile_name: str = "custom"
    source_joint_names: tuple[str, ...] = ()
    key_link_names: tuple[str, ...] = ()
    history_steps: int = 5
    step_dt: float = 0.02
    command_smoothing_seconds: float = 0.25
    features: tuple[str, ...] = DEFAULT_FEATURES
    clip_weights: dict[str, float] | None = None
    file_pattern: str = "*.npz"
    quaternion_order: str = "xyzw"


@dataclass
class _MotionClip:
    name: str
    frames: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    key_links_pos_b: torch.Tensor
    motion_command: torch.Tensor


@dataclass
class ReferenceMotionState:
    """A coherent batch of root, joint, and key-link states from expert clips."""

    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    key_links_pos_b: torch.Tensor
    motion_command: torch.Tensor


def _finite_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[0] < 2:
        raise ValueError("Motion clips need at least two frames to derive velocities.")
    delta = (values[1:] - values[:-1]) / dt
    return torch.cat((delta, delta[-1:]), dim=0)


def _smooth_signal(values: torch.Tensor, window_size: int) -> torch.Tensor:
    """Centered moving average with replicated boundaries."""
    if window_size <= 1:
        return values
    if window_size % 2 == 0:
        window_size += 1
    padding = window_size // 2
    channels_first = values.transpose(0, 1).unsqueeze(0)
    padded = F.pad(channels_first, (padding, padding), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=window_size, stride=1).squeeze(0).transpose(0, 1)


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    rotated = _quat_multiply(_quat_multiply(_quat_conjugate(quaternion), pure), quaternion)
    return rotated[..., 1:]


def _angular_velocity_world(quaternion: torch.Tensor, dt: float) -> torch.Tensor:
    relative = _quat_multiply(quaternion[1:], _quat_conjugate(quaternion[:-1]))
    relative = torch.where(relative[:, :1] < 0.0, -relative, relative)
    vector = relative[:, 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, relative[:, :1].clamp(min=1.0e-8))
    axis = vector / vector_norm.clamp(min=1.0e-8)
    velocity = axis * angle / dt
    velocity = torch.where(vector_norm > 1.0e-7, velocity, torch.zeros_like(velocity))
    return torch.cat((velocity, velocity[-1:]), dim=0)


def _resample_linear(values: torch.Tensor, source_dt: float, target_dt: float) -> torch.Tensor:
    duration = source_dt * (values.shape[0] - 1)
    sample_count = int(np.floor(duration / target_dt + 1.0e-7)) + 1
    times = torch.arange(sample_count, device=values.device, dtype=torch.float32) * target_dt
    location = times / source_dt
    lower = torch.floor(location).long().clamp(max=values.shape[0] - 1)
    upper = (lower + 1).clamp(max=values.shape[0] - 1)
    blend = (location - lower).reshape((-1,) + (1,) * (values.ndim - 1))
    return torch.lerp(values[lower], values[upper], blend)


def _resample_quaternion(values: torch.Tensor, source_dt: float, target_dt: float) -> torch.Tensor:
    duration = source_dt * (values.shape[0] - 1)
    sample_count = int(np.floor(duration / target_dt + 1.0e-7)) + 1
    times = torch.arange(sample_count, device=values.device, dtype=torch.float32) * target_dt
    location = times / source_dt
    lower = torch.floor(location).long().clamp(max=values.shape[0] - 1)
    upper = (lower + 1).clamp(max=values.shape[0] - 1)
    blend = (location - lower).unsqueeze(-1)
    q0 = values[lower]
    q1 = values[upper]
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    angle = torch.acos(dot)
    sin_angle = torch.sin(angle)
    scale0 = torch.sin((1.0 - blend) * angle) / sin_angle.clamp(min=1.0e-7)
    scale1 = torch.sin(blend * angle) / sin_angle.clamp(min=1.0e-7)
    slerp = scale0 * q0 + scale1 * q1
    nlerp = torch.lerp(q0, q1, blend)
    result = torch.where(dot > 0.9995, nlerp, slerp)
    return torch.nn.functional.normalize(result, dim=-1)


class MotionDataset:
    """Load expert clips once and sample fixed-length motion windows on-device."""

    def __init__(self, cfg: MotionDatasetCfg, device: str | torch.device) -> None:
        if not cfg.profile_name.strip():
            raise ValueError("profile_name must be non-empty.")
        if cfg.history_steps < 1:
            raise ValueError("history_steps must be positive.")
        if cfg.step_dt <= 0.0:
            raise ValueError("step_dt must be positive.")
        if cfg.command_smoothing_seconds < 0.0:
            raise ValueError("command_smoothing_seconds must be non-negative.")
        self.cfg = cfg
        self.device = torch.device(device)
        self.feature_slices: dict[str, slice] = {}
        motion_dir = Path(os.path.expandvars(os.path.expanduser(cfg.motion_dir)))
        paths = sorted(motion_dir.glob(cfg.file_pattern)) if motion_dir.is_dir() else []
        if not paths:
            raise FileNotFoundError(
                f"No AMP motion files matching '{cfg.file_pattern}' were found in '{motion_dir}'. "
                "Set UNITREE_AMP_MOTION_DIR or copy .npz clips into the configured directory."
            )

        self.clips = [self._load_clip(path) for path in paths]
        weights_cfg = cfg.clip_weights or {}
        available_names = {clip.name for clip in self.clips}
        missing_names = sorted(
            name for name, weight in weights_cfg.items() if weight > 0.0 and name not in available_names
        )
        if missing_names:
            # 默认严格报错：少一段专家数据会悄悄改变风格分布，训练出来的
            # 步态和作业要求的不一致，却不会有任何报错提示。
            #
            # 但冒烟测试时手上往往只有少数几段（完整 ACCAD 走跑集需要先做
            # 实践 7 的 GMR 重定向），此时允许显式放宽，只用已有片段跑通链路。
            # 必须显式设 UNITREE_AMP_ALLOW_MISSING_CLIPS=1，不会被误触发。
            if os.environ.get("UNITREE_AMP_ALLOW_MISSING_CLIPS", "0") == "1":
                print(
                    f"[AMP] 警告：profile '{cfg.profile_name}' 缺少 {len(missing_names)} 段专家数据，"
                    f"已按 UNITREE_AMP_ALLOW_MISSING_CLIPS=1 忽略：{', '.join(missing_names)}\n"
                    f"[AMP] 仅用现有 {len(self.clips)} 段继续: 这只适合冒烟测试，"
                    "正式训练必须补齐，否则风格分布与作业要求不符。"
                )
                weights_cfg = {k: v for k, v in weights_cfg.items() if k in available_names}
                if not weights_cfg:
                    weights_cfg = {clip.name: 1.0 for clip in self.clips}
            else:
                raise ValueError(f"AMP clip weights reference missing clips: {', '.join(missing_names)}.")
        weights = [weights_cfg.get(clip.name, 1.0 if not weights_cfg else 0.0) for clip in self.clips]
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("AMP clip weights must be non-negative and contain at least one positive value.")
        self.clip_probabilities = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.clip_probabilities /= self.clip_probabilities.sum()
        self.frame_dim = self.clips[0].frames.shape[1]
        if any(clip.frames.shape[1] != self.frame_dim for clip in self.clips):
            raise ValueError("All AMP clips must produce the same feature dimension.")

    def _load_clip(self, path: Path) -> _MotionClip:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if "fps" not in arrays:
            raise ValueError(f"Motion file '{path}' is missing required field 'fps'.")
        fps = float(np.asarray(arrays["fps"]).item())
        if fps <= 0.0:
            raise ValueError(f"Motion file '{path}' has invalid fps={fps}.")

        joint_key = "joint_pos" if "joint_pos" in arrays else "dof_pos" if "dof_pos" in arrays else None
        if joint_key is None:
            raise ValueError(f"Motion file '{path}' is missing 'joint_pos' (or 'dof_pos').")
        name_key = "joint_names" if "joint_names" in arrays else "dof_names" if "dof_names" in arrays else None
        if name_key is None:
            if not self.cfg.source_joint_names:
                raise ValueError(
                    f"Motion file '{path}' is missing 'joint_names' (or 'dof_names'); "
                    "configure source_joint_names for this dataset format."
                )
            source_names = list(self.cfg.source_joint_names)
        else:
            source_names = [str(name) for name in arrays[name_key].tolist()]
        if len(source_names) != arrays[joint_key].shape[1]:
            raise ValueError(
                f"Motion file '{path}' contains {arrays[joint_key].shape[1]} joint columns but "
                f"{len(source_names)} source joint names were provided."
            )
        missing = [name for name in self.cfg.joint_names if name not in source_names]
        if missing:
            raise ValueError(f"Motion file '{path}' is missing configured joints: {missing}.")
        joint_indices = [source_names.index(name) for name in self.cfg.joint_names]

        source_dt = 1.0 / fps
        joint_pos = torch.as_tensor(arrays[joint_key], dtype=torch.float32, device=self.device)[:, joint_indices]
        joint_pos = _resample_linear(joint_pos, source_dt, self.cfg.step_dt)

        root_pos_array = arrays.get("root_pos", np.zeros((len(arrays[joint_key]), 3), dtype=np.float32))
        root_pos = torch.as_tensor(root_pos_array, dtype=torch.float32, device=self.device)
        root_pos = _resample_linear(root_pos, source_dt, self.cfg.step_dt)

        quat_key = next((key for key in ("root_quat_wxyz", "root_quat", "root_rot") if key in arrays), None)
        if quat_key is None:
            root_quat = torch.zeros(len(arrays[joint_key]), 4, dtype=torch.float32, device=self.device)
            root_quat[:, 0] = 1.0
        else:
            root_quat = torch.as_tensor(arrays[quat_key], dtype=torch.float32, device=self.device)
            order = "wxyz" if quat_key == "root_quat_wxyz" else self.cfg.quaternion_order
            if order == "xyzw":
                root_quat = root_quat[:, [3, 0, 1, 2]]
            elif order != "wxyz":
                raise ValueError(f"Unsupported quaternion order '{order}'.")
            root_quat = torch.nn.functional.normalize(root_quat, dim=-1)
        root_quat = _resample_quaternion(root_quat, source_dt, self.cfg.step_dt)

        joint_vel = _finite_difference(joint_pos, self.cfg.step_dt)
        root_vel_world = _finite_difference(root_pos, self.cfg.step_dt)
        root_vel_body = _quat_rotate_inverse(root_quat, root_vel_world)
        root_ang_world = _angular_velocity_world(root_quat, self.cfg.step_dt)
        root_ang_body = _quat_rotate_inverse(root_quat, root_ang_world)
        motion_command = torch.cat((root_vel_body[:, :2], root_ang_body[:, 2:3]), dim=-1)
        smoothing_window = max(1, round(self.cfg.command_smoothing_seconds / self.cfg.step_dt))
        motion_command = _smooth_signal(motion_command, smoothing_window)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(root_quat.shape[0], -1)
        projected_gravity = _quat_rotate_inverse(root_quat, gravity_world)

        if self.cfg.key_link_names:
            if "local_body_pos" not in arrays or "link_body_list" not in arrays:
                raise ValueError(
                    f"Motion file '{path}' is missing configured key links because "
                    "'local_body_pos' or 'link_body_list' is absent."
                )
            source_link_names = [str(name) for name in np.asarray(arrays["link_body_list"]).tolist()]
            missing_links = [name for name in self.cfg.key_link_names if name not in source_link_names]
            if missing_links:
                raise ValueError(f"Motion file '{path}' is missing configured key links: {missing_links}.")
            link_indices = [source_link_names.index(name) for name in self.cfg.key_link_names]
            key_links_pos_b = torch.as_tensor(
                arrays["local_body_pos"][:, link_indices, :], dtype=torch.float32, device=self.device
            )
            key_links_pos_b = _resample_linear(key_links_pos_b, source_dt, self.cfg.step_dt)
            key_link_features = key_links_pos_b.flatten(start_dim=1)
        else:
            key_links_pos_b = torch.empty(root_pos.shape[0], 0, 3, device=self.device)
            key_link_features = torch.empty(root_pos.shape[0], 0, device=self.device)

        feature_values = {
            "base_lin_vel": root_vel_body,
            "base_ang_vel": root_ang_body,
            "projected_gravity": projected_gravity,
            "base_height": root_pos[:, 2:3],
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "key_links_pos_b": key_link_features,
            "motion_command": motion_command,
        }
        unknown = [name for name in self.cfg.features if name not in feature_values]
        if unknown:
            raise ValueError(f"Unsupported AMP motion features: {unknown}.")
        feature_slices: dict[str, slice] = {}
        offset = 0
        for name in self.cfg.features:
            width = feature_values[name].shape[1]
            feature_slices[name] = slice(offset, offset + width)
            offset += width
        if self.feature_slices and feature_slices != self.feature_slices:
            raise ValueError("All AMP clips must use the same feature layout.")
        self.feature_slices = feature_slices
        frames = torch.cat([feature_values[name] for name in self.cfg.features], dim=-1)
        if frames.shape[0] < self.cfg.history_steps:
            raise ValueError(
                f"Motion file '{path}' has {frames.shape[0]} resampled frames, fewer than history_steps="
                f"{self.cfg.history_steps}."
            )
        return _MotionClip(
            name=path.stem,
            frames=frames,
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel=root_vel_world,
            root_ang_vel=root_ang_world,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            key_links_pos_b=key_links_pos_b,
            motion_command=motion_command,
        )

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample chronological windows and flatten history for the discriminator."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        clip_ids = torch.multinomial(self.clip_probabilities, batch_size, replacement=True)
        windows = torch.empty(
            batch_size,
            self.cfg.history_steps,
            self.frame_dim,
            dtype=torch.float32,
            device=self.device,
        )
        offsets = torch.arange(self.cfg.history_steps, device=self.device)
        for clip_id in torch.unique(clip_ids).tolist():
            output_ids = torch.nonzero(clip_ids == clip_id, as_tuple=False).squeeze(-1)
            clip = self.clips[clip_id]
            max_start = clip.frames.shape[0] - self.cfg.history_steps
            starts = torch.randint(max_start + 1, (output_ids.numel(),), device=self.device)
            windows[output_ids] = clip.frames[starts[:, None] + offsets[None, :]]
        return windows.flatten(start_dim=1)

    @torch.no_grad()
    def sample_reference_state(
        self,
        batch_size: int,
        command_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    ) -> ReferenceMotionState:
        """Sample complete frames for coherent simulator reference-state resets."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        frame_candidates: list[torch.Tensor] | None = None
        probabilities = self.clip_probabilities
        if command_ranges is not None:
            if len(command_ranges) != 3 or any(len(bounds) != 2 for bounds in command_ranges):
                raise ValueError("command_ranges must contain x, y, and yaw (min, max) pairs.")
            frame_candidates = []
            supported = torch.zeros(len(self.clips), dtype=torch.bool, device=self.device)
            for clip_id, clip in enumerate(self.clips):
                valid = torch.ones(clip.motion_command.shape[0], dtype=torch.bool, device=self.device)
                for axis, (lower, upper) in enumerate(command_ranges):
                    if lower > upper:
                        raise ValueError(f"Invalid command range ({lower}, {upper}) for axis {axis}.")
                    valid &= (clip.motion_command[:, axis] >= lower) & (clip.motion_command[:, axis] <= upper)
                candidates = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                frame_candidates.append(candidates)
                supported[clip_id] = candidates.numel() > 0
            if not torch.any(supported):
                raise ValueError(f"No expert reference frames satisfy command ranges {command_ranges}.")
            probabilities = probabilities * supported
            probabilities = probabilities / probabilities.sum()

        clip_ids = torch.multinomial(probabilities, batch_size, replacement=True)
        root_pos = torch.empty(batch_size, 3, device=self.device)
        root_quat = torch.empty(batch_size, 4, device=self.device)
        root_lin_vel = torch.empty(batch_size, 3, device=self.device)
        root_ang_vel = torch.empty(batch_size, 3, device=self.device)
        joint_pos = torch.empty(batch_size, len(self.cfg.joint_names), device=self.device)
        joint_vel = torch.empty_like(joint_pos)
        key_links_pos_b = torch.empty(batch_size, len(self.cfg.key_link_names), 3, device=self.device)
        motion_command = torch.empty(batch_size, 3, device=self.device)

        for clip_id in torch.unique(clip_ids).tolist():
            output_ids = torch.nonzero(clip_ids == clip_id, as_tuple=False).squeeze(-1)
            clip = self.clips[clip_id]
            if frame_candidates is None:
                frame_ids = torch.randint(clip.frames.shape[0], (output_ids.numel(),), device=self.device)
            else:
                candidates = frame_candidates[clip_id]
                selected = torch.randint(candidates.numel(), (output_ids.numel(),), device=self.device)
                frame_ids = candidates[selected]
            root_pos[output_ids] = clip.root_pos[frame_ids]
            root_quat[output_ids] = clip.root_quat[frame_ids]
            root_lin_vel[output_ids] = clip.root_lin_vel[frame_ids]
            root_ang_vel[output_ids] = clip.root_ang_vel[frame_ids]
            joint_pos[output_ids] = clip.joint_pos[frame_ids]
            joint_vel[output_ids] = clip.joint_vel[frame_ids]
            key_links_pos_b[output_ids] = clip.key_links_pos_b[frame_ids]
            motion_command[output_ids] = clip.motion_command[frame_ids]

        return ReferenceMotionState(
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            key_links_pos_b=key_links_pos_b,
            motion_command=motion_command,
        )
