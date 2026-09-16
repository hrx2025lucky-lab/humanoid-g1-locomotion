"""Manager-based environment hooks required by AMP."""

from __future__ import annotations

from dataclasses import replace

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .motion_dataset import MotionDataset


class AMPManagerBasedRLEnv(ManagerBasedRLEnv):
    """Expose expert samples and capture style state immediately before reset."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        self._amp_capture_ready = False
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        current_frame = self.observation_manager.compute_group("amp", update_history=False)
        if current_frame.shape[-1] != self.amp_expert_sampler.frame_dim:
            raise ValueError(
                "Simulator AMP frame dimension "
                f"{current_frame.shape[-1]} does not match expert frame dimension "
                f"{self.amp_expert_sampler.frame_dim}."
            )
        self.terminal_amp_frames = torch.zeros_like(current_frame)
        self._amp_capture_ready = True

    def load_managers(self) -> None:
        """Load the expert dataset before the first reset event can request RSI."""
        super().load_managers()
        joint_names = tuple(self.scene["robot"].data.joint_names)
        dataset_cfg = replace(
            self.cfg.amp_motion,
            joint_names=joint_names,
            history_steps=self.cfg.amp_history_steps,
            step_dt=self.step_dt,
        )
        dataset_cfg = self.cfg.motion_source.apply_to_dataset_cfg(dataset_cfg)
        if dataset_cfg.profile_name != self.cfg.motion_style:
            raise ValueError(
                f"AMP environment selected '{self.cfg.motion_style}', "
                f"but motion source selected '{dataset_cfg.profile_name}'."
            )
        self.amp_expert_sampler = MotionDataset(dataset_cfg, device=self.device)

    def _reset_idx(self, env_ids):
        self._amp_pending_reference_commands = None
        if self._amp_capture_ready and env_ids is not None and len(env_ids) > 0:
            current_frame = self.observation_manager.compute_group("amp", update_history=False)
            self.terminal_amp_frames[env_ids] = current_frame[env_ids]
        super()._reset_idx(env_ids)
        pending = self._amp_pending_reference_commands
        if pending is not None:
            command_env_ids, commands, command_name = pending
            command_term = self.command_manager.get_term(command_name)
            command_term.vel_command_b[command_env_ids] = commands
            if hasattr(command_term, "is_standing_env"):
                command_term.is_standing_env[command_env_ids] = False
            if hasattr(command_term, "is_heading_env"):
                command_term.is_heading_env[command_env_ids] = False
            self._amp_pending_reference_commands = None


ManagerBasedAmpEnv = AMPManagerBasedRLEnv


__all__ = ["AMPManagerBasedRLEnv", "ManagerBasedAmpEnv"]
