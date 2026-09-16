from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def estimated_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the latest actor-side estimated base linear velocity.

    The PPO pipeline updates this buffer every step before actor evaluation.
    When unavailable (e.g., at reset/export), this term defaults to zeros.
    """
    if not hasattr(env, "_estimated_lin_vel_obs"):
        env._estimated_lin_vel_obs = torch.zeros(env.num_envs, 3, device=env.device)
    return env._estimated_lin_vel_obs
