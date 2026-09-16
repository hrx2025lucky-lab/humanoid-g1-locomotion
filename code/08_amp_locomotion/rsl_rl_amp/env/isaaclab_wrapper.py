"""Duck-typed Isaac Lab adapters for the local learning interfaces."""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl_amp.env.vec_env import VecEnv


class IsaacLabVecEnvWrapper(VecEnv):
    """Adapt an Isaac Lab Gym environment to :class:`rsl_rl_amp.env.VecEnv`."""

    def __init__(self, env, clip_actions: float | None = None) -> None:
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        self._latest_raw_obs, self._latest_extras = self.env.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def step_dt(self) -> float:
        return float(self.unwrapped.step_dt)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def _tensor_dict(self, observations: dict[str, torch.Tensor]) -> TensorDict:
        return TensorDict(observations, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._tensor_dict(dict(self._latest_raw_obs))

    def reset(self) -> tuple[TensorDict, dict]:
        self._latest_raw_obs, extras = self.env.reset()
        self._latest_extras = extras
        return self.get_observations(), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        raw_obs, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).long()
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        self._latest_raw_obs = raw_obs
        self._latest_extras = extras
        return self._tensor_dict(dict(raw_obs)), rewards, dones, extras

    def close(self):
        return self.env.close()


class AMPIsaacLabVecEnvWrapper(IsaacLabVecEnvWrapper):
    """Maintain AMP history while preserving the physical terminal frame."""

    def __init__(
        self,
        env,
        history_steps: int | None = None,
        amp_group: str = "amp",
        clip_actions: float | None = None,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)
        self.amp_group = amp_group
        self.history_steps = int(history_steps or getattr(self.cfg, "amp_history_steps", 1))
        if self.history_steps < 1:
            raise ValueError("AMP history_steps must be positive.")
        initial_frame = self._latest_raw_obs[self.amp_group]
        if initial_frame.ndim != 2:
            raise ValueError(
                f"Raw AMP observation must be [num_envs, frame_dim], got {tuple(initial_frame.shape)}."
            )
        self._amp_history = initial_frame.unsqueeze(1).repeat(1, self.history_steps, 1)

    @property
    def amp_expert_sampler(self):
        return self.unwrapped.amp_expert_sampler

    def get_observations(self) -> TensorDict:
        observations = dict(self._latest_raw_obs)
        observations[self.amp_group] = self._amp_history
        return self._tensor_dict(observations)

    def reset(self) -> tuple[TensorDict, dict]:
        self._latest_raw_obs, extras = self.env.reset()
        initial_frame = self._latest_raw_obs[self.amp_group]
        self._amp_history = initial_frame.unsqueeze(1).repeat(1, self.history_steps, 1)
        return self.get_observations(), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        raw_obs, rewards, terminated, truncated, extras = self.env.step(actions)
        dones_bool = terminated | truncated
        dones = dones_bool.long()
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        reset_frame = raw_obs[self.amp_group]
        transition_frame = reset_frame.clone()
        if torch.any(dones_bool):
            transition_frame[dones_bool] = self.unwrapped.terminal_amp_frames[dones_bool]
        transition_history = torch.roll(self._amp_history, shifts=-1, dims=1)
        transition_history[:, -1] = transition_frame

        next_history = transition_history.clone()
        if torch.any(dones_bool):
            next_history[dones_bool] = reset_frame[dones_bool].unsqueeze(1).repeat(1, self.history_steps, 1)
        self._amp_history = next_history
        self._latest_raw_obs = raw_obs
        self._latest_extras = extras

        transition_obs = dict(raw_obs)
        transition_obs[self.amp_group] = transition_history
        return self._tensor_dict(transition_obs), rewards, dones, extras
