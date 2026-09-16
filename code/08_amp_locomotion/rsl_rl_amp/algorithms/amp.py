"""PPO with an adversarial motion-prior reward and discriminator update."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl_amp.algorithms.discriminator import AMPDiscriminator
from rsl_rl_amp.algorithms.amp_stability import DiscriminatorBalanceController
from rsl_rl_amp.algorithms.ppo import PPO
from rsl_rl_amp.storage import ReplayBuffer
from rsl_rl_amp.utils import resolve_obs_groups


class AMP(PPO):
    """Augment PPO with a learned style reward from expert motion windows."""

    checkpoint_version = 3
    prior_scope = "task_agnostic_state_transition_v1"

    def __init__(self, *args, amp_cfg: dict, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not amp_cfg:
            raise ValueError("AMP requires a non-empty amp_cfg block.")
        self.amp_cfg = amp_cfg
        self.amp_observation_groups = list(amp_cfg["observation_groups"])
        self.frame_dim = int(amp_cfg["frame_dim"])
        self.history_steps = int(amp_cfg["history_steps"])
        self.step_dt = float(amp_cfg["step_dt"])
        self.expert_sampler = amp_cfg["expert_sampler"]
        self.motion_profile = str(amp_cfg.get("motion_profile", "unspecified"))

        discriminator_cfg = dict(amp_cfg.get("discriminator", {}))
        self.discriminator = AMPDiscriminator(
            frame_dim=self.frame_dim,
            history_steps=self.history_steps,
            **discriminator_cfg,
        ).to(self.device)
        self._raw_discriminator = self.discriminator
        discriminator_weight_decay = float(amp_cfg.get("discriminator_weight_decay", 0.0))
        output_weight_decay = float(
            amp_cfg.get("discriminator_output_weight_decay", discriminator_weight_decay)
        )
        output_parameters = list(self.discriminator.network[-1].parameters())
        output_parameter_ids = {id(parameter) for parameter in output_parameters}
        trunk_parameters = [
            parameter
            for parameter in self.discriminator.parameters()
            if id(parameter) not in output_parameter_ids
        ]
        self.discriminator_optimizer = torch.optim.AdamW(
            [
                {"params": trunk_parameters, "weight_decay": discriminator_weight_decay},
                {"params": output_parameters, "weight_decay": output_weight_decay},
            ],
            lr=float(amp_cfg.get("discriminator_learning_rate", 1.0e-4)),
        )
        self.discriminator_max_grad_norm = float(amp_cfg.get("discriminator_max_grad_norm", 1.0))
        self.discriminator_batch_size = int(amp_cfg.get("discriminator_batch_size", 8192))
        self.normalization_batch_size = int(amp_cfg.get("normalization_batch_size", 8192))
        if self.discriminator_batch_size < 1 or self.normalization_batch_size < 1:
            raise ValueError("AMP discriminator and normalization batch sizes must be positive.")
        max_discriminator_updates = int(amp_cfg.get("discriminator_updates", 4))
        balance_cfg = dict(amp_cfg.get("discriminator_balance", {}))
        self.discriminator_balance = DiscriminatorBalanceController(
            max_updates=max_discriminator_updates,
            **balance_cfg,
        )
        self.replay_buffer = ReplayBuffer(
            capacity=int(amp_cfg.get("replay_buffer_size", 200_000)),
            feature_dim=self.frame_dim * self.history_steps,
            device=self.device,
        )
        self.last_task_rewards = torch.zeros(self.storage.num_envs, device=self.device)
        self.last_style_rewards = torch.zeros(self.storage.num_envs, device=self.device)
        self.last_mixed_rewards = torch.zeros(self.storage.num_envs, device=self.device)
        self.last_reward_components = {
            "task": self.last_task_rewards,
            "style": self.last_style_rewards,
            "mixed": self.last_mixed_rewards,
        }
        self.last_discriminator_scores = torch.zeros(self.storage.num_envs, device=self.device)
        self._amp_history_age = torch.zeros(
            self.storage.num_envs, dtype=torch.long, device=self.device
        )
        self._normalization_policy_samples: list[torch.Tensor] = []
        self._normalization_samples_per_step = math.ceil(
            self.normalization_batch_size / self.storage.num_transitions_per_env
        )
        self._reset_amp_metrics()

    def _reset_amp_metrics(self) -> None:
        self._amp_metric_count = 0
        self._amp_style_reward_sum = 0.0
        self._amp_task_reward_sum = 0.0
        self._amp_discriminator_score_sum = 0.0
        self._amp_valid_window_count = 0
        self._amp_window_candidate_count = 0
        self._amp_style_contribution_sum = 0.0
        self._amp_task_contribution_sum = 0.0

    def _amp_sequence(self, obs: TensorDict) -> torch.Tensor:
        sequences = []
        for group in self.amp_observation_groups:
            value = obs[group]
            if value.ndim != 3:
                raise ValueError(
                    f"AMP observation group '{group}' must have shape [batch, history, features], got {tuple(value.shape)}."
                )
            if value.shape[1] != self.history_steps:
                raise ValueError(
                    f"AMP observation group '{group}' has {value.shape[1]} history steps; expected {self.history_steps}."
                )
            sequences.append(value)
        return torch.cat(sequences, dim=-1)

    def _remember_fresh_policy_samples(self, flattened_windows: torch.Tensor) -> None:
        """Keep a bounded per-step sample of new policy windows for normalization."""
        count = min(flattened_windows.shape[0], self._normalization_samples_per_step)
        if count == 0:
            return
        if count < flattened_windows.shape[0]:
            indices = torch.randperm(flattened_windows.shape[0], device=self.device)[:count]
            flattened_windows = flattened_windows[indices]
        self._normalization_policy_samples.append(flattened_windows.detach())

    @torch.no_grad()
    def _expert_samples_for(self, policy_samples: torch.Tensor) -> torch.Tensor:
        """Draw an unbiased expert batch independent of policy goals or state."""
        return self.expert_sampler.sample(policy_samples.shape[0]).to(self.device)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        amp_sequence = self._amp_sequence(obs)
        next_age = self._amp_history_age + 1
        valid_windows = next_age >= self.history_steps
        style_rewards = torch.zeros(amp_sequence.shape[0], device=self.device)
        scores = torch.zeros_like(style_rewards)
        if torch.any(valid_windows):
            valid_sequence = amp_sequence[valid_windows]
            valid_style_rewards, valid_scores = self.discriminator.style_reward(valid_sequence, self.step_dt)
            style_rewards[valid_windows] = valid_style_rewards
            scores[valid_windows] = valid_scores
            flattened_windows = valid_sequence.reshape(valid_sequence.shape[0], -1)
            self.replay_buffer.append(flattened_windows)
            self._remember_fresh_policy_samples(flattened_windows)
        self.last_task_rewards = rewards
        self.last_style_rewards = style_rewards
        self.last_discriminator_scores = scores
        self._amp_metric_count += style_rewards.numel()
        self._amp_style_reward_sum += float(style_rewards.sum())
        self._amp_task_reward_sum += float(rewards.sum())
        self._amp_discriminator_score_sum += float(scores.sum())
        self._amp_valid_window_count += int(valid_windows.sum())
        self._amp_window_candidate_count += valid_windows.numel()
        task_weight = self.discriminator.task_reward_weight
        self._amp_style_contribution_sum += float(
            ((1.0 - task_weight) * style_rewards).abs().sum()
        )
        self._amp_task_contribution_sum += float((task_weight * rewards).abs().sum())
        # r_t = (1 - alpha) * r_amp + alpha * r_task
        mixed_rewards = self.discriminator.mix_rewards(style_rewards, rewards)
        # 历史不足 history_steps 帧的环境拿不到有意义的判别器分数
        # （style_rewards 那一路是补的 0），此时必须退回纯任务奖励。
        # 否则 episode 刚重置的那几步会被灌进一个"风格极差"的假信号。
        mixed_rewards = torch.where(valid_windows, mixed_rewards, rewards)
        self.last_mixed_rewards = mixed_rewards
        self.last_reward_components = {
            "task": self.last_task_rewards,
            "style": self.last_style_rewards,
            "mixed": self.last_mixed_rewards,
        }
        super().process_env_step(obs, mixed_rewards, dones, extras)
        self._amp_history_age = torch.where(dones.bool(), torch.zeros_like(next_age), next_age)

    @torch.no_grad()
    def _probe_discriminator(self, batch_size: int) -> dict[str, float]:
        policy_samples = self.replay_buffer.sample(batch_size)
        expert_samples = self._expert_samples_for(policy_samples)
        policy_scores = self.discriminator(policy_samples)
        expert_scores = self.discriminator(expert_samples)
        metrics = {
            "policy_score": float(policy_scores.mean()),
            "expert_score": float(expert_scores.mean()),
            "policy_score_std": float(policy_scores.std(unbiased=False)),
            "expert_score_std": float(expert_scores.std(unbiased=False)),
        }
        if self.is_multi_gpu:
            values = torch.tensor(list(metrics.values()), device=self.device)
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
            values /= self.gpu_world_size
            metrics = dict(zip(metrics, values.tolist()))
        return metrics

    @torch.no_grad()
    def _update_amp_normalizer(self) -> None:
        if not self._normalization_policy_samples:
            return
        policy_samples = torch.cat(self._normalization_policy_samples, dim=0)
        if policy_samples.shape[0] > self.normalization_batch_size:
            indices = torch.randperm(policy_samples.shape[0], device=self.device)[
                : self.normalization_batch_size
            ]
            policy_samples = policy_samples[indices]
        expert_samples = self._expert_samples_for(policy_samples)
        self.discriminator.update_normalization(policy_samples, expert_samples)
        self._normalization_policy_samples.clear()

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        self._update_amp_normalizer()

        score_keys = ("policy_score", "expert_score", "policy_score_std", "expert_score_std")
        probe = {key: 0.0 for key in score_keys}
        updates = 0
        if len(self.replay_buffer) > 0:
            probe_size = min(self.discriminator_batch_size, len(self.replay_buffer))
            probe = self._probe_discriminator(probe_size)
            updates = self.discriminator_balance.recommended_updates(
                probe["policy_score"], probe["expert_score"]
            )

        totals = {
            "total": 0.0,
            "classification": 0.0,
            "logit_regularization": 0.0,
            "gradient_penalty": 0.0,
            "policy_score": 0.0,
            "expert_score": 0.0,
            "policy_score_std": 0.0,
            "expert_score_std": 0.0,
        }
        actual_updates = 0
        for _ in range(updates):
            policy_samples = self.replay_buffer.sample(self.discriminator_batch_size)
            expert_samples = self._expert_samples_for(policy_samples)
            losses = self.discriminator.loss(policy_samples, expert_samples)
            self.discriminator_optimizer.zero_grad()
            losses["total"].backward()
            if self.is_multi_gpu:
                self._reduce_discriminator_gradients()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.discriminator_max_grad_norm)
            self.discriminator_optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach())
            actual_updates += 1
            if self.discriminator_balance.recommended_updates(
                float(losses["policy_score"]), float(losses["expert_score"])
            ) == 0:
                break

        divisor = max(1, actual_updates)
        score_metrics = (
            {key: totals[key] / divisor for key in score_keys}
            if actual_updates
            else probe
        )
        style_mean = self._amp_style_reward_sum / max(1, self._amp_metric_count)
        task_mean = self._amp_task_reward_sum / max(1, self._amp_metric_count)
        style_task_ratio = self._amp_style_contribution_sum / max(
            1.0e-8, self._amp_task_contribution_sum
        )
        loss_dict.update(
            {
                "amp_discriminator": totals["total"] / divisor,
                "amp_classification": totals["classification"] / divisor,
                "amp_logit_regularization": totals["logit_regularization"] / divisor,
                "amp_gradient_penalty": totals["gradient_penalty"] / divisor,
                "amp_policy_score": score_metrics["policy_score"],
                "amp_expert_score": score_metrics["expert_score"],
                "amp_policy_score_std": score_metrics["policy_score_std"],
                "amp_expert_score_std": score_metrics["expert_score_std"],
                "amp_score_gap": score_metrics["expert_score"] - score_metrics["policy_score"],
                "amp_style_reward": style_mean,
                "amp_task_reward": task_mean,
                "amp_discriminator_score": self._amp_discriminator_score_sum / max(1, self._amp_metric_count),
                "amp_valid_window_fraction": self._amp_valid_window_count
                / max(1, self._amp_window_candidate_count),
                "amp_normalizer_count": float(self.discriminator.normalizer.count.item()),
                "amp_style_task_ratio": style_task_ratio,
                "amp_discriminator_updates": float(actual_updates),
            }
        )
        self._reset_amp_metrics()
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        state = super().save()
        state.update(
            {
                "amp_discriminator_state_dict": self._raw_discriminator.state_dict(),
                "amp_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
                "amp_motion_profile": self.motion_profile,
                "amp_checkpoint_version": self.checkpoint_version,
                "amp_history_steps": self.history_steps,
                "amp_reward_mode": self.discriminator.reward_mode,
                "amp_prior_scope": self.prior_scope,
            }
        )
        return state

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        restore_amp = load_cfg is None or load_cfg.get("amp", True)
        restore_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
        checkpoint_profile = str(loaded_dict.get("amp_motion_profile", "unspecified"))
        if restore_amp and strict and "amp_discriminator_state_dict" in loaded_dict:
            checkpoint_version = loaded_dict.get("amp_checkpoint_version")
            if checkpoint_version != self.checkpoint_version:
                raise ValueError(
                    "Cannot strictly load a legacy AMP checkpoint without the current stability metadata; "
                    "start a fresh AMP run or load only actor/critic weights with amp=False."
                )
            checkpoint_history = int(loaded_dict.get("amp_history_steps", -1))
            if checkpoint_history != self.history_steps:
                raise ValueError(
                    f"AMP checkpoint uses history length {checkpoint_history}, but the current learner expects "
                    f"{self.history_steps}."
                )
            checkpoint_reward_mode = str(loaded_dict.get("amp_reward_mode", "unspecified"))
            if checkpoint_reward_mode != self.discriminator.reward_mode:
                raise ValueError(
                    f"AMP checkpoint reward mode '{checkpoint_reward_mode}' does not match current mode "
                    f"'{self.discriminator.reward_mode}'."
                )
            checkpoint_prior_scope = str(loaded_dict.get("amp_prior_scope", "unspecified"))
            if checkpoint_prior_scope != self.prior_scope:
                raise ValueError(
                    f"AMP checkpoint prior scope '{checkpoint_prior_scope}' does not match current prior scope "
                    f"'{self.prior_scope}'. Start a fresh run."
                )
        if (
            restore_amp
            and strict
            and self.motion_profile != "unspecified"
            and checkpoint_profile != "unspecified"
            and checkpoint_profile != self.motion_profile
        ):
            raise ValueError(
                f"AMP checkpoint profile '{checkpoint_profile}' does not match current profile '{self.motion_profile}'."
            )
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if restore_amp and "amp_discriminator_state_dict" in loaded_dict:
            self._raw_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
        if restore_optimizer and "amp_optimizer_state_dict" in loaded_dict:
            self.discriminator_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
        return load_iteration

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        states = [self._raw_discriminator.state_dict()]
        torch.distributed.broadcast_object_list(states, src=0)
        self._raw_discriminator.load_state_dict(states[0])

    def _reduce_discriminator_gradients(self) -> None:
        gradients = [parameter.grad for parameter in self.discriminator.parameters() if parameter.grad is not None]
        if not gradients:
            return
        flattened = torch.cat([gradient.reshape(-1) for gradient in gradients])
        torch.distributed.all_reduce(flattened, op=torch.distributed.ReduceOp.SUM)
        flattened /= self.gpu_world_size
        offset = 0
        for gradient in gradients:
            count = gradient.numel()
            gradient.copy_(flattened[offset : offset + count].view_as(gradient))
            offset += count

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "AMP":
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg.get("obs_groups", {}), ["amp"])
        amp_groups = cfg["obs_groups"]["amp"]
        history_steps = None
        frame_dim = 0
        for group in amp_groups:
            value = obs[group]
            if value.ndim != 3:
                raise ValueError(
                    f"AMP observation group '{group}' must have shape [batch, history, features], got {tuple(value.shape)}."
                )
            if history_steps is None:
                history_steps = value.shape[1]
            elif value.shape[1] != history_steps:
                raise ValueError("All AMP observation groups must use the same history length.")
            frame_dim += value.shape[2]

        sampler = getattr(env, "amp_expert_sampler", None)
        if sampler is None and hasattr(env, "unwrapped"):
            sampler = getattr(env.unwrapped, "amp_expert_sampler", None)
        if sampler is None:
            raise ValueError("AMP environment must expose an 'amp_expert_sampler' with a sample(batch_size) method.")

        step_dt = getattr(env, "step_dt", None)
        if step_dt is None and hasattr(env, "unwrapped"):
            step_dt = getattr(env.unwrapped, "step_dt", None)
        if step_dt is None:
            raise ValueError("AMP environment must expose step_dt.")

        amp_cfg = cfg["algorithm"].setdefault("amp_cfg", {})
        amp_cfg.update(
            {
                "observation_groups": amp_groups,
                "frame_dim": frame_dim,
                "history_steps": history_steps,
                "step_dt": step_dt,
                "expert_sampler": sampler,
                "motion_profile": getattr(getattr(sampler, "cfg", None), "profile_name", "unspecified"),
            }
        )
        return PPO.construct_algorithm(obs, env, cfg, device)  # type: ignore[return-value]
