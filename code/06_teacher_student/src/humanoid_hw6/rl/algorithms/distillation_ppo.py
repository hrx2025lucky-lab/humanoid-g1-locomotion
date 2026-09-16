from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO

from humanoid_hw6.rl.models.student_teacher import StudentTeacherActor


@dataclass
class DistillationOutput:
  """Scalar distillation loss plus detached diagnostics for logging."""

  loss: torch.Tensor
  metrics: dict[str, float] = field(default_factory=dict)


class DistillationPPO(PPO):
  """Shared PPO update loop for teacher-regularized student training."""

  actor: StudentTeacherActor
  distillation_counter_name: str

  def _require_loaded_teacher(self, coef: float, task_name: str) -> None:
    if coef > 0.0 and not self.actor.loaded_teacher:
      raise ValueError(
        f"Teacher weights are not loaded. Provide --agent.teacher-checkpoint-file "
        f"before running {task_name}."
      )

  def _compute_distillation_output(self, batch) -> DistillationOutput:
    raise NotImplementedError

  def update(self) -> dict[str, float]:  # noqa: C901
    if self.rnd is not None or self.symmetry is not None:
      raise NotImplementedError(
        "DistillationPPO supports the HW6 setup without RND/symmetry."
      )

    distill_coef = self._current_distillation_coef()
    self._require_loaded_teacher(distill_coef, self.__class__.__name__)

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_distill_loss = 0.0
    distill_metric_sums: dict[str, float] = {}

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
      )
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)

          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size

          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()

          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      if distill_coef > 0.0:
        distill_output = self._compute_distillation_output(batch)
        distill_loss = distill_output.loss
        for key, value in distill_output.metrics.items():
          distill_metric_sums[key] = distill_metric_sums.get(key, 0.0) + float(value)
      else:
        distill_loss = torch.zeros((), device=self.device)

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
        + distill_coef * distill_loss
      )

      self.optimizer.zero_grad()
      loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_distill_loss += distill_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    metrics = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "distill_coef": distill_coef,
    }
    metrics.update(self._format_distillation_metrics(mean_distill_loss / num_updates))
    for key, total in distill_metric_sums.items():
      metrics[key] = total / num_updates
    self.storage.clear()
    self._on_distillation_update_end()
    return metrics

  def _current_distillation_coef(self) -> float:
    raise NotImplementedError

  def _format_distillation_metrics(self, mean_distill_loss: float) -> dict[str, float]:
    return {"distill": mean_distill_loss}

  def _on_distillation_update_end(self) -> None:
    return None

  def save(self) -> dict:
    saved_dict = super().save()
    saved_dict["distillation_state"] = {
      "counter_name": self.distillation_counter_name,
      "num_updates": getattr(self, self.distillation_counter_name),
    }
    return saved_dict

  def _restore_distillation_state(self, loaded_dict: dict) -> None:
    state = loaded_dict.get("distillation_state")
    if state is None:
      # Old unresumed course runs saved the zero-based index after each update.
      # Legacy resumed chains can repeat that index; migrate those checkpoints
      # using verified rollout counters before continuing a formal experiment.
      import warnings
      count = int(loaded_dict["iter"]) + 1
      warnings.warn(
        "Legacy checkpoint has no distillation_state; using iter + 1. "
        "Verify/migrate the update count before resuming a chained run.",
        RuntimeWarning,
        stacklevel=2,
      )
    else:
      if state["counter_name"] != self.distillation_counter_name:
        raise ValueError("Checkpoint distillation method does not match this algorithm")
      count = state["num_updates"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
      raise ValueError("Distillation update count must be a nonnegative integer")
    setattr(self, self.distillation_counter_name, count)

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if (
      "student_state_dict" in loaded_dict
      and "teacher_state_dict" in loaded_dict
      and "actor_state_dict" not in loaded_dict
    ):
      if load_cfg is None:
        load_cfg = {
          "actor": True,
          "critic": False,
          "optimizer": False,
          "iteration": False,
          "rnd": False,
        }
      if load_cfg.get("actor"):
        self.actor.load_distillation_state(
          loaded_dict["student_state_dict"],
          loaded_dict["teacher_state_dict"],
          strict=strict,
        )
      return load_cfg.get("iteration", False)

    load_iteration = super().load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self._restore_distillation_state(loaded_dict)
    # Upstream restores optimizer groups but not PPO's adaptive-LR scalar.
    if load_cfg is None or load_cfg.get("optimizer", False):
      self.learning_rate = float(self.optimizer.param_groups[0]["lr"])
    return load_iteration
