"""HW6 KL Matching 蒸馏算法。

本文件包含 TODO 7/7。
快速定位：grep -n "【作业 TODO" src/humanoid_hw6/rl/algorithms/
"""

from __future__ import annotations

import torch

from humanoid_hw6.rl.algorithms.distillation_ppo import DistillationOutput, DistillationPPO
from humanoid_hw6.rl.algorithms.distillation_utils import (
  diagonal_gaussian_kl,
  gaussian_matching_metrics,
  linear_anneal,
)


class KlMatchingPPO(DistillationPPO):
  """PPO with an additional KL-matching loss on the student actor."""

  distillation_counter_name = "num_kl_updates"

  def __init__(
    self,
    actor,
    critic,
    storage,
    *args,
    kl_coef: float = 0.1,
    kl_coef_min: float = 0.0,
    kl_coef_anneal_iters: int = 10_000,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, *args, **kwargs)
    self.kl_coef_start = float(kl_coef)
    self.kl_coef_min = float(kl_coef_min)
    self.kl_coef_anneal_iters = int(kl_coef_anneal_iters)
    self.num_kl_updates = 0

  def _current_distillation_coef(self) -> float:
    # >>> HOMEWORK_TODO_7A_START
    # 【作业 TODO 7/7 · 系数退火】KL matching 的 distill_coef
    # 与 Action Matching 同构，只是退火到 kl_coef_min 而非 0:
    # 保留一个下限意味着即使训练后期也维持轻微的分布约束，防止 Student
    # 在 PPO 回报的驱动下漂移到与 Teacher 完全不同的行为模式。
    return linear_anneal(
      self.kl_coef_start, self.kl_coef_min, self.num_kl_updates, self.kl_coef_anneal_iters
    )
    # <<< HOMEWORK_TODO_7A_END

  def _compute_distillation_output(self, batch) -> DistillationOutput:
    # >>> HOMEWORK_TODO_7B_START
    # 【作业 TODO 7/7 · 蒸馏集成】KL matching distillation
    #
    # 与 Action Matching 的本质区别：
    #   Action Matching 只对齐分布的均值（动作本身）
    #   KL Matching   对齐整个高斯分布（均值 + 标准差）
    # 后者额外传递了 Teacher 的"不确定性结构",在哪些状态下它也拿不准、
    # 该保留多大的探索幅度。这个信息对 Student 后续用 PPO 继续改进很有价值。
    #
    # 顺序有讲究：必须先取 teacher 参数，再读 student 参数。
    # output_distribution_params 是 actor 上一次前向留下的状态，
    # 由 DistillationPPO 的主更新循环在调用本方法前刚刚写入
    #（见 distillation_ppo.py 中 self.actor(...) 后取 output_distribution_params）。
    # 若在这里再调一次 self.actor(...)，会覆盖掉那次前向的分布参数，
    # 导致 PPO 的 surrogate loss 与蒸馏 loss 基于不同的前向结果。
    # 所以这里只读不算。
    with torch.no_grad():
      teacher_mean, teacher_std = self.actor.teacher_distribution_params(batch.observations)
    student_mean, student_std = self.actor.output_distribution_params

    loss = diagonal_gaussian_kl(teacher_mean, teacher_std, student_mean, student_std)
    metrics = gaussian_matching_metrics(teacher_mean, teacher_std, student_mean, student_std)
    return DistillationOutput(loss=loss, metrics=metrics)
    # <<< HOMEWORK_TODO_7B_END

  def _format_distillation_metrics(self, mean_distill_loss: float) -> dict[str, float]:
    return {"kl": mean_distill_loss}

  def _on_distillation_update_end(self) -> None:
    self.num_kl_updates += 1
