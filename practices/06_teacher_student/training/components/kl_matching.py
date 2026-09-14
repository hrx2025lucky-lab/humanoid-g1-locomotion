"""HW6 KL Matching 蒸馏算法。

本文件包含 实验 7/7。
快速定位：grep -n "【实验 实验" src/humanoid_hw6/rl/algorithms/
"""
from __future__ import annotations
import torch
from humanoid_hw6.rl.algorithms.distillation_ppo import DistillationOutput, DistillationPPO
from humanoid_hw6.rl.algorithms.distillation_utils import diagonal_gaussian_kl, gaussian_matching_metrics, linear_anneal

class KlMatchingPPO(DistillationPPO):
    """PPO with an additional KL-matching loss on the student actor."""
    distillation_counter_name = 'num_kl_updates'

    def __init__(self, actor, critic, storage, *args, kl_coef: float=0.1, kl_coef_min: float=0.0, kl_coef_anneal_iters: int=10000, **kwargs) -> None:
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.kl_coef_start = float(kl_coef)
        self.kl_coef_min = float(kl_coef_min)
        self.kl_coef_anneal_iters = int(kl_coef_anneal_iters)
        self.num_kl_updates = 0

    def _current_distillation_coef(self) -> float:
        return linear_anneal(self.kl_coef_start, self.kl_coef_min, self.num_kl_updates, self.kl_coef_anneal_iters)

    def _compute_distillation_output(self, batch) -> DistillationOutput:
        with torch.no_grad():
            (teacher_mean, teacher_std) = self.actor.teacher_distribution_params(batch.observations)
        (student_mean, student_std) = self.actor.output_distribution_params
        loss = diagonal_gaussian_kl(teacher_mean, teacher_std, student_mean, student_std)
        metrics = gaussian_matching_metrics(teacher_mean, teacher_std, student_mean, student_std)
        return DistillationOutput(loss=loss, metrics=metrics)

    def _format_distillation_metrics(self, mean_distill_loss: float) -> dict[str, float]:
        return {'kl': mean_distill_loss}

    def _on_distillation_update_end(self) -> None:
        self.num_kl_updates += 1
