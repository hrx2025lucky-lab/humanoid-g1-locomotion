"""HW6 Action Matching 蒸馏算法。

本文件包含 实验 6/7。
快速定位：grep -n "【实验 实验" src/humanoid_hw6/rl/algorithms/
"""
from __future__ import annotations
import torch
from humanoid_hw6.rl.algorithms.distillation_ppo import DistillationOutput, DistillationPPO
from humanoid_hw6.rl.algorithms.distillation_utils import action_matching_metrics, action_regression_loss, linear_anneal

class ActionMatchingPPO(DistillationPPO):
    """PPO with an additional teacher action-matching loss on the student actor."""
    distillation_counter_name = 'num_bc_updates'

    def __init__(self, actor, critic, storage, *args, bc_coef_start: float=1.0, bc_coef_end: float=0.0, bc_anneal_iters: int=10000, bc_loss_type: str='mse', **kwargs) -> None:
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.bc_coef_start = float(bc_coef_start)
        self.bc_coef_end = float(bc_coef_end)
        self.bc_anneal_iters = int(bc_anneal_iters)
        self.bc_loss_type = str(bc_loss_type)
        self.num_bc_updates = 0

    def _current_distillation_coef(self) -> float:
        return linear_anneal(self.bc_coef_start, self.bc_coef_end, self.num_bc_updates, self.bc_anneal_iters)

    def _compute_distillation_output(self, batch) -> DistillationOutput:
        with torch.no_grad():
            teacher_actions = self.actor.teacher_forward(batch.observations)
        student_actions = self.actor(batch.observations)
        loss = action_regression_loss(student_actions, teacher_actions, self.bc_loss_type)
        metrics = action_matching_metrics(student_actions, teacher_actions)
        return DistillationOutput(loss=loss, metrics=metrics)

    def _format_distillation_metrics(self, mean_distill_loss: float) -> dict[str, float]:
        return {'bc': mean_distill_loss}

    def _on_distillation_update_end(self) -> None:
        self.num_bc_updates += 1
