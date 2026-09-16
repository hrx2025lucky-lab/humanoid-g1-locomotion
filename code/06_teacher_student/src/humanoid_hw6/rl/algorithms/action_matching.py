"""HW6 Action Matching 蒸馏算法。

本文件包含 TODO 6/7。
快速定位：grep -n "【作业 TODO" src/humanoid_hw6/rl/algorithms/
"""

from __future__ import annotations

import torch

from humanoid_hw6.rl.algorithms.distillation_ppo import DistillationOutput, DistillationPPO
from humanoid_hw6.rl.algorithms.distillation_utils import (
  action_matching_metrics,
  action_regression_loss,
  linear_anneal,
)


class ActionMatchingPPO(DistillationPPO):
  """PPO with an additional teacher action-matching loss on the student actor."""

  distillation_counter_name = "num_bc_updates"

  def __init__(
    self,
    actor,
    critic,
    storage,
    *args,
    bc_coef_start: float = 1.0,
    bc_coef_end: float = 0.0,
    bc_anneal_iters: int = 10_000,
    bc_loss_type: str = "mse",
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, *args, **kwargs)
    self.bc_coef_start = float(bc_coef_start)
    self.bc_coef_end = float(bc_coef_end)
    self.bc_anneal_iters = int(bc_anneal_iters)
    self.bc_loss_type = str(bc_loss_type)
    self.num_bc_updates = 0

  def _current_distillation_coef(self) -> float:
    # >>> HOMEWORK_TODO_6A_START
    # 【作业 TODO 6/7 · 系数退火】Action matching 的 distill_coef
    # 用 num_bc_updates 而不是环境步数：退火节奏应当跟随参数更新次数，
    # 这样改 num_envs 或 rollout 长度时 schedule 的语义保持不变。
    return linear_anneal(
      self.bc_coef_start, self.bc_coef_end, self.num_bc_updates, self.bc_anneal_iters
    )
    # <<< HOMEWORK_TODO_6A_END

  def _compute_distillation_output(self, batch) -> DistillationOutput:
    # >>> HOMEWORK_TODO_6B_START
    # 【作业 TODO 6/7 · 蒸馏集成】Action matching distillation
    #
    # Teacher 前向必须包在 no_grad 里。StudentTeacherActor 虽然已把 teacher
    # 参数设为 requires_grad=False，但不加 no_grad 仍会为 teacher 的中间激活
    # 构建计算图并保留下来,白白占显存，且在某些写法下会把梯度错误地
    # 引流回共享模块。teacher 在这里的角色是数据源，不是网络。
    #
    # 注意 Teacher 与 Student 吃的是同一个 batch.observations。
    # 二者内部各自取自己需要的切片：teacher 用带未来 20 帧参考动作的
    # 特权观测，student 用只含当前帧的受限观测。所以这里不能手动分发观测，
    # 交给 actor 内部处理即可,手动切分正是助教点名的高频错误。
    with torch.no_grad():
      teacher_actions = self.actor.teacher_forward(batch.observations)
    student_actions = self.actor(batch.observations)

    loss = action_regression_loss(student_actions, teacher_actions, self.bc_loss_type)
    metrics = action_matching_metrics(student_actions, teacher_actions)
    return DistillationOutput(loss=loss, metrics=metrics)
    # <<< HOMEWORK_TODO_6B_END

  def _format_distillation_metrics(self, mean_distill_loss: float) -> dict[str, float]:
    return {"bc": mean_distill_loss}

  def _on_distillation_update_end(self) -> None:
    self.num_bc_updates += 1
