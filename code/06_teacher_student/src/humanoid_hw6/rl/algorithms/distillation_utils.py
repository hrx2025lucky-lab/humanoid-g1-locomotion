"""HW6 蒸馏共享数学工具。

本文件包含 TODO 1/7 至 TODO 5/7。
快速定位：grep -n "【作业 TODO" src/humanoid_hw6/rl/algorithms/
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_anneal(start: float, end: float, step: int, duration: int) -> float:
  """在 ``duration`` 步内，从 ``start`` 线性退火到 ``end``。"""
  # >>> HOMEWORK_TODO_1_START
  # 【作业 TODO 1/7】线性蒸馏系数退火
  #
  # 蒸馏系数控制"听老师的"与"自己探索"之间的权重：
  #   训练初期 coef 大 → Student 主要模仿 Teacher，快速获得可用策略
  #   训练后期 coef 小 → 让 PPO 的环境回报主导，Student 适应自己受限的观测
  #
  # 为什么必须退火而非固定：Teacher 能看到未来 20 帧参考动作，Student 看不到。
  # 在信息不对称的前提下，Student 永远无法完全复制 Teacher 的动作:
  # 硬压着它去匹配，反而阻止它学出"在只有当前帧时该怎么办"的补偿策略。
  # 蒸馏过强限制适应，过弱则得不到有效指导，退火是二者的折中。
  #
  # duration <= 0 直接返回终值：既避免除零，也让"关闭退火"有一个明确入口。
  if duration <= 0:
    return end
  progress = min(max(step / duration, 0.0), 1.0)
  return start + (end - start) * progress
  # <<< HOMEWORK_TODO_1_END


def action_regression_loss(
  student_actions: torch.Tensor,
  teacher_actions: torch.Tensor,
  loss_type: str = "mse",
) -> torch.Tensor:
  """student 与 teacher action 之间的标量回归损失。"""
  # >>> HOMEWORK_TODO_2_START
  # 【作业 TODO 2/7】Action 回归损失
  #
  # 这是 Action Matching 的监督信号：让 Student 在同一状态下输出接近 Teacher 的动作。
  #
  # MSE  对误差平方加权，大偏差被放大，收敛快但对离群动作敏感
  # Huber 小误差处同 MSE、大误差处退化为 L1，对 Teacher 偶发的极端动作更稳健
  #
  # 注意 loss 只对 student_actions 反传,teacher_actions 是 no_grad 下算出的
  # 常量张量（见 TODO 6），这里不需要也不能对它求梯度。
  if loss_type == "mse":
    return F.mse_loss(student_actions, teacher_actions)
  if loss_type == "huber":
    return F.huber_loss(student_actions, teacher_actions)
  raise ValueError(f"Unsupported loss_type '{loss_type}', expected 'mse' or 'huber'.")
  # <<< HOMEWORK_TODO_2_END


def action_matching_metrics(
  student_actions: torch.Tensor,
  teacher_actions: torch.Tensor,
) -> dict[str, float]:
  """用于日志记录的 action 级诊断指标（不参与梯度）。"""
  # >>> HOMEWORK_TODO_3_START
  # 【作业 TODO 3/7】Action matching 诊断指标
  #
  # 只用于 TensorBoard 日志，不参与反向传播,所以整段包在 no_grad 里，
  # 并用 .item() 取出 Python float，彻底切断计算图。
  # 若忘了 detach，这些张量会把整张图挂在 metrics 字典里不被释放，显存缓慢泄漏。
  #
  # 为什么 loss 之外还要单独记 MAE/RMSE：MSE loss 是被平方放大过的，
  # 数值大小不直观。MAE 与动作本身同量纲，能直接回答"平均差多少"；
  # RMSE 对大偏差更敏感，两者一起看能判断误差是均匀分布还是被少数离群点主导。
  #
  # 这与实践 2 的教训一致：只看总 loss 会错过任务层面的真实表现。
  with torch.no_grad():
    diff = student_actions - teacher_actions
    return {
      "action_mae": diff.abs().mean().item(),
      "action_rmse": diff.pow(2).mean().sqrt().item(),
    }
  # <<< HOMEWORK_TODO_3_END


def diagonal_gaussian_kl(
  teacher_mean: torch.Tensor,
  teacher_std: torch.Tensor,
  student_mean: torch.Tensor,
  student_std: torch.Tensor,
  std_eps: float = 1e-6,
) -> torch.Tensor:
  """对角 Gaussian 分布的 mean KL(teacher || student)。"""
  # >>> HOMEWORK_TODO_4_START
  # 【作业 TODO 4/7】对角 Gaussian 的 analytic KL
  #
  # 逐维闭式解（两个一维正态之间的 KL）：
  #     log(σ_s/σ_t) + (σ_t² + (μ_t - μ_s)²) / (2σ_s²) - 1/2
  # 对角协方差意味着各维独立，所以对最后一维求和即得联合分布的 KL，
  # 再对 batch 取均值得到标量 loss。
  #
  # 方向必须是 forward KL(teacher ‖ student)，不能写反。
  # forward KL 在 teacher 概率高而 student 概率低的地方给出巨大惩罚，
  # 迫使 student 的分布覆盖teacher 的全部支撑集（mass-covering）。
  # reverse KL(student ‖ teacher) 则是 mode-seeking：student 会缩到
  # teacher 的某一个峰上，丢掉其余行为模式。对蒸馏而言我们要的是前者:
  # Student 应当继承 Teacher 的完整行为分布，而不是只学会其中一种走法。
  #
  # clamp_min 是数值保护：std 若为 0，log(0) = -inf 且除零得 nan，
  # 一旦污染梯度，整个网络的参数会在一次更新内全部变成 nan。
  #
  # 这个 KL 与 PPO 的 desired_kl 完全无关：那个衡量新旧策略的差异、
  # 用于自适应调学习率；这个是 Student 与 Teacher 之间的蒸馏损失。
  t_d = teacher_std.clamp_min(std_eps)
  s_d = student_std.clamp_min(std_eps)
  kl = (
    torch.log(s_d / t_d)
    + (t_d.pow(2) + (teacher_mean - student_mean).pow(2)) / (2.0 * s_d.pow(2))
    - 0.5
  )
  return kl.sum(dim=-1).mean()
  # <<< HOMEWORK_TODO_4_END


def gaussian_matching_metrics(
  teacher_mean: torch.Tensor,
  teacher_std: torch.Tensor,
  student_mean: torch.Tensor,
  student_std: torch.Tensor,
) -> dict[str, float]:
  """用于日志记录的 Gaussian 参数诊断指标（不参与梯度）。"""
  # >>> HOMEWORK_TODO_5_START
  # 【作业 TODO 5/7】Gaussian matching 诊断指标
  #
  # 把 KL 这个标量拆成两个可解释的分量：
  #   mean_rmse  两个分布中心差多远,Student 的"平均动作"学得像不像
  #   std_rmse   两个分布宽度差多少,Student 的"探索程度"是否匹配
  #
  # 单看 KL 无法区分这两种失配。常见现象是 mean 已经贴得很近、
  # 但 student_std 明显偏大（Student 因为观测受限而保留了更多不确定性），
  # 此时 KL 仍然不小，却不代表动作学得差。分开看才能定位问题。
  #
  # 同 TODO 3：no_grad + .item()，纯日志用途，不进计算图。
  with torch.no_grad():
    return {
      "mean_rmse": (student_mean - teacher_mean).pow(2).mean().sqrt().item(),
      "std_rmse": (student_std - teacher_std).pow(2).mean().sqrt().item(),
    }
  # <<< HOMEWORK_TODO_5_END
