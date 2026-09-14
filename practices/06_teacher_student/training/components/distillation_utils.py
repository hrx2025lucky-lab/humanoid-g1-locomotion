"""HW6 蒸馏共享数学工具。

本文件包含 实验 1/7 至 实验 5/7。
快速定位：grep -n "【实验 实验" src/humanoid_hw6/rl/algorithms/
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def linear_anneal(start: float, end: float, step: int, duration: int) -> float:
    """在 ``duration`` 步内，从 ``start`` 线性退火到 ``end``。"""
    if duration <= 0:
        return end
    progress = min(max(step / duration, 0.0), 1.0)
    return start + (end - start) * progress

def action_regression_loss(student_actions: torch.Tensor, teacher_actions: torch.Tensor, loss_type: str='mse') -> torch.Tensor:
    """student 与 teacher action 之间的标量回归损失。"""
    if loss_type == 'mse':
        return F.mse_loss(student_actions, teacher_actions)
    if loss_type == 'huber':
        return F.huber_loss(student_actions, teacher_actions)
    raise ValueError(f"Unsupported loss_type '{loss_type}', expected 'mse' or 'huber'.")

def action_matching_metrics(student_actions: torch.Tensor, teacher_actions: torch.Tensor) -> dict[str, float]:
    """用于日志记录的 action 级诊断指标（不参与梯度）。"""
    with torch.no_grad():
        diff = student_actions - teacher_actions
        return {'action_mae': diff.abs().mean().item(), 'action_rmse': diff.pow(2).mean().sqrt().item()}

def diagonal_gaussian_kl(teacher_mean: torch.Tensor, teacher_std: torch.Tensor, student_mean: torch.Tensor, student_std: torch.Tensor, std_eps: float=1e-06) -> torch.Tensor:
    """对角 Gaussian 分布的 mean KL(teacher || student)。"""
    t_d = teacher_std.clamp_min(std_eps)
    s_d = student_std.clamp_min(std_eps)
    kl = torch.log(s_d / t_d) + (t_d.pow(2) + (teacher_mean - student_mean).pow(2)) / (2.0 * s_d.pow(2)) - 0.5
    return kl.sum(dim=-1).mean()

def gaussian_matching_metrics(teacher_mean: torch.Tensor, teacher_std: torch.Tensor, student_mean: torch.Tensor, student_std: torch.Tensor) -> dict[str, float]:
    """用于日志记录的 Gaussian 参数诊断指标（不参与梯度）。"""
    with torch.no_grad():
        return {'mean_rmse': (student_mean - teacher_mean).pow(2).mean().sqrt().item(), 'std_rmse': (student_std - teacher_std).pow(2).mean().sqrt().item()}
