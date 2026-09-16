from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch


def _load_implementation(module_path: str):
  # Exercise the implementation used by training against the numerical oracles
  # below. The course's instructor_solutions directory is not part of this repo.
  return importlib.import_module(f"humanoid_hw6.{module_path}")


def test_distillation_utils_reference() -> None:
  reference = _load_implementation("rl.algorithms.distillation_utils")

  assert abs(reference.linear_anneal(1.0, 0.1, 5, 10) - 0.55) < 1e-6
  assert abs(reference.linear_anneal(1.0, 0.1, 5, 0) - 0.1) < 1e-6

  student = torch.ones(4, 3, requires_grad=True)
  teacher = torch.zeros(4, 3)
  loss = reference.action_regression_loss(student, teacher, "mse")
  loss.backward()
  assert student.grad is not None
  assert teacher.grad is None

  metrics = reference.action_matching_metrics(student.detach(), teacher)
  assert set(metrics) == {"action_mae", "action_rmse"}

  teacher_mean = torch.zeros(2, 2)
  teacher_std = torch.ones(2, 2)
  student_mean = torch.zeros(2, 2)
  student_std = torch.full((2, 2), 2.0)
  forward_kl = reference.diagonal_gaussian_kl(
    teacher_mean, teacher_std, student_mean, student_std
  )
  reverse_kl = reference.diagonal_gaussian_kl(
    student_mean, student_std, teacher_mean, teacher_std
  )
  assert forward_kl.item() > 0.0
  assert reverse_kl.item() > 0.0
  assert not torch.allclose(forward_kl, reverse_kl)

  gaussian_metrics = reference.gaussian_matching_metrics(
    teacher_mean, teacher_std, student_mean, student_std
  )
  assert set(gaussian_metrics) == {"mean_rmse", "std_rmse"}


def test_action_matching_reference_output() -> None:
  reference = _load_implementation("rl.algorithms.action_matching")

  class _Actor:
    loaded_teacher = True

    def teacher_forward(self, obs):
      return torch.zeros(4, 3)

    def __call__(self, obs):
      return torch.ones(4, 3)

  alg = SimpleNamespace(
    bc_coef_start=1.0,
    bc_coef_end=0.1,
    bc_anneal_iters=10,
    num_bc_updates=5,
    bc_loss_type="mse",
    actor=_Actor(),
  )
  output = reference.ActionMatchingPPO._compute_distillation_output(
    alg, SimpleNamespace(observations={})
  )
  assert torch.allclose(output.loss, torch.tensor(1.0))
  assert abs(output.metrics["action_mae"] - 1.0) < 1e-6


def test_kl_matching_reference_output() -> None:
  reference = _load_implementation("rl.algorithms.kl_matching")
  actor = SimpleNamespace(loaded_teacher=True)
  actor.output_distribution_params = (torch.zeros(4, 3), torch.ones(4, 3))
  actor.teacher_distribution_params = lambda obs: (
    torch.zeros(4, 3),
    torch.ones(4, 3),
  )
  alg = SimpleNamespace(
    kl_coef_start=0.2,
    kl_coef_min=0.05,
    kl_coef_anneal_iters=10,
    num_kl_updates=5,
    actor=actor,
  )
  output = reference.KlMatchingPPO._compute_distillation_output(
    alg, SimpleNamespace(observations={})
  )
  assert torch.allclose(output.loss, torch.tensor(0.0), atol=1e-6)
  assert abs(reference.KlMatchingPPO._current_distillation_coef(alg) - 0.125) < 1e-6
