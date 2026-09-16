from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCAD_YAML = (
  REPO_ROOT
  / "src/humanoid_hw6/config/g1/motion_data_cfg_g1_accad_walk.yaml"
)


def test_accad_motion_yaml_points_to_subset() -> None:
  text = ACCAD_YAML.read_text()
  assert "g1_accad_walk" in text


def test_accad_subset_has_25_clips() -> None:
  clip_dir = REPO_ROOT / "assets/motions/g1_accad_walk"
  npz_files = sorted(clip_dir.glob("*.npz"))
  assert len(npz_files) == 25


def test_humanoid_hw6_tasks_registered() -> None:
  from mjlab.tasks.registry import list_tasks

  import humanoid_hw6.config.g1  # noqa: F401

  tasks = set(list_tasks())
  expected = {
    "Mjlab-Humanoid-HW6-Teacher-G1",
    "Mjlab-Humanoid-HW6-Student-Action-Matching-G1",
    "Mjlab-Humanoid-HW6-Student-KL-Matching-G1",
  }
  assert expected.issubset(tasks)



def test_provided_teacher_checkpoint_exists() -> None:
  teacher_ckpt = REPO_ROOT / "checkpoints/g1_hw6_teacher/model_latest.pt"
  teacher_onnx = REPO_ROOT / "checkpoints/g1_hw6_teacher/g1_hw6_teacher.onnx"
  assert teacher_ckpt.is_file()
  assert teacher_onnx.is_file()


def test_student_runner_cfg_defaults_to_hw6_teacher() -> None:
  from humanoid_hw6.config.g1.rl_cfg import (
    DEFAULT_TEACHER_CHECKPOINT,
    unitree_g1_student_action_matching_runner_cfg,
    unitree_g1_student_kl_matching_runner_cfg,
  )

  assert DEFAULT_TEACHER_CHECKPOINT == "checkpoints/g1_hw6_teacher/model_latest.pt"
  assert (
    unitree_g1_student_action_matching_runner_cfg().teacher_checkpoint_file
    == DEFAULT_TEACHER_CHECKPOINT
  )
  assert (
    unitree_g1_student_kl_matching_runner_cfg().teacher_checkpoint_file
    == DEFAULT_TEACHER_CHECKPOINT
  )

