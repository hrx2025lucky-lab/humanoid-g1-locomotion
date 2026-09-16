"""RL configuration for the Unitree G1 Humanoid HW6 tasks."""

import os

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from humanoid_hw6.rl import (
  ActionMatchingPpoAlgorithmCfg,
  KlMatchingPpoAlgorithmCfg,
  OnPolicyRunnerCfg,
  StudentOnPolicyRunnerCfg,
)

DEFAULT_TEACHER_CHECKPOINT = "checkpoints/g1_hw6_teacher/model_latest.pt"


def _wandb_tags(*extra: str) -> tuple[str, ...]:
  return extra


def unitree_g1_teacher_ppo_runner_cfg() -> OnPolicyRunnerCfg:
  return OnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      class_name="humanoid_hw6.rl.models.teacher:TeacherActor",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="humanoid_hw6.rl.models.teacher:TeacherCritic",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_hw6_teacher",
    wandb_project="humanoid_hw6",
    wandb_tags=_wandb_tags("humanoid_hw6", "teacher", "privileged"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
    obs_groups={"actor": ("policy",), "critic": ("critic",)},
  )


# ── 超参对齐开关（用于把两组蒸馏对照变成严格单因素）────────────────────────
# 两组原本除蒸馏目标外还差三处超参（lr 5e-4/3e-4、entropy_coef 0.0025/0.005、
# desired_kl 0.005/0.008），因此实测出的差距无法归因到蒸馏目标本身。
# 置 HW6_ALIGN_HPARAMS=1 后两组共用下面这套值，唯一变量才只剩蒸馏目标。
_ALIGN = os.environ.get("HW6_ALIGN_HPARAMS", "0") == "1"
_ALIGNED = dict(learning_rate=3.0e-4, entropy_coef=0.005, desired_kl=0.008)


def _hp(name: str, own):
  """对齐模式下返回统一值，否则返回该组原有的值。"""
  return _ALIGNED[name] if _ALIGN else own


def unitree_g1_student_action_matching_runner_cfg() -> StudentOnPolicyRunnerCfg:
  return StudentOnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      class_name="humanoid_hw6.rl.models.student_teacher:StudentTeacherActor",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.4,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="rsl_rl.models:MLPModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=ActionMatchingPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=_hp("entropy_coef", 0.0025),
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=_hp("learning_rate", 5.0e-4),
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=_hp("desired_kl", 0.005),
      max_grad_norm=1.0,
      bc_coef_start=1.0,
      bc_coef_end=0.05,
      bc_anneal_iters=20_000,
      bc_loss_type="mse",
    ),
    experiment_name="g1_hw6_student_action_matching" + ("_aligned" if _ALIGN else ""),
    wandb_project="humanoid_hw6",
    wandb_tags=_wandb_tags("humanoid_hw6", "student", "action_matching"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=20_000,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "teacher": ("teacher_policy",),
    },
    teacher_checkpoint_file=DEFAULT_TEACHER_CHECKPOINT,
  )


def unitree_g1_student_kl_matching_runner_cfg() -> StudentOnPolicyRunnerCfg:
  return StudentOnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      class_name="humanoid_hw6.rl.models.student_teacher:StudentTeacherActor",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.4,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="rsl_rl.models:MLPModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=KlMatchingPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.008,
      max_grad_norm=1.0,
      kl_coef=0.1,
      kl_coef_min=0.01,
      kl_coef_anneal_iters=60_000,
    ),
    experiment_name="g1_hw6_student_kl_matching" + ("_aligned" if _ALIGN else ""),
    wandb_project="humanoid_hw6",
    wandb_tags=_wandb_tags("humanoid_hw6", "student", "kl_matching"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=20_000,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "teacher": ("teacher_policy",),
    },
    teacher_checkpoint_file=DEFAULT_TEACHER_CHECKPOINT,
  )
