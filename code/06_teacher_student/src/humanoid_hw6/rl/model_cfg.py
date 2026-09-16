from __future__ import annotations

import math

from mjlab.rl import RslRlVecEnvWrapper

from humanoid_hw6.mdp.motion.base import MotionCommand


def infer_motion_dims(
  env: RslRlVecEnvWrapper, representation: str
) -> tuple[int, int] | None:
  env_unwrapped = env.unwrapped
  motion_term = env_unwrapped.command_manager.get_term("motion")
  if not isinstance(motion_term, MotionCommand):
    return None
  if not motion_term.has_command_representation(representation):
    return None
  motion_obs = motion_term.get_command_representation(representation)
  motion_obs_dim = int(motion_obs.shape[-1])
  motion_steps = len(motion_term.future_sampling_step_offsets)
  if motion_steps <= 0:
    motion_steps = 1
  return motion_obs_dim, motion_steps


def obs_group_name(train_cfg: dict, obs_set: str, default: str) -> str:
  obs_groups = train_cfg.get("obs_groups", {})
  group_names = obs_groups.get(obs_set, (default,))
  if isinstance(group_names, str):
    return group_names
  if len(group_names) == 0:
    return default
  return str(group_names[0])


def configure_model_cfg(env, train_cfg: dict) -> None:
  if not isinstance(env, RslRlVecEnvWrapper):
    return

  actor_cfg = train_cfg.get("actor", {})
  critic_cfg = train_cfg.get("critic", {})

  actor_class = str(actor_cfg.get("class_name", ""))
  critic_class = str(critic_cfg.get("class_name", ""))

  motion_dims = infer_motion_dims(env, "default")
  teacher_dims = infer_motion_dims(env, "teacher")

  if (
    actor_class == "humanoid_hw6.rl.models.teacher:TeacherActor"
    and motion_dims is not None
  ):
    observation_manager = env.unwrapped.observation_manager
    obs_dims = observation_manager.group_obs_dim
    actor_group = obs_group_name(train_cfg, "actor", "actor")
    actor_terms = observation_manager.active_terms[actor_group]
    actor_term_dims = observation_manager._group_obs_term_dim[actor_group]
    flat_term_dims = {
      name: int(math.prod(dims))
      for name, dims in zip(actor_terms, actor_term_dims, strict=False)
    }
    history_dim = flat_term_dims["history"]
    current_dim = int(obs_dims[actor_group][0]) - int(motion_dims[0]) - history_dim
    single_motion_dim = motion_dims[0] // max(motion_dims[1], 1)
    history_step_dim = single_motion_dim + current_dim
    if history_dim % history_step_dim != 0:
      raise ValueError(
        "TeacherActor history dimension mismatch: "
        f"history_dim={history_dim}, history_step_dim={history_step_dim}."
      )
    actor_cfg.setdefault("motion_obs_dim", motion_dims[0])
    actor_cfg.setdefault("motion_steps", motion_dims[1])
    actor_cfg.setdefault("proprio_obs_dim", current_dim)
    actor_cfg.setdefault("history_steps", max(history_dim // history_step_dim, 1))
    actor_cfg.setdefault("motion_latent_dim", 64)
    actor_cfg.setdefault("history_latent_dim", 128)
    actor_cfg.setdefault("motion_conv_channels", (48, 24))
    actor_cfg.setdefault("motion_conv_kernel_sizes", (6, 4))
    actor_cfg.setdefault("motion_conv_strides", (2, 2))
    actor_cfg.setdefault("history_conv_channels", (64, 32))
    actor_cfg.setdefault("history_conv_kernel_sizes", (4, 2))
    actor_cfg.setdefault("history_conv_strides", (2, 1))
    actor_cfg.setdefault("layer_norm", True)

  if (
    critic_class == "humanoid_hw6.rl.models.teacher:TeacherCritic"
    and motion_dims is not None
  ):
    critic_cfg.setdefault("motion_obs_dim", motion_dims[0])
    critic_cfg.setdefault("motion_steps", motion_dims[1])
    critic_cfg.setdefault("motion_latent_dim", 64)
    critic_cfg.setdefault("motion_conv_channels", (48, 24))
    critic_cfg.setdefault("motion_conv_kernel_sizes", (6, 4))
    critic_cfg.setdefault("motion_conv_strides", (2, 2))
    critic_cfg.setdefault("layer_norm", True)

  if (
    actor_class == "humanoid_hw6.rl.models.student_teacher:StudentTeacherActor"
    and teacher_dims is not None
  ):
    observation_manager = env.unwrapped.observation_manager
    obs_dims = observation_manager.group_obs_dim
    actor_group = obs_group_name(train_cfg, "actor", "actor")
    teacher_group = obs_group_name(train_cfg, "teacher", "teacher_policy")
    actor_terms = observation_manager.active_terms[actor_group]
    actor_term_dims = observation_manager._group_obs_term_dim[actor_group]
    flat_term_dims = {
      name: int(math.prod(dims))
      for name, dims in zip(actor_terms, actor_term_dims, strict=False)
    }
    command_dim = flat_term_dims["command"]
    history_dim = flat_term_dims["history"]
    current_dim = int(obs_dims[actor_group][0]) - command_dim - history_dim
    history_step_dim = command_dim + current_dim
    if history_dim % history_step_dim != 0:
      raise ValueError(
        "StudentTeacherActor history dimension mismatch: "
        f"history_dim={history_dim}, history_step_dim={history_step_dim}."
      )
    actor_cfg.setdefault("current_motion_obs_dim", command_dim)
    actor_cfg.setdefault("proprio_obs_dim", current_dim)
    actor_cfg.setdefault("history_steps", max(history_dim // history_step_dim, 1))
    actor_cfg.setdefault("history_latent_dim", 128)
    actor_cfg.setdefault("history_conv_channels", (64, 32))
    actor_cfg.setdefault("history_conv_kernel_sizes", (4, 2))
    actor_cfg.setdefault("history_conv_strides", (2, 1))
    actor_cfg.setdefault("layer_norm", True)
    actor_cfg.setdefault(
      "teacher_distribution_cfg",
      {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    )
    teacher_terms = observation_manager.active_terms[teacher_group]
    teacher_term_dims = observation_manager._group_obs_term_dim[teacher_group]
    teacher_flat_term_dims = {
      name: int(math.prod(dims))
      for name, dims in zip(teacher_terms, teacher_term_dims, strict=False)
    }
    teacher_command_dim = teacher_flat_term_dims["command"]
    teacher_history_dim = teacher_flat_term_dims["history"]
    teacher_current_dim = (
      int(obs_dims[teacher_group][0]) - teacher_command_dim - teacher_history_dim
    )
    teacher_motion_steps = max(teacher_command_dim // max(command_dim, 1), 1)
    actor_cfg.setdefault("teacher_motion_obs_dim", teacher_command_dim)
    actor_cfg.setdefault("teacher_motion_steps", teacher_motion_steps)
    actor_cfg.setdefault("teacher_proprio_obs_dim", teacher_current_dim)
    actor_cfg.setdefault("teacher_motion_latent_dim", 64)
    actor_cfg.setdefault("teacher_motion_conv_channels", (48, 24))
    actor_cfg.setdefault("teacher_motion_conv_kernel_sizes", (6, 4))
    actor_cfg.setdefault("teacher_motion_conv_strides", (2, 2))
    actor_cfg.setdefault("teacher_history_latent_dim", 128)
    actor_cfg.setdefault("teacher_history_conv_channels", (64, 32))
    actor_cfg.setdefault("teacher_history_conv_kernel_sizes", (4, 2))
    actor_cfg.setdefault("teacher_history_conv_strides", (2, 1))
    actor_cfg.setdefault("teacher_layer_norm", True)

  train_cfg["actor"] = actor_cfg
  train_cfg["critic"] = critic_cfg
