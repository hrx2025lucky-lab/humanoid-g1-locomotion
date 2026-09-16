from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_height_ablation_blind_actor_env_cfg,
  unitree_g1_flat_height_ablation_no_reward_env_cfg,
  unitree_g1_flat_height_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg, unitree_g1_velocity_height_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-VelocityHeight-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_height_env_cfg(),
  play_env_cfg=unitree_g1_flat_height_env_cfg(play=True),
  rl_cfg=unitree_g1_velocity_height_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# ── 消融实验（作业 §7 要求 ≥2 组，公平对照：仅改一个因素）──
# 共用 baseline 的 rl_cfg，保证 seed / num_steps_per_env / 网络结构完全一致。

register_mjlab_task(
  # 消融 A：切断"奖励高度跟踪"这一段（track_base_height.weight = 0）
  task_id="Mjlab-VelocityHeight-Flat-Unitree-G1-NoHeightRew",
  env_cfg=unitree_g1_flat_height_ablation_no_reward_env_cfg(),
  play_env_cfg=unitree_g1_flat_height_ablation_no_reward_env_cfg(play=True),
  rl_cfg=unitree_g1_velocity_height_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  # 消融 B：切断"Actor 观测到指令"这一段（Actor 移除 height_command，Critic 保留）
  task_id="Mjlab-VelocityHeight-Flat-Unitree-G1-BlindActor",
  env_cfg=unitree_g1_flat_height_ablation_blind_actor_env_cfg(),
  play_env_cfg=unitree_g1_flat_height_ablation_blind_actor_env_cfg(play=True),
  rl_cfg=unitree_g1_velocity_height_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
