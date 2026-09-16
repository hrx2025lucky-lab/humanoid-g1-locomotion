from mjlab.tasks.registry import register_mjlab_task

from humanoid_hw6.rl import OnPolicyRunner, StudentOnPolicyRunner

from .env_cfgs import unitree_g1_student_env_cfg, unitree_g1_teacher_env_cfg
from .rl_cfg import (
  unitree_g1_student_action_matching_runner_cfg,
  unitree_g1_student_kl_matching_runner_cfg,
  unitree_g1_teacher_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Teacher-G1",
  env_cfg=unitree_g1_teacher_env_cfg(),
  play_env_cfg=unitree_g1_teacher_env_cfg(play=True),
  rl_cfg=unitree_g1_teacher_ppo_runner_cfg(),
  runner_cls=OnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Student-Action-Matching-G1",
  env_cfg=unitree_g1_student_env_cfg(),
  play_env_cfg=unitree_g1_student_env_cfg(play=True),
  rl_cfg=unitree_g1_student_action_matching_runner_cfg(),
  runner_cls=StudentOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Humanoid-HW6-Student-KL-Matching-G1",
  env_cfg=unitree_g1_student_env_cfg(),
  play_env_cfg=unitree_g1_student_env_cfg(play=True),
  rl_cfg=unitree_g1_student_kl_matching_runner_cfg(),
  runner_cls=StudentOnPolicyRunner,
)
