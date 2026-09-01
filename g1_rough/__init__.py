"""G1 29-DOF 粗糙地形速度跟踪任务（实践 2）。

这个包通过 symlink 挂进 unitree_rl_lab 的
``tasks/locomotion/robots/g1/29dof/rough``，
由 IsaacLab 的 ``import_packages`` 递归发现并自动 import，
因此**不需要修改 unitree_rl_lab 的任何现有文件**。
"""

import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
