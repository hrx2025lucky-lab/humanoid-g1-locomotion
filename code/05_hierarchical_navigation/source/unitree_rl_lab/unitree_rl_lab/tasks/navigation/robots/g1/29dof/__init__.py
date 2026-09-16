import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Navigation-HRL-Extension",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NavigationV5MixedObstaclePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Navigation-HRL-Baseline",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal",
        "play_env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NavigationV5CompactSingleGoalPPORunnerCfg",
    },
)

# Part 2 单因素对照组：继承 Baseline，仅把「固定竞技场」换成「每 episode 随机重排障碍」。
# 课程自带的 HRL-Extension 与 Baseline 相差五处（布局/课程/观测维度/目标模式/终止条件），
# 其中观测维度不同使两组网络输入层都不一样，无法归因到单一因素。
# 本组共用 Baseline 的 PPORunnerCfg，保证 seed / num_steps_per_env / 网络结构完全一致。
gym.register(
    id="Unitree-G1-29dof-Navigation-HRL-RandomArena",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5RandomArenaEnvCfg_SingleGoal",
        "play_env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5RandomArenaEnvCfg_SingleGoal_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NavigationV5CompactSingleGoalPPORunnerCfg",
    },
)
