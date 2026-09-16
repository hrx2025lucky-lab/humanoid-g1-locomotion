import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLegacyObsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLegacyObsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LegacyPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Heightfield-Warmup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightfieldWarmupLegacyObsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightfieldWarmupLegacyObsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LegacyPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-EstimatedObs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Heightfield-Warmup-EstimatedObs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightfieldWarmupEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightfieldWarmupPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Heightfield-Warmup-Aligned",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_warmup_aligned:RobotHeightfieldWarmupAlignedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_warmup_aligned:RobotHeightfieldWarmupAlignedPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LegacyPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Heightfield-Warmup-Aligned-URDF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_warmup_aligned:RobotHeightfieldWarmupAlignedUrdfEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_warmup_aligned:RobotHeightfieldWarmupAlignedUrdfPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LegacyPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Heightfield-Warmup-DrCurriculum",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightfieldWarmupDrCurriculumLegacyObsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightfieldWarmupDrCurriculumLegacyObsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LegacyPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Heightfield-Warmup-DrCurriculum-EstimatedObs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightfieldWarmupDrCurriculumEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightfieldWarmupDrCurriculumPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
