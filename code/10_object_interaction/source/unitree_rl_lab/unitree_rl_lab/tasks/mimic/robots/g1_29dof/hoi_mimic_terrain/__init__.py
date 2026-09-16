import gymnasium as gym

_HOI_TERRAIN_ENV_ENTRY = f"{__name__}.hoi_terrain_rl_env:HoiTerrainMimicRLEnv"

gym.register(
    id="Unitree-G1-29dof-Mimic-HOI_terrain",
    entry_point=_HOI_TERRAIN_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
# ------------------------ Temporary DR experiments ------------------------
gym.register(
    id="Unitree-G1-29dof-Mimic-HOI_terrain-FT-DR-StageA",
    entry_point=_HOI_TERRAIN_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg_ft:RobotFineTuneStageAEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg_ft:RobotFineTunePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:HoiTerrainFineTunePpoRunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Mimic-HOI_terrain-FT-DR-StageB",
    entry_point=_HOI_TERRAIN_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg_ft:RobotFineTuneStageBEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg_ft:RobotFineTunePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:HoiTerrainFineTunePpoRunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Mimic-HOI_terrain-FT-DR-StageC",
    entry_point=_HOI_TERRAIN_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg_ft:RobotFineTuneStageCEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg_ft:RobotFineTunePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:HoiTerrainFineTunePpoRunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Mimic-HOI_terrain-SCRATCH-DR-Conservative",
    entry_point=_HOI_TERRAIN_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg_ft:RobotScratchDrConservativeEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg_ft:RobotFineTunePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:HoiTerrainFineTunePpoRunnerCfg",
    },
)
