# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Mimic tasks (HOI terrain, dance, etc.): default W&B logging; use ``--logger tensorboard`` to opt out."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = ""  # same as task name
    logger = "wandb"
    wandb_project = "unitree_hoi_mimic"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3, # learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HoiTerrainFineTunePpoRunnerCfg(BasePPORunnerCfg):
    """PPO runner for HOI terrain fine-tune tasks.

    Use the same ``experiment_name`` as ``Unitree-G1-29dof-Mimic-HOI_terrain`` so
    ``train.py`` resolves ``--resume --load_run`` under the existing pretrained log root
    (``logs/rsl_rl/unitree_g1_29dof_mimic_hoi_terrain/``). New fine-tune runs still get
    distinct folders via ``--run_name`` and the timestamp prefix.
    """

    experiment_name = "unitree_g1_29dof_mimic_hoi_terrain"


@configclass
class HoiTerrainPerceptivePpoRunnerCfg(BasePPORunnerCfg):
    """PPO runner for HOI terrain perceptive task."""

    experiment_name = "unitree_g1_29dof_mimic_hoi_terrain_perceptive"
    wandb_project = "unitree_hoi_mimic_terrain_perceptive"


@configclass
class HoiTerrainPerceptiveRaycastPpoRunnerCfg(BasePPORunnerCfg):
    """PPO runner for HOI terrain perceptive RayCaster task."""

    experiment_name = "unitree_g1_29dof_mimic_hoi_terrain_perceptive_raycast"
    wandb_project = "unitree_hoi_mimic_terrain_perceptive_raycast"
