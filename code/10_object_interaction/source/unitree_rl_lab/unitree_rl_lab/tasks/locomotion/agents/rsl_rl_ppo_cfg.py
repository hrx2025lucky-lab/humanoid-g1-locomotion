# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        velocity_estimator_cfg={
            "enabled": True,
            "loss_coef": 0.1,
            "loss_type": "mse",
            "target_obs_group": "estimator_target",
            "output_obs_group": "estimated_lin_vel",
            "input_obs_group": "estimator_input",
            "class_name": "VelocityEstimatorModel",
            "output_dim": 3,
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
        },
    )

    def __post_init__(self):
        """Allow fast ablations via environment-variable overrides."""
        estimator_cfg = self.algorithm.velocity_estimator_cfg
        if estimator_cfg is None:
            return

        estimator_enable = os.getenv("G1_ESTIMATOR_ENABLE")
        if estimator_enable is not None:
            estimator_cfg["enabled"] = estimator_enable.strip().lower() in {"1", "true", "yes", "on"}

        loss_coef = os.getenv("G1_ESTIMATOR_LOSS_COEF")
        if loss_coef is not None:
            estimator_cfg["loss_coef"] = float(loss_coef)


@configclass
class LegacyPPORunnerCfg(BasePPORunnerCfg):
    """Runner config for legacy observation tasks (no explicit estimator observation)."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.velocity_estimator_cfg = None
