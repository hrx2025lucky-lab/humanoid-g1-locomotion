"""PPO and AMP hyperparameters for the G1 flat-ground tasks."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlAMPAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "AMP"
    amp_cfg: dict = MISSING


@configclass
class RslRlAMPActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic configuration with trainable exploration covariance."""

    learn_std: bool = True


@configclass
class G1AMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    experiment_name = "unitree_g1_29dof_amp"
    amp_motion_profile: str = "mixed"
    # actor / critic / amp 三路各自读哪些 observation group。
    # amp 这一路是必需的：amp.py 用 obs_groups["amp"] 取判别器输入
    # （resolve_obs_groups(..., ["amp"]) 会强制校验该键存在）。
    #
    # 三者读的是不同东西：
    #   policy: 策略能观测到的量（带噪、含指令）
    #   critic: 可含特权信息，只在训练时用
    #   amp: 无噪的 80 维风格特征，必须与专家数据同构
    # 把 amp 混进 actor 会让策略直接看到判别器输入，等于泄题。
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "amp": ["amp"],
    }

    policy = RslRlAMPActorCriticCfg(
        init_noise_std=0.8,
        noise_std_type="scalar",
        learn_std=True,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlAMPAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg={
            "replay_buffer_size": 200_000,
            "discriminator_updates": 4,
            "discriminator_batch_size": 8192,
            "normalization_batch_size": 8192,
            "discriminator_learning_rate": 1.0e-4,
            "discriminator_weight_decay": 1.0e-4,
            "discriminator_output_weight_decay": 1.0e-3,
            "discriminator_max_grad_norm": 1.0,
            "discriminator_balance": {
                "enabled": True,
                "saturation_threshold": 0.7,
                "saturated_updates": 0,
            },
            "discriminator": {
                "hidden_dims": [512, 256],
                "activation": "elu",
                "reward_mode": "lsq",
                "classification_margin": 0.8,
                "logit_regularization": 0.05,
                "reward_scale": 5.0,
                "task_reward_weight": 0.4,
                "gradient_penalty": 10.0,
            },
        },
    )


@configclass
class G1AMPWalkRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_walk"
    amp_motion_profile: str = "walk"

    def __post_init__(self):
        self.algorithm.amp_cfg["discriminator"]["task_reward_weight"] = 0.6


@configclass
class G1AMPRunRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_run"
    amp_motion_profile: str = "run"


@configclass
class G1AMPOmniRunRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_omni_run"
    amp_motion_profile: str = "omni_run"


@configclass
class G1AMPWalkToRunRunnerCfg(G1AMPRunnerCfg):
    # 独立实验名：checkpoint 与日志目录由它决定，与 Walk / Run 共用会互相覆盖。
    experiment_name = "unitree_g1_29dof_amp_walk_to_run"
    # 必须与环境侧、专家数据侧用同一个 profile，否则判别器拿走跑数据、
    # 环境却按纯 walk 采指令，风格奖励会一直压制正确的高速步态。
    amp_motion_profile: str = "walk_to_run"

    def __post_init__(self):
        # task_reward_weight = alpha，见 discriminator.mix_rewards：
        #   r_t = (1 - alpha) * r_amp + alpha * r_task
        # 走跑切换要同时管住"速度跟得上"（任务）和"步态像人"（风格），
        # 取 0.5 让两者等权：
        #   - 比 Walk 的 0.6 低，因为走跑切换更依赖专家示范的步态转换；
        #   - 比基类的 0.4 高，因为必须真的跟上高速指令才谈得上"跑"。
        self.algorithm.amp_cfg["discriminator"]["task_reward_weight"] = 0.5


@configclass
class G1AMPDanceRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_dance"
    amp_motion_profile: str = "dance"


@configclass
class G1AMPMixedRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_mixed"
    amp_motion_profile: str = "mixed"


@configclass
class G1AMPPlayRunnerCfg(G1AMPRunnerCfg):
    experiment_name = "unitree_g1_29dof_amp_play"
    amp_motion_profile: str = "auto"


G1FlatRslRlOnPolicyRunnerAmpCfg = G1AMPRunnerCfg
