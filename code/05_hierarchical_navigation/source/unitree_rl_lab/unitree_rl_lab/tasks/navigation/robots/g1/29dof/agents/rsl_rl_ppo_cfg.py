import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class NavigationV5MixedObstaclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 8
    max_iterations = 30000
    save_interval = 50
    experiment_name = "unitree_g1_29dof_navigation_hrl_extension"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=float(os.environ.get("NAV_ENTROPY_COEF", 0.005)),
        # 课程原值 0.005。实测 baseline 的动作噪声 std 在前 100 iter 就从
        # init_noise_std=0.2 塌到 0.07 并再未回升，探索过早收窄。
        # 与 NAV_COMMAND_SMOOTHING 是两个独立因子：前者控制 PPO 愿不愿意保持
        # 随机性，后者控制随机性能不能传到低层（EMA 会把不相关噪声衰减到 22%，
        # 见 scripts/probe_p5_exploration.py 的实测 0.202 vs 理论 0.229）。
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class NavigationV5CompactSingleGoalPPORunnerCfg(NavigationV5MixedObstaclePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_navigation_hrl_baseline"
    # 探索修复的几组共用同一个 Baseline 任务（只差环境变量），
    # 不打 tag 的话 checkpoint 全落进同名目录、只能靠时间戳分辨。
    # train.py 会把它拼成 {时间戳}_{run_name}。
    run_name = os.environ.get("NAV_RUN_NAME", "")
