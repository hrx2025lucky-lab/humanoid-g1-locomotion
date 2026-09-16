"""Configuration adapters for Isaac Lab and native ``rsl_rl_amp`` users."""

from __future__ import annotations

import copy
from dataclasses import MISSING
from typing import Any


_ALGORITHM_TYPES = {
    "PPO": "rsl_rl_amp.algorithms:PPO",
    "AMP": "rsl_rl_amp.algorithms:AMP",
}


def _value_or(value: Any, fallback: Any) -> Any:
    return fallback if value is None or value is MISSING else value


def normalize_runner_config(cfg: dict) -> dict:
    """Return a native runner configuration without mutating ``cfg``.

    Isaac Lab releases that use the classic RSL-RL API expose one combined
    ``policy`` block.  The local learner uses separate actor and critic model
    blocks.  Keeping this conversion at the runner boundary prevents simulator
    configuration types from leaking into the learning library.
    """

    normalized = copy.deepcopy(cfg)
    if "actor" in normalized and "critic" in normalized:
        return normalized

    policy = normalized.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Runner configuration requires either actor/critic blocks or a legacy policy block.")

    policy_type = policy.get("class_name", "ActorCritic")
    if policy_type not in {"ActorCritic", "MLPModel", "rsl_rl_amp.models:MLPModel"}:
        raise ValueError(
            f"Legacy policy class '{policy_type}' is not supported by the AMP compatibility adapter. "
            "Use explicit actor and critic model blocks for recurrent or convolutional models."
        )

    empirical = bool(_value_or(normalized.get("empirical_normalization"), False))
    activation = policy.get("activation", "elu")
    normalized["actor"] = {
        "class_name": "rsl_rl_amp.models:MLPModel",
        "hidden_dims": list(policy.get("actor_hidden_dims", (256, 256, 256))),
        "activation": activation,
        "obs_normalization": bool(_value_or(policy.get("actor_obs_normalization"), empirical)),
        "distribution_cfg": {
            "class_name": "rsl_rl_amp.modules:GaussianDistribution",
            "init_std": float(policy.get("init_noise_std", 1.0)),
            "std_type": policy.get("noise_std_type", "scalar"),
            "learn_std": bool(policy.get("learn_std", True)),
        },
    }
    normalized["critic"] = {
        "class_name": "rsl_rl_amp.models:MLPModel",
        "hidden_dims": list(policy.get("critic_hidden_dims", (256, 256, 256))),
        "activation": activation,
        "obs_normalization": bool(_value_or(policy.get("critic_obs_normalization"), empirical)),
    }

    obs_groups = normalized.setdefault("obs_groups", {})
    if "actor" not in obs_groups:
        obs_groups["actor"] = list(obs_groups.get("policy", ["policy"]))
    if "critic" not in obs_groups:
        obs_groups["critic"] = list(obs_groups.get("critic", obs_groups["actor"]))

    algorithm = normalized.setdefault("algorithm", {})
    algorithm_type = algorithm.get("class_name", "PPO")
    algorithm["class_name"] = _ALGORITHM_TYPES.get(algorithm_type, algorithm_type)
    algorithm.setdefault("rnd_cfg", None)
    algorithm.setdefault("symmetry_cfg", None)
    normalized.pop("policy", None)
    return normalized
