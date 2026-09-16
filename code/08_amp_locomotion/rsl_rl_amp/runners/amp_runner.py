"""Runner type used to select AMP training from task configuration."""

from __future__ import annotations

from rsl_rl_amp.algorithms import AMP
from rsl_rl_amp.runners.on_policy_runner import OnPolicyRunner


def validate_amp_motion_profile(env, train_cfg: dict) -> str:
    """Validate the agent, environment, and loaded expert dataset agree on style."""
    expected = str(train_cfg.get("amp_motion_profile", "")).strip().lower()
    if not expected:
        raise ValueError("AMP runner configuration must declare a non-empty 'amp_motion_profile'.")
    env_cfg = getattr(env, "cfg", None)
    environment_profile = str(getattr(env_cfg, "motion_style", "")).strip().lower()
    sampler = getattr(env, "amp_expert_sampler", None)
    if sampler is None and hasattr(env, "unwrapped"):
        sampler = getattr(env.unwrapped, "amp_expert_sampler", None)
    dataset_cfg = getattr(sampler, "cfg", None)
    dataset_profile = str(getattr(dataset_cfg, "profile_name", "")).strip().lower()
    if not environment_profile:
        raise ValueError("AMP environment configuration must declare a non-empty 'motion_style'.")
    if not dataset_profile:
        raise ValueError("AMP expert dataset must declare a non-empty 'profile_name'.")
    if expected != "auto" and expected != environment_profile:
        raise ValueError(
            f"AMP agent expects '{expected}' motion data, but environment selected '{environment_profile}'."
        )
    if environment_profile != dataset_profile:
        raise ValueError(
            f"AMP environment selected '{environment_profile}', but dataset loaded '{dataset_profile}'."
        )
    return environment_profile


class AMPRunner(OnPolicyRunner):
    """On-policy runner whose configured algorithm is expected to be :class:`AMP`."""

    alg: AMP

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.amp_motion_profile = validate_amp_motion_profile(env, train_cfg)
        train_cfg["amp_motion_profile"] = self.amp_motion_profile
        train_cfg["algorithm"].setdefault("amp_cfg", {})["motion_profile"] = self.amp_motion_profile
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)


__all__ = ["AMPRunner", "validate_amp_motion_profile"]
