"""Unitree locomotion curriculum utilities.

Terrain curriculum uses :func:`isaaclab_tasks.manager_based.locomotion.velocity.mdp.terrain_levels_vel`
(re-exported as ``mdp.terrain_levels_vel``). It updates ``TerrainImporter`` origins when the robot walks far
enough or under-performs relative to the velocity command.

Command curriculum (:func:`lin_vel_cmd_levels`) widens ``base_velocity`` sampling ``ranges`` toward
``limit_ranges`` when mean episodic tracking reward exceeds ``reward_term.weight * reward_fraction``.
The gate runs when ``env.common_step_counter % env.max_episode_length == 0`` (global env steps, not
aligned per parallel env).

Angular velocity command curriculum (:func:`ang_vel_cmd_levels`) uses the same pattern for
``ranges.ang_vel_z`` but is not registered in the default :class:`CurriculumCfg` in
``velocity_env_cfg.py``; add ``ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)`` there to enable it.

Reset / push DR curriculum (:func:`pose_reset_dr_levels`, :func:`push_robot_dr_levels`) widen
``reset_base`` pose and ``push_robot`` velocity ranges toward configured limits using the same gate.
:func:`push_robot_dr_levels` can also shorten ``push_robot.interval_range_s`` by linear interpolation
from a start pair (seconds) to a minimum (end) pair, with per-gate progress and rounded seconds.
"""

from __future__ import annotations

import torch
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_PUSH_VEL_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_PUSH_INTERVAL_LERP_PROGRESS_ATTR = "_push_robot_dr_interval_lerp_t"


def _curriculum_tracking_gate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    reward_fraction: float,
) -> bool:
    """True when the global step hits an episode boundary and tracking reward is high enough."""
    if env.common_step_counter % env.max_episode_length != 0:
        return False
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    return bool(reward > reward_term.weight * reward_fraction)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    """Widen ``ang_vel_z`` command range; register in ``CurriculumCfg`` to use (not enabled by default)."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def pose_reset_dr_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    reward_fraction: float = 0.8,
    pose_step: Mapping[str, float] | None = None,
    pose_range_limits: Mapping[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Expand ``reset_base`` ``pose_range`` toward ``pose_range_limits`` when the tracking gate passes."""
    pose_step = dict(pose_step or {})
    pose_range_limits = dict(pose_range_limits or {})
    term_cfg = env.event_manager.get_term_cfg("reset_base")
    pose_range: dict[str, Any] = dict(term_cfg.params.get("pose_range", {}))

    if _curriculum_tracking_gate(env, env_ids, reward_term_name, reward_fraction):
        for key in _POSE_KEYS:
            if key not in pose_range_limits:
                continue
            step = float(pose_step.get(key, 0.0))
            if step <= 0.0:
                continue
            lim_lo, lim_hi = pose_range_limits[key]
            cur_lo, cur_hi = pose_range.get(key, (0.0, 0.0))
            new_lo = max(float(cur_lo) - step, lim_lo)
            new_hi = min(float(cur_hi) + step, lim_hi)
            pose_range[key] = (new_lo, new_hi)
        term_cfg.params["pose_range"] = pose_range

    # Log max horizontal spawn half-range for visibility
    if "x" in pose_range:
        half = max(abs(pose_range["x"][0]), abs(pose_range["x"][1]))
    else:
        half = 0.0
    return torch.tensor(half, device=env.device)


def push_robot_dr_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    reward_fraction: float = 0.8,
    push_velocity_step: Mapping[str, float] | None = None,
    push_velocity_limits: Mapping[str, tuple[float, float]] | None = None,
    push_interval_lerp_start_s: tuple[float, float] | None = None,
    push_interval_lerp_end_s: tuple[float, float] | None = None,
    push_interval_lerp_delta: float = 0.0,
    push_interval_round_decimals: int = 0,
) -> torch.Tensor:
    """Expand ``push_robot`` ``velocity_range`` toward limits; optionally shorten ``interval_range_s`` by lerp.

    Interval curriculum: progress ``t`` in ``[0, 1]`` stored on the event term cfg; each passing gate adds
    ``push_interval_lerp_delta``. Current interval endpoints are ``round((1-t)*start + t*end, decimals)``.
    """
    push_velocity_step = dict(push_velocity_step or {})
    push_velocity_limits = dict(push_velocity_limits or {})
    term_cfg = env.event_manager.get_term_cfg("push_robot")
    vel_range: dict[str, Any] = dict(term_cfg.params.get("velocity_range", {}))

    if _curriculum_tracking_gate(env, env_ids, reward_term_name, reward_fraction):
        for key in _PUSH_VEL_KEYS:
            if key not in push_velocity_limits:
                continue
            step = float(push_velocity_step.get(key, 0.0))
            if step <= 0.0:
                continue
            lim_lo, lim_hi = push_velocity_limits[key]
            cur_lo, cur_hi = vel_range.get(key, (0.0, 0.0))
            new_lo = max(float(cur_lo) - step, lim_lo)
            new_hi = min(float(cur_hi) + step, lim_hi)
            vel_range[key] = (new_lo, new_hi)
        term_cfg.params["velocity_range"] = vel_range

        use_interval_lerp = (
            push_interval_lerp_start_s is not None
            and push_interval_lerp_end_s is not None
            and float(push_interval_lerp_delta) > 0.0
            and hasattr(term_cfg, "interval_range_s")
            and term_cfg.interval_range_s is not None
        )
        if use_interval_lerp:
            s_lo, s_hi = (float(push_interval_lerp_start_s[0]), float(push_interval_lerp_start_s[1]))
            e_lo, e_hi = (float(push_interval_lerp_end_s[0]), float(push_interval_lerp_end_s[1]))
            t_prev = float(getattr(term_cfg, _PUSH_INTERVAL_LERP_PROGRESS_ATTR, 0.0))
            t = min(1.0, t_prev + float(push_interval_lerp_delta))
            setattr(term_cfg, _PUSH_INTERVAL_LERP_PROGRESS_ATTR, t)
            nd = int(push_interval_round_decimals)
            n_lo = round((1.0 - t) * s_lo + t * e_lo, nd)
            n_hi = round((1.0 - t) * s_hi + t * e_hi, nd)
            if n_lo > n_hi:
                n_lo, n_hi = n_hi, n_lo
            n_lo = max(n_lo, 0.05)
            n_hi = max(n_hi, n_lo)
            term_cfg.interval_range_s = (float(n_lo), float(n_hi))

    # Log max planar push half-magnitude
    mag = 0.0
    if "x" in vel_range:
        mag = max(mag, max(abs(vel_range["x"][0]), abs(vel_range["x"][1])))
    if "y" in vel_range:
        mag = max(mag, max(abs(vel_range["y"][0]), abs(vel_range["y"][1])))
    return torch.tensor(mag, device=env.device)
