"""Coherent reference-state initialization for G1 AMP environments."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def reset_from_reference_motion(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    probability: float = 1.0,
    root_height_offset: float = 0.03,
    synchronize_command: bool = False,
    command_name: str = "base_velocity",
    fixed_axis_tolerance: float = 0.15,
) -> None:
    """Reset root and joints from the same sampled expert frame.

    Environments outside the RSI probability are reset to the asset defaults,
    so this single event fully replaces the ordinary root and joint events.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"RSI probability must be in [0, 1], got {probability}.")
    if env_ids.numel() == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    count = env_ids.numel()
    use_reference = torch.rand(count, device=env.device) < probability

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    if torch.any(use_reference):
        ids = torch.nonzero(use_reference, as_tuple=False).squeeze(-1)
        command_ranges = None
        if synchronize_command:
            command_term = env.command_manager.get_term(command_name)
            configured_ranges = command_term.cfg.ranges
            command_ranges_list = []
            for lower, upper in (
                configured_ranges.lin_vel_x,
                configured_ranges.lin_vel_y,
                configured_ranges.ang_vel_z,
            ):
                if lower == upper:
                    command_ranges_list.append((lower - fixed_axis_tolerance, upper + fixed_axis_tolerance))
                else:
                    command_ranges_list.append((lower, upper))
            command_ranges = tuple(command_ranges_list)
        reference = env.amp_expert_sampler.sample_reference_state(
            ids.numel(), command_ranges=command_ranges
        )
        reference_pos = reference.root_pos.clone()
        reference_pos[:, :2] = 0.0
        reference_pos[:, 2] += root_height_offset
        reference_pos += env.scene.env_origins[env_ids[ids]]
        root_state[ids, :3] = reference_pos
        root_state[ids, 3:7] = reference.root_quat
        root_state[ids, 7:10] = reference.root_lin_vel
        root_state[ids, 10:13] = reference.root_ang_vel
        joint_pos[ids] = reference.joint_pos
        joint_vel[ids] = reference.joint_vel
        if synchronize_command:
            env._amp_pending_reference_commands = (
                env_ids[ids].clone(),
                reference.motion_command.clone(),
                command_name,
            )

    joint_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])
    velocity_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel.clamp_(-velocity_limits, velocity_limits)
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


__all__ = ["reset_from_reference_motion"]
