from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ground_height_from_scan(sensor: RayCaster, fallback: float=0.0) -> torch.Tensor:
    """从高度扫描点云估计每个环境脚下的地面高度，忽略打空的射线。

    Args:
        sensor: 高度扫描传感器。``data.ray_hits_w`` 形状 (num_envs, num_rays, 3)，
            打空的射线该行为 ``inf``。
        fallback: 一个环境的射线**全部**打空时的回退值。默认 0.0，
            即退化成基线的世界系原点假设——这是最保守的选择：
            此时行为与修改前完全一致，不会引入新的失败模式。

    Returns:
        形状 (num_envs,) 的地面高度。

    实现要点：不能用 ``torch.mean`` 后再 ``nan_to_num``——inf 会先污染整个均值，
    补救时真实信息已经丢了。必须**先掩掉 inf 再求均值**。
    """
    z = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(z)
    n_valid = valid.sum(dim=1)
    z_sum = torch.where(valid, z, torch.zeros_like(z)).sum(dim=1)
    mean_valid = z_sum / n_valid.clamp(min=1)
    return torch.where(n_valid > 0, mean_valid, torch.full_like(mean_valid, fallback))

def base_height_l2_safe(env: ManagerBasedRLEnv, target_height: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    """按地形调整目标高度的 base height L2 惩罚，带 inf 防护。

    与官方 ``base_height_l2(sensor_cfg=...)`` 的唯一区别是地面高度经
    :func:`ground_height_from_scan` 估计，射线打空不会污染结果。

    ``target_height`` 的含义随之变成**离地高度**而不是世界系高度，
    数值不用改：平地上两者相等，G1 的 0.78 m 依然成立。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ground = ground_height_from_scan(sensor)
    return torch.square(asset.data.root_pos_w[:, 2] - ground - target_height)

def foot_clearance_reward_rough(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ground = ground_height_from_scan(sensor).unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - ground
    foot_z_target_error = torch.square(foot_z - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def root_height_below_minimum_adaptive(env: ManagerBasedRLEnv, minimum_height: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    """离地高度低于阈值时终止（官方版本只支持平地）。

    判据从 ``root_z < minimum_height`` 换成
    ``root_z - ground < minimum_height``，其中 ``ground`` 由高度扫描测得。

    射线全打空时 ``ground`` 回退为 0.0，判据自动退化成官方的世界系版本，
    因此这个改动在最坏情况下也不会比原来更差。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ground = ground_height_from_scan(sensor)
    return asset.data.root_pos_w[:, 2] - ground < minimum_height

class IllegalResetContact(ManagerTermBase):
    """P2 §5.8: terminate after repeated non-foot contact within an episode.

    Count control steps with a contact in the sensor history. Counts accumulate
    until reset, as in the assignment example; a clean step does not erase them.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.illegal_contact_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)

    def __call__(self, env, sensor_cfg: SceneEntityCfg, threshold: float=1.0, episode_length_threshold: int=5) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        contacts = (torch.linalg.vector_norm(forces, dim=-1).amax(dim=1) > threshold).any(dim=1)
        self.illegal_contact_counter += contacts.int()
        return (self.illegal_contact_counter >= episode_length_threshold) & (env.episode_length_buf >= episode_length_threshold)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.illegal_contact_counter[env_ids] = 0
