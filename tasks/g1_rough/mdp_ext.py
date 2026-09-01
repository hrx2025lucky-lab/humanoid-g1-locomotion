"""高度相关 MDP 项的扩展实现（本实践自写，零侵入）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题：世界系绝对高度在非平地上不成立
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基线 `velocity_env_cfg.py` 里两个 height 项都拿机器人根节点的**世界系 z**
直接和常数比：

    RewardsCfg.base_height      = base_height_l2(target_height=0.78)
    TerminationsCfg.base_height = root_height_below_minimum(minimum_height=0.2)

平地上地面 z ≡ 0，所以「世界系高度」和「离地高度」是同一个数，两者等价。
换到 ROUGH_TERRAINS_CFG 之后这个前提消失：倒金字塔楼梯和下坡的地面本身
低于 0，机器人**姿态完全正常**却因为「世界系 z 太低」被判摔倒。

16 环境冒烟测试实测：`Episode_Termination/base_height = 0.25`，
四分之一的 episode 是被这条误杀的。误杀不仅浪费采样，还会让策略学到
「不要走进低洼地形」这种完全错误的行为。

IsaacLab 官方对此是知情的，`root_height_below_minimum` 的 docstring 自己写着：

    This is currently only supported for flat terrains,
    i.e. the minimum height is in the world frame.

修法：用高度扫描传感器测出脚下真实地面高度，判据从
「机器人在世界里多高」换成「机器人相对脚下地面多高」。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么不直接用官方的 base_height_l2(sensor_cfg=...)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

官方奖励项**确实**支持 `sensor_cfg`，但内部是：

    adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)

没有任何 inf 防护。射线打空时 `ray_hits_w` 是 `inf`——这不是猜测，
IsaacLab 自己的可视化回调里就写着：

    # remove possible inf values
    viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]

`torch.mean` 只要输入里有一个 inf，整个环境的结果就是 inf
→ 奖励 inf → 反传出 NaN → 优化器把 NaN 写进权重 → 之后每一步输出都是 NaN。
它**不会抛异常**，只会让 `Policy/mean_noise_std` 变成 nan 而训练看起来还在跑。

终止项更直接：`root_height_below_minimum` 连 `sensor_cfg` 参数都没有，
只能自己实现。

所以这里两个都自己写，共用一个带防护的地面高度估计。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
零侵入
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

`RewTerm(func=...)` / `DoneTerm(func=...)` 接受任意可调用对象，不要求函数
来自 `isaaclab.envs.mdp`。因此本文件的函数可以直接顶替官方实现，
`IsaacLab` 与 `unitree_rl_lab` 一行都不用改。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ground_height_from_scan(sensor: RayCaster, fallback: float = 0.0) -> torch.Tensor:
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


def base_height_l2_safe(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
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


def root_height_below_minimum_adaptive(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """离地高度低于阈值时终止（官方版本只支持平地）。

    判据从 ``root_z < minimum_height`` 换成
    ``root_z - ground < minimum_height``，其中 ``ground`` 由高度扫描测得。

    射线全打空时 ``ground`` 回退为 0.0，判据自动退化成官方的世界系版本，
    因此这个改动在最坏情况下也不会比原来更差。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ground = ground_height_from_scan(sensor)
    return (asset.data.root_pos_w[:, 2] - ground) < minimum_height
