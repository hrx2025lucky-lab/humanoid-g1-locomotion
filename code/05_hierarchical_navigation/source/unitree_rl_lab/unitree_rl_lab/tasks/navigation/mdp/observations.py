from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.envs.mdp.observations import height_scan as _height_scan
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def last_high_level_command(env: ManagerBasedRLEnv, action_name: str = "pre_trained_policy_action") -> torch.Tensor:
    """Previous high-level velocity command after command clipping."""
    # 提示：高层策略需要观察真正传给低层策略的速度指令，而不是裁剪前的原始动作。
    # 可通过 action_manager 按名称取得 action term，并读取其 processed_actions。
    # >>> HOMEWORK_TODO_1_START
    # 必须返回 processed_actions 而非 raw_actions。
    # 高层策略的输出会被逐维裁剪到低层熟悉的速度范围（见 TODO 4），
    # 真正送进低层策略的是裁剪后的值。若这里返回裁剪前的原始动作，
    # 高层就会以为"我上一步下的指令是 2.5 m/s"，而低层实际只收到 1.0 m/s:
    # 观测与现实脱节，高层学到的因果关系是错的。
    # 动作饱和时这个差别最大，而饱和恰恰是训练早期的常态。
    return env.action_manager.get_term(action_name).processed_actions
    # <<< HOMEWORK_TODO_1_END


def low_level_last_action(env: ManagerBasedRLEnv, action_name: str = "pre_trained_policy_action") -> torch.Tensor:
    """Last 29-DoF joint action emitted by the frozen low-level policy."""
    return env.action_manager.get_term(action_name).low_level_actions


def command_distance(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """2D distance to the active pose command."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1, keepdim=True)


def _height_scan_grid_shape(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> tuple[int, int]:
    """Return (ny, nx) ray-grid shape matching Isaac Lab ``grid_pattern`` with ordering ``xy``."""
    sensor = env.scene.sensors[sensor_cfg.name]
    pattern = sensor.cfg.pattern_cfg
    device = env.device
    x = torch.arange(start=-pattern.size[0] / 2, end=pattern.size[0] / 2 + 1.0e-9, step=pattern.resolution, device=device)
    y = torch.arange(start=-pattern.size[1] / 2, end=pattern.size[1] / 2 + 1.0e-9, step=pattern.resolution, device=device)
    return y.numel(), x.numel()


def height_scan_pooled(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    pool_size: int = 2,
) -> torch.Tensor:
    """Max-pooled height scan: 2D grid from the scanner, then ``pool_size`` max-pool, flattened."""
    # 提示：先调用 _height_scan 得到扁平射线高度，再依据 (ny, nx) 恢复二维网格。
    # 为适配 F.max_pool2d，需要添加通道维；池化后再展平为 (num_envs, -1)。
    # >>> HOMEWORK_TODO_2_START
    # 为什么要池化：V5 扫描区 5.0×3.0 m @ 0.12 m 分辨率 = 42×26 = 1092 根射线。
    # 直接进网络会让第一层权重爆炸式增长，且相邻射线高度高度相关、信息冗余。
    # 2×2 最大池化后降到 21×13 = 273 维，配合其余 103 维本体/目标观测得到 376 维。
    #
    # 关键：必须先还原成二维网格再池化，不能对扁平向量直接做一维池化。
    # 射线在扁平向量里按 (y, x) 行主序排列，相邻元素是"同一行左右相邻的两点"，
    # 但每 nx 个元素就跨到下一行。一维池化会把行尾和下一行行首混在一起，
    # 得到的空间布局完全错乱,而且不会报错，只会让策略读到无意义的地形图。
    #
    # 取 max 而非 mean：导航关心的是"前方有没有障碍物"，
    # 最大值保留了网格内最高的那个点（障碍），均值会把孤立的细柱子抹平。
    # 对避障来说漏掉一根柱子的代价远大于高估地形高度。
    flat = _height_scan(env, sensor_cfg, offset=offset)  # (num_envs, ny * nx)
    ny, nx = _height_scan_grid_shape(env, sensor_cfg)
    # F.max_pool2d 要求 (N, C, H, W)，这里补一个单通道维
    grid = flat.view(-1, 1, ny, nx)
    pooled = F.max_pool2d(grid, kernel_size=pool_size)
    return pooled.flatten(start_dim=1)
    # <<< HOMEWORK_TODO_2_END
