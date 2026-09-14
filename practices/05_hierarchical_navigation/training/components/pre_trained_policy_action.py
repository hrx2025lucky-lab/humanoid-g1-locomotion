from __future__ import annotations
from dataclasses import MISSING
from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
_LOW_LEVEL_ACTION_LIMIT = 10.0

class PreTrainedPolicyAction(ActionTerm):
    """Run a frozen low-level locomotion policy from high-level velocity commands."""
    cfg: PreTrainedPolicyActionCfg

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_low_level_action():
            if hasattr(env, 'episode_length_buf'):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._processed_actions
        cfg.low_level_observations.velocity_commands.params = dict()
        _action_obs_name = 'last_action' if hasattr(cfg.low_level_observations, 'last_action') else 'actions'
        _action_obs_term = getattr(cfg.low_level_observations, _action_obs_name)
        _action_obs_term.func = lambda dummy_env: last_low_level_action()
        _action_obs_term.params = dict()
        self._low_level_obs_manager = ObservationManager({'ll_policy': cfg.low_level_observations}, env)
        self._clip_lower = torch.tensor([limit[0] for limit in cfg.velocity_clip], device=self.device)
        self._clip_upper = torch.tensor([limit[1] for limit in cfg.velocity_clip], device=self.device)
        self._counter = 0

    def reset(self, env_ids: Sequence[int] | None=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.low_level_actions[env_ids] = 0.0
        self._counter = 0
        self._low_level_obs_manager.reset(env_ids)
        self._low_level_action_term.reset(env_ids)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        clipped = torch.clamp(actions, self._clip_lower, self._clip_upper)
        alpha = float(getattr(self.cfg, 'command_smoothing', 1.0))
        if alpha >= 1.0:
            self._processed_actions[:] = clipped
        else:
            self._processed_actions.mul_(1.0 - alpha).add_(clipped, alpha=alpha)
            if hasattr(self._env, 'episode_length_buf'):
                just_reset = self._env.episode_length_buf == 0
                if just_reset.any():
                    self._processed_actions[just_reset] = clipped[just_reset]

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group('ll_policy', update_history=True)
            with torch.inference_mode():
                raw_ll = self.policy(low_level_obs)
            torch.clamp(raw_ll, -_LOW_LEVEL_ACTION_LIMIT, _LOW_LEVEL_ACTION_LIMIT, out=self.low_level_actions)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, 'base_vel_goal_visualizer'):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = '/Visuals/Actions/velocity_goal'
                marker_cfg.markers['arrow'].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = '/Visuals/Actions/velocity_current'
                marker_cfg.markers['arrow'].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        elif hasattr(self, 'base_vel_goal_visualizer'):
            self.base_vel_goal_visualizer.set_visibility(False)
            self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        (vel_des_arrow_scale, vel_des_arrow_quat) = self._resolve_xy_velocity_to_arrow(self.processed_actions[:, :2])
        (vel_arrow_scale, vel_arrow_quat) = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.base_vel_goal_visualizer.cfg.markers['arrow'].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return (arrow_scale, arrow_quat)

@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for the frozen low-level policy action term."""
    class_type: type[ActionTerm] = PreTrainedPolicyAction
    asset_name: str = MISSING
    policy_path: str = MISSING
    low_level_decimation: int = 4
    low_level_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING
    velocity_clip: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-0.5, 1.0), (-0.5, 0.5), (-0.5, 0.5))
    debug_vis: bool = True
    command_smoothing: float = 1.0
    '高层指令送入低层前的 EMA 平滑系数（1.0 = 不平滑，保持原行为）。\n\n    低层策略是在 `resampling_time_range=(10,10)` 秒、即**指令每 10 秒才换一次**\n    的分布上训练的；而 HRL 的高层每个 env step（0.2 s）就重新采样一次，\n    快了 50 倍。实测这个分布偏移会让机器人平均每 22.6 步就摔一次，\n    而下发恒定指令时它能稳定行走 —— 摔倒的原因是指令抖动，不是指令幅度。\n\n    每次摔倒触发 `termination_penalty = -400`，使 episode 回报稳定在 -400 量级\n    （实践 2 的量级只有 ~26）。MSE 价值损失按平方放大，初始 value loss 因此\n    高达 6×10^4，critic 在第 2 个 iteration 就发散到 10^27，\n    最终 float32 溢出使权重变 NaN，训练以 `normal expects all elements of\n    std >= 0.0` 崩溃。\n\n    EMA 让低层看到缓慢变化的指令，同时保留高层的响应能力：\n    α=0.1 在 5 Hz 下对应约 1.9 s 的时间常数。\n    '
