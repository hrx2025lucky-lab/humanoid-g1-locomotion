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


# 冻结低层策略的输出限幅。正常关节动作在 ±1 量级，±10 远超合理范围，
# 只用于截断策略在分布外姿态下吐出的极端值（实测可达 1.7e6）。
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
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # 提示：
        # 1. 用 read_file + torch.jit.load 加载随仓库提供的低层策略，并切换到 eval 模式。
        # 2. 将低层观测中的 velocity_commands 绑定到裁剪后的高层动作。
        # 3. 将 last_action（或兼容名称 actions）绑定到上一次低层关节动作。
        # 4. 最后创建只含 ll_policy 组的 ObservationManager。
        # >>> HOMEWORK_TODO_3_START
        # 加载冻结的低层策略。read_file 支持本地路径与远程 URI，
        # torch.jit.load 读的是 TorchScript 归档,低层策略是训练完导出的静态图，
        # 没有 Python 源码依赖，这正是它能被当作"技能库"复用的原因。
        #
        # .eval() 关掉 dropout/BN 的训练态。这里还有一层含义：低层网络不参与
        # 高层的梯度更新。分层 RL 的前提就是低层能力固定，高层只学"下什么指令"；
        # 若低层跟着一起更新，高层刚学会的映射关系下一步就失效了，训练无法收敛。
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        # 把低层观测里的两项"重新接线"到本 action term 的内部状态。
        #
        # velocity_commands：低层训练时它来自 CommandManager 的 base_velocity 指令，
        # 但导航环境里根本没有这个 command,速度指令现在由高层策略产生。
        # 所以要把它的 func 换成"返回高层的输出"，并清空原来的 params
        #（原 params 是 {"command_name": "base_velocity"}，不清会被当多余参数传进去）。
        #
        # 绑定 _processed_actions 而不是 _raw_actions：低层实际执行的是裁剪后的指令。
        # IsaacLab 官方示例绑的是 raw，那样低层观测里的"我收到的指令"会与它真正
        # 被驱动的值不一致，动作饱和时尤其严重。
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._processed_actions
        cfg.low_level_observations.velocity_commands.params = dict()

        # last_action：低层需要看到"自己上一步输出的 29 维关节动作"。
        # 这个值由本 term 维护（low_level_actions），不能走默认的 mdp.last_action:
        # 那个返回的是高层的 3 维动作，维度和语义都不对。
        # 兼容两种字段名：本仓库低层配置用 last_action，IsaacLab 官方示例用 actions。
        _action_obs_name = "last_action" if hasattr(cfg.low_level_observations, "last_action") else "actions"
        _action_obs_term = getattr(cfg.low_level_observations, _action_obs_name)
        _action_obs_term.func = lambda dummy_env: last_low_level_action()
        _action_obs_term.params = dict()

        # 单独建一个只含 ll_policy 组的 ObservationManager。
        # 它与环境主观测管理器互不干扰：主管理器算高层的 376 维，
        # 这个算低层的 (6 项 × 5 帧历史)。低层的输入契约必须原封不动，
        # 所以这里用的是 deepcopy 自低层训练配置的 group（见 TODO 6）。
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        # <<< HOMEWORK_TODO_3_END

        self._clip_lower = torch.tensor([limit[0] for limit in cfg.velocity_clip], device=self.device)
        self._clip_upper = torch.tensor([limit[1] for limit in cfg.velocity_clip], device=self.device)
        self._counter = 0

    def reset(self, env_ids: Sequence[int] | None = None):
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
        # >>> HOMEWORK_TODO_4_START
        # 两份动作各有用途，不能只留一份：
        #   _raw_actions       高层网络的原始输出，用于日志/诊断（看动作是否长期饱和）
        #   _processed_actions 裁剪到低层熟悉范围后的指令，这才是真正执行的
        #
        # 为什么必须裁剪：低层策略是在 vx∈[-0.5,1.0]、vy/ωz∈[-0.5,0.5] 的指令分布上
        # 训练出来的。高层是个没有输出限幅的高斯策略，训练早期完全可能吐出 5.0 m/s。
        # 把分布外的指令喂给低层，它的行为不可预测（通常是直接摔倒），
        # 高层会因此收到一个与自身决策无关的噪声回报，学不到有效策略。
        # 裁剪等于给两层之间的接口加了一道契约保护。
        #
        # 用 [:] 原地写入而不是重新赋值：这两个张量在 __init__ 里预分配，
        # 且被 TODO 3 的闭包 (lambda: self._processed_actions) 捕获。
        # 若在这里 self._processed_actions = ... 重新绑定对象，
        # 闭包仍指向旧张量，低层会一直读到初始的零指令。
        self._raw_actions[:] = actions
        clipped = torch.clamp(actions, self._clip_lower, self._clip_upper)
        alpha = float(getattr(self.cfg, "command_smoothing", 1.0))
        if alpha >= 1.0:
            self._processed_actions[:] = clipped
        else:
            # EMA 平滑：低层是在指令每 10 秒才变一次的分布上训练的，
            # 直接把每 0.2 秒重采样的高层输出喂进去属于分布外输入。
            # 仍然用 [:] 原地写入: 闭包捕获的是这个张量对象本身。
            self._processed_actions.mul_(1.0 - alpha).add_(clipped, alpha=alpha)
            # 环境重置时必须清零，否则新 episode 会继承上一条轨迹的指令惯性
            if hasattr(self._env, "episode_length_buf"):
                just_reset = self._env.episode_length_buf == 0
                if just_reset.any():
                    self._processed_actions[just_reset] = clipped[just_reset]
        # <<< HOMEWORK_TODO_4_END

    def apply_actions(self):
        # >>> HOMEWORK_TODO_5_START
        # 分层控制的多时间尺度：
        #   高层 planner   每个 env step 出一次速度指令        (~10 Hz)
        #   低层 locomotion 每 low_level_decimation=4 步推理一次 (~50 Hz)
        #   PD / 物理步     每个物理步都要执行                  (~200 Hz)
        #
        # 注意 apply_actions() 在 if 之外：低层网络虽然只在计数器归零时前向一次，
        # 但它算出的关节目标必须在随后的每个物理步持续驱动 PD 控制器。
        # 若把 apply_actions 写进 if 里，关节目标只在 1/4 的物理步被写入，
        # 其余 3 步执行器收不到指令，机器人会抽搐。这是作业讲解点名的高频错误。
        #
        # inference_mode 比 no_grad 更彻底：它连版本计数都不记录，
        # 对这种"每步都跑、永不反传"的冻结网络能省下可观的显存与开销。
        if self._counter % self.cfg.low_level_decimation == 0:
            # update_history=True 不可省略。
            # 它默认是 False，此时 ObservationManager 走 compute_group 里的
            # elif 分支：每次都新建一个 CircularBuffer 并用当前帧填满 5 槽
            #（那句 `circular_buffer = CircularBuffer(...)` 只重新绑定局部变量，
            # 没有存回 self._group_obs_term_history_buffer，所以永远进不了
            # update 分支）。结果低层看到的"5 帧历史"是当前帧重复 5 次，
            # 它永远拿不到时间差分信息。
            #
            # IsaacLab 正常环境走的是 observation_manager.compute(update_history=True)
            #（manager_based_rl_env.py:238）。官方 navigation 示例照搬到这里时不传
            # 该参数是无害的: 因为官方低层观测 history_length=0，此参数是 no-op。
            # 但本课程的低层策略用 5 帧历史（low_level_env_cfg.py 的 history_length=5），
            # 照搬就成了静默 bug：量级完全正常，只有时间结构是错的，
            # 所以基于均值/最大值的对比全都发现不了。
            low_level_obs = self._low_level_obs_manager.compute_group(
                "ll_policy", update_history=True
            )
            with torch.inference_mode():
                raw_ll = self.policy(low_level_obs)
            # 在源头限幅。低层是冻结的 TorchScript，机器人进入其训练分布之外的
            # 姿态（摔倒过程中）时会吐出极端值,实测达到 1.7e6。
            # 这个值有两条去路，都会出事：
            #   1. 经 low_level_last_action 进高层观测 → V(s) 爆炸 → critic 发散
            #   2. 经 PD 控制器变成巨大力矩 → 把机器人掀翻 → 更多摔倒 → 正反馈
            # 正常关节动作在 ±1 量级（×action_scale=0.25 后为 ±0.25 rad），
            # ±10 远超合理范围，截断不会损失任何有效信号。
            torch.clamp(raw_ll, -_LOW_LEVEL_ACTION_LIMIT, _LOW_LEVEL_ACTION_LIMIT,
                        out=self.low_level_actions)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1
        # <<< HOMEWORK_TODO_5_END

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "base_vel_goal_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)

                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)

            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.processed_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for the frozen low-level policy action term."""

    class_type: type[ActionTerm] = PreTrainedPolicyAction
    asset_name: str = MISSING
    policy_path: str = MISSING
    low_level_decimation: int = 4
    low_level_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING
    velocity_clip: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.5, 1.0),
        (-0.5, 0.5),
        (-0.5, 0.5),
    )
    debug_vis: bool = True
    command_smoothing: float = 1.0
    """高层指令送入低层前的 EMA 平滑系数（1.0 = 不平滑，保持原行为）。

    低层策略是在 `resampling_time_range=(10,10)` 秒、即指令每 10 秒才换一次
    的分布上训练的；而 HRL 的高层每个 env step（0.2 s）就重新采样一次，
    快了 50 倍。实测这个分布偏移会让机器人平均每 22.6 步就摔一次，
    而下发恒定指令时它能稳定行走: 摔倒的原因是指令抖动，不是指令幅度。

    每次摔倒触发 `termination_penalty = -400`，使 episode 回报稳定在 -400 量级
    （实践 2 的量级只有 ~26）。MSE 价值损失按平方放大，初始 value loss 因此
    高达 6×10^4，critic 在第 2 个 iteration 就发散到 10^27，
    最终 float32 溢出使权重变 NaN，训练以 `normal expects all elements of
    std >= 0.0` 崩溃。

    EMA 让低层看到缓慢变化的指令，同时保留高层的响应能力：
    α=0.1 在 5 Hz 下对应约 1.9 s 的时间常数。
    """
