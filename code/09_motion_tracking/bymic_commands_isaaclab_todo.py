from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(
        self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device
        )
        self.joint_vel = torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device
        )
        self._body_pos_w = torch.tensor(
            data["body_pos_w"], dtype=torch.float32, device=device
        )
        self._body_quat_w = torch.tensor(
            data["body_quat_w"], dtype=torch.float32, device=device
        )
        self._body_lin_vel_w = torch.tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device
        )
        self._body_ang_vel_w = torch.tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device
        )
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name
        )
        self.motion_anchor_body_index = self.cfg.body_names.index(
            self.cfg.anchor_body_name
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        self.motion = MotionLoader(
            self.cfg.motion_file, self.body_indexes, device=self.device
        )
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = (
            int(
                self.motion.time_step_total
                // (1 / (env.cfg.decimation * env.cfg.sim.dt))
            )
            + 1
        )
        self.bin_failed_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_rot"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_lin_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_ang_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_prob"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_bin"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        # 自适应采样：让训练重点落在"机器人经常摔倒的动作片段"上。
        # 整段动作被切成 bin_count 个时间区间，失败次数多的区间获得更高的
        # 起始帧采样概率,相当于对着难点反复练，而不是每次都从头均匀重放。

        # 1. 读取失败环境。terminated 是"非超时终止"的掩码，即真正的失败
        #（摔倒/姿态越界），超时结束不算失败。
        episode_failed = self._env.termination_manager.terminated[env_ids]

        # 2. 把动作帧号映射到 bin。整数除法天然向下取整，再 clamp 防止
        # time_steps 恰好等于 total 时算出越界的 bin_count。
        current_bin_index = (self.time_steps * self.bin_count) // max(
            self.motion.time_step_total, 1
        )
        current_bin_index = torch.clamp(current_bin_index, 0, self.bin_count - 1)

        # 3. 统计本轮各 bin 的失败次数。
        # minlength 必不可少：没有它 bincount 只统计到"出现过的最大 bin"，
        # 返回张量长度会随数据变化，与 bin_failed_count 形状对不上。
        self._current_bin_failed[:] = torch.bincount(
            current_bin_index[env_ids][episode_failed], minlength=self.bin_count
        )

        # 4. 基础概率 = 历史失败热度 + 均匀探索项。
        # 均匀项保证任何 bin 概率都不为 0：否则一个从未失败过的区间永远
        # 不会被采到，策略会逐渐遗忘那段动作（catastrophic forgetting）。
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(
            self.bin_count
        )

        # 5. 卷积平滑。失败是稀疏事件，某个 bin 失败往往意味着它前后
        # 一小段都不好走。把热度向邻域扩散，训练才会覆盖整个困难区段
        # 而不是死磕单帧。kernel 是几何衰减序列（adaptive_lambda^i）且已归一化。
        # 右侧 padding 后，当前输出会读取右侧输入；对应非因果扩散核。
        sampling_probabilities = torch.nn.functional.conv1d(
            torch.nn.functional.pad(
                sampling_probabilities.view(1, 1, self.bin_count),
                (0, self.cfg.adaptive_kernel_size - 1),
            ),
            self.kernel.view(1, 1, -1),
        ).view(-1)

        # 6. 归一化成概率分布。
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # 7. 按概率抽 bin。replacement=True：多个环境可以落在同一个困难区间。
        sampled_bins = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )

        # 8. bin 内再加 [0,1) 的随机偏移后换算回帧号。
        # 没有这个偏移，所有落在同一 bin 的环境会从完全相同的帧起步，
        # 采样多样性退化成 bin_count 个离散点。
        #
        # 先转为整段动作的连续相位，再按 T-1 缩放；与 MJLab 参考实现一致。
        # 不能先用 T // bin_count 截断区间宽度，否则非整除长度会丢掉尾段。
        # 采样的是起始时刻；末帧通常作为后续跟踪的终点，不必单独选作起点。
        offsets = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        sampled_time_steps = (
            (sampled_bins + offsets) / self.bin_count
            * (self.motion.time_step_total - 1)
        )
        self.time_steps[env_ids] = torch.clamp(
            sampled_time_steps, 0, self.motion.time_step_total - 1
        ).long()

        # 9. 采样分布的诊断指标。这三个量用来判断采样是否"塌缩":
        # 若归一化熵长期接近 0、top1_bin 长期不动，说明训练卡在同一个
        # 片段反复失败，其余动作正在被遗忘。此时应检查 termination 是否过严。
        #
        # 熵除以 log(bin_count) 归一化到 [0,1]：均匀分布得 1，
        # 全部概率集中在一个 bin 得 0。这样不同 bin_count 的实验可以横向比较。
        # clamp_min(1e-12) 防止 log(0)；max(bin_count,2) 防止 bin_count=1 时除零。
        #
        # top1_bin 同样归一化为 imax/bin_count，表示"最难的片段在整段动作的
        # 什么位置"（0=开头，接近 1=结尾），与动作总长无关，便于跨实验对比。
        probabilities = sampling_probabilities.clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum()
        normalized_entropy = entropy / math.log(max(self.bin_count, 2))
        top1_prob, top1_bin = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = normalized_entropy
        self.metrics["sampling_top1_prob"][:] = top1_prob
        self.metrics["sampling_top1_bin"][:] = top1_bin.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(
            *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids],
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_ori[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

    def _update_command(self):
        # 每个控制步推进一帧参考动作，并把参考动作"对齐"到机器人当前所在的
        # 位置与朝向,否则参考轨迹固定在世界原点，机器人一旦偏离就再也追不上。

        # 1. 时间前进一帧。
        self.time_steps += 1

        # 2. 播放到结尾的环境重新采样起始帧（内部走自适应采样）。
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        # 3. anchor 是"参考动作与机器人对齐所用的基准刚体"（通常是躯干）。
        # 下面要对每个 body 做同样的变换，先把 anchor 量沿 body 维复制。
        body_count = len(self.cfg.body_names)
        anchor_pos_w_repeat = self.anchor_pos_w.unsqueeze(1).repeat(1, body_count, 1)
        anchor_quat_w_repeat = self.anchor_quat_w.unsqueeze(1).repeat(1, body_count, 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w.unsqueeze(1).repeat(
            1, body_count, 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w.unsqueeze(1).repeat(
            1, body_count, 1
        )

        # 4. 位置对齐：x/y 用机器人的真实位置，z 用参考动作的高度。
        # 这是本实践的一个关键设计,水平方向不要求机器人走到参考轨迹的
        # 绝对坐标上（那等于同时考核导航），只要求它把动作"就地做对"；
        # 但竖直方向必须跟参考，否则蹲下/跳起这类高度变化就无从考核。
        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]

        # 5. 姿态对齐：只保留 yaw。
        # quat_mul(robot_anchor, inv(motion_anchor)) 得到"从参考朝向转到机器人
        # 当前朝向"的旋转；yaw_quat 把它投影成纯 yaw。
        # 为什么只留 yaw：roll/pitch 是机器人自己的平衡姿态，属于被考核内容，
        # 若一并对齐就等于把姿态误差抹掉了，跟踪奖励会失去意义。
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )

        # 6. 构造对齐后的目标位姿：先把参考 body 位置转成相对 anchor 的偏移，
        # 用 delta_ori_w 旋转到机器人当前朝向，再平移到 delta_pos_w。
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

        # 7. 用 EMA 更新失败热度。alpha 越大越看重最近一轮，越小则记忆越长。
        # 用滑动平均而非直接累加：策略在进步，早期的失败记录会过时，
        # 一直累加会让采样长期困在"曾经很难但现在已经学会"的片段上。
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1.0 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )

        # 8. 清空本轮临时计数，避免同一批失败被下一轮 EMA 重复计入。
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/current/anchor"
                    )
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/goal/anchor"
                    )
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/current/" + name
                            )
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + name
                            )
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
            else:
                return
            
    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(
            self.robot_anchor_pos_w, self.robot_anchor_quat_w
        )
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(
                self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i]
            )
            self.goal_body_visualizers[i].visualize(
                self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i]
            )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
