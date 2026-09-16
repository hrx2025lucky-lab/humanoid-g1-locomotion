"""Explicit and student-editable G1 AMP train/Play environment configurations."""

from __future__ import annotations

import importlib

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from unitree_rl_lab.tasks.locomotion import mdp

from ...motion_dataset import MotionDatasetCfg
from ...reference_reset import reset_from_reference_motion
from .motion_cfg import (
    G1_AMP_KEY_LINK_NAMES,
    G1DanceMotionCfg,
    G1MixedMotionCfg,
    G1MotionSourceCfg,
    G1OmniRunMotionCfg,
    G1RunMotionCfg,
    G1WalkMotionCfg,
    G1WalkToRunMotionCfg,
    make_motion_dataset_cfg,
)


_velocity_cfg = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
)
VelocityObservationsCfg = _velocity_cfg.ObservationsCfg
VelocityEventCfg = _velocity_cfg.EventCfg
VelocityCurriculumCfg = _velocity_cfg.CurriculumCfg
VelocityRobotEnvCfg = _velocity_cfg.RobotEnvCfg


def key_link_positions_in_base(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return selected body origins expressed in the robot root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    count = body_pos_w.shape[1]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, count, -1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, count, -1)
    positions_b = quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
    return positions_b.flatten(start_dim=1)


@configclass
class AMPObservationsCfg(VelocityObservationsCfg):
    @configclass
    class AMPCfg(ObsGroup):
        # 单帧 AMP 特征 d_amp = 3 + 3 + 3 + 1 + 29 + 29 + 4×3 = 80。
        #
        # 顺序必须与 motion_dataset.py 顶部的 AMP_FEATURE_ORDER 逐项对齐：
        #   base_lin_vel → base_ang_vel → projected_gravity → base_height
        #   → joint_pos → joint_vel → key_links_pos_b
        # 判别器把策略帧和专家帧当同一分布比较，顺序错位不会报错，
        # 只会让判别器学到"第 i 维含义不同"这种伪差异: 训练能跑但风格奖励无意义。
        #
        # 这里全部用基座坐标系的量（而非世界系）：专家数据来自 GMR 重定向，
        # 世界系位置和朝向与仿真里的机器人无关，只有基座系特征才可比。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_height = ObsTerm(func=mdp.base_pos_z)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        key_links_pos_b = ObsTerm(
            func=key_link_positions_in_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=list(G1_AMP_KEY_LINK_NAMES))},
        )

        def __post_init__(self):
            # 判别器要区分的是"动作风格"，不是"传感器噪声"。
            # 若给策略侧加噪而专家侧无噪，判别器可以靠噪声本身区分两者，
            # 风格奖励就退化成了"噪声检测器"。
            self.enable_corruption = False
            # 拼成单个 [B, 80] 向量，AMPRunner 再按 history_steps 堆成 [B, H, 80]
            self.concatenate_terms = True

    amp: AMPCfg = AMPCfg()


@configclass
class AMPEventCfg(VelocityEventCfg):
    reset_reference = EventTerm(
        func=reset_from_reference_motion,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "probability": 0.5, "root_height_offset": 0.03},
    )


@configclass
class AMPCurriculumCfg(VelocityCurriculumCfg):
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)


@configclass
class G1AMPFlatEnvCfg(VelocityRobotEnvCfg):
    """Common flat-ground AMP wiring; motion subclasses below contain task values."""

    observations: AMPObservationsCfg = AMPObservationsCfg()
    events: AMPEventCfg = AMPEventCfg()
    curriculum: AMPCurriculumCfg = AMPCurriculumCfg()
    motion_style: str = "mixed"
    motion_source: G1MotionSourceCfg = G1MixedMotionCfg()
    amp_history_steps: int = 3
    amp_motion: MotionDatasetCfg = make_motion_dataset_cfg(G1MixedMotionCfg())

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        self.amp_motion = make_motion_dataset_cfg(self.motion_source, self.amp_history_steps)
        self.amp_motion.step_dt = self.sim.dt * self.decimation
        self.events.reset_base = None
        self.events.reset_robot_joints = None


@configclass
class G1AMPWalkEnvCfg(G1AMPFlatEnvCfg):
    """Forward, backward, and turning walking."""

    motion_style: str = "walk"
    motion_source: G1WalkMotionCfg = G1WalkMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_reference.params["probability"] = 0.7

        self.rewards.track_lin_vel_xy.weight = 1.5
        self.rewards.track_ang_vel_z.weight = 0.5
        self.rewards.action_rate.weight = -0.03
        self.rewards.base_linear_velocity.weight = -2.0
        self.rewards.base_angular_velocity.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height.weight = -10.0
        self.rewards.gait.weight = 0.5
        self.rewards.feet_clearance.weight = 1.0
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_waists.weight = -1.0
        self.rewards.joint_deviation_legs.weight = -1.0

        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 1.3)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.6, 0.6)


@configclass
class G1AMPRunEnvCfg(G1AMPWalkEnvCfg):
    """Pure steady forward running used as the AMP diagnostic baseline."""

    motion_style: str = "run"
    motion_source: G1RunMotionCfg = G1RunMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy.weight = 1.8
        self.rewards.base_linear_velocity = None
        self.rewards.base_angular_velocity = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.base_height = None
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waists = None
        self.rewards.joint_deviation_legs = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (1.0, 4.2)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1AMPOmniRunEnvCfg(G1AMPRunEnvCfg):
    """Steady-run-dominant motion with commanded left and right turns."""

    motion_style: str = "omni_run"
    motion_source: G1OmniRunMotionCfg = G1OmniRunMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.2, 1.2)


@configclass
class G1AMPWalkToRunEnvCfg(G1AMPWalkEnvCfg):
    """Task-goal-driven reverse walk, walk, transition, run, and turning task."""

    motion_style: str = "walk_to_run"
    motion_source: G1WalkToRunMotionCfg = G1WalkToRunMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_reference.params["probability"] = 0.5
        self.events.reset_reference.params["synchronize_command"] = True
        self.rewards.track_lin_vel_xy.weight = 1.8
        self.rewards.action_rate.weight = -0.015
        self.rewards.joint_vel.weight = -0.0005
        self.rewards.joint_acc.weight = -1.25e-7
        self.rewards.feet_slide.weight = -0.1
        self.rewards.base_linear_velocity.weight = -0.5
        self.rewards.base_angular_velocity.weight = -0.02
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height.weight = -2.0
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waists = None
        self.rewards.joint_deviation_legs = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 4.2)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.2, 1.2)


@configclass
class G1AMPDanceEnvCfg(G1AMPFlatEnvCfg):
    """Near-stationary Irish and Salsa motion."""

    motion_style: str = "dance"
    motion_source: G1DanceMotionCfg = G1DanceMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_reference.params["probability"] = 0.8
        self.rewards.track_lin_vel_xy.weight = 0.05
        self.rewards.track_ang_vel_z.weight = 0.05
        self.rewards.alive.weight = 0.12
        self.rewards.joint_acc.weight = -1.0e-7
        self.rewards.action_rate.weight = -0.01
        self.rewards.dof_pos_limits.weight = -0.5
        self.rewards.base_linear_velocity = None
        self.rewards.base_angular_velocity = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.base_height = None
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waists = None
        self.rewards.joint_deviation_legs = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.5, 0.5)
        self.terminations.bad_orientation.params["limit_angle"] = 1.31


@configclass
class G1AMPMixedEnvCfg(G1AMPFlatEnvCfg):
    motion_style: str = "mixed"
    motion_source: G1MixedMotionCfg = G1MixedMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 3.2)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.5, 0.5)


def _set_common_play_settings(cfg) -> None:
    cfg.scene.num_envs = 32
    cfg.scene.env_spacing = 2.5
    cfg.episode_length_s = 40.0
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.curriculum.ang_vel_cmd_levels = None
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


@configclass
class G1AMPWalkPlayEnvCfg(G1AMPWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)


@configclass
class G1AMPRunPlayEnvCfg(G1AMPRunEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1AMPOmniRunPlayEnvCfg(G1AMPOmniRunEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)


@configclass
class G1AMPWalkToRunPlayEnvCfg(G1AMPWalkToRunEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)


@configclass
class G1AMPWalkToRunFullPlayEnvCfg(G1AMPWalkToRunPlayEnvCfg):
    """训练完成后的完整能力评估：一次覆盖走、跑、转弯的全速度区间。

    与普通 Play 的区别：
    普通 `G1AMPWalkToRunPlayEnvCfg` 的 lin_vel_x 只到 1.2，那是训练时的指令范围，
    用来确认"训练分布内表现正常"。而作业要验收的是走↔跑切换能力，
    必须让指令同时穿过低速行走区和高速奔跑区，否则根本触发不了步态切换。

    速度区间取三个已有 Play 配置的并集：
      Walk     lin_vel_x (-0.7, 1.0)   ← 低速行走，含后退
      Run      lin_vel_x ( 1.5, 2.5)   ← 高速奔跑
      OmniRun  ang_vel_z (-0.4, 0.4)   ← 左右转弯
    合并后 lin_vel_x = (-0.7, 2.5) 是连续区间，采样时会自然经过 1.0~1.5 这段
    "既不算走也不算跑"的过渡带: 走跑切换是否自然，正是在这一段看出来的。
    """

    def __post_init__(self):
        super().__post_init__()
        # 课程会随训练进度收窄/放宽指令范围；评估必须用固定的完整范围，
        # 否则不同 checkpoint 之间的结果不可比。
        # 父类已置 None，这里显式重申，避免将来父类改动导致评估范围被悄悄改掉。
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        # 观测噪声与外力扰动已由 _set_common_play_settings 关闭
        # （enable_corruption=False、base_external_force_torque/push_robot=None），
        # 评估看的是策略本身的能力，不该被随机扰动掺进来。


@configclass
class G1AMPDancePlayEnvCfg(G1AMPDanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1AMPMixedPlayEnvCfg(G1AMPMixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)


@configclass
class G1AMPPlayEnvCfg(G1AMPMixedPlayEnvCfg):
    """Default Play configuration for the mixed task ID."""


# Reference-project compatibility aliases.
G1AmpFlatEnvCfg = G1AMPWalkEnvCfg
G1AmpFlatEnvCfg_run = G1AMPRunEnvCfg
G1AmpFlatEnvCfg_dance = G1AMPDanceEnvCfg
G1AmpFlatEnvCfg_PLAY = G1AMPPlayEnvCfg
