"""Readable G1 expert-motion selections shared by train and Play environments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ...motion_dataset import DEFAULT_FEATURES, MotionDatasetCfg


AMP_ROOT = Path(__file__).parents[2]
DATA_ROOT = Path(os.environ.get("UNITREE_AMP_MOTION_ROOT", AMP_ROOT / "data")).expanduser()

G1_REFERENCE_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

G1_AMP_KEY_LINK_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


@dataclass
class G1MotionSourceCfg:
    """A named directory and explicit sampling weights for one G1 motion style."""

    profile_name: str
    motion_dir: str
    clip_weights: dict[str, float] = field(default_factory=dict)
    file_pattern: str = "*.npz"
    features: tuple[str, ...] = DEFAULT_FEATURES

    def __post_init__(self) -> None:
        if not self.profile_name.strip():
            raise ValueError("G1 motion profile_name must be non-empty.")
        if not self.motion_dir:
            raise ValueError(f"G1 motion profile '{self.profile_name}' must define motion_dir.")
        if not self.clip_weights or sum(self.clip_weights.values()) <= 0.0:
            raise ValueError(f"G1 motion profile '{self.profile_name}' needs positive clip weights.")
        if any(weight < 0.0 for weight in self.clip_weights.values()):
            raise ValueError(f"G1 motion profile '{self.profile_name}' contains a negative clip weight.")

    def apply_to_dataset_cfg(self, dataset_cfg: MotionDatasetCfg) -> MotionDatasetCfg:
        dataset_cfg.motion_dir = self.motion_dir
        dataset_cfg.profile_name = self.profile_name
        dataset_cfg.clip_weights = dict(self.clip_weights)
        dataset_cfg.file_pattern = self.file_pattern
        return dataset_cfg


@dataclass
class G1WalkMotionCfg(G1MotionSourceCfg):
    profile_name: str = "walk"
    motion_dir: str = str(DATA_ROOT / "walk")
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "B3_-_walk1_stageii": 4.0,
        "B5_-_walk_backwards_stageii": 3.0,
        "B9_-_walk_turn_left_(90)_stageii": 1.0,
        "B10_-_walk_turn_left_(45)_stageii": 1.0,
        "B12_-_walk_turn_right_(90)_stageii": 1.0,
        "B13_-_walk_turn_right_(45)_stageii": 1.0,
    })


@dataclass
class G1RunMotionCfg(G1MotionSourceCfg):
    profile_name: str = "run"
    motion_dir: str = str(DATA_ROOT / "run")
    clip_weights: dict[str, float] = field(default_factory=lambda: {"C3_-_Run_stageii": 1.0})


@dataclass
class G1OmniRunMotionCfg(G1MotionSourceCfg):
    profile_name: str = "omni_run"
    motion_dir: str = str(DATA_ROOT / "run")
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "C3_-_Run_stageii": 6.0,
        "C11_-__run_turn_left_(90)_stageii": 1.5,
        "C14_-__run_turn_right__(90)_stageii": 1.0,
        "C15_-__run_turn_right__(45)_stageii": 0.5,
    })


@dataclass
class G1WalkToRunMotionCfg(G1MotionSourceCfg):
    """Task-agnostic walking, running, transition, reverse, and turning clips."""

    profile_name: str = "walk_to_run"
    motion_dir: str = str(DATA_ROOT / "mixed")
    # 2026-09-07：与实际重定向产物同步。
    #
    # 原表引用的 B3_-_walk1 / B5_-_walk_backwards / B9_-_walk_turn_left(90)
    # 出自 ACCAD 的 Male1Walking_c3d 子集，而课程百度网盘发的 ACCAD 包里
    # 没有这个目录（只有 Male2Walking_c3d，且里面全是 hop/leap 类）。
    # 照原表启动会直接抛
    #   ValueError: AMP clip weights reference missing clips: ...
    # 这正是实践 7 已知检查项,「motion_cfg.py 中的文件名和
    # 采样权重是否与生成的数据同步」。
    #
    # C14 的文件名也对不上：源文件是 C14_-_run_turn_right_90_stageii
    #（单下划线、无括号），原表写的是 C14_-__run_turn_right__(90)_stageii。
    # 同一批数据里 C11 反而是带括号的（来自 Female1Running），
    # 两个子集的命名风格不同，不能按一种模式套。
    #
    # 三类覆盖（数据覆盖的硬要求）在当前这套里是齐的：
    #   走路   B1_-_stand_to_walk
    #   跑步   C3_-_Run / C3_-_run / C2_-_Run_to_stand / C6_-_stand_to_run_backwards
    #   切换   C5_-_walk_to_run
    #   转弯   B12_-_walk_turn_right(90) / C11_-_run_turn_left(90) / C14_-_run_turn_right_90
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        # 走路与站走过渡：ACCAD 包里没有纯 walk 片段，
        # 这两条是课程自带的样例数据，正好补上这一类
        "B1_-_stand_to_walk_stageii": 4.0,
        "B12_-_walk_turn_right_(90)_stageii": 1.0,
        # 跑步（权重给高，本 profile 的主体）
        "C3_-_Run_stageii": 5.0,
        "C3_-_run_stageii": 2.0,
        # 走↔跑切换，是 walk_to_run 这个 profile 的核心
        "C5_-_walk_to_run_stageii": 3.0,
        "C2_-_Run_to_stand_stageii": 1.0,
        "C6_-_stand_to_run_backwards_stageii": 1.0,
        # 转弯
        "C11_-__run_turn_left_(90)_stageii": 1.0,
        "C14_-_run_turn_right_90_stageii": 1.0,
    })


@dataclass
class G1P7SegmentsMotionCfg(G1MotionSourceCfg):
    """Explicit P7 walk/run/turn selection; short clips, not a P8 training default."""

    profile_name: str = "p7_segments_20260911"
    motion_dir: str = str(DATA_ROOT / "p7_segments_20260911")
    # Walk/run are disclosed slices of complete self-retargeted C4/C5 sources.
    # C16 is the complete right turn with a separately recorded root-Z correction.
    # Equal clip weights are for course loading/inspection, not tuned for training.
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "walk_C4_f050_105": 1.0,
        "run_C5_f120_161": 1.0,
        "turn_right_C16_full": 1.0,
    })


@dataclass
class G1DanceMotionCfg(G1MotionSourceCfg):
    profile_name: str = "dance"
    motion_dir: str = str(DATA_ROOT / "dance")
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "irish_dance_stageii": 0.5,
        "salsa_1_stageii": 1.0,
        "salsa_stageii": 1.0,
    })


@dataclass
class G1MixedMotionCfg(G1MotionSourceCfg):
    profile_name: str = "mixed"
    motion_dir: str = os.environ.get("UNITREE_AMP_MOTION_DIR", str(DATA_ROOT / "mixed"))
    clip_weights: dict[str, float] = field(default_factory=lambda: {
        "B3_-_walk1_stageii": 2.0,
        "B5_-_walk_backwards_stageii": 1.0,
        "B9_-_walk_turn_left_(90)_stageii": 1.0,
        "B12_-_walk_turn_right_(90)_stageii": 1.0,
        "C11_-__run_turn_left_(90)_stageii": 1.0,
        "C14_-__run_turn_right__(90)_stageii": 1.0,
        "C2_-_Run_to_stand_stageii": 1.0,
        "C5_-_walk_to_run_stageii": 1.0,
    })


MOTION_CONFIGS = {
    "walk": G1WalkMotionCfg,
    "run": G1RunMotionCfg,
    "omni_run": G1OmniRunMotionCfg,
    "walk_to_run": G1WalkToRunMotionCfg,
    "p7_segments_20260911": G1P7SegmentsMotionCfg,
    "dance": G1DanceMotionCfg,
    "mixed": G1MixedMotionCfg,
}


def make_motion_source(profile_name: str) -> G1MotionSourceCfg:
    try:
        return MOTION_CONFIGS[profile_name.strip().lower()]()
    except KeyError as error:
        choices = ", ".join(sorted(MOTION_CONFIGS))
        raise ValueError(f"Unknown G1 AMP motion profile '{profile_name}'. Available: {choices}.") from error


def make_motion_dataset_cfg(source: G1MotionSourceCfg, history_steps: int = 3) -> MotionDatasetCfg:
    return MotionDatasetCfg(
        motion_dir=source.motion_dir,
        joint_names=(),
        profile_name=source.profile_name,
        source_joint_names=G1_REFERENCE_JOINT_NAMES,
        key_link_names=G1_AMP_KEY_LINK_NAMES,
        history_steps=history_steps,
        step_dt=0.02,
        quaternion_order="xyzw",
        clip_weights=dict(source.clip_weights),
        file_pattern=source.file_pattern,
        features=source.features,
    )


# Compatibility names retained for existing student configs.
MotionSourceCfg = G1MotionSourceCfg
G1WalkMotionSourceCfg = G1WalkMotionCfg
G1RunMotionSourceCfg = G1RunMotionCfg
G1OmniRunMotionSourceCfg = G1OmniRunMotionCfg
G1WalkToRunMotionSourceCfg = G1WalkToRunMotionCfg
G1DanceMotionSourceCfg = G1DanceMotionCfg
G1MixedMotionSourceCfg = G1MixedMotionCfg
G1_walk_Cfg = G1WalkMotionCfg
G1_Run_Cfg = G1RunMotionCfg
G1_dance_Cfg = G1DanceMotionCfg
