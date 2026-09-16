from __future__ import annotations

import os

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.hoi_mimic_terrain import tracking_env_cfg as blind_cfg
from unitree_rl_lab.tasks.mimic.sensors import HoiMergedTerrainRayCasterCfg


@configclass
class RobotSceneCfg(blind_cfg.RobotSceneCfg):
    # TODO(student): replace None with a HoiMergedTerrainRayCasterCfg.
    # Configure the torso-mounted, yaw-aligned grid scanner described in
    # HOI_MIMIC_HOMEWORK.md. Keep the supplied RayCaster implementation unchanged.
    height_scanner: HoiMergedTerrainRayCasterCfg | None = HoiMergedTerrainRayCasterCfg(
        # 挂在 torso 上，扫描区随躯干平移
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        # yaw 对齐：网格只跟随朝向、始终保持水平。
        # 用默认的 "base" 会让扫描面随俯仰翻转，机器人一低头高度场就整体失真。
        ray_alignment="yaw",
        # 射线起点抬到 20 m 高处垂直向下打。
        # 起点若在躯干附近，射线会打到机器人自身、或起点已埋进障碍物内部导致 miss。
        # 注意这个 offset 是传感器的几何起点，与观测项里的 offset=0.5 无关，
        # 后者是高度基准的数值平移。
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        # 1.6 / 0.1 = 16，边界含两端 → 17×17 = 289 个扫描点。
        # 写成 16×16=256 不会报错，只是观测维度悄悄少 264 维。
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.6)),
        terrain_prim_path="{ENV_REGEX_NS}/HOI_Terrain",
        # RayCasterCfg.mesh_prim_paths 是 MISSING（必填）。
        # 对这个 HOI RayCaster 它只是存放烘焙 mesh 的键名:
        # hoi_merged_terrain_ray_caster.py:328 把合并后的地形 mesh
        # 存进 self.meshes[mesh_prim_paths[0]]，基类再按同一个键取用。
        # 不设会在第一次 _update_buffers_impl 时 KeyError: '/World/ground'，
        # 而且是在环境构建完之后才炸，看着像地形没加载。
        # 沿用仓库里 h1/go2/g1 velocity 任务的惯例值。
        mesh_prim_paths=["/World/ground"],
        # 与 blind 任务共用同一份地形 metadata，避免两边地形定义漂移
        metadata_file=blind_cfg.TERRAIN_META_FILE,
        use_mjcf_boxes_mesh=True,
        # 补地面：否则 box 之外的射线全部 miss，高度场出现大片 inf
        include_ground_plane=True,
        # 地形位姿固定，不必每次 reset 重建 mesh
        rebake_on_reset=False,
        # 作业要求提交 RayCaster 命中点截图，那需要 debug_vis=True 才画得出射线。
        # 但可视化会拖慢训练，所以做成开关：截图时设
        #   HOI_RAYCAST_DEBUG_VIS=1
        # 训练默认关闭。
        debug_vis=os.getenv("HOI_RAYCAST_DEBUG_VIS", "0").lower() in ("1", "true", "yes"),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        # TODO(student): replace None with the policy height_scanner ObsTerm.
        # The policy term uses mdp.height_scan, clipping, history, and uniform noise.
        height_scanner: ObsTerm | None = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            # policy 侧加噪：actor 必须在带噪观测下学会鲁棒，否则 sim2real 会垮
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-1.0, 5.0),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=blind_cfg.PROPRIO_HISTORY_LENGTH
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=blind_cfg.PROPRIO_HISTORY_LENGTH
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        # TODO(student): replace None with the critic height_scanner ObsTerm.
        # Match the policy height semantics and history, but do not add noise.
        height_scanner: ObsTerm | None = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            # critic 不加噪: 非对称 actor-critic 的关键：
            # critic 用无噪真值算价值，actor 在带噪观测下学策略。
            # 这里若跟着加 Unoise，特权观测就白设了。
            clip=(-1.0, 5.0),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        actions = ObsTerm(func=mdp.last_action, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


def _check_homework_todos(cfg) -> None:
    """Fail with a focused message until the three assignment TODOs are complete."""
    if cfg.scene.height_scanner is None:
        raise NotImplementedError(
            "TODO(student): configure RobotSceneCfg.height_scanner as described in HOI_MIMIC_HOMEWORK.md."
        )
    if cfg.observations.policy.height_scanner is None:
        raise NotImplementedError(
            "TODO(student): add the policy height_scanner observation described in HOI_MIMIC_HOMEWORK.md."
        )
    if cfg.observations.critic.height_scanner is None:
        raise NotImplementedError(
            "TODO(student): add the critic height_scanner observation described in HOI_MIMIC_HOMEWORK.md."
        )


@configclass
class RobotEnvCfg(blind_cfg.RobotEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=4.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _check_homework_todos(self)
        self.scene.height_scanner.env_spacing = float(self.scene.env_spacing)


@configclass
class RobotPlayEnvCfg(blind_cfg.RobotPlayEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=4.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _check_homework_todos(self)
        self.scene.height_scanner.env_spacing = float(self.scene.env_spacing)
