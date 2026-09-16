import copy
import math
import os
from importlib import import_module
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.navigation import mdp

_low_level_env_cfg = import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.low_level_env_cfg")
LowLevelObservationsCfg = _low_level_env_cfg.LowLevelObservationsCfg
LowLevelRobotEnvCfg = _low_level_env_cfg.LowLevelRobotEnvCfg
LowLevelRobotSceneCfg = _low_level_env_cfg.LowLevelRobotSceneCfg

# Resolve the packaged pretrained low-level policy relative to the unitree_rl_lab repo root.
# __file__ = .../source/unitree_rl_lab/unitree_rl_lab/tasks/navigation/robots/g1/29dof/navigation_env_cfg.py
_REPO_ROOT = Path(__file__).resolve().parents[8]
_COURSE_LOW_LEVEL_POLICY_PATH = str(_REPO_ROOT / "pretrained" / "g1_29dof_lowlevel" / "policy.pt")

# 默认改用我们自己在同一套 URDF 机器人上训出来的平地速度策略。
#
# 课程自带的 pretrained/g1_29dof_lowlevel/policy.pt 是在原作者的 USD 机器人上
# 训练的。它在零指令下能让本机的 URDF 机器人稳稳站住（投影重力 -1.000），
# 观测维度也完全一致（480→29），但一旦下发持续的速度指令就频繁摔倒。
#
# 同一份高层配置、同一套限幅与指令平滑，只换低层策略，实测（120 iter）：
#     低层策略            episode length    bad_orientation
#     课程 pretrained        16 ~ 18            0.97
#     自训（URDF）          136 ~ 141            0.12      ← 8 倍改善
#
# 这个差别是决定性的：episode 上限 150 步（30 s），机器人约 0.5 m/s。
# 16 步 = 3.2 s 只能走 1.6 m，而目标在 2~5 m 外: 根本来不及走到就摔了，
# 无论怎么调奖励都学不会导航。140 步才让高层有机会学。
#
# 具体成因未做进一步隔离（USD/URDF 的物理属性差异、训练指令分布差异都有可能），
# 这里只陈述可复现的实测差距。设 UNITREE_G1_LOW_LEVEL_POLICY_PATH
# 指向课程原策略即可复现对照。
_OWN_LOW_LEVEL_POLICY_PATH = os.path.expanduser(
    "~/unitree_rl_lab/logs/rsl_rl"
    "/unitree_g1_29dof_velocity/2026-08-31_10-49-33/exported/policy.pt"
)
_DEFAULT_LOW_LEVEL_POLICY_PATH = (
    _OWN_LOW_LEVEL_POLICY_PATH
    if os.path.exists(_OWN_LOW_LEVEL_POLICY_PATH)
    else _COURSE_LOW_LEVEL_POLICY_PATH
)
LOW_LEVEL_POLICY_PATH = os.environ.get("UNITREE_G1_LOW_LEVEL_POLICY_PATH", _DEFAULT_LOW_LEVEL_POLICY_PATH)

LOW_LEVEL_ENV_CFG = LowLevelRobotEnvCfg()
V3_MAX_CYLINDER_OBSTACLES = 8
V3_CYLINDER_RADIUS = 0.4
V3_CYLINDER_HEIGHT = 2.0
V4_MAX_MAZE_OBSTACLES = 81
V4_MAZE_CYLINDER_RADIUS = 0.35
V4_MAZE_CYLINDER_HEIGHT = 2.0

NAVIGATION_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)

NAVIGATION_V2_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(56.0, 56.0),
    border_width=4.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


V4_OBSTACLE_COUNT_LEVELS = (0, 45, 65, 81, 120)

# ⚠️ 不要用环境变量改这个数来做"关掉障碍"的隔离实验。
# 它决定的是物理生成多少个刚体，改成 0 会让布局代码崩溃：
#   RuntimeError: value tensor of shape [N,120,7] cannot be broadcast to [N,0,7]
#   （布局侧仍按 120 个写位姿）
# 正确做法是控制激活数量：MixedObstacleLayout.sample_layout(num_active=0)，
# 课程 V5_OBSTACLE_COUNT_LEVELS 的第一档就是 0，走的正是这条路。
V5_MAX_MIXED_OBSTACLES = 120
V5_OBSTACLE_COUNT_LEVELS = (0, 50, 80, 100, 120)
V5_HEIGHT_SCAN_SIZE = (5.0, 3.0)
V5_HEIGHT_SCAN_RESOLUTION = 0.12


def make_cylinder_obstacle_collection(
    max_obstacles: int = V3_MAX_CYLINDER_OBSTACLES,
    cylinder_radius: float = V3_CYLINDER_RADIUS,
    cylinder_height: float = V3_CYLINDER_HEIGHT,
) -> RigidObjectCollectionCfg:
    """Create fixed slots for sparse cylinder obstacles in every environment."""
    obstacle_cfgs = {}
    for obstacle_id in range(max_obstacles):
        obstacle_cfgs[f"cylinder_{obstacle_id}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/cylinder_obstacle_{obstacle_id}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CylinderCfg(
                radius=cylinder_radius,
                height=cylinder_height,
                axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.85)),
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=obstacle_cfgs)


def make_low_level_inference_observations() -> LowLevelObservationsCfg.PolicyCfg:
    """Create a no-corruption copy of the exact low-level policy observation group."""
    # >>> HOMEWORK_TODO_6_START
    # 必须 deepcopy 低层训练时的 observation group，绝不能手写重建。
    # 低层策略是个 TorchScript 静态图，它的输入是按训练时的 term 顺序、
    # 缩放系数和 5 帧历史拼出来的一个扁平向量。顺序或缩放差一点，
    # 网络不会报错，只会读到错位的物理量然后输出乱七八糟的关节角。
    # 这就是作业讲解说的"低层输入契约",契约的唯一可靠来源是低层自己的配置。
    #
    # deepcopy 而非直接引用：下面要改 noise/corruption，直接改会污染
    # LOW_LEVEL_ENV_CFG 这个模块级单例，影响同进程内其他使用它的配置。
    observations = copy.deepcopy(LOW_LEVEL_ENV_CFG.observations.policy)

    # 推理时关掉观测噪声。训练低层时加噪是为了让它对真机传感器误差鲁棒；
    # 现在低层是被当作一个确定性的"技能模块"来调用，再注入噪声只会让
    # 高层观测到的低层行为变得随机，增加高层的学习难度而无任何收益。
    observations.enable_corruption = False

    # enable_corruption=False 已足以在 ObservationManager 层面跳过加噪，
    # 这里再把四个本体感知项的 noise 显式置空，做双保险:
    # 它们是低层观测里仅有的带 noise 的项（velocity_commands 与 last_action 本就没有）。
    observations.base_ang_vel.noise = None
    observations.projected_gravity.noise = None
    observations.joint_pos_rel.noise = None
    observations.joint_vel_rel.noise = None

    # 注意不要动 history_length：低层训练时用的就是 5 帧，
    # deepcopy 已带过来，改了就破坏契约。
    return observations
    # <<< HOMEWORK_TODO_6_END


@configclass
class NavigationSceneCfg(LowLevelRobotSceneCfg):
    """Flat navigation scene with G1, height scanner, and contact sensors."""

    def __post_init__(self):
        self.terrain.terrain_generator = copy.deepcopy(NAVIGATION_FLAT_TERRAIN_CFG)
        self.terrain.max_init_terrain_level = 0
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.curriculum = False


@configclass
class NavigationV2SceneCfg(NavigationSceneCfg):
    """Larger flat scene for long-range multi-goal navigation."""

    def __post_init__(self):
        self.terrain.terrain_generator = copy.deepcopy(NAVIGATION_V2_FLAT_TERRAIN_CFG)
        self.terrain.max_init_terrain_level = 0
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.curriculum = False


@configclass
class NavigationV3MazeSceneCfg(NavigationV2SceneCfg):
    """V2-sized flat scene with sparse cylinder obstacles and multi-mesh height scans."""

    def __post_init__(self):
        super().__post_init__()
        self.cylinder_obstacles = make_cylinder_obstacle_collection()
        self.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=[
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/cylinder_obstacle_.*",
                    track_mesh_transforms=True,
                ),
            ],
        )


@configclass
class NavigationV4FixedMazeSceneCfg(NavigationV2SceneCfg):
    """V2-sized random dense cylinder arena with enlarged height scans."""

    def __post_init__(self):
        super().__post_init__()
        self.cylinder_obstacles = make_cylinder_obstacle_collection(
            max_obstacles=V4_MAX_MAZE_OBSTACLES,
            cylinder_radius=V4_MAZE_CYLINDER_RADIUS,
            cylinder_height=V4_MAZE_CYLINDER_HEIGHT,
        )
        self.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.4, 1.6]),
            debug_vis=False,
            mesh_prim_paths=[
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/cylinder_obstacle_.*",
                    track_mesh_transforms=True,
                ),
            ],
        )


@configclass
class NavigationV5MixedObstacleSceneCfg(NavigationV2SceneCfg):
    """Dense mixed cylinder/box arena with long-range height scans."""

    def __post_init__(self):
        super().__post_init__()
        self.mixed_obstacles = mdp.make_mixed_obstacle_collection()
        self.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=V5_HEIGHT_SCAN_RESOLUTION,
                size=list(V5_HEIGHT_SCAN_SIZE),
            ),
            debug_vis=False,
            mesh_prim_paths=[
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/cylinder_obstacle_.*",
                    track_mesh_transforms=True,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/box_obstacle_.*",
                    track_mesh_transforms=True,
                ),
            ],
        )


@configclass
class NavigationEventCfg:
    """Reset-only events for deterministic v1 navigation training."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class NavigationV3MazeEventCfg:
    """Reset events that place obstacles before resetting the robot and command."""

    randomize_cylinders = EventTerm(
        func=mdp.randomize_cylinder_layout,
        mode="reset",
        params={
            "layout_cfg": mdp.CylinderObstacleLayoutCfg(
                obstacle_asset_name="cylinder_obstacles",
                max_obstacles=V3_MAX_CYLINDER_OBSTACLES,
                cylinder_radius=V3_CYLINDER_RADIUS,
                cylinder_height=V3_CYLINDER_HEIGHT,
                soft_margin=0.6,
                min_center_separation=2.0,
                arena_half_extent=28.0,
                arena_margin=3.0,
                max_resample_tries=128,
            ),
            "default_num_active": 0,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_obstacle_aware,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "robot_radius": 0.5,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


V5_FIXED_MIXED_LAYOUT_CFG = mdp.MixedObstacleLayoutCfg(
    obstacle_asset_name="mixed_obstacles",
    max_obstacles=V5_MAX_MIXED_OBSTACLES,
    soft_margin=0.4,
    min_center_separation=1.1,
    arena_half_extent=28.0,
    arena_margin=3.0,
    max_resample_tries=256,
    exclude_origin=False,
)


@configclass
class NavigationV5FixedArenaEventCfg:
    """Sticky per-env maps baked once at startup from three fixed arena templates."""

    assign_arena_maps = EventTerm(
        func=mdp.assign_fixed_mixed_arena_layout,
        mode="startup",
        params={"layout_cfg": V5_FIXED_MIXED_LAYOUT_CFG},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_obstacle_aware,
        mode="reset",
        params={
            "pose_range": {"x": (-25.0, 25.0), "y": (-25.0, 25.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "robot_radius": 0.5,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class NavigationV5MixedObstacleEventCfg:
    """Reset events for heterogeneous dense obstacles."""

    randomize_obstacles = EventTerm(
        func=mdp.randomize_mixed_obstacle_layout,
        mode="reset",
        params={
            "layout_cfg": mdp.MixedObstacleLayoutCfg(
                obstacle_asset_name="mixed_obstacles",
                max_obstacles=V5_MAX_MIXED_OBSTACLES,
                soft_margin=0.4,
                min_center_separation=1.1,
                arena_half_extent=28.0,
                arena_margin=3.0,
                max_resample_tries=256,
                exclude_origin=False,
            ),
            "default_num_active": 0,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_obstacle_aware,
        mode="reset",
        params={
            "pose_range": {"x": (-25.0, 25.0), "y": (-25.0, 25.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "robot_radius": 0.5,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class NavigationV4FixedMazeEventCfg:
    """Reset events that place random dense cylinders before resetting the robot."""

    randomize_cylinders = EventTerm(
        func=mdp.randomize_cylinder_layout,
        mode="reset",
        params={
            "layout_cfg": mdp.CylinderObstacleLayoutCfg(
                obstacle_asset_name="cylinder_obstacles",
                max_obstacles=V4_MAX_MAZE_OBSTACLES,
                cylinder_radius=V4_MAZE_CYLINDER_RADIUS,
                cylinder_height=V4_MAZE_CYLINDER_HEIGHT,
                soft_margin=0.4,
                min_center_separation=1.45,
                arena_half_extent=28.0,
                arena_margin=3.0,
                max_resample_tries=256,
                exclude_origin=False,
            ),
            "default_num_active": 0,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_obstacle_aware,
        mode="reset",
        params={
            "pose_range": {"x": (-25.0, 25.0), "y": (-25.0, 25.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "robot_radius": 0.5,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class NavigationActionsCfg:
    """High-level action is a velocity command consumed by the frozen low-level policy."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.JointPositionAction,
        low_level_observations=make_low_level_inference_observations(),
        velocity_clip=((-0.5, 1.0), (-0.5, 0.5), (-0.5, 0.5)),
        # 低层策略训练时指令每 10 秒才换一次，而高层每 0.2 秒重采样一次（快 50 倍）。
        # 不平滑时实测平均每 22.6 步摔一次，每次摔倒吃 -400 的 termination_penalty，
        # 使 critic 在第 2 个 iteration 就发散（value loss 10^27 → NaN → 训练崩溃）。
        # α=0.1 在 5 Hz 下时间常数约 1.9 s，实测摔倒率降到每 120.8 步一次。
        # 设为 1.0 可关闭平滑，复现修复前的行为做对照。
        # 默认改回 1.0（关闭平滑）。这个 EMA 是 §10 为保护低层加的补丁，
        # 而 §14 找到真因后它已成多余：修复 update_history 后，
        # 关闭平滑的 oracle 测试到达率 46.9%、摔倒率 0.0%。
        # 保留平滑反而有害: §11 实测它把探索速度压到 1/5
        # （0.045 vs 0.223 m/s，理论 sqrt(a/(2-a))=0.229）。
        command_smoothing=float(os.environ.get("NAV_COMMAND_SMOOTHING", 1.0)),
        debug_vis=True,
    )


@configclass
class NavigationCommandsCfg:
    """One 2D pose target per episode, sampled in a 2-5 m ring."""

    pose_command: mdp.RingPose2dCommandCfg = mdp.RingPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(30.0, 30.0),
        debug_vis=True,
        success_radius=1.0,
        ranges=mdp.RingPose2dCommandCfg.Ranges(distance=(2.0, 5.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationV2CommandsCfg:
    """Long-range 2D pose targets resampled only when reached."""

    pose_command: mdp.RingPose2dCommandCfg = mdp.RingPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        success_radius=0.5,
        update_goal_on_success=True,
        ranges=mdp.RingPose2dCommandCfg.Ranges(distance=(5.0, 25.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationV3MazeCommandsCfg(NavigationV2CommandsCfg):
    """Obstacle-aware long-range targets for sparse maze training."""

    pose_command: mdp.RingPose2dCommandCfg = mdp.RingPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        success_radius=0.5,
        update_goal_on_success=True,
        obstacle_filter=True,
        max_obstacle_resample_tries=128,
        ranges=mdp.RingPose2dCommandCfg.Ranges(distance=(5.0, 25.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationV4FixedMazeCommandsCfg:
    """Arena-wide targets that force traversal through dense obstacle fields."""

    pose_command: mdp.ArenaPose2dCommandCfg = mdp.ArenaPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        success_radius=0.5,
        update_goal_on_success=True,
        obstacle_filter=True,
        max_obstacle_resample_tries=128,
        arena_half_extent=25.0,
        arena_margin=0.5,
        ranges=mdp.ArenaPose2dCommandCfg.Ranges(distance=(0.0, 0.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationV5CompactSingleGoalCommandsCfg:
    """One obstacle-aware robot-relative target per episode for V5 compact arenas."""

    pose_command: mdp.RingPose2dCommandCfg = mdp.RingPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(30.0, 30.0),
        debug_vis=False,
        success_radius=0.5,
        update_goal_on_success=False,
        obstacle_filter=True,
        max_obstacle_resample_tries=128,
        ranges=mdp.RingPose2dCommandCfg.Ranges(distance=(5.0, 10.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationObservationsCfg:
    """Observation groups for high-level navigation PPO."""

    @configclass
    class PolicyCfg(ObsGroup):
        # >>> HOMEWORK_TODO_7_START
        # 高层观测 = 目标 + 本体状态 + 控制历史（局部地图由子类补上）。
        # 基类这 8 项共 3+3+3+4+3+29+29+29 = 103 维；
        # V5Compact 子类再加 height_scan_pooled 的 273 维，合计 376 维。
        #
        # 分组理解：
        # 1. 本体运动状态,高层要知道机器人当前实际走多快，才能算出该加速还是该减速。
        #    注意这是实际速度，与下面的 last_high_level_command（期望速度）不同，
        #    两者之差正是低层的跟踪误差，高层可据此感知"低层是不是跟不上了"。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # 2. 任务目标,目标点在机体系下的相对位姿（4 维）。
        #    用相对量而非世界坐标：导航策略应当学"朝目标偏左就右转"这种
        #    与绝对位置无关的规律，这样才能泛化到任意起点和地图。
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

        # 3. 控制历史,上一条裁剪后的高层指令（见 TODO 1）。
        #    让高层知道自己刚下了什么命令，有助于输出平滑的速度序列，
        #    避免指令在相邻步之间剧烈跳变把低层带崩。
        last_high_level_command = ObsTerm(func=mdp.last_high_level_command)

        # 4. 关节状态 + 低层输出,高层借此感知低层的内部状态。
        #    low_level_last_action 是冻结低层策略上一步吐出的 29 维关节动作，
        #    它间接反映了低层当前的"努力程度"：若关节动作已经很大而实际速度上不去，
        #    说明低层已接近能力边界，高层应该降低指令而不是继续加码。
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # clip 是必需的，不是保险。低层是冻结的 TorchScript，机器人进入
        # 训练分布之外的姿态（摔倒过程中）时它会吐出极端值: 实测达到 1.7e6，
        # 而其余观测项都在个位数量级。这个值同时流向两处：
        #   1. 高层观测 → V(s)=W·obs 随之爆炸 → value loss 10^25 → 权重 NaN
        #   2. PD 控制器 → 巨大力矩把机器人掀翻 → 更多摔倒 → 正反馈
        # 正常关节动作量级在 ±1（乘 action_scale=0.25 后是 ±0.25 rad），
        # ±10 已经远超合理范围，不会截掉任何有效信号。
        low_level_last_action = ObsTerm(func=mdp.low_level_last_action, clip=(-10.0, 10.0))
        # <<< HOMEWORK_TODO_7_END

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_height = ObsTerm(func=mdp.base_pos_z)
        command_distance = ObsTerm(func=mdp.command_distance, params={"command_name": "pose_command"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class NavigationRewardsCfg:
    """Progress reward plus sparse success and fall penalty."""

    # 课程原值 -400。它隐含假设低层策略足够鲁棒、摔倒是小概率事件；
    # 但本机这套低层（hw5 自带的预训练 TorchScript）在高层输出随机指令时
    # 平均每 143 步就摔一次，一个 150 步的 episode 里摔倒概率约 65%。
    #
    # 按归一化单位（Episode_Reward/* = episode 和 ÷ 30 s）算边际激励：
    #     走完全程 3.5 m 的进展奖励  +0.117
    #     到达目标的 success_bonus  +3.333
    #     摔一次的 termination      -13.333   ← 比走完全程大 114 倍
    # 于是 E[移动] = 0.117 + 3.333·P(到达) - 13.333·0.65 < 0 恒成立，
    # 需要 P(到达) > 256% 才划算: 数学上不可能。
    # 「站着不动」严格占优，策略必然收敛到它（实测 200 iter 时
    # position_progress ≈ 0、goals_reached = 0、error_pos_2d 不降）。
    #
    # 这与实践 2 的"原地踏步"是同一类问题，修法也一样：
    # 不是把奖励调大，而是改变边际激励比，让"尝试移动"变成正期望。
    # -20 是使 E[移动] 在 P(到达)=10% 时刚好转正的量级。
    # 设 NAV_TERMINATION_PENALTY=-400 可复现课程原值做对照。
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=float(os.environ.get("NAV_TERMINATION_PENALTY", -20.0)),
        params={"term_keys": ["base_height", "bad_orientation"]},
    )
    position_progress = RewTerm(
        func=mdp.pose_command_progress,
        weight=1.0,
        params={"command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        # std 决定这个"稠密奖励"在多大范围内有梯度。
        # Baseline 的目标在 5~10 m 外采样（ranges.distance=(5.0,10.0)），
        # 而课程原值 std=0.1~0.2 时该奖励在整个工作范围内恒等于 0
        # （1-tanh(5/0.2) 与 1-tanh(10/0.2) 都是 0.000000，差值 0），
        # 也就是说它完全没有提供"靠近目标更好"的梯度。
        # 于是策略能看到的唯一非零信号是动作惩罚（action_magnitude -0.0014，
        # 比 position_progress 的 +0.0002 大 7 倍），最优解就是输出零指令不动。
        # 实测 2298 iter 里 error_pos_2d 完全平坦在 7.5、goals_reached 恒为 0。
        #
        # 按真实区间选 std：
        #     std    @10m     @5m    10m→5m 梯度
        #     0.2   0.0000  0.0000     0.0000   ← 课程原值，无梯度
        #     2.0   0.0001  0.0134     0.0133   ← 仍然太弱
        #     5.0   0.0360  0.2384     0.2024   ← 采用
        # 设 NAV_TRACKING_STD=0.1 可复现课程原值做对照。
        params={
            "std": float(os.environ.get("NAV_TRACKING_STD", 5.0)),
            "command_name": "pose_command",
        },
    )
    success_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=100.0,
        params={"command_name": "pose_command"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class NavigationV2RewardsCfg:
    """Long-range progress reward with sparse per-waypoint success bonus."""

    # Baseline / RandomArena 实际继承的是这个类（经 V5MixedObstacleRewardsCfg），
    # 分析与理由见上面 NavigationRewardsCfg 处的注释。
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=float(os.environ.get("NAV_TERMINATION_PENALTY", -20.0)),
        params={"term_keys": ["base_height", "bad_orientation"]},
    )
    position_progress = RewTerm(
        func=mdp.pose_command_progress,
        weight=2.0,
        params={"command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        # Baseline / RandomArena 实际继承的是这一处（经 V5MixedObstacleRewardsCfg）。
        # 原值 0.1 比上面那处的 0.2 还窄：目标在 5~10 m 外，
        # 而 1-tanh(5/0.1) 已经是 0.000000: 整个工作范围零梯度。
        # 分析见上面 NavigationRewardsCfg 处的注释。
        params={
            "std": float(os.environ.get("NAV_TRACKING_STD", 5.0)),
            "command_name": "pose_command",
        },
    )
    success_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=50.0,
        params={"command_name": "pose_command"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class NavigationV3MazeRewardsCfg(NavigationV2RewardsCfg):
    """V2 rewards plus a virtual soft-constraint around cylinder obstacles."""

    obstacle_soft_zone = RewTerm(func=mdp.obstacle_soft_zone_penalty, weight=-2.0)


@configclass
class NavigationV5MixedObstacleRewardsCfg(NavigationV2RewardsCfg):
    """V2 rewards with stronger soft-constraint around mixed obstacles."""

    obstacle_soft_zone = RewTerm(func=mdp.obstacle_soft_zone_penalty, weight=-6.0)


@configclass
class NavigationV3MazeCurriculumCfg:
    """Ramp sparse cylinders from open field to eight active obstacles."""

    obstacle_count = CurrTerm(func=mdp.obstacle_count_levels)


@configclass
class NavigationV4FixedMazeCurriculumCfg:
    """Ramp random dense cylinder count across the full arena."""

    obstacle_count = CurrTerm(
        func=mdp.obstacle_count_levels,
        params={"level_counts": V4_OBSTACLE_COUNT_LEVELS},
    )


@configclass
class NavigationV5MixedObstacleCurriculumCfg:
    """Ramp mixed obstacle count across the full arena."""

    obstacle_count = CurrTerm(
        func=mdp.obstacle_count_levels,
        params={"level_counts": V5_OBSTACLE_COUNT_LEVELS},
    )


@configclass
class NavigationV5ObservationsCfg(NavigationObservationsCfg):
    """Navigation observations with enlarged height-scan clip range."""

    @configclass
    class PolicyCfg(NavigationObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.5, 1.5),
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_height = ObsTerm(func=mdp.base_pos_z)
        command_distance = ObsTerm(func=mdp.command_distance, params={"command_name": "pose_command"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class NavigationV5CompactObservationsCfg(NavigationV5ObservationsCfg):
    """V5 observations with 2x2 max-pooled height scan (273-d instead of 1092-d)."""

    @configclass
    class PolicyCfg(NavigationV5ObservationsCfg.PolicyCfg):
        height_scan = None
        height_scan_pooled = ObsTerm(
            func=mdp.height_scan_pooled,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "pool_size": 2},
            clip=(-1.5, 1.5),
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_height = ObsTerm(func=mdp.base_pos_z)
        command_distance = ObsTerm(func=mdp.command_distance, params={"command_name": "pose_command"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class NavigationTerminationsCfg:
    """Episode ends on timeout, success, or low-level stability failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached, params={"command_name": "pose_command", "threshold": 1.0})
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class NavigationV2TerminationsCfg:
    """Episode ends on timeout or low-level stability failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class NavigationV5CompactSingleGoalTerminationsCfg:
    """Episode ends on timeout, first goal success, or low-level stability failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached, params={"command_name": "pose_command", "threshold": 0.5})
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """High-level G1 navigation task over a frozen low-level locomotion policy."""

    scene: NavigationSceneCfg = NavigationSceneCfg(num_envs=4096, env_spacing=8.0)
    actions: NavigationActionsCfg = NavigationActionsCfg()
    observations: NavigationObservationsCfg = NavigationObservationsCfg()
    events: NavigationEventCfg = NavigationEventCfg()
    commands: NavigationCommandsCfg = NavigationCommandsCfg()
    rewards: NavigationRewardsCfg = NavigationRewardsCfg()
    terminations: NavigationTerminationsCfg = NavigationTerminationsCfg()

    def __post_init__(self):
        self.decimation = self.actions.pre_trained_policy_action.low_level_decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = (
            self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
        )


@configclass
class NavigationV2EnvCfg(NavigationEnvCfg):
    """Harder multi-goal G1 navigation task over the frozen low-level policy."""

    scene: NavigationV2SceneCfg = NavigationV2SceneCfg(num_envs=4096, env_spacing=60.0)
    commands: NavigationV2CommandsCfg = NavigationV2CommandsCfg()
    rewards: NavigationV2RewardsCfg = NavigationV2RewardsCfg()
    terminations: NavigationV2TerminationsCfg = NavigationV2TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0


@configclass
class NavigationV3MazeEnvCfg(NavigationV2EnvCfg):
    """Sparse cylinder maze task with curriculum from scratch."""

    scene: NavigationV3MazeSceneCfg = NavigationV3MazeSceneCfg(num_envs=4096, env_spacing=60.0)
    events: NavigationV3MazeEventCfg = NavigationV3MazeEventCfg()
    commands: NavigationV3MazeCommandsCfg = NavigationV3MazeCommandsCfg()
    rewards: NavigationV3MazeRewardsCfg = NavigationV3MazeRewardsCfg()
    curriculum: NavigationV3MazeCurriculumCfg = NavigationV3MazeCurriculumCfg()


@configclass
class NavigationV4FixedMazeEnvCfg(NavigationV2EnvCfg):
    """Random dense cylinder arena with obstacle-count curriculum from scratch."""

    scene: NavigationV4FixedMazeSceneCfg = NavigationV4FixedMazeSceneCfg(num_envs=4096, env_spacing=60.0)
    events: NavigationV4FixedMazeEventCfg = NavigationV4FixedMazeEventCfg()
    commands: NavigationV4FixedMazeCommandsCfg = NavigationV4FixedMazeCommandsCfg()
    rewards: NavigationV3MazeRewardsCfg = NavigationV3MazeRewardsCfg()
    curriculum: NavigationV4FixedMazeCurriculumCfg = NavigationV4FixedMazeCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # 4096 envs × up to 81 kinematic cylinders exceeds default GPU broadphase buffers.
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_max_rigid_contact_count = 2**24


@configclass
class NavigationV5MixedObstacleEnvCfg(NavigationV2EnvCfg):
    """Random dense mixed obstacle arena with long-range perception."""

    scene: NavigationV5MixedObstacleSceneCfg = NavigationV5MixedObstacleSceneCfg(num_envs=4096, env_spacing=60.0)
    events: NavigationV5MixedObstacleEventCfg = NavigationV5MixedObstacleEventCfg()
    commands: NavigationV4FixedMazeCommandsCfg = NavigationV4FixedMazeCommandsCfg()
    observations: NavigationV5ObservationsCfg = NavigationV5ObservationsCfg()
    rewards: NavigationV5MixedObstacleRewardsCfg = NavigationV5MixedObstacleRewardsCfg()
    curriculum: NavigationV5MixedObstacleCurriculumCfg = NavigationV5MixedObstacleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # 4096 envs × 120 kinematic obstacles needs larger broadphase buffers than V4.
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**25
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**27
        self.sim.physx.gpu_max_rigid_contact_count = 2**25


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact(NavigationV5MixedObstacleEnvCfg):
    """V5 mixed arena with max-pooled height scan and fixed per-env baked arena maps."""

    events: NavigationV5FixedArenaEventCfg = NavigationV5FixedArenaEventCfg()
    curriculum = None
    observations: NavigationV5CompactObservationsCfg = NavigationV5CompactObservationsCfg()

    def __post_init__(self):
        # Call V5 directly so PLAY.__post_init__ (randomize_obstacles) is not invoked via MRO.
        NavigationV5MixedObstacleEnvCfg.__post_init__(self)
        self.actions.pre_trained_policy_action.debug_vis = False
        self.commands.pose_command.debug_vis = False


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal(NavigationV5MixedObstacleEnvCfg_Compact):
    """V5 compact arena with one robot-relative goal and success termination."""

    commands: NavigationV5CompactSingleGoalCommandsCfg = NavigationV5CompactSingleGoalCommandsCfg()
    terminations: NavigationV5CompactSingleGoalTerminationsCfg = NavigationV5CompactSingleGoalTerminationsCfg()


@configclass
class NavigationV5RandomArenaEnvCfg_SingleGoal(NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal):
    """Part 2 单因素对照组：固定竞技场 → 每个 episode 随机重排障碍。

    ─────────────────────────────────────────────────────────────────────
    为什么另建一组，而不直接用课程自带的 HRL-Extension
    ─────────────────────────────────────────────────────────────────────
    作业讲解要求 Part 2「只改变一个主要因素并保持公平对照」，并举例
    「baseline 使用静态地图，extension 仅增加随机障碍」。

    但课程自带的 Baseline 与 Extension 之间实际相差 五处：

        项目          Baseline(Compact_SingleGoal)   Extension(MixedObstacle)
        障碍布局      固定模板(startup 烘焙)          每次 reset 随机
        课程学习      无 (curriculum = None)         障碍数 0→120 递增
        观测          池化 height_scan 273 维         未池化 1092 维
        目标          单目标 RingPose                 arena-wide 多目标
        终止          含 goal_reached success         不同

    其中观测维度不同意味着两组的策略网络输入层都不一样，已经不是
    同一个模型，无法归因到任何单一因素。它们是两个难度不同的任务，
    不是一组消融。

    本类继承 Baseline，只替换 events，其余（观测/目标/终止/课程/
    奖励/网络/超参）全部原样继承，使唯一变量是「障碍布局是否每个
    episode 重排」。
    """

    events: NavigationV5MixedObstacleEventCfg = NavigationV5MixedObstacleEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # 必须显式设置 default_num_active。
        # randomize_mixed_obstacle_layout 的逻辑是：
        #     num_active = getattr(env, "obstacle_num_active", None)
        #     active_counts = default_num_active if num_active is None else num_active[env_ids]
        # Baseline 靠 startup 事件 assign_fixed_mixed_arena_layout 写入
        # env.obstacle_num_active；本组换掉 events 后没有该 startup 事件，
        # 又没有 curriculum 去设置它，若不显式指定就会落到默认值 0:
        # 得到一个空场地，那样比较的就成了"有障碍 vs 无障碍"，
        # 与设计意图完全相反。
        #
        # 取 V5_MAX_MIXED_OBSTACLES 是为了与 Baseline 对齐：固定模板烘焙时
        # target_count = min(layout_cfg.max_obstacles, V5_MAX_MIXED_OBSTACLES) = 120。
        # 两组都受同样的 min_center_separation / soft_margin 约束，
        # 实际落位数会自然接近，从而把变量限制在"布局是否重排"这一点上。
        self.events.randomize_obstacles.params["default_num_active"] = V5_MAX_MIXED_OBSTACLES


@configclass
class NavigationV5MixedObstacleEnvCfg_PLAY(NavigationV5MixedObstacleEnvCfg):
    viewer: ViewerCfg = ViewerCfg(
        eye=(-3.5, 0.0, 2.0),
        lookat=(1.0, 0.0, 0.8),
        resolution=(1920, 1080),
        origin_type="asset_body",
        env_index=0,
        asset_name="robot",
        body_name="torso_link",
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 60.0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.curriculum = None
        if hasattr(self.events, "randomize_obstacles"):
            self.events.randomize_obstacles.params["default_num_active"] = V5_MAX_MIXED_OBSTACLES


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact_PLAY(NavigationV5MixedObstacleEnvCfg_Compact, NavigationV5MixedObstacleEnvCfg_PLAY):
    # Top-down camera: follows robot root; eye/lookat are world-frame offsets (+Z is up).
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        resolution=(1920, 1080),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        body_name=None,
    )

    events: NavigationV5FixedArenaEventCfg = NavigationV5FixedArenaEventCfg()

    def __post_init__(self):
        # Call Compact (not super()) so V5 __post_init__ runs and sets decimation / episode_length_s.
        # PLAY.__post_init__ is skipped intentionally (it targets randomize_obstacles on the old event cfg).
        NavigationV5MixedObstacleEnvCfg_Compact.__post_init__(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 60.0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.curriculum = None
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.debug_vis = True
        self.commands.pose_command.debug_vis = True


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_PLAY(
    NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal, NavigationV5MixedObstacleEnvCfg_Compact_PLAY
):
    # Top-down camera: follows robot root; eye/lookat are world-frame offsets (+Z is up).
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        resolution=(1920, 1080),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        body_name=None,
    )

    events: NavigationV5FixedArenaEventCfg = NavigationV5FixedArenaEventCfg()

    def __post_init__(self):
        NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal.__post_init__(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 60.0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.curriculum = None
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.debug_vis = True
        self.commands.pose_command.debug_vis = True


@configclass
class NavigationV5RandomArenaEnvCfg_SingleGoal_PLAY(
    NavigationV5RandomArenaEnvCfg_SingleGoal, NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_PLAY
):
    """对照组的 play 配置：与 baseline 的 PLAY 完全一致，只保留随机障碍事件。"""

    events: NavigationV5MixedObstacleEventCfg = NavigationV5MixedObstacleEventCfg()

    def __post_init__(self):
        # 显式走 baseline PLAY 的初始化（num_envs=16、关噪声、关 curriculum、开 debug_vis），
        # 再补上本组特有的障碍数设置。不走 MRO 默认链是因为
        # NavigationV5RandomArenaEnvCfg_SingleGoal.__post_init__ 会先执行，
        # 随后被 PLAY 的设置覆盖，default_num_active 就丢了。
        NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_PLAY.__post_init__(self)
        self.events.randomize_obstacles.params["default_num_active"] = V5_MAX_MIXED_OBSTACLES
