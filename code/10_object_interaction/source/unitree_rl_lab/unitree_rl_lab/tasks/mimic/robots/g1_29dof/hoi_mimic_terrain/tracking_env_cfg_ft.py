# from __future__ import annotations

# import os

# """HOI terrain mimic fine-tune DR.

# Physical + root-velocity ranges follow a common Table V-style schedule (uniform):
# static/dynamic friction, restitution, split default joint offsets (ankle vs non-ankle),
# torso CoM offset, and additive root velocity kicks with event interval ~ U[1,3] s.

# Note: ``push_by_setting_velocity`` applies an instantaneous root velocity increment when
# the interval fires; it does not model a sustained push over Δt seconds like some papers.
# """

# from isaaclab.managers import EventTermCfg as EventTerm
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.utils import configclass

# import unitree_rl_lab.tasks.mimic.mdp as mdp
# from .tracking_env_cfg import EventCfg as BaseEventCfg
# from .tracking_env_cfg import RobotEnvCfg, RobotPlayEnvCfg

# # Table V-style contact / inertia DR (uniform ranges as in the paper).
# _PAPER_FRICTION_STATIC = (0.4, 1.3)
# _PAPER_FRICTION_DYNAMIC = (0.4, 1.1)
# _PAPER_RESTITUTION = (0.0, 0.5)

# # Default joint offsets: non-ankle vs ankle (rad).
# _PAPER_JOINT_DEFAULT_NON_ANKLE = (-0.01, 0.01)
# _PAPER_JOINT_DEFAULT_ANKLE = (-0.1, 0.1)

# # Torso CoM offset (m): dx smaller than dy, dz per Table V.
# _PAPER_TORSO_COM = {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}

# # Root velocity perturbations (additive); keys match ``push_by_setting_velocity``.
# # Paper: v_x,v_y ~ U[-0.1,0.1], v_z ~ U[-0.05,0.05], ω ~ U[-0.1,0.1]; push interval ~ U[1,3] s.
# _PAPER_ROOT_VELOCITY_PUSH = {
#     "x": (-0.05, 0.05),
#     "y": (-0.05, 0.05),
#     "z": (-0.05, 0.05),
#     "roll": (-0.1, 0.1),
#     "pitch": (-0.1, 0.1),
#     "yaw": (-0.1, 0.1),
# }
# _PAPER_PUSH_INTERVAL_S = (2.0, 8.0)


# def _env_float(name: str, default: float) -> float:
#     raw = os.getenv(name)
#     if raw is None:
#         return default
#     try:
#         return float(raw)
#     except ValueError as exc:
#         raise ValueError(f"Environment variable {name} must be a float, got {raw!r}") from exc


# def _scaled_velocity_range(
#     base: dict[str, tuple[float, float]],
#     scale: float,
# ) -> dict[str, tuple[float, float]]:
#     """Scale symmetric push ranges conservatively around zero."""
#     out: dict[str, tuple[float, float]] = {}
#     for key, (low, high) in base.items():
#         out[key] = (low * scale, high * scale)
#     return out


# # Stage-B conservative transfer knobs (default to a gentler push regime than Table V full range).
# # Keep Stage-A physical DR unchanged; only push term is staged/tunable.
# _STAGE_B_PUSH_SCALE = _env_float("FT_STAGE_B_PUSH_SCALE", 0.6)
# _STAGE_B_PUSH_INTERVAL_S = (
#     _env_float("FT_STAGE_B_PUSH_INTERVAL_MIN_S", 2.0),
#     _env_float("FT_STAGE_B_PUSH_INTERVAL_MAX_S", 4.0),
# )
# _STAGE_B_ROOT_VELOCITY_PUSH = _scaled_velocity_range(_PAPER_ROOT_VELOCITY_PUSH, _STAGE_B_PUSH_SCALE)
# # --------------------------------------------------------------------
# # Scratch conservative task knobs (all Stage-B terms active from iteration 0).
# # Hardcoded for easier manual tuning in one place.
# _SCRATCH_PUSH_INTERVAL_S = (
#     _env_float("SCRATCH_PUSH_INTERVAL_MIN_S", 1.0),
#     _env_float("SCRATCH_PUSH_INTERVAL_MAX_S", 5.0),
# )
# # Robot: match ``EventCfg.physics_material`` in ``tracking_env_cfg.py``.
# _SCRATCH_ROBOT_FRICTION_STATIC = (0.4, 1.3)
# _SCRATCH_ROBOT_FRICTION_DYNAMIC = (0.4, 1.1)
# _SCRATCH_ROBOT_RESTITUTION = (0.0, 0.5)
# # Ground + HOI boxes: mild friction DR; keep restitution tiny (stiff contacts / sim2sim).
# _SCRATCH_TERRAIN_FRICTION_STATIC = (0.9, 1.1)
# _SCRATCH_TERRAIN_FRICTION_DYNAMIC = (0.9, 1.1)
# _SCRATCH_TERRAIN_RESTITUTION = (0.0, 0.15)
# _SCRATCH_JOINT_DEFAULT_NON_ANKLE = (-0.005, 0.005)
# _SCRATCH_JOINT_DEFAULT_ANKLE = (-0.015, 0.015)
# _SCRATCH_TORSO_COM = {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.015, 0.015)}
# _SCRATCH_ROOT_VELOCITY_PUSH = {
#     "x": (-0.00, 0.00),
#     "y": (-0.00, 0.00),
#     "z": (-0.00, 0.00),
#     # "roll": (-0.04, 0.04),
#     # "pitch": (-0.04, 0.04),
#     # "yaw": (-0.04, 0.04),
# }

# # ------------------------ Temporary DR experiments ------------------------
# @configclass
# class FineTuneStageAEventCfg(BaseEventCfg):
#     """Table V physical DR only (no root pushes). Friction on robot + ground + HOI terrain."""

#     physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "static_friction_range": _PAPER_FRICTION_STATIC,
#             "dynamic_friction_range": _PAPER_FRICTION_DYNAMIC,
#             "restitution_range": _PAPER_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )

#     ground_physics_material = EventTerm(
#         func=mdp.randomize_terrain_importer_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("terrain"),
#             "static_friction_range": _PAPER_FRICTION_STATIC,
#             "dynamic_friction_range": _PAPER_FRICTION_DYNAMIC,
#             "restitution_range": _PAPER_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )
#     hoi_terrain_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("hoi_terrain", body_names=".*"),
#             "static_friction_range": _PAPER_FRICTION_STATIC,
#             "dynamic_friction_range": _PAPER_FRICTION_DYNAMIC,
#             "restitution_range": _PAPER_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )

#     add_joint_default_pos_non_ankle = EventTerm(
#         func=mdp.randomize_joint_default_pos,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*ankle).*$"]),
#             "pos_distribution_params": _PAPER_JOINT_DEFAULT_NON_ANKLE,
#             "operation": "add",
#         },
#     )
#     add_joint_default_pos_ankle = EventTerm(
#         func=mdp.randomize_joint_default_pos,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"]),
#             "pos_distribution_params": _PAPER_JOINT_DEFAULT_ANKLE,
#             "operation": "add",
#         },
#     )

#     base_com = EventTerm(
#         func=mdp.randomize_rigid_body_com,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
#             "com_range": _PAPER_TORSO_COM,
#         },
#     )


# @configclass
# class FineTuneStageBEventCfg(BaseEventCfg):
#     """Conservative Stage-2: freeze physical DR, add tunable interval root velocity pushes."""

#     physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "static_friction_range": _PAPER_FRICTION_STATIC,
#             "dynamic_friction_range": _PAPER_FRICTION_DYNAMIC,
#             "restitution_range": _PAPER_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )

#     ground_physics_material = EventTerm(
#         func=mdp.randomize_terrain_importer_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("terrain"),
#             "static_friction_range": _PAPER_FRICTION_STATIC,
#             "dynamic_friction_range": _PAPER_FRICTION_DYNAMIC,
#             "restitution_range": _PAPER_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )
#     hoi_terrain_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("hoi_terrain", body_names=".*"),
#             "static_friction_range": _PAPER_FRICTION_STATIC,
#             "dynamic_friction_range": _PAPER_FRICTION_DYNAMIC,
#             "restitution_range": _PAPER_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )

#     add_joint_default_pos_non_ankle = EventTerm(
#         func=mdp.randomize_joint_default_pos,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*ankle).*$"]),
#             "pos_distribution_params": _PAPER_JOINT_DEFAULT_NON_ANKLE,
#             "operation": "add",
#         },
#     )
#     add_joint_default_pos_ankle = EventTerm(
#         func=mdp.randomize_joint_default_pos,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"]),
#             "pos_distribution_params": _PAPER_JOINT_DEFAULT_ANKLE,
#             "operation": "add",
#         },
#     )

#     base_com = EventTerm(
#         func=mdp.randomize_rigid_body_com,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
#             "com_range": _PAPER_TORSO_COM,
#         },
#     )

#     push_root_velocity = EventTerm(
#         func=mdp.push_by_setting_velocity,
#         mode="interval",
#         interval_range_s=_STAGE_B_PUSH_INTERVAL_S,
#         params={"velocity_range": _STAGE_B_ROOT_VELOCITY_PUSH},
#     )


# @configclass
# class FineTuneStageCEventCfg(BaseEventCfg):
#     """Same Table V DR as Stage B; Stage C env adds motion-reference velocity caps in ``RobotFineTuneStageCEnvCfg``."""

#     physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "static_friction_range": _PAPER_FRICTION_STATIC,
#             "dynamic_friction_range": _PAPER_FRICTION_DYNAMIC,
#             "restitution_range": _PAPER_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )

#     ground_physics_material = EventTerm(
#         func=mdp.randomize_terrain_importer_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("terrain"),
#             "static_friction_range": _PAPER_FRICTION_STATIC,
#             "dynamic_friction_range": _PAPER_FRICTION_DYNAMIC,
#             "restitution_range": _PAPER_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )
#     hoi_terrain_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("hoi_terrain", body_names=".*"),
#             "static_friction_range": _PAPER_FRICTION_STATIC,
#             "dynamic_friction_range": _PAPER_FRICTION_DYNAMIC,
#             "restitution_range": _PAPER_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )

#     add_joint_default_pos_non_ankle = EventTerm(
#         func=mdp.randomize_joint_default_pos,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*ankle).*$"]),
#             "pos_distribution_params": _PAPER_JOINT_DEFAULT_NON_ANKLE,
#             "operation": "add",
#         },
#     )
#     add_joint_default_pos_ankle = EventTerm(
#         func=mdp.randomize_joint_default_pos,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"]),
#             "pos_distribution_params": _PAPER_JOINT_DEFAULT_ANKLE,
#             "operation": "add",
#         },
#     )

#     base_com = EventTerm(
#         func=mdp.randomize_rigid_body_com,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
#             "com_range": _PAPER_TORSO_COM,
#         },
#     )

#     push_root_velocity = EventTerm(
#         func=mdp.push_by_setting_velocity,
#         mode="interval",
#         interval_range_s=_PAPER_PUSH_INTERVAL_S,
#         params={"velocity_range": _PAPER_ROOT_VELOCITY_PUSH},
#     )


# @configclass
# class ScratchConservativeDrEventCfg(BaseEventCfg):
#     """Scratch DR task: Stage-B term coverage with conservative defaults from iteration 0."""

#     physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "static_friction_range": _SCRATCH_ROBOT_FRICTION_STATIC,
#             "dynamic_friction_range": _SCRATCH_ROBOT_FRICTION_DYNAMIC,
#             "restitution_range": _SCRATCH_ROBOT_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )

#     ground_physics_material = EventTerm(
#         func=mdp.randomize_terrain_importer_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("terrain"),
#             "static_friction_range": _SCRATCH_TERRAIN_FRICTION_STATIC,
#             "dynamic_friction_range": _SCRATCH_TERRAIN_FRICTION_DYNAMIC,
#             "restitution_range": _SCRATCH_TERRAIN_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )
#     hoi_terrain_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("hoi_terrain", body_names=".*"),
#             "static_friction_range": _SCRATCH_TERRAIN_FRICTION_STATIC,
#             "dynamic_friction_range": _SCRATCH_TERRAIN_FRICTION_DYNAMIC,
#             "restitution_range": _SCRATCH_TERRAIN_RESTITUTION,
#             "num_buckets": 64,
#         },
#     )

#     add_joint_default_pos_non_ankle = EventTerm(
#         func=mdp.randomize_joint_default_pos,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*ankle).*$"]),
#             "pos_distribution_params": _SCRATCH_JOINT_DEFAULT_NON_ANKLE,
#             "operation": "add",
#         },
#     )
#     add_joint_default_pos_ankle = EventTerm(
#         func=mdp.randomize_joint_default_pos,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"]),
#             "pos_distribution_params": _SCRATCH_JOINT_DEFAULT_ANKLE,
#             "operation": "add",
#         },
#     )

#     base_com = EventTerm(
#         func=mdp.randomize_rigid_body_com,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
#             "com_range": _SCRATCH_TORSO_COM,
#         },
#     )

#     push_root_velocity = EventTerm(
#         func=mdp.push_by_setting_velocity,
#         mode="interval",
#         interval_range_s=_SCRATCH_PUSH_INTERVAL_S,
#         params={"velocity_range": _SCRATCH_ROOT_VELOCITY_PUSH},
#     )


# @configclass
# class RobotFineTuneStageAEnvCfg(RobotEnvCfg):
#     """Resume fine-tune stage A with mild DR."""

#     events: FineTuneStageAEventCfg = FineTuneStageAEventCfg()


# @configclass
# class RobotFineTuneStageBEnvCfg(RobotEnvCfg):
#     """Resume fine-tune stage B with medium DR."""

#     events: FineTuneStageBEventCfg = FineTuneStageBEventCfg()


# @configclass
# class RobotFineTuneStageCEnvCfg(RobotEnvCfg):
#     """Resume fine-tune stage C with stronger DR."""

#     events: FineTuneStageCEventCfg = FineTuneStageCEventCfg()

#     def __post_init__(self):
#         super().__post_init__()
#         # Motion-reference root pose: keep tight (terrain coupling); Table V does not add pose noise here.
#         self.commands.motion.pose_range = {
#             "x": (0.0, 0.0),
#             "y": (0.0, 0.0),
#             "z": (0.0, 0.0),
#             "roll": (0.0, 0.0),
#             "pitch": (0.0, 0.0),
#             "yaw": (0.0, 0.0),
#         }
#         # Align commanded root velocity sampling with Table V root velocity perturbation ranges.
#         self.commands.motion.velocity_range = dict(_PAPER_ROOT_VELOCITY_PUSH)


# @configclass
# class RobotScratchDrConservativeEnvCfg(RobotEnvCfg):
#     """Scratch training env with conservative Stage-B-style DR enabled from the beginning."""

#     events: ScratchConservativeDrEventCfg = ScratchConservativeDrEventCfg()


# @configclass
# class RobotFineTunePlayEnvCfg(RobotPlayEnvCfg):
#     """Play config paired with fine-tune stages."""

#     pass
