"""Unitree G1 HW6 environment configurations."""

import math
import os
from pathlib import Path

from mjlab.actuator import DelayedActuatorCfg
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION_WITHOUT_SELF,
)
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from humanoid_hw6.mdp import (
  FutureJointRefAnchorRpMotionCommandCfg,
  JointRefAnchorRpMotionCommandCfg,
  TeacherStudentJointRefAnchorRpMotionCommandCfg,
)
from humanoid_hw6.student_env_cfg import make_student_env_cfg
from humanoid_hw6.teacher_env_cfg import make_teacher_env_cfg

# 默认动作源可用环境变量切换，便于实践 9 P2 复用同一套跟踪框架：
#   HW6_MOTION_SOURCE=motion_data_cfg_hw9_dance.yaml  → 训练舞蹈轨迹跟踪
# 不写死路径是为了不影响实践 6 已经跑完的两组对照。
_MOTION_CFG_NAME = os.environ.get("HW6_MOTION_SOURCE", "motion_data_cfg.yaml")
DEFAULT_MOTION_SOURCE = str(
  Path(__file__).resolve().with_name(_MOTION_CFG_NAME)
)
G1_PELVIS_MASS_KG = 3.813
G1_PELVIS_MASS_DELTA_KG = 3.0
G1_PELVIS_ALPHA_RANGE = (
  0.5 * math.log((G1_PELVIS_MASS_KG - G1_PELVIS_MASS_DELTA_KG) / G1_PELVIS_MASS_KG),
  0.5 * math.log((G1_PELVIS_MASS_KG + G1_PELVIS_MASS_DELTA_KG) / G1_PELVIS_MASS_KG),
)
G1_TRACKED_BODY_NAMES = (
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

G1_COMPARISON_KEY_BODY_NAMES = (
  "left_knee_link",
  "left_ankle_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def _make_g1_delayed_actuators(
  base_actuators: tuple[DelayedActuatorCfg, ...],
) -> tuple[DelayedActuatorCfg, ...]:
  return tuple(
    DelayedActuatorCfg(
      base_cfg=actuator_cfg,
      delay_target="position",
      delay_min_lag=0,
      delay_max_lag=4,
      delay_hold_prob=1.0,
    )
    for actuator_cfg in base_actuators
  )


def _apply_unitree_g1_overrides(
  cfg: ManagerBasedRlEnvCfg,
  play: bool,
) -> ManagerBasedRlEnvCfg:
  """Apply Unitree G1 robot/sensor/DR overrides to a HW6 task template."""

  robot_cfg = get_g1_robot_cfg()
  robot_cfg.collisions = (FULL_COLLISION_WITHOUT_SELF,)
  assert robot_cfg.articulation is not None
  robot_cfg.articulation = EntityArticulationInfoCfg(
    actuators=_make_g1_delayed_actuators(robot_cfg.articulation.actuators),
    soft_joint_pos_limit_factor=robot_cfg.articulation.soft_joint_pos_limit_factor,
  )
  cfg.scene.entities = {"robot": robot_cfg}

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Keep this commented out while using the no-self-collision G1 preset.
  # self_collision_cfg = ContactSensorCfg(
  #   name="self_collision",
  #   primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
  #   secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
  #   fields=("found",),
  #   reduce="none",
  #   num_slots=1,
  # )
  cfg.scene.sensors = (feet_ground_cfg,)
  # cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = dict(G1_ACTION_SCALE)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(
    motion_cmd,
    (
      FutureJointRefAnchorRpMotionCommandCfg,
      JointRefAnchorRpMotionCommandCfg,
      TeacherStudentJointRefAnchorRpMotionCommandCfg,
    ),
  )
  motion_cmd.motion_file = DEFAULT_MOTION_SOURCE
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.root_body_name = "pelvis"
  motion_cmd.body_names = G1_TRACKED_BODY_NAMES
  motion_cmd.sampling_mode = "adaptive"

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  for observation_group in cfg.observations.values():
    if observation_group is None or "friction_coeff" not in observation_group.terms:
      continue
    observation_group.terms["friction_coeff"].params[
      "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  if "base_mass" in cfg.events:
    cfg.events["base_mass"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.events["base_mass"].params["alpha_range"] = G1_PELVIS_ALPHA_RANGE
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  if "push_end_effector" in cfg.events:
    cfg.events["push_end_effector"].params["asset_cfg"].body_names = (
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    )
    # G1 palm sites are defined at +X = 0.08 in each wrist-yaw link frame.
    cfg.events["push_end_effector"].params["body_point_offset"] = (0.08, 0.0, 0.0)
    cfg.events["push_end_effector"].params["randomize_application_point"] = True
    cfg.events["push_end_effector"].params["application_point_delta_range"] = (
      (-0.04, 0.06),
      (-0.01, 0.01),
      (-0.03, 0.03),
    )

  if "foot_slip" in cfg.rewards:
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = (
      "left_foot",
      "right_foot",
    )

  if "ee_body_pos" in cfg.terminations:
    cfg.terminations["ee_body_pos"].params["body_names"] = (
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    )

  cfg.viewer.body_name = "pelvis"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    for observation_group in cfg.observations.values():
      if observation_group is not None:
        observation_group.enable_corruption = False
    motion_expiration_termination = cfg.terminations.get("motion_ref_expired")
    cfg.terminations.clear()
    if motion_expiration_termination is not None:
      cfg.terminations["motion_ref_expired"] = motion_expiration_termination
    cfg.events.pop("push_robot", None)
    cfg.events.pop("action_delay", None)
    # Uncomment to disable "push_end_effector" event.
    # cfg.events.pop("push_end_effector", None)

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    motion_cmd.log_active_motion = True

  return cfg


def unitree_g1_teacher_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 HW6 teacher configuration."""
  return _apply_unitree_g1_overrides(make_teacher_env_cfg(), play=play)


def unitree_g1_student_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 student distillation configuration."""
  return _apply_unitree_g1_overrides(make_student_env_cfg(), play=play)
