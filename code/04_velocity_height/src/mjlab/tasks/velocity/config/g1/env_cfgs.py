"""Unitree G1 velocity environment configurations.

Homework TODOs in this file: 3, 4, 5  (of 10 total)
Function: unitree_g1_flat_height_env_cfg()
Index: 本文件 · grep: 【实现要点
"""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import (
  UniformBaseHeightCommandCfg,
  UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

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
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  velocity_cmd = cfg.commands["velocity"]
  assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
  velocity_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    velocity_cmd = cfg.commands["velocity"]
    assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
    velocity_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    velocity_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  return unitree_g1_base_env_cfg(play=play)


def unitree_g1_flat_height_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity + height command configuration."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  # >>> IMPL_3_START
  # ==============================================================================
  # 【实现要点 3/10】注册 base_height 命令
  # ==============================================================================
  # 蹲姿行走的闭环有三段，缺一不可：
  #   TODO 3 采样高度指令  →  TODO 4 Actor 观测到指令  →  TODO 5/6 奖励高度跟踪
  # 助教点名的最常见错误就是只做了第三段（写了高度奖励），却漏了第二段。
  # 那样策略根本不知道目标高度是多少，只能从历史观测里瞎猜，训练目标不成立。
  #
  # 键名 "base_height" 必须与 TODO 4 的 command_name、TODO 5 的 command_name
  # 三处完全一致,CommandManager 靠这个字符串查表，拼错不会报错，
  # 只会在 get_command 时返回 None 触发断言。
  #
  # resampling_time_range 取 (3.0, 8.0) 与速度指令保持一致：高度和速度指令
  # 各自独立重采样，策略会遇到"边加速边站起""边减速边下蹲"等各种组合，
  # 这正是双指令 MDP 要覆盖的状态空间。
  cfg.commands["base_height"] = UniformBaseHeightCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    debug_vis=True,  # Viser 里画出橙色目标球与青色实际球，便于肉眼验收
    ranges=UniformBaseHeightCommandCfg.Ranges(height=(0.45, 0.80)),
  )
  # <<< IMPL_3_END

  # >>> IMPL_4_START
  # ==============================================================================
  # 【实现要点 4/10】接入 height_command 观测（蹲姿/高度任务核心观测）
  # ==============================================================================
  # 闭环的第二段。generated_commands 是通用观测函数，按 command_name 从
  # CommandManager 取出当前指令张量塞进观测向量，这里是 1 维（h_cmd）。
  #
  # Actor 和 Critic 都要加：
  #   Actor  必须看到,它要根据目标高度决定蹲多低，这是任务输入不是特权信息
  #   Critic 也要看到,价值函数得知道"当前指令是什么"才能正确估计这个状态的期望回报
  # 特权信息（如 TODO 10 的 foot_contact）才是只给 Critic 的，指令不是特权信息。
  height_command_obs = ObservationTermCfg(
    func=envs_mdp.generated_commands,
    params={"command_name": "base_height"},
  )
  cfg.observations["actor"].terms["height_command"] = height_command_obs
  cfg.observations["critic"].terms["height_command"] = height_command_obs
  # <<< IMPL_4_END

  # >>> IMPL_5_START
  # ==============================================================================
  # 【实现要点 5/10】注册 track_base_height 奖励项（蹲姿/高度任务核心奖励）
  # ==============================================================================
  # 闭环的第三段，接上 TODO 6 实现的高斯跟踪奖励。
  #
  # weight=1.0 按 PDF §2.4 的奖励表。注意它比速度跟踪的 2.0 小一半:
  # 高度是附加任务，行走本身才是主任务；若高度权重压过速度，策略会倾向于
  # 站着不动把高度稳住（实践 2 刚踩过这个坑：静止是很多奖励项的隐藏最优解）。
  #
  # std=0.1 是这一项的尺度（weight 是重要性，std 是"多大误差算差"）。
  # 高度指令区间只有 0.45~0.80 m 共 0.35 m 宽，比速度指令的量程窄得多，
  # 所以尺度要比速度跟踪的 sqrt(0.25)=0.5 严得多：
  #   std=0.1 时，差 0.1 m 得 exp(-1)=0.37 分；差 0.05 m 得 exp(-0.25)=0.78 分
  #   若沿用 0.5，差 0.1 m 仍有 exp(-0.04)=0.96 分，几乎没有区分度
  cfg.rewards["track_base_height"] = RewardTermCfg(
    func=mdp.track_base_height,
    weight=1.0,
    params={"command_name": "base_height", "std": 0.1},
  )
  # <<< IMPL_5_END

  return cfg


# ═══════════════════════════════════════════════════════════════════════════
# 消融实验变体（消融设计要求 ≥2 组）
# ═══════════════════════════════════════════════════════════════════════════
#
# 蹲姿行走的因果链是一条三段闭环：
#     1. 采样高度指令  →  2. Actor 观测到指令  →  3. 奖励高度跟踪
# 下面两组各切断其中一段，用来验证"缺一不可"这个结论。
#
# 公平对照原则：每组只改一个因素，其余（seed / num_envs / iteration /
# num_steps_per_env / 所有其他奖励权重）全部保持与 baseline 一致。
#
# 评估指标（项目指定）：error_height、总回报、h_cmd=0.5 m 下的稳定行走时长。


def unitree_g1_flat_height_ablation_no_reward_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """消融 A: 切断第 3. 段：保留指令与观测，但不奖励高度跟踪。

  预期：策略能"看见"目标高度却没有动机去跟踪它，骨盆高度会停在
  它自己觉得最舒服的位置（通常是让速度跟踪奖励最大化的姿态），
  error_height 显著高于 baseline 且与 h_cmd 无相关性。

  用 weight=0 而不是删除该项：RewardManager 不跳过零权重项，
  TensorBoard 里仍会记录 track_base_height（恒为 0），
  且 error_height 指标照常输出，两组曲线可以直接叠图对比。
  """
  cfg = unitree_g1_flat_height_env_cfg(play=play)
  cfg.rewards["track_base_height"].weight = 0.0
  return cfg


def unitree_g1_flat_height_ablation_blind_actor_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """消融 B: 切断第 2. 段：奖励照给，但 Actor 看不到高度指令。

  这是更能说明问题的一组。奖励函数仍在惩罚"高度不对"，可策略的输入里
  没有目标高度这一维，它无从知道该蹲到多低。

  理论预期：策略只能收敛到一个折中的固定高度,让 h_cmd 在整个区间上的
  期望误差最小，即区间中点 (0.45+0.80)/2 = 0.625 m。此时
      E|h_cmd - 0.625| = (0.80-0.45)/4 = 0.0875 m
  所以 error_height 应当收敛到约 0.0875 m 且不随指令变化。
  这个可解析的预测值让消融结论可证伪，而不只是"效果更差"。

  只从 Actor 移除，Critic 保留。
  我们要测的是"Actor 知不知道目标"，不是"系统里有没有这个信息"。
  Critic 保留能让价值估计质量与 baseline 一致，把变量隔离在策略网络
  这一侧,否则 critic 也变瞎，优势函数的方差变化会污染对照。
  这与文档里"指令不是特权信息"的论述不矛盾：此处是刻意制造的病态配置，
  用于反证观测接线的必要性。
  """
  cfg = unitree_g1_flat_height_env_cfg(play=play)
  cfg.observations["actor"].terms.pop("height_command")
  return cfg
