"""G1 粗糙地形环境配置。

Step 1: 注册链路（已通过）
Step 2: 换地形: 平地 → ROUGH_TERRAINS_CFG（已通过）
Step 3: 修 base_height: 世界系绝对高度 → 相对地形高度（已通过）
Step 4: 感知空间: 把 height_scanner 接进 policy / critic 观测（当前）
后续: 奖励调整，每加一项跑一次冒烟测试。
"""
import copy
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from unitree_rl_lab.tasks.locomotion import mdp
from . import mdp_ext
from ..velocity_env_cfg import RobotEnvCfg

@configclass
class G1RoughEnvCfg(RobotEnvCfg):
    """粗糙地形训练配置。"""

    def __post_init__(self):
        terrain_cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
        self.scene.terrain.terrain_generator = terrain_cfg
        self.scene.terrain.max_init_terrain_level = terrain_cfg.num_rows - 1
        super().__post_init__()
        scan = SceneEntityCfg('height_scanner')
        self.rewards.base_height.func = mdp_ext.base_height_l2_safe
        self.rewards.base_height.params = {'target_height': 0.78, 'sensor_cfg': scan}
        self.terminations.base_height.func = mdp_ext.root_height_below_minimum_adaptive
        self.terminations.base_height.params = {'minimum_height': 0.2, 'sensor_cfg': scan}
        self.terminations.illegal_reset_contact = DoneTerm(func=mdp_ext.IllegalResetContact, time_out=True, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['torso_link']), 'threshold': 1.0, 'episode_length_threshold': 5})
        scan_obs = ObsTerm(func=mdp.height_scan, params={'sensor_cfg': scan}, clip=(-1.0, 5.0), noise=Unoise(n_min=-0.1, n_max=0.1))
        self.observations.policy.height_scanner = scan_obs
        self.observations.critic.height_scanner = copy.deepcopy(scan_obs)
        self.observations.critic.height_scanner.noise = None
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.gait.weight = 0.0
        self.rewards.feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.5, params={'command_name': 'base_velocity', 'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*ankle_roll.*'), 'threshold': 0.5})
        self.rewards.stand_still = RewTerm(func=mdp.stand_still, weight=-1.0, params={'command_name': 'base_velocity'})
        self.rewards.joint_deviation_arms.weight = -0.5
        self.rewards.joint_deviation_waists.weight = -2.0
        self.rewards.joint_deviation_legs.weight = -2.0
        self.rewards.base_height.weight = -20.0
        self.rewards.feet_clearance.func = mdp_ext.foot_clearance_reward_rough
        self.rewards.feet_clearance.params = {'std': 0.05, 'tanh_mult': 2.0, 'target_height': 0.1, 'asset_cfg': SceneEntityCfg('robot', body_names='.*ankle_roll.*'), 'sensor_cfg': scan}

@configclass
class G1RoughPlayEnvCfg(G1RoughEnvCfg):
    """播放配置。

    必须继承 ``G1RoughEnvCfg`` 而不是基线的 ``RobotPlayEnvCfg``:
    后者继承自 ``RobotEnvCfg``，走那条链会丢掉本文件对训练配置做的全部改动。
    代价是要手动重复基线 play 的差异项。
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 4
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.viewer.origin_type = 'asset_root'
        self.viewer.asset_name = 'robot'
        self.viewer.env_index = 0
        self.viewer.eye = (6.0, 6.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)
