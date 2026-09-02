"""G1 粗糙地形环境配置。

Step 1: 注册链路（已通过）
Step 2: 换地形 —— 平地 → ROUGH_TERRAINS_CFG（已通过）
Step 3: 修 base_height —— 世界系绝对高度 → 相对地形高度（已通过）
Step 4: 感知空间 —— 把 height_scanner 接进 policy / critic 观测（当前）
后续: 奖励调整，每加一项跑一次冒烟测试。
"""

import copy

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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
        # ───────────────────────── Step 2: 地形 ─────────────────────────
        # deepcopy：ROUGH_TERRAINS_CFG 是模块级全局单例，直接引用后再改它的字段
        # （父类会写 curriculum，play 配置会写 num_rows）会污染同进程内的其他配置。
        terrain_cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)

        # 必须在 super() 之前赋值：父类 __post_init__ 检测到 curriculum.terrain_levels
        # 存在时，才会把 terrain_generator.curriculum 置 True。
        # TerrainGeneratorCfg.curriculum 默认是 False，赋值晚了这行逻辑就跑空。
        self.scene.terrain.terrain_generator = terrain_cfg
        # 初始难度上限。terrain_importer 用 randint(0, max_init_level+1) 给每个 env
        # **均匀随机**分配难度 —— 不是"从最简单开始"，而是一开始就铺满全部难度。
        self.scene.terrain.max_init_terrain_level = terrain_cfg.num_rows - 1

        super().__post_init__()

        # ──────────────── Step 3: base_height 改用相对地形高度 ────────────────
        # 基线的两个 height 项都拿根节点世界系 z 和常数比，这在平地上等价于
        # 离地高度（地面 z≡0），换到粗糙地形就不成立：倒金字塔楼梯和下坡的
        # 地面本身低于 0，机器人姿态正常却被判摔倒。
        # 冒烟测试实测 Episode_Termination/base_height = 0.25，误杀掉 1/4 的 episode。
        #
        # 放在 super() 之后：父类 __post_init__ 不碰 rewards/terminations
        #（只改 decimation、sim 参数、传感器周期、terrain curriculum），
        # 所以这里覆盖是安全的，且顺序上明确表达"在基线之上做修改"。
        #
        # 传感器不用新建：基线场景第 68 行已有 height_scanner（1.6×1.0 网格，
        # 分辨率 0.1 → 17×11=187 根射线，挂 torso_link 上方 20 m 向下打）。
        # 它此前只被 update_period 初始化过，没有任何 MDP 项在用。
        scan = SceneEntityCfg("height_scanner")

        # 奖励：官方 base_height_l2 虽然支持 sensor_cfg，但内部 torch.mean 无 inf
        # 防护，一根射线打空就会让整个环境的奖励变 inf → 梯度 NaN → 静默崩溃。
        self.rewards.base_height.func = mdp_ext.base_height_l2_safe
        self.rewards.base_height.params = {"target_height": 0.78, "sensor_cfg": scan}

        # 终止：官方 root_height_below_minimum 连 sensor_cfg 都没有，
        # docstring 自己写着 "only supported for flat terrains"。
        self.terminations.base_height.func = mdp_ext.root_height_below_minimum_adaptive
        self.terminations.base_height.params = {"minimum_height": 0.2, "sensor_cfg": scan}

        # ─────────────── Step 4: 感知空间 —— 接入地形高度图 ───────────────
        # 实践 2 的核心要求。基线的 policy 观测里只有本体感受（角速度、重力
        # 投影、速度指令、关节位置/速度、上一步动作），机器人对脚下地形一无所知，
        # 只能靠"踩到了才知道"被动响应。粗糙地形上这远远不够。
        #
        # 传感器早就在场景里（第 68 行 height_scanner），基线却把观测项注释掉了
        # （velocity_env_cfg.py:225-228，而且只写在 CriticCfg 里）。这里启用它。
        #
        # 为什么可以直接给实例赋值而不用重新定义 ObsGroup 子类：
        # ObservationManager._prepare_terms 用的是 group_cfg.__dict__.items()
        #（observation_manager.py:523），遍历的是**实例字典**而非 dataclass 字段，
        # 所以动态赋值会被正常收集。且 Python 3.7+ 的 __dict__ 保持插入顺序，
        # 新项自动排在末尾 —— 这一点对 sim2sim 是硬约束：MuJoCo 侧按同样顺序
        # 手工拼观测向量，顺序错了不会报错，只会让策略读到错位的数值。
        #
        # 用官方 mdp.height_scan 而不是自己写：它的语义是
        #   sensor.data.pos_w[:, 2] - ray_hits_w[..., 2] - offset
        # 一度怀疑 pos_w 含 RayCasterCfg.offset 的 20 m（那样值会是 ~19.9，
        # 被 clip 压成常数 5.0，187 维观测全废且不报错）。实测否定了这个猜测：
        #   pos_w_z - torso_z = 0.0，height_scan 值域 [0.325, 0.536]，
        #   clip 后仍有 390 个不同值。
        # 原因是 cfg.offset 只加在射线起点 ray_starts（ray_caster.py:224，
        # 目的是从高处往下打避免被机器人自身挡住），不进 data.pos_w；
        # RayCaster 内部那个同名 self._offset 是 mesh→刚体的相对变换，纯属撞名。
        #
        # clip 在这里是双保险：既压住异常值域，又能把射线打空时的 inf
        # 截成 5.0（torch.clip(inf, -1, 5) == 5.0），不会污染网络输入。
        # 奖励和终止项没有 clip 这层保护，所以 Step 3 才必须自己做 inf 掩码。
        scan_obs = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": scan},
            clip=(-1.0, 5.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        # policy：加噪。真机高度图来自 LiDAR/深度相机，有测量误差和标定漂移，
        # ±0.1 m 的均匀噪声迫使策略不去依赖单根射线的精确值。
        # policy 组 enable_corruption=True，噪声才会真正生效。
        self.observations.policy.height_scanner = scan_obs
        # critic：同样接入但不加噪。critic 只在训练时用，是特权网络，
        # 它的信息集必须 ⊇ policy，否则价值估计比策略还"瞎"，优势函数失真。
        self.observations.critic.height_scanner = copy.deepcopy(scan_obs)
        self.observations.critic.height_scanner.noise = None

        # ─────────────── Step 5: 奖励权重 —— 打破"站着不动"局部最优 ───────────────
        # 背景：第一次训练（10000 iter）跑出来的策略学会了原地踏步，11.98 s 净位移
        # < 0.3 m。terrain_levels 从 4.92 一路掉到 0（升级判据是净位移 > 4 m，
        # 它做不到就逐级降档），error_vel_xy 反而从 0.135 涨到 0.615。
        #
        # 根因不是训练不足——track_lin_vel_xy 在 4000 iter 就到 0.600，
        # 之后 6000 iter 只挪到 0.628。是奖励函数把"站着"设成了最优解。
        # 实测各项终值（Episode_Reward，单位=每秒平均）：
        #
        #     feet_clearance   +0.8228   ← 脚不动时 tanh(0)=0 → exp(0)=1，白拿满分
        #     track_lin_vel_xy +0.6355   ← 上限只有 1.0
        #     gait             +0.5014   ← 上限 0.5，双脚长期着地也能拿约 0.55/1.0
        #     alive            +0.1448   ← 上限 0.15，拿了 96.5%
        #     action_rate      -0.4281   ← 这些惩罚**只在动起来时才产生**
        #     joint_vel        -0.1334
        #
        # 站着不动能白拿 feet_clearance+gait+alive ≈ 1.47，且几乎不付惩罚；
        # 走起来 track 最多再多拿 0.36，却要付出全部动作类惩罚。走路是亏本买卖。
        #
        # 修法来自课程《第二章作业讲解》p5/p6 的推荐权重表。核心思路不是
        # "把奖励调大"，而是**改变边际激励比**：让"多走一点"带来的收益
        # 明显盖过它引发的动作惩罚。

        # 速度跟踪 1.0→3.0、0.5→2.0。这是本次改动里最关键的一项。
        # track_lin_vel_xy_yaw_frame_exp = exp(-‖v_cmd - v_actual‖²/std²)，值域 (0,1]，
        # 权重就是它的上限。原来跟踪从 60% 提到 100% 只多拿 0.4，现在多拿 1.2，
        # 足以覆盖 action_rate 等惩罚的增量。官方参考曲线该项终值 2.2626/3.0。
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_ang_vel_z.weight = 2.0

        # 关掉固定步态相位奖励。feet_gait 按 0.8 s 周期、相位阈值 0.55 要求
        # 每条腿"该支撑时着地、该摆动时腾空"。粗糙地形上机器人需要临时改步频、
        # 延长某只脚支撑时间来救失衡，固定相位是束缚。
        # 附带好处：双脚静止时它仍能拿约 0.55，是"站着不动"的收益来源之一。
        # 用 weight=0 而不是删除——RewardManager 不会跳过零权重项，
        # TensorBoard 里仍可见（恒为 0），方便和上一次训练对照。
        self.rewards.gait.weight = 0.0

        # 新增：抬脚腾空奖励。这是整个奖励函数里**唯一直接给"迈步"发钱**的项。
        # feet_air_time_positive_biped 只在 single_stance（恰好一只脚着地）时计分，
        # 双脚同时着地（站立）和双脚同时腾空（跳跃）都是 0，天然反"原地不动"。
        # 内部还有 reward *= (‖cmd_xy‖ > 0.1)，零指令时不发钱，不会鼓励空踏步。
        # threshold=0.5 给腾空时间封顶，避免策略靠"单腿长时间悬空"刷分。
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                "threshold": 0.5,
            },
        )

        # 新增：低速指令下的静止约束。stand_still 返回 Σ|q - q_default|
        # 并乘以 (‖cmd‖ < 0.1) 的掩码，配负权重即"指令接近零时偏离默认站姿要挨罚"。
        # 作用不是让它更爱站着，恰恰相反：它把"停"定义成一个**具体姿态**，
        # 让"走"和"停"成为两个可区分的模式。缺了它，策略可以用同一套
        # 原地小幅摆动应付所有指令——这正是第一次训练的失败形态。
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )

        # 姿态约束加严。粗糙地形上手臂/腰/腿乱摆会显著抬高重心晃动，
        # 让 height_scan 的观测和真实落脚点失配。官方表给的是 -0.5/-2.0/-2.0。
        self.rewards.joint_deviation_arms.weight = -0.5
        self.rewards.joint_deviation_waists.weight = -2.0
        self.rewards.joint_deviation_legs.weight = -2.0

        # 机身高度惩罚 -10→-20。Step 3 已把它换成相对地形高度的安全版，
        # 加倍权重是为了在 track 权重提到 3.0 之后，机器人不会为了跑快而弓身下蹲
        # ——蹲着重心低、更容易通过速度跟踪拿分，但那不是我们要的步态。
        self.rewards.base_height.weight = -20.0

        # feet_clearance 换成相对地形高度的版本（详见 mdp_ext 中的推导）。
        # 官方版 target_height=0.1 是世界系绝对高度，在台阶上会把"正确抬脚"
        # 判成误差。权重维持 1.0 不变，只修基准。
        self.rewards.feet_clearance.func = mdp_ext.foot_clearance_reward_rough
        self.rewards.feet_clearance.params = {
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": scan,
        }


@configclass
class G1RoughPlayEnvCfg(G1RoughEnvCfg):
    """播放配置。

    必须继承 ``G1RoughEnvCfg`` 而不是基线的 ``RobotPlayEnvCfg`` ——
    后者继承自 ``RobotEnvCfg``，走那条链会丢掉本文件对训练配置做的全部改动。
    代价是要手动重复基线 play 的差异项。
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # 基线 play 用 num_rows=2，但行=难度等级，2 行只有难度 0.0/0.5 两档，
        # 录像里会全是最简单地形。粗糙地形下给 5 行才看得到中高难度。
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 4
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # 播放相机跟随选中的机器人。
        # 保留跟随而不是固定机位：跟随才能看清步态细节（抬脚高度、落脚时机）。
        # 但把机位从 (3,3,2.5) 拉到 (6,6,4)：第一次录像用近机位的教训是
        # 机器人被锁在画面正中，既看不出净位移、也框不进周围地形，
        # 只能靠背景台阶的像素偏移倒推它走了多远。拉远后单帧内可同时看到
        # 机器人和它脚下/前方的地形块，位移和地形难度都能直接读出来。
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (6.0, 6.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)
