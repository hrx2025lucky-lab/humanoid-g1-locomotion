"""G1 粗糙地形环境配置。

Step 1: 注册链路（已通过）
Step 2: 换地形 —— 平地 → ROUGH_TERRAINS_CFG（已通过）
Step 3: 修 base_height —— 世界系绝对高度 → 相对地形高度（当前）
后续: 高度扫描观测 → 奖励调整，每加一项跑一次冒烟测试。
"""

import copy

from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

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
        # 播放相机跟随选中的机器人
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (3.0, 3.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.8)
