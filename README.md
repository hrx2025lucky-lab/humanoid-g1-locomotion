# Humanoid Locomotion — 人形机器人强化学习运动控制

人形机器人运控的训练、验证与跨仿真器部署。每个任务以**零侵入**方式挂进第三方训练框架，
第三方仓库保持零改动，`git log` 里只有自己写的代码。

仿真栈：Isaac Sim 5.1 / Isaac Lab 2.3.2 / rsl-rl 3.1.2 / MuJoCo 3.12。
本体：Unitree G1（29 DoF）。

---

## 路线图

覆盖从环境搭建、MDP 设计、模仿学习到跨仿真器部署的完整技术链路：

| # | 主题 | 技术点 | 状态 |
|---|---|---|---|
| 1 | 仿真环境搭建与基础验证 | Isaac Sim / Isaac Lab / MuJoCo 三栈打通 | ✅ |
| 2 | 粗糙地形行走 | 地形课程、高度扫描感知、奖励与终止项重设计 | 🚧 |
| 3 | 动作空间与 Sim2Sim 部署 | 动作空间设计、跨仿真器迁移 | ⬜ |
| 4 | 蹲姿行走策略 | 速度 + 骨盆高度的 MDP 设计 | ⬜ |
| 5 | 分层强化学习导航 | 高层导航策略 + 底层运控策略 | ⬜ |
| 6 | 教师–学生蒸馏 | 全身运动跟踪、特权信息蒸馏 | ⬜ |
| 7 | 人体动作重定向 | 人体动捕 → G1 关节空间 | ⬜ |
| 8 | AMP 拟人走跑 | 对抗式动作先验 | ⬜ |
| 9 | 轨迹追踪训练 | 运动跟踪关键函数实现 | ⬜ |
| 10 | 人–物交互运动跟踪 | HOI | ⬜ |
| 11 | 跑酷策略与 Sim2Sim 验证 | 高动态动作 + 部署验证 | ⬜ |

## 已实现

| 目录 | 内容 | 对应主题 |
|---|---|---|
| [`tasks/g1_rough/`](tasks/g1_rough) | 粗糙地形行走：地形课程 + 高度扫描感知 + 奖励重设计 | 2 |
| [`sim2sim/`](sim2sim) | Isaac Lab → MuJoCo 策略迁移，六项对齐验证 | 1 |

## 仓库结构

```
tasks/       每个训练任务一个独立包，通过 symlink 挂进训练框架的任务树
sim2sim/     跨仿真器部署验证：把训练好的策略搬到另一个物理引擎独立复现
scripts/     训练/录像等可复现的操作脚本
docs/        调参与问题定位记录
```

新增任务时在 `tasks/` 下建目录，配一个 `install_overlay.sh` 完成挂载即可，
不需要改动本仓库以外的任何文件。

## 记录

| 文档 | 内容 |
|---|---|
| [`docs/实践2_奖励调参记录.md`](docs/实践2_奖励调参记录.md) | 第一次训练学出"原地踏步"策略的完整定位过程：如何从 `Episode_Reward/*` 分项拆解识别局部最优，以及 10 项权重为什么这么改 |
| [`docs/实践3_HoST增量动作空间.md`](docs/实践3_HoST增量动作空间.md) | 增量动作空间 `q*=q_cur+αa` 与残差式 `q*=q_def+αa` 的本质差别，以及为什么接触状态频繁切换的任务必须用前者；含 MuJoCo 浮动基座状态读取与 POMDP 历史观测的实现要点 |

---

## `tasks/g1_rough/` — 粗糙地形行走

在六种程序化生成地形（金字塔楼梯 / 倒金字塔楼梯 / 随机方块 / 随机起伏 / 上坡 / 下坡）
上训练速度跟踪策略，10 级难度课程 × 20 列样本。

### 设计要点：零侵入挂载

策略配置不写进 `unitree_rl_lab` 的目录树，而是作为独立包通过 symlink 挂载：

```
tasks/g1_rough/  ──symlink──▶  unitree_rl_lab/.../g1/29dof/rough
```

Isaac Lab 的 `import_packages()` 会递归 import 所有**目录包**，因此挂载后任务自动注册，
**第三方仓库零改动**。两个前置假设在写代码前用最小实验验证过：

1. `pkgutil.iter_modules` 能否发现 symlink 目录 → 能，识别为 package
2. 相对导入能否穿过 `29dof` 这种**非法 Python 标识符**（数字开头）→ 能，
   相对导入做的是 `__package__` 字符串拼接，不走标识符校验

安装：

```bash
./tasks/g1_rough/install_overlay.sh
# 验证（纯 Python，无需启动 Isaac Sim）
python scripts/list_envs.py | grep Rough
```

训练：

```bash
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Rough --headless
```

### 已定位的问题

**世界系绝对高度在非平地上不成立。**

基线配置的终止条件与奖励项都基于机器人根节点的世界系 z 坐标：

```python
# isaaclab/envs/mdp/terminations.py
def root_height_below_minimum(...):
    """...
    Note:
        This is currently only supported for flat terrains,
        i.e. the minimum height is in the world frame.
    """
    return asset.data.root_pos_w[:, 2] < minimum_height
```

在下沉地形（倒金字塔楼梯、下坡）上地面本身低于 0，机器人姿态正常却被判定为"摔倒"。
16 环境冒烟测试中该项贡献 **25% 的 episode 终止**。

修法是改用高度扫描传感器测量**相对地形**的高度差，而非世界系绝对值。

---

## `sim2sim/` — 跨仿真器部署验证

把 Isaac Lab 训练出的策略搬进 MuJoCo 独立复现，用来暴露训练与部署之间的隐式约定。
必须逐项对齐的六件事，错任何一项机器人立刻摔倒：

| # | 对齐项 | 坑在哪 |
|---|---|---|
| ① | 关节顺序 | MuJoCo(SDK) 顺序 ≠ Isaac Lab(asset) 顺序，需要显式映射表 |
| ② | 默认姿态 | 网络输出是**相对默认姿态的偏移**，不是绝对角度 |
| ③ | 动作尺度 | `目标角 = 默认角 + scale × 网络输出` |
| ④ | PD 参数 | Isaac Lab 内置 PD，MuJoCo 侧要自己算力矩 |
| ⑤ | 观测顺序 | 拼接顺序、缩放系数、历史帧数逐项一致 |
| ⑥ | 控制频率 | `decimation × sim_dt = policy_dt` |

这六项没有任何一项会报错——错了只是行为不对，所以只能靠逐项核对配置文件来保证。

路径通过环境变量定位，无硬编码：

```bash
export ROXAN_ROOT=/path/to/workspace      # 仓库与资产的公共根
export RL_LAB_RUN_DIR=/path/to/run        # 训练输出目录（含 exported/policy.pt）
export MUJOCO_SCENE=/path/to/scene.xml    # MuJoCo 场景
python sim2sim/sim2sim_flat.py
```

---

## 出处声明

- `sim2sim/sim2sim_flat.py` 参考课程示例实现的结构独立编写（502 行 vs 322 行，
  完全相同的有效行占 15.4%，且均为受功能约束的样板行：变量赋值、函数签名、字典键）。
  课程原始代码不包含在本仓库中。
- `unitree_rl_lab`、`IsaacLab` 为第三方项目，本仓库不包含其代码，
  也未对其做任何修改（见上文零侵入设计）。
