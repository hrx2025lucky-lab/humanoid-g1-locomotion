# Humanoid Locomotion — 人形机器人强化学习运动控制

Unitree G1（29 DoF）的运控全栈实践：从地形行走、分层导航，到动作重定向、
对抗式模仿与跨仿真器部署。**11 个任务，106 次提交，每个结论都有可复现的验证。**

仿真栈：Isaac Sim 5.1 / Isaac Lab 2.3.2 / mjlab（MuJoCo Warp）/ MuJoCo 3.12 / rsl-rl 3.1.2

每个任务以**零侵入**方式挂进第三方训练框架，第三方仓库保持零改动，
`git log` 里只有自己写的代码。

## 几个代表性的结果

| | |
|---|---|
| **分层导航到达率 0% → 97.7%** | 定位到上游框架 `compute_group(update_history=False)` 导致低层观测的 5 帧历史退化为当前帧重复 5 次 —— 不改变张量形状、不报错、不产生 NaN，排查了十一轮 |
| **蒸馏对照结论从 21.3% 修正到 13.4%** | 首轮对照超参混杂，对齐后重跑发现那 21.3% 里有 7.9 个百分点来自超参而非蒸馏目标 |
| **一次录像推翻了「已收敛」的结论** | 实践 9 的 `mean_reward` 由负转正（−0.75 → 6.98），但真实跟踪误差在恶化（0.932 → 1.834）—— 根因是参考动作的关节列序是**广度优先**、而模型是深度优先，用正运动学交叉验证定位，修正后误差 **0.178 m → 0.0011 m**（163×），关节超限 11/29 → 1/29 |
| **对照官方评分细则的闭环审计** | 把 11 份作业 PDF 的评分点变成 111 条可复现断言，只查产物（代码实现、训练日志、checkpoint、验证脚本实跑结果），不查"文档里提没提" |

> 这些不是"跑通了 demo"，而是**发现并解释了本不该出现的行为**。
> 详细的排查过程都写在各实践文档里。

---

## 路线图

覆盖从环境搭建、MDP 设计、模仿学习到跨仿真器部署的完整技术链路：

| # | 主题 | 技术点 | 关键结果 | 状态 |
|---|---|---|---|---|
| 1 | 仿真环境搭建与基础验证 | Isaac Sim / Isaac Lab / MuJoCo 三栈打通 | 三套环境实测版本已核实 | ✅ |
| 2 | 粗糙地形行走 | 地形课程、高度扫描感知、奖励与终止项重设计 | track **2.236/3.0**（官方参考 2.263，达 98%） | ✅ |
| 3 | 动作空间与 Sim2Sim 部署 | 增量式动作空间、跨仿真器迁移 | 30 项断言全通过 | ✅ |
| 4 | 蹲姿行走策略 | 速度 + 骨盆高度的双指令 MDP | 三组消融：切断任一段误差劣化 **5 倍** | ✅ |
| 5 | 分层强化学习导航 | 高层导航策略 + 冻结低层运控策略 | 到达率 **97.7%**；难度扩展对照差距仅 0.64% | ✅ |
| 6 | 教师–学生蒸馏 | 全身运动跟踪、特权信息蒸馏 | KL 比 Action 跟踪误差低 **13.4%**（公平对照） | ✅ |
| 7 | 人体动作重定向 | SMPL-X 人体动捕 → G1 关节空间 | 端到端跑通，产物 16 项校验全过 | ✅ |
| 8 | AMP 拟人走跑 | 对抗式动作先验、判别器风格奖励 | 训练中 | 🚧 |
| 9 | 轨迹追踪训练 | BeyondMimic 自适应采样 | reward **1.88 → 8.34**（跑满 20000 轮） | ✅ |
| 10 | 人–物交互运动跟踪 | HOI + RayCaster 地形感知 | 三组 TODO 完成，审计 24/24 | 🚧 |
| 11 | 跑酷策略与 Sim2Sim 验证 | 深度图流水线 + 六项接口对齐 | 深度管线 15/15 验证通过 | 🚧 |

图例：✅ 完成　🚧 代码完成、训练/验收进行中

### 工程质量

对照课程官方评分细则做的闭环审计（`scripts/audit_against_rubric.py`，
把 11 份作业 PDF 的评分点变成可复现断言）：

```
通过 111 · 待改进 0 · 不合规 0
```

审计不查"文档里提没提"，只查产物：代码里的实现、训练日志、
checkpoint、验证脚本的实跑结果。

### 仿真栈

| 环境 | 仿真器 | 用于 |
|---|---|---|
| `envs/isaaclab` | IsaacSim 5.1.0 / IsaacLab 0.54.2 / MuJoCo 3.12.0 | 实践 1,2,3,5,8,9,10,11 |
| `hw4_mjlab/.venv` | MuJoCo 3.8.1 / **MuJoCo Warp** 3.9.0.1 | 实践 4 |
| `hw6_distill/.venv` | MuJoCo 3.6.0 / MuJoCo Warp 3.6.0 | 实践 6, 9 |
| `envs/gmr` | MuJoCo + mink（IK 求解） | 实践 7 |

> `mjwarp` 是 MuJoCo 的 GPU 并行版（**不是 MJX**），`mjlab` 是建在其上的 RL 框架。
> 详见 [`docs/mjlab与MuJoCoWarp说明.md`](docs/mjlab与MuJoCoWarp说明.md)。

## 已实现

| 目录 | 内容 | 对应主题 |
|---|---|---|
| [`tasks/g1_rough/`](tasks/g1_rough) | 粗糙地形行走：地形课程 + 高度扫描感知 + 奖励重设计 | 2 |
| [`sim2sim/`](sim2sim) | 跨仿真器部署与验证：量化评估、断言测试 | 1 · 2 · 3 |
| [`scripts/`](scripts) | 插件编译、训练收尾、消融对照、流水线编排 | 2 · 4 · 5 · 6 |

## 从哪看起

**想快速了解做了什么** → 上面的[路线图](#路线图)表格

**想看排查过程的深度**（推荐面试官看这几篇）：
- [`docs/实践5_分层强化学习导航.md`](docs/实践5_分层强化学习导航.md)
  §12~§18 —— 十一轮排除、oracle 测试切分问题域、根因是一个默认参数
- [`docs/实践9_自适应采样与轨迹跟踪.md`](docs/实践9_自适应采样与轨迹跟踪.md)
  附二 —— 从"效果差"查到"指数核饱和"，中间两轮结论都被自己推翻
- [`docs/实践2_实验报告.md`](docs/实践2_实验报告.md)
  §6 —— 所有指标都好看，但录像显示机器人在原地踏步

**想看工程方法**：
- [`scripts/verify_training_outcome.py`](scripts/verify_training_outcome.py)
  —— **按真实任务指标验收，不看 reward**。能自动区分"训练有 bug"与
  "设计权衡"：判据是奖励函数在当前误差量级上还有没有梯度
- [`scripts/check_stale_imports.py`](scripts/check_stale_imports.py)
  —— **不启动 IsaacSim 就查出"训练能跑、回放起不来"**。一晚上撞了三次
  同一处上游 API 迁移（模块从 `isaaclab.utils` 挪到 `isaaclab_rl.utils`），
  因为 `train.py` 不引用它，问题只在交付录像时才暴露。
  难点全在消除误报：查子模块会执行父包 `__init__`（需要 IsaacSim 的 carb），
  所以只对顶层包用 `find_spec`、子模块改查文件系统；
  还要补上启动脚本里 `export PYTHONPATH` 的运行时路径。
  用 5 组已知答案自检，做到零误报且不漏检
- [`scripts/audit_against_rubric.py`](scripts/audit_against_rubric.py)
  —— 把评分细则变成 111 条可复现断言
- [`scripts/check_deliverables.py`](scripts/check_deliverables.py)
  —— 交付材料清单。刻意防"假阳性"：冒烟存的 `model_3.pt` 不算训练成果、
  报错的日志不算跑通、"无数据"不等于"通过"
- [`scripts/assess_training_budget.py`](scripts/assess_training_budget.py)
  —— 判断训练量是否充分，判据是收敛而非轮数

**想复现**：
- [`docs/录像命令_按实践分列.md`](docs/录像命令_按实践分列.md)
  —— 每个实践的 play 录像命令，任务名与 checkpoint 路径都核实过
- [`docs/mjlab与MuJoCoWarp说明.md`](docs/mjlab与MuJoCoWarp说明.md)
  —— 三套仿真环境的区别与选择理由

> `teaching/` 目录下另有 11 份教学文档，是给自己复习用的原理讲解，
> 与这里的实践记录分工不同。

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
| [`docs/待解决卡点清单.md`](docs/待解决卡点清单.md) | **卡住时看这个** — 四个外部依赖卡点，每条写清「要什么 / 放哪里 / 怎么验证解决了」 |
| [`docs/00_实践总览.md`](docs/00_实践总览.md) | **先看这个** — 11 个实践各自在解决什么问题、如何串联成一条从"会走"到"会跳舞"的技术链，以及贯穿全课程的 5 条通用经验 |
| [`docs/实践2_实验报告.md`](docs/实践2_实验报告.md) | 粗糙地形行走的完整实验报告：地形/感知/判据/奖励四层改动、height_scan 的扫描区域与 187 维构成、观测 480→1415 维、10000 iter 最终结果、sim2sim 部署验证，以及"`--resume` 会丢失环境侧课程进度"这一发现 |
| [`docs/实践2_奖励调参记录.md`](docs/实践2_奖励调参记录.md) | 第一次训练学出"原地踏步"策略的完整定位过程：如何从 `Episode_Reward/*` 分项拆解识别局部最优，以及 10 项权重为什么这么改 |
| [`docs/实践3_HoST增量动作空间.md`](docs/实践3_HoST增量动作空间.md) | 增量动作空间 `q*=q_cur+αa` 与残差式 `q*=q_def+αa` 的本质差别，以及为什么接触状态频繁切换的任务必须用前者；含 MuJoCo 浮动基座状态读取与 POMDP 历史观测的实现要点 |
| [`docs/实践4_双指令MDP设计.md`](docs/实践4_双指令MDP设计.md) | 三组消融验证「采样→观测→奖励」三段闭环缺一不可（切断任一段误差劣化 5 倍）；一个自带校准物的指标 bug；速度+骨盆高度双指令 MDP：为什么"按指令做某事"必须打通采样→观测→奖励三段闭环；高度用世界系而速度用机体系的坐标系对照；非对称 Actor-Critic 中什么算特权信息 |
| [`docs/实践5_分层强化学习导航.md`](docs/实践5_分层强化学习导航.md) | **训练发散的完整定位过程**：三个隔离实验推翻三个合理假设，最后由「学习率已在下限」逼出真因——冻结低层在分布外吐出 1.7e6 污染 critic；分层 RL 的两层接口设计：低层 TorchScript 的输入契约为什么只能 deepcopy 不能手写；裁剪前后动作在三处的一致性；`apply_actions` 为何必须在 decimation 判断之外 |
| [`docs/实践6_教师学生蒸馏.md`](docs/实践6_教师学生蒸馏.md) | 2×2 对照（蒸馏目标 × 超参）给出可归因结论：KL 确实更优但只差 13.5%，首轮看到的 21.4% 里有 7.9 个百分点来自超参；教师-学生蒸馏：forward KL 的 mass-covering 与 reverse KL 的 mode-seeking 之别；为何冻结的 Teacher 仍需 `no_grad`；蒸馏系数退火对抗的信息不对称 |
| [`docs/实践7_运动重定向.md`](docs/实践7_运动重定向.md) | 重定向产物的 22 项验证：四元数 xyzw/wxyz 判定、关节限位、时序连续性、支撑相检测；以及为什么装 GMR 前要小心 mujoco 版本 |
| [`docs/实践11_跑酷与深度感知.md`](docs/实践11_跑酷与深度感知.md) | 深度图五步流水线：为什么必须最近邻插值、无效点 0 为何会被读成「紧贴镜头的墙」、模糊为何不能太强；15 项验证 |
| [`docs/实践9_自适应采样与轨迹跟踪.md`](docs/实践9_自适应采样与轨迹跟踪.md) | 自适应采样如何把训练算力集中到失败率高的动作片段；bin 宽度必须整数除法的原因；参考动作对齐为何只对 z 和 yaw |

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

### 验证工具

部署侧的正确性不能靠看录像 —— 实践 2 已经栽过一次：`episode_length` 与
`reward` 全线上涨，录像里机器人却在原地踏步。所以这里的每个工具都输出**数字**。

| 脚本 | 作用 |
|---|---|
| [`eval_rough_headless.py`](sim2sim/eval_rough_headless.py) | 粗糙地形策略的无头量化评估：观测契约核对 + 机体系指令跟踪 + 姿态存活判定。以课程 checkpoint 为对照基准 |
| [`verify_practice7_motion.py`](sim2sim/verify_practice7_motion.py) | 重定向动作数据的 22 项验证。产物会被实践 9/10/11 当参考轨迹，错了会一路传下去 |
| [`verify_practice11_depth.py`](sim2sim/verify_practice11_depth.py) | 深度图处理流水线的 15 项验证（resize/crop/inpaint/blur/normalize） |
| [`verify_practice3.py`](sim2sim/verify_practice3.py) | HoST 增量动作空间的 30 项断言。零动作检查一次证明"是增量式"且"不是残差式"；含 `mj_data.qpos` 视图/副本陷阱 |
| [`sim2sim_flat.py`](sim2sim/sim2sim_flat.py) | Isaac Lab → MuJoCo 的独立复现 |

```bash
python sim2sim/eval_rough_headless.py --run-dir /path/to/run
python sim2sim/verify_practice3.py --full
```

### raycaster 插件

粗糙地形的 `height_scanner` 依赖第三方 MuJoCo 插件，必须在 MuJoCo 源码树内编译，
且版本与 Python 运行时严格一致。插件与 MuJoCo 3.12 之间有三处 API 断裂
（`mjtnum.h` 改名、`mjthread.h` 与旧线程 API 删除、`mjPLUGIN_LIB_INIT` 变带参宏），
全流程固化在幂等脚本里：

```bash
./scripts/build_raycaster_plugin.sh          # 打补丁 → 编译 → 加载验证
./scripts/build_raycaster_plugin.sh verify   # 只验证已有产物
```

---

## `scripts/` — 训练编排

单个 4096 环境的训练会把 3090 打到 85~97% 利用率，并行只会互相拖慢，
因此所有训练串行排队：

```bash
./scripts/run_pipeline.sh        # 等 GPU 空闲 → 实践2收尾 → 实践4 → 实践5 → 实践6
ITERS=3000 ./scripts/run_pipeline.sh
```

| 脚本 | 作用 |
|---|---|
| `run_pipeline.sh` | 轮询 GPU 占用，按顺序编排下面这些步骤 |
| `finish_p2.sh` | 实践 2 收尾：指标提取（自动串联续训分段）/ 录像 / sim2sim 前置检查 |
| `record_play.sh` | 录满一整个 episode 并导出 `policy.pt` / `policy.onnx` |
| `run_p4_ablation.sh` | 实践 4 三组消融（单因素对照） |
| `run_p5p6_compare.sh` | 实践 5 / 6 的对照训练 |
| `compare_p4_ablation.py` | 实践 4 消融结果对比，含指标自校准与可证伪的理论预测 |
| `compare_p6_distill.py` | 实践 6 的 2×2 对照：纵向看超参效应、横向看方法差异，对齐组未跑完时拒绝下结论 |
| `compare_p5_navigation.py` | 实践 5 对照，把「表面指标」与「真实任务指标」分开列，内置「站着不动」自检 |
| `watch_p5.sh` | 实践 5 训练健康巡检，每 10 分钟一次 |
| `diagnose_p5_lowlevel.py` | 实践 5 分层控制的分层诊断：底层站立 → 接线验证 → NaN/量级溯源 |

---

## 出处声明

- `sim2sim/sim2sim_flat.py` 参考课程示例实现的结构独立编写（502 行 vs 322 行，
  完全相同的有效行占 15.4%，且均为受功能约束的样板行：变量赋值、函数签名、字典键）。
  课程原始代码不包含在本仓库中。
- `unitree_rl_lab`、`IsaacLab` 为第三方项目，本仓库不包含其代码，
  也未对其做任何修改（见上文零侵入设计）。
