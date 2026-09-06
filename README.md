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
| 2 | 粗糙地形行走 | 地形课程、高度扫描感知、奖励与终止项重设计 | ✅ |
| 3 | 动作空间与 Sim2Sim 部署 | 动作空间设计、跨仿真器迁移 | ✅ |
| 4 | 蹲姿行走策略 | 速度 + 骨盆高度的 MDP 设计 | ✅ |
| 5 | 分层强化学习导航 | 高层导航策略 + 底层运控策略 | 🚧 |
| 6 | 教师–学生蒸馏 | 全身运动跟踪、特权信息蒸馏 | ✅ |
| 7 | 人体动作重定向 | 人体动捕 → G1 关节空间 | 🚧 |
| 8 | AMP 拟人走跑 | 对抗式动作先验 | ⬜ |
| 9 | 轨迹追踪训练 | 运动跟踪关键函数实现 | ✅ |
| 10 | 人–物交互运动跟踪 | HOI | ⬜ |
| 11 | 跑酷策略与 Sim2Sim 验证 | 高动态动作 + 部署验证 | 🚧 |

图例：✅ 完成　🚧 代码完成、训练/验收进行中　⬜ 未开始

### 已确认的外部卡点

| # | 卡在哪 | 需要的人工动作 |
|---|---|---|
| 7 | 缺三样：SMPL-X 模型、ACCAD 动作数据、GMR 运行环境 | 下载两个网盘包（见 `docs/下载包放置指引.md`）；GMR 环境已在后台安装 |
| 8 | 只缺 1 条跑步类专家动作 | 等实践 7 重定向 1 条 `Running` 动作即可 |
| 10 | 代码包未下载 | 下载 `pan.baidu.com/s/1fbSggWFaxc_mL-ZXyYMANg` 提取码 `nk8r` → `shenlan_hw/HOI_Mimic/` |

> 实践 11 曾被判为"需另建 Python 3.12 环境"，2026-09-06 实测推翻：
> 那是 setup.py 的过度声明，现有 `envs/isaaclab` 直接可用，只补装了
> `pytorch_kinematics`。详见 `docs/待解决卡点清单.md` §②。

## 已实现

| 目录 | 内容 | 对应主题 |
|---|---|---|
| [`tasks/g1_rough/`](tasks/g1_rough) | 粗糙地形行走：地形课程 + 高度扫描感知 + 奖励重设计 | 2 |
| [`sim2sim/`](sim2sim) | 跨仿真器部署与验证：量化评估、断言测试 | 1 · 2 · 3 |
| [`scripts/`](scripts) | 插件编译、训练收尾、消融对照、流水线编排 | 2 · 4 · 5 · 6 |

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
