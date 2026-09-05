# mjlab 与 MuJoCo Warp 是什么

> 实践 4、6、9 用的是这套栈，实践 1、2、3、5、7、8、10、11 用 Isaac Sim/Lab。
> 本文的每条结论都来自本机代码实测，不是转述宣传材料。

## 一句话

- **MuJoCo Warp（mjwarp）** = MuJoCo 物理引擎的 **GPU 并行版**
- **mjlab** = 建在 mjwarp 之上的 **RL 环境框架**，架构照搬 Isaac Lab 的 Manager 模式

用一个类比：

| | 物理引擎 | RL 框架 |
|---|---|---|
| NVIDIA 栈 | PhysX（在 Isaac Sim 里） | **Isaac Lab** |
| MuJoCo 栈 | **MuJoCo Warp** | **mjlab** |

两栈解决的是同一个问题：让几千个机器人同时在 GPU 上跑物理，供 RL 采样。

## MuJoCo Warp

普通 MuJoCo（`import mujoco`）是 **CPU 单实例**的：一次算一个机器人。
训练 RL 需要几千个并行环境，CPU 版会成为瓶颈。

MuJoCo Warp 用 NVIDIA 的 **Warp**（一个把 Python 函数编译成 CUDA kernel 的库）
把 MuJoCo 的求解器搬到了 GPU 上。本机实测的调用方式：

```python
# mjlab/sim/sim.py
import mujoco_warp as mjwarp

self._wp_model = mjwarp.put_model(self._mj_model)   # CPU 模型 → GPU
mjwarp.step(self.wp_model, self.wp_data)            # 一次 step 推进所有环境
```

`put_model` 这个名字来自 JAX 的 `device_put` 传统 —— 把数据搬到加速器上。

**它不是 MJX。** MJX 是 MuJoCo 的 JAX 后端，走的是 XLA 编译；
mjwarp 走的是 Warp/CUDA。本机 jax 完全没装，实践 4/6/9 照跑不误。
这一点我最初判断错过，实测后更正。

## mjlab

mjlab 把 mjwarp 包装成 RL 环境框架。它的 Manager 列表和 Isaac Lab **几乎一一对应**：

```
mjlab/managers/          IsaacLab 的对应物
  action_manager.py      ActionManager
  command_manager.py     CommandManager
  curriculum_manager.py  CurriculumManager
  event_manager.py       EventManager
  observation_manager.py ObservationManager
  metrics_manager.py     （IsaacLab 把 metrics 放在 CommandManager 里）
```

所以从 Isaac Lab 迁到 mjlab，**心智模型几乎不用换**：
一样是「写配置类 → Manager 组装环境 → RL 库训练」。

区别在细节：

| | Isaac Lab | mjlab |
|---|---|---|
| 物理 | PhysX | MuJoCo Warp |
| 资产格式 | USD（可从 URDF 转） | MJCF（`.xml`） |
| 启动 | 必须先 `AppLauncher` 起 Isaac Sim | 直接 import，无独立仿真器进程 |
| 渲染 | Omniverse RTX | MuJoCo 自带 viewer |
| 启动耗时 | 3~5 分钟 | 秒级 |
| 配置写法 | `@configclass` + dataclass | 同左 |

**启动速度是最直观的差异**：Isaac Lab 每次跑脚本都要等 Isaac Sim 起来
（本轮多次因此吃亏，见 `实践5_分层强化学习导航.md` §13.1），
mjlab 几秒就开跑，调试循环快得多。

## 为什么课程要用两套

不是重复劳动，是刻意的：

1. **接触主流的两条技术路线**。工业界 Isaac 系与学术界 MuJoCo 系都在用，
   两边的资产格式、API 风格、调试手段都不同。
2. **sim2sim 需要第二个仿真器**。策略在 Isaac Sim 里训好，
   拿到 MuJoCo 里还能走，才说明学到的是控制策略而非某个求解器的数值特性。
   这是实践 2、3、11 的核心验收点。
3. **不同任务适配不同引擎**。运动跟踪类任务（实践 6、9）需要频繁重置到
   参考动作的任意帧，mjlab 在这方面更轻便。

## 本机三个环境的实测版本

**三套环境互相隔离，版本各不相同**——这不是疏忽，是必须：

| 环境 | MuJoCo | MuJoCo Warp | torch | 用于 |
|---|---|---|---|---|
| `envs/isaaclab` | **3.12.0** | 1.15.0 (warp-lang) | 2.7.0 | 1,2,3,5,7,8,10,11 |
| `hw4_mjlab/.venv` | 3.8.1 | **3.9.0.1** | 2.9.0 | 4 |
| `hw6_distill/.venv` | 3.6.0 | **3.6.0** | 2.10.0 | 6, 9 |

三个 torch 版本（2.7 / 2.9 / 2.10）互不兼容，这正是必须用独立 venv 的原因。
也是实践 11 卡住的根源：`instinct_rl` 要 torch 2.11 + numpy≥2，
装进 `envs/isaaclab` 会毁掉实践 1~10。

> **`envs/isaaclab` 的 mujoco 锁死在 3.12.0**：实践 2 的 raycaster 插件是按它
> 编译的，升级立刻 ABI 不兼容（`plugin mujoco.sensor.ray_caster not found`）。
> 装任何新包前先 `pip install --dry-run` 确认不会带动它。

## 一个容易混淆的点

`warp-lang` 和 `mujoco_warp` 是两个包：

- **warp-lang**：NVIDIA 的通用 GPU 编程库（Python → CUDA kernel）。
  Isaac Lab 也装了它（1.15.0），用来写自定义 kernel，与 MuJoCo 无关。
- **mujoco_warp**：MuJoCo 官方基于 warp-lang 实现的 GPU 物理后端。

所以 `envs/isaaclab` 里有 `warp-lang` **不代表**它能跑 mjlab —— 它没装 `mujoco_warp`。
