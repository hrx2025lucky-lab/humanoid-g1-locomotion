# 实践 10 · 基于 PHP 的 HOI 人-物交互运动跟踪

> **状态（2026-09-06 21:5x 更新）**：代码包已下载，**三组 TODO 已全部实现**，
> 审计 24 项通过 · 0 不合规。剩下的是 smoke train 与正式训练（等 GPU）。
>
> 本文前半部分是下载前写的实施预案（把官方规格拆成可核对的清单），
> 后半部分 §七 记录实际实现时遇到的问题。

---

## 一、任务是什么

给 Unitree G1 29DoF 的 HOI（Human-Object Interaction）运动跟踪任务
**接上 RayCaster 地形感知**。

原本的 `Unitree-G1-29dof-Mimic-HOI_terrain` 是 **blind** 任务 ——
机器人靠本体感知做动作跟踪，看不见地形。本作业把 height scanner 接进去，
让 policy 和 critic 都能看到脚下 1.6×1.6 m 的高度图。

只需完成**三组 TODO**，环境、奖励、PPO 都是现成的。

## 二、三组 TODO 的逐条规格

官方分值 20 + 20 + 20 = 60，其余 40 分是 smoke train、可视化、训练回放、代码质量。

### TODO 1 · metadata loader（20 分）

**文件**：`source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/hoi_height_scan.py`

补两个函数。验收重点是「数据读取、字段校验、full_size 转换、shape、dtype、device 正确」。

#### `_normalize_box_record(entry, idx)`

| # | 要求 | 容易漏的点 |
|---|---|---|
| a | 从 entry 读 `pos` / `quat` / `half_size` | |
| b | 没有 `half_size` 但有 `full_size` 时，**每个边长除以 2** | 是 full→half，别搞反 |
| c | 校验 `pos` 长度为 3 | |
| d | 校验 `quat` 长度为 4，**顺序保持 wxyz** | 不要转成 xyzw（实践 7 那边才要转） |
| e | 校验 `half_size` 长度为 3 | |
| f | 数据非法抛 `ValueError`，**错误信息里必须带 box 下标 `idx`** | 这是明确的评分点，别只写"invalid box" |

返回 `pos`(3) / `quat`(4, wxyz) / `half_size`(3)。

#### `_load_boxes(metadata_file, device)`

| # | 要求 |
|---|---|
| a | **UTF-8** 打开 metadata_file 读 JSON |
| b | 读 `mjcf_boxes` 字段 |
| c | 确认它是**非空 list** |
| d | 每个 box 调 `_normalize_box_record()` |
| e | 转成 `torch.float32` tensor |
| f | tensor 必须落在**调用方传入的 device** |

返回三个 tensor：

```
box_pos_local   [num_boxes, 3]
box_quat_local  [num_boxes, 4]
box_half        [num_boxes, 3]
```

**性能约束**（官方明确写了）：JSON 不能在每个仿真 step 重复读。
RayCaster 只在 mesh bake 阶段调 loader，可按 `(metadata_file, device)` 做缓存，
但**不要把逐帧文件 I/O 引入 observation 计算**。

> `_build_grid_xy()`、`compute_hoi_height_scan()`、`hoi_height_scan()`
> **不在作业范围**，不要动。

### TODO 2 · RayCaster height scanner 配置（20 分）

**文件**：`.../g1_29dof/hoi_mimic_terrain_perceptive_raycast/tracking_env_cfg.py`

把 `RobotSceneCfg.height_scanner` 从 `None` 换成 `HoiMergedTerrainRayCasterCfg`：

| 配置项 | 值 | 官方给的理由 |
|---|---|---|
| `prim_path` | `{ENV_REGEX_NS}/Robot/torso_link` | 扫描器随 torso 移动 |
| `ray_alignment` | `"yaw"` | 网格保持水平，只跟随朝向 |
| `pattern_cfg` | `GridPatternCfg(resolution=0.1, size=[1.6, 1.6])` | **17×17 = 289 点** |
| `terrain_prim_path` | 每个环境中的 `HOI_Terrain` | 找到要烘焙的地形 prim |
| `metadata_file` | `blind_cfg.TERRAIN_META_FILE` | 与 blind task 共用同一份 metadata |
| `use_mjcf_boxes_mesh` | `True` | 用 mjcf_boxes 构建 mesh |
| `include_ground_plane` | `True` | 否则 box 之外的射线全 miss |
| `rebake_on_reset` | `False` | 地形位姿固定，不必每次 reset 重建 |
| **`offset`** | **`OffsetCfg(pos=(0.0, 0.0, 20.0))`** | **射线起点抬到 20 m 高处垂直向下测高，减少自碰、内嵌与 miss** |

> ⚠️ **`OffsetCfg(z=20)` 是参考答案强调了三次的点**（作业讲解第 57、93、113 行，
> 以及"作业优化与常见问题"第 1 条："与参考写法一致，起点更稳"）。
> 官方作业正文的配置表里没有它，只有讲解里提 —— 光看作业 PDF 会漏掉。
>
> **注意这里有两套 offset，别混**：
>
> | | 是什么 | 值 |
> |---|---|---|
> | `RayCasterCfg.offset` | 传感器的**几何起点**，射线从哪儿发出 | `pos=(0,0,20)` |
> | `ObsTerm(offset=...)` | **观测的高度基准**，扫描高度减去它再喂给网络 | `0.5` |
>
> 一个是物理位置，一个是数值平移，同名但完全无关。

**算一遍点数**：`size 1.6 / resolution 0.1 = 16`，`16 + 1 = 17` → 17×17 = 289。
边界含两端，所以是 17 不是 16 —— 这个 off-by-one 直接决定观测维度对不对。

### TODO 3 · observation 接入（20 分）

**同一个文件**。分别补 `ObservationsCfg.PolicyCfg.height_scanner`
和 `ObservationsCfg.PrivilegedCfg.height_scanner`。

两者共同：

```python
func           = mdp.height_scan
sensor_cfg     = SceneEntityCfg("height_scanner")
offset         = 0.5
clip           = (-1.0, 5.0)
history_length = blind_cfg.PROPRIO_HISTORY_LENGTH
```

差异 —— **policy 加噪，critic 不加**：

```python
# 只有 PolicyCfg
noise = Unoise(n_min=-0.02, n_max=0.02)
```

这是标准的非对称 actor-critic：critic 训练时可以用特权信息（无噪真值），
actor 必须在带噪观测下学会鲁棒，否则 sim2real 会垮。
实践 6 的教师-学生蒸馏是同一思路的另一种实现。

**观测维度自检**：参考答案直接给了确定值 —— `PROPRIO_HISTORY_LENGTH = 8`，
所以 **289 × 8 = 2312**。接完 TODO3 打出实际维度对一下，不等于 2312 就是哪里错了。

常见错法是 `size 1.6 / resolution 0.1 = 16` 就当 16×16=256，
漏了边界含两端应该是 17 —— 256×8=2048，差 264 维，网络照样能建起来不报错。

## 三、和已完成实践的复用关系

| 技术点 | 已有实现 | 能否直接搬 |
|---|---|---|
| RayCaster + GridPattern | 实践 2 `hw2_sim2sim`、实践 5 `height_scan_pooled` | 配置写法可参考，参数不同 |
| `mdp.height_scan` 的 offset/clip 语义 | 实践 5 用过 `offset=0.5` | 语义相同，直接对齐 |
| policy 加噪 / critic 不加噪 | 实践 6 教师-学生 | 思路相同 |
| 观测 history_length | **实践 5 的根因就在这**（§14） | ⚠️ 见下 |

> ⚠️ **实践 5 的教训必须带过来**：`history_length` 只是声明观测要存几帧，
> 真正让缓冲区滚动的是 `compute_group(..., update_history=True)`。
> 实践 5 因为漏传这个参数，5 帧历史退化成"当前帧重复 5 次"，
> 排查了十一轮才找到 —— 因为它不改变任何张量形状、不报错、不改变量级。
>
> 实践 10 走的是 IsaacLab 标准 `ManagerBasedRLEnv`（不是 HRL 那种自建低层
> obs manager），正常路径会传 `update_history=True`，**大概率不受影响**。
> 但接完 TODO 3 后值得花一分钟验证，方法见实践 5 文档 §14 的
> `verify_p5_history_frozen.py`：抓两帧观测，比较历史槽位是否真的在滚动。

## 四、验收顺序（官方建议，别跳步）

官方明确写了：「如果 blind 任务无法创建，优先检查 Isaac Lab 安装、任务注册、
数据路径和 rsl_rl，**不要先调试 RayCaster TODO**」。

```bash
cd "${PROJECT_ROOT}" && source set_project_root.sh

# ① 可选：blind 基线 smoke train —— 先证明基础环境是好的
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-HOI_terrain \
    --num_envs 64 --max_iterations 10 --headless --logger tensorboard

# ② 必做：感知任务 smoke train
#    验收：连续运行、无 NaN/Inf、无路径/shape 错误、无 NotImplementedError

# ③ RayCaster 可视化：截图命中点，检查网格是否随 yaw 转动、高度统计是否合理

# ④ 正式训练 → checkpoint → play 回放
```

这个顺序的价值在于**把"基础环境坏了"和"我的 TODO 写错了"分开**。
实践 5 排查时吃过亏：一开始没有隔离变量，把环境问题当成算法问题查了很久。

## 五、评分表（对照用）

| 项目 | 分值 | 验收重点 |
|---|---|---|
| TODO 1 metadata loader | 20 | 读取、字段校验、full_size 转换、shape/dtype/device |
| TODO 2 RayCaster 配置 | 20 | 挂载、扫描网格、地形 mesh、reset 策略 |
| TODO 3 observation 接入 | 20 | height 语义、history、noise、clipping |
| Smoke train 与数值检查 | 10 | 连续运行且无 NaN/Inf |
| RayCaster 可视化 | 10 | 扫描点、网格运动方式、高度统计合理 |
| 训练与回放 | 15 | 生成 checkpoint 并成功播放，tracking 表现合理 |
| 代码质量与改动范围 | 5 | 实现清晰、无无关改动 |

**提交物**：补全的 `hoi_height_scan.py`、补全的 `tracking_env_cfg.py`、
smoke train 终端输出、RayCaster 命中点截图/录屏、正式训练日志 + checkpoint 路径
+ 回放截图/视频。提交整个项目时还要附 `git diff --stat`。

## 五点五、参考答案额外提的（作业 PDF 里没有）

作业讲解（参考答案）除了三组 TODO，还给了一批只在讲解里出现的要求和建议。

### 提交时的六个注意事项

参考答案"作业优化与常见问题"逐条点名，都是**会丢分但容易忽略**的：

1. RayCaster 补 `OffsetCfg(z=20)`，与参考写法一致
2. `_load_boxes` 按 `(metadata_file, device)` 缓存 tensor
   —— 注意讲解特意澄清："**用 numpy 做校验 ≠ 会每 step 读盘；反复读盘是因为未缓存**"
3. 提交的必须是**感知版** `tracking_env_cfg`，别交错成 `HOI_Box` 的 cfg
4. smoke train 要交**完整训练终端日志**，不要只交 `[HEIGHT_SCAN]` 打印截图
5. 正式训练要交 `.pt` checkpoint + 回放视频/截图
6. play 异常时**先关 curriculum/DR 再查** —— 代码对齐不代表策略已收敛

第 6 条值得展开：实践 5 排查时吃过同类亏 —— 没隔离变量就查算法，
把环境/课程设置的问题当成实现 bug 追了很久。

### blind vs perceptive 消融（讲解建议的验收方式）

参考答案把它列为"把当前感知链路做扎实"的收尾动作：
**同一段 motion 下，比较 blind 与 perceptive 两版的跟踪误差与终止率**。

这比"感知版能跑起来"强得多 —— 后者只证明代码没崩，
前者才证明**height scan 真的被策略用上了**。

官方建议的 blind 基线命令本来只是用来排查环境问题的：

```bash
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-HOI_terrain \
    --num_envs 64 --max_iterations 10 --headless --logger tensorboard
```

把它跑满同样轮数，就成了对照组。**两组训练量必须对齐** ——
实践 6 栽过这个跟头（KL 组 1515 轮 vs action 组 2999 轮），
实践 5 的对照脚本现在会在 iter 相差超 20% 时拒绝给结论。

### 其它可选进阶

- 调扫描几何 `size` / `resolution`：更大视野 vs 更细网格的算力权衡
- 试不同 `history_length`：更长历史有助于看趋势，但增大观测与网络负担
- 关注 hit 分布：box 顶面 vs ground，**避免大面积 inf 污染梯度**
- 分析 height scan 时序：攀爬前后局部高度场怎么变、策略是否用上了
- 记录 motion error / anchor 与 ee 终止 / reward 分解，
  用来定位瓶颈在**感知**还是**跟踪**

### 这个作业在论文里的位置

对应论文 *Perceptive Humanoid Parkour — Chaining Dynamic Human Skills via Motion Matching*
中「**特权状态 + height scan 的 motion-tracking expert**」这一段。

完整链路是：

```
OmniRetarget → Motion Matching → height-scan expert → 深度图 student（DAgger/PPO）→ 真机
                                  ↑ 本作业在这里
```

打通后可继续向 student 蒸馏（正是实践 6 的技术）、技能组合与真机迁移推进
（实践 11 的深度图流水线就是 student 侧）。

---

## 七、实现记录（2026-09-06）

### 三组 TODO 都已完成

| TODO | 文件 | 状态 |
|---|---|---|
| 1 metadata loader | `mdp/hoi_height_scan.py` | ✅ `_normalize_box_record` + `_load_boxes` |
| 2 RayCaster 配置 | `.../perceptive_raycast/tracking_env_cfg.py` | ✅ 9 项配置齐全 |
| 3 观测接入 | 同上 | ✅ policy 加噪 / critic 不加 |

审计：`python3 scripts/audit_against_rubric.py --practice 10` → **24 项通过 · 0 不合规**。

### 踩的坑：把 `OffsetCfg(z=20)` 判成了与 `clip` 冲突

写 TODO2 时按参考答案加了 `OffsetCfg(pos=(0,0,20))`，随即怀疑它与
作业要求的 `clip=(-1.0, 5.0)` 冲突。推理链是这样的：

```
height_scan = pos_w[:,2] - ray_hits_w[...,2] - offset      # isaaclab observations.py:300
若 pos_w.z 含 +20  →  平地 20.25 / 台阶 19.95 / 障碍 19.65
clip 到 (-1,5) 后  →  全部变成 5.00，高度场信息完全丢失
```

数值一算，三种地形 clip 后完全相同，看着像是实锤。

**但结论是错的。** 转折点是发现 `OffsetCfg(pos=(0,0,20))` 是 IsaacLab
的标准惯例写法 —— h1、go2、g1 的 `velocity_env_cfg` 和官方
`interactive_scene_cfg` 全都这么写，且配的正是 `clip=(-1,1)`。
**如果我的推理成立，官方所有地形任务的 height scan 都是废的。**

回去读代码，找到了错处：

```python
# ray_caster.py:243  —— 我误以为这里在应用 cfg.offset
pos_w, quat_w = math_utils.combine_frame_transforms(
    pos_w, quat_w, self._offset[0][env_ids], self._offset[1][env_ids])
```

`self._offset` 不是 `cfg.offset`。它来自
`_obtain_trackable_prim_view()`，文档写得很清楚：
「the relative pose between the mesh and its corresponding physics prim」
—— 是 **mesh 相对物理刚体的位姿修正**，与配置项无关。

`cfg.offset` 真正的落点在第 224 行：

```python
offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
self.ray_starts += offset_pos          # ← 只加到射线起点
```

所以：**`cfg.offset` 只抬高射线起点，不进 `pos_w`**。
`height_scan` 的基准仍是 torso 高度，两者完全兼容。

修正后：

| 地形 | height_scan | clip 后 |
|---|---|---|
| 平地 | 0.25 | 0.25 |
| 0.3 m 台阶 | −0.05 | −0.05 |
| 0.6 m 障碍 | −0.35 | −0.35 |

> **这次的教训**：数值算得再干净，也只是在验证「假设 A 成立时会怎样」，
> 不能反过来证明 A 成立。真正救回来的是那个**外部矛盾**——
> "官方所有任务都这么写"与"这么写是错的"不可能同时为真。
>
> 遇到"我发现官方/参考答案错了"的时刻，先假设是自己读错了。
> 这次是第二次了：上一次是实践 11 判定"必须另建 20GB 环境"，
> 同样是只读了声明没读实现。

### TODO1 的实测

用真实 metadata 验证（`datasets/hoi_mimic_data/*.terrain.json`）：

```
✅ climb_15 的 2 个 box：pos(2,3) quat(2,4) half(2,3)  dtype=float32
✅ 缓存二次命中 2.3 µs
✅ climb_00 的 mjcf_boxes 为空 → 正确抛 ValueError
✅ full_size[2,4,6] → half_size[1,2,3]
✅ pos 长度错 / quat 长度错 / 缺 half_size → 报错都带下标 42
```

**`climb_00` 那份 metadata 的 `mjcf_boxes` 真的是空的**，正好撞上作业
要求的"非空校验"。它不是默认地形（默认是 `climb_15`），所以严格抛错是对的。

顺带一提，测这两个函数时不能直接 import 模块 —— 文件顶部
`import isaaclab.utils.math` 会连带拉 `pxr`，而 `pxr` 要 IsaacSim 运行时。
把那一行替换掉再 `exec` 就能单测纯函数部分。

### 还没做的

- [ ] smoke train（等 GPU，实践 5/11/8 排在前面）
- [ ] RayCaster 命中点可视化截图
- [ ] 正式训练 + checkpoint + 回放

---

## 六、下载后的第一步

```bash
cd "/home/limx/workspace/Roxan_warmup/motion control/humanoid_practice/g1_locomotion"
python3 scripts/check_downloads.py          # 确认放对位置
python3 scripts/audit_against_rubric.py --practice 10
```

审计脚本已经按上面的评分细则写好断言，会告诉你哪几条还没做到。

---

## 八、smoke 训练崩了：一个"写一个字典、读另一个字典"的 bug（2026-09-07）

### 现象

```
File ".../isaaclab/sensors/ray_caster/ray_caster.py", line 304, in _update_buffers_impl
    mesh=RayCaster.meshes[self.cfg.mesh_prim_paths[0]],
KeyError: '/World/ground'
```

`/World/ground` 正是我在 `tracking_env_cfg.py:44` 填的 `mesh_prim_paths`。
第一反应是"路径填错了"——**这个方向是错的**。

### 排除法：为什么不是路径填错

看基类的注册逻辑（`ray_caster.py:170-211`）：

```python
for mesh_prim_path in self.cfg.mesh_prim_paths:
    ...
    if mesh_prim is None or not mesh_prim.IsValid():
        raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")   # ← 路径错走这里
    ...
    RayCaster.meshes[mesh_prim_path] = wp_mesh
```

**路径填错会抛 `RuntimeError`，不是 `KeyError`。**
拿到 `KeyError` 说明：注册那段代码压根没往字典里写这个键。

### 真因：实例属性把类属性遮蔽了

本任务用的是子类 `HoiMergedTerrainRayCaster`，它把烘焙推迟到第一次更新时做，
于是重写了 `_initialize_warp_meshes`：

```python
# 修复前
def _initialize_warp_meshes(self):
    self.meshes = {}            # ← 问题在这一行
    self._hoi_mesh_baked = False
```

而基类的声明是（`ray_caster.py:60`）：

```python
meshes: ClassVar[dict[str, wp.Mesh]] = {}
```

`meshes` 是**类属性**。`self.meshes = {}` 不会清空类属性，
而是**新建一个同名实例属性把它遮蔽掉**。之后：

| 代码 | 实际操作的字典 |
|---|---|
| 子类 `self.meshes[key] = wp_mesh`（`:328` 烘焙结果） | **实例**字典 |
| 基类 `RayCaster.meshes[key]`（`:304` 读取） | **类**字典 |

写进一个字典，从另一个字典里取，必然 `KeyError`。
而且这个 bug **没有任何静态检查能发现**——两边语法都完全合法。

### 最小复现（不需要 IsaacSim，几行就能验证）

```python
from typing import ClassVar

class Base:
    meshes: ClassVar[dict] = {}
    def read(self, k): return Base.meshes[k]      # 模拟 ray_caster.py:304

class Old(Base):
    def init(self): self.meshes = {}              # 修复前
    def bake(self, k): self.meshes[k] = "mesh"

o = Old(); o.init(); o.bake("/World/ground")
o.read("/World/ground")     # KeyError: '/World/ground'
print(o.meshes is Base.meshes)   # False ← 两个不同的字典
```

实跑输出与线上报错逐字一致。**能用十行纯 Python 复现的问题，
就不要占着 GPU 反复试** —— 当时 GPU 正被实践 11 的训练占满，
等一轮验证要十几个小时。

### 修法

```python
def _initialize_warp_meshes(self):
    self.meshes = RayCaster.meshes   # 绑定同一个类字典（是引用，不是拷贝）
    self._hoi_mesh_baked = False

def ensure_mesh_baked(self) -> None:
    key = self.cfg.mesh_prim_paths[0]
    if not getattr(self, "_hoi_mesh_baked", False) or key not in self.meshes:
        self._bake_hoi_merged_mesh()
```

第二处是顺带加固：原来只看 `_hoi_mesh_baked` 这个布尔标志位，
但 `meshes` 是全局共享的类字典，基类 `__del__` 在最后一个 RayCaster 销毁时会
`RayCaster.meshes.clear()`（`ray_caster.py:428`）。清空之后标志位仍是 `True`
而字典已空，会再次 `KeyError`。所以判据要落在**字典里到底有没有**，
而不是"我以为我烘焙过了"。

> 可复用的教训：**在子类里给一个名字赋值前，先确认基类有没有同名的类属性。**
> Python 的赋值语义是"绑定名字"，不是"修改对象"；
> `self.x = {}` 和 `self.x[k] = v` 在有类属性时行为完全不同——
> 后者会穿透到类属性，前者会遮蔽它。

### 连带修掉的检查漏洞

这次 smoke 明明失败了，`check_deliverables.py` 却报：

```
✅ smoke train 终端日志    1 个，最新 p10_smoke.log
```

因为它只判断"日志文件存不存在"——**而失败的日志同样是个文件**。
于是"根本没跑起来"被报成了"材料齐全"。已改为要求日志里真的出现
`Learning iteration N/`，否则提取异常类型报出来：

```
❌ smoke train 终端日志    1 份日志都没跑起训练循环（p10_smoke.log→KeyError）
```

同一批还修了另外两个同类假阳性：

| 检查项 | 原来的判据 | 假阳性表现 | 现在的判据 |
|---|---|---|---|
| checkpoint | 有 `model_*.pt` 文件 | 冒烟存的 `model_3.pt` 也算数 | 解析文件名轮数，< 100 判为冒烟 |
| 消融实验 | 有 ≥3 个 `.pt` 文件 | 单次训练就能存出几十个 | 按 run 目录去重，要 3 次**独立**训练 |
| 效果验收 | 无 problems 即"全部通过" | 没训练过的实践也算通过 | "无数据"单列为**未验证**，不计入通过 |

> 这三个是同一种错误：**把"没有证据说明它坏"当成"有证据说明它好"。**
> 对交付检查来说，假阳性比漏报危险得多——漏报只是多看一眼，
> 假阳性会让人以为做完了而停止追查。

---

## 九、正式训练：4096 环境会 OOM（2026-09-07）

修好 RayCaster 后 smoke（64 环境）通过，正式训练用配置默认的 4096
却在**第 0 轮结束时**炸掉：

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 24.00 MiB. GPU 0 has a total capacity of 23.54 GiB
of which 33.56 MiB is free.
```

**为什么这个任务特别吃显存**：观测维度远超普通 locomotion。

| 观测组 | 维度 | 其中 height_scanner |
|---|---|---|
| policy | 3120 | **2312**（= 289 射线 × 8 帧历史） |
| critic | 3252 | 2312 |

rollout 缓冲区是 `num_steps_per_env × num_envs × obs_dim`：

```
24 × 4096 × (3120 + 3252) × 4 B ≈ 2.5 GB
```

PPO 每轮还要保存旧动作、旧价值、优势等，更新阶段峰值再翻几倍。

**注意"只差 24 MiB"这个细节**：它说明 4096 不是差得远，而是刚好越线。
这种情况更不能贴着上限跑——第 0 轮只是最低水位，PPO 更新阶段的峰值更高，
勉强跑起来也可能在第几百轮才崩，那时已经烧掉几小时。
所以默认降到 **2048**（显存对半），而不是试 3072 去榨那点余量。

实测 2048：跑到 320 轮，显存稳定 20.5 GB，**零 OOM**。

**脚本层面加了自动降档**（`run_p10_full.sh`）：

```bash
while [ "$rc" -ne 0 ] && grep -q "OutOfMemoryError" "$T_LOG" && [ "$NUM_ENVS" -gt 512 ]; do
    NUM_ENVS=$(( NUM_ENVS / 2 ))
    log "⚠️ CUDA OOM，降到 --num_envs $NUM_ENVS 重试"
    sleep 30    # 等上一个进程把显存真正还回来
    run_train "$NUM_ENVS"
    rc=$?
done
```

显存占用主要由 `num_envs` 线性决定，减半基本能过；退到 512 就停，
再小没有训练意义。这样一次 OOM 不会浪费整夜。

---

## 十、回放录像：两处上游 API 变更（2026-09-07）

实践 5 和实践 8 的视频一直录不出来，**但训练都是正常的**。
原因是两个 `play.py` 各自撞上一处上游 API 变更，而 `train.py` 不引用它们，
所以问题只在回放路径上暴露：

| 实践 | 报错 | 原因 | 修法 |
|---|---|---|---|
| 8 | `ModuleNotFoundError: No module named 'isaaclab.utils.pretrained_checkpoint'` | 模块挪到了 `isaaclab_rl.utils` | try 新路径，except 回退旧路径 |
| 5 | `ImportError: cannot import name 'handle_deprecated_rsl_rl_cfg'` | 新版删除了这个兼容垫片 | 可选导入，缺失时退化成 no-op |

第二个能安全退化，是因为那个垫片本来就只做"把过时字段翻译成新字段"，
而我们的 cfg 本身就是新格式，对它来说是恒等变换。

> 教训：**训练能跑不代表回放能跑。** 两条代码路径的依赖并不相同，
> 交付材料里的视频要单独验证，不能因为训练正常就默认回放没问题。

### 顺带修掉的第三个问题：play 录完不退出

各框架的 `play.py` 录完视频后都会进入交互式 viewer
（mjlab 是 `play.py:227` 的 `NativeMujocoViewer.run()`），**永不自行退出**。
实测视频 8 分钟就落盘了，进程却一直挂到 1 小时超时才被杀——
三段消融于是从 25 分钟变成 3 小时。

解法是在录像脚本里加看门狗 `play_until_video()`：轮询目标目录的 mp4，
**文件大小连续两次不变**就判定写完并结束进程。

判据用"大小不再变化"而不是"文件出现"很关键：mp4 是边写边落盘的，
刚出现时还在写，此时杀掉只会得到一个截断的坏文件。
