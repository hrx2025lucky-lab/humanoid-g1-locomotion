# 实践 10 教学文档 · HOI 人-物交互跟踪（给盲眼策略装上"地形眼睛"）

> 面试复习稿。目标：把 RayCaster 高度扫描从"为什么需要"讲到"每一项数值的含义"，
> 重点讲透那个**同名但无关的两套 offset**，以及我自己踩的一个"数值算得很干净、
> 结论却是错的"的坑——这是本实践最好的面试素材。

> **训练状态（如实说明）**：本实践的**代码已全部补完**（3 个 TODO），
> 离线审计 **24/24 通过**（纯函数单测 + 配置校验）。但**尚未启动整机训练**——
> GPU 正被实践 8 的 AMP 正式训练占着（见实践 8 文档）。所以本文能给"实现正确性"
> 的证据，暂时给不了"训练收敛曲线"。文中不会把没跑的结果当成跑过的。

---

## 0. 一句话概括

原本的 HOI 跟踪任务是"盲人爬台阶"——策略只知道自己的关节状态和要模仿的动作，
**看不见脚下地形**。我们给它加一套向下打的"激光测距网格"（RayCaster），
在机器人躯干下方铺 17×17=289 条向下的射线测地面高度，把这张"高度图"当成观测，
策略于是能感知台阶、障碍，做出爬高、跨越这类人-物交互动作。

---

## 1. 问题从哪来（Why）

HOI = Human-Object Interaction，人和物体交互（比如爬箱子、跨越障碍）。原始的
`HOI_terrain` 任务是"盲跟踪"：策略观测里有本体状态（关节角/速度、姿态）和要模仿的
参考动作锚点，但**没有任何地形信息**。

**不加地形感知会怎样**：机器人只能靠"参考动作告诉我该抬多高腿"去硬爬。一旦地形
和参考动作对不齐（台阶高度变了、箱子位置偏了），它无从知道，一脚踩空或撞上去。
就像蒙着眼按记忆爬楼梯——楼梯一改就摔。

所以这个实践的任务是：**给盲眼策略装上地形眼睛**。作业把它拆成 3 个 TODO：
①地形 metadata 加载（20 分）、②RayCaster 传感器配置（20 分）、③策略+critic 观测接入（20 分）。

---

## 2. 核心概念（零基础可懂）

### RayCaster（射线投射测高度）

- **生活化类比**：拿一把**激光测距笔**从高处垂直向下照地面，读数就是"从笔到地面
  有多远"。在机器人躯干下方铺一整片这样的笔（网格排布），就得到一张"脚下地形深浅图"。
- **严格定义**：RayCaster 是一个传感器，从一组起点沿固定方向发射射线，计算射线与
  场景网格（mesh）的交点，返回命中点坐标。向下发射时，命中点的 z 就是该处地面高度。
- **在本实践中**：`HoiMergedTerrainRayCasterCfg`（`tracking_env_cfg.py:20`），
  挂在躯干、随偏航角对齐（`ray_alignment="yaw"`），网格向下打。

### 为什么用 RayCaster 而不是"直接读地形高度"

- **类比**：真机上没有"上帝视角的地形数据库"，只有传感器（深度相机/激光）实测。
  仿真里用 RayCaster 模拟这种"实测"，才能和真机对齐。
- 直接从仿真内部读高度场是"作弊"——真机上拿不到，训出来的策略没法部署。
  RayCaster 打在**烘焙好的地形 mesh** 上，模拟的是真实测距。

### GridPattern（网格扫描点）

- **类比**：在躯干正下方地面上画一张**渔网**，每个网格交点发一条向下的射线。
- **定义**：`GridPatternCfg(resolution=0.1, size=(1.6, 1.6))` 生成一片
  1.6m×1.6m、间隔 0.1m 的网格点。
- **点数怎么算**（§3 详解）：`1.6 / 0.1 = 16` 个间隔，**含两端边界**是 16+1=17 个点，
  于是 17×17 = **289** 条射线。这个 off-by-one 直接决定观测维度对不对。

### mesh bake（网格烘焙/合并）

- **类比**：射线要打在"实体墙"上才有回声。地形在物理引擎里可能是很多零散的碰撞体，
  RayCaster 需要一张**统一的三角网格**去求交点。"烘焙"就是把这些几何**预先合并成
  一张静态 mesh**，之后每帧快速求交，不用实时重算。
- **在本实践中**：`HoiMerged...` 里的 "Merged" 就是指把 HOI 场景里的地形箱体
  （由 metadata 描述）合并进一张可供射线求交的 mesh。TODO1 的 metadata 加载
  就是为这个合并提供箱体的位置/朝向/尺寸。

### 非对称 actor-critic（policy 加噪、critic 不加）

- **类比**：练兵时给**士兵（policy）**发的是**有雪花的夜视仪**（带噪观测），
  但**参谋（critic）**看的是**清晰卫星图**（无噪特权观测）。士兵练的是"图像模糊也能走"，
  参谋评估得准，训练更稳。
- **在本实践中**：policy 的 height_scan 加 `Unoise(-0.02, 0.02)`（`tracking_env_cfg.py:65`），
  critic 的 height_scan **不加噪**（`:97`，注释明写"这里若跟着加 Unoise，特权观测就白设了"）。
- **去掉这个非对称会怎样**：若 critic 也加噪，价值估计变噪、方差变大，训练更慢更抖；
  若 policy 不加噪，训出的策略只会用"完美地形图"，真机上传感器一有噪声就崩。

---

## 3. 算法原理

### 3.1 GridPattern 为什么是 289 而不是 256

```
size = 1.6 m,  resolution = 0.1 m
每边间隔数 = 1.6 / 0.1 = 16
每边点数   = 16 + 1 = 17     ← 关键：含首尾两个边界点
总点数     = 17 × 17 = 289
```

- **"+1"从哪来**：一段长 1.6m 的线以 0.1m 为步长采样，是"栅栏与栅栏柱"问题——
  16 个间隔对应 **17 根柱子**。漏掉这个 +1 就变成 16×16=256。
- **去掉 +1 会怎样**：256×8=2048，网络照建不报错，但每一维的空间含义悄悄错位，
  地形图和真实位置对不上，是**静默错误**。所以接完必须打印实际维度核对
  `289 × history 8 = 2312`。

### 3.2 height_scan 每一项的含义

IsaacLab 的 `mdp.height_scan`（`observations.py:300`）：

```
height_scan = pos_w.z − hit_z − offset
```

- `pos_w.z`：**传感器（躯干）在世界系的高度**。
- `hit_z`：这条射线**命中地面点的高度**。
- `pos_w.z − hit_z`：**躯干离该处地面的竖直距离**（躯干有多高）。
- `offset`：**高度基准**（这里 0.5）。减掉它，是把"离地高度"换算成"相对某基准的偏差"，
  让平地时读数落在 0 附近而不是一个大常数，网络更好学。
- **去掉 offset 会怎样**：所有读数整体平移一个常数，网络得自己学着减掉这个偏置，
  没坏但没必要。真正的作用是让不同地形的相对高差落在 clip 区间 `(-1, 5)` 的有效段内。

举例（修正后，offset=0.5）：

| 地形 | 躯干离地 pos_w.z−hit_z | height_scan | clip(-1,5) 后 |
|---|---|---|---|
| 平地 | 0.75 | 0.25 | 0.25 |
| 0.3m 台阶（脚下更高）| 0.45 | −0.05 | −0.05 |
| 0.6m 障碍 | 0.15 | −0.35 | −0.35 |

三种地形 clip 后**各不相同**，高度场信息保留下来了——这正是它该有的样子。
（为什么强调这点，见 §5 的坑。）

### 3.3 两套 offset：同名，但完全无关

这是本实践最容易混、也最值得在面试讲的点。代码里有两个都叫 `offset` 的东西：

| 名字 | 位置 | 是什么 | 值 |
|---|---|---|---|
| `RayCasterCfg.OffsetCfg(pos=...)` | 传感器配置 | **射线的几何起点**（射线从哪儿发出） | `(0,0,20)` |
| `ObsTerm(..., params={"offset":...})` | 观测项 | **观测的高度基准**（扫描高度减去它） | `0.5` |

- **前者 `(0,0,20)`**：把射线起点抬到躯干上方 20m 处，再垂直向下打。为什么抬这么高？
  减少射线起点被机器人自身几何"内嵌/自碰"、以及从太低处发射打不到远处地形的问题。
  它**只加到射线起点 `ray_starts`**（`ray_caster.py:224`：`self.ray_starts += offset_pos`），
  **不进入 `pos_w`**，所以不影响 height_scan 的数值。
- **后者 `0.5`**：是 §3.2 里那个高度基准，直接进 height_scan 公式。

**它们只是恰好都叫 offset，作用毫不相干。** 把两者搞混，就会推出"抬高 20m 会让
height_scan 全部被 clip 成 5、地形信息全丢"这种看似有理实则错误的结论——见 §5。

---

## 4. 代码走读

### ① RayCaster 传感器配置（TODO2，`tracking_env_cfg.py:20`）

```python
height_scanner = HoiMergedTerrainRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # 挂在躯干
    ray_alignment="yaw",                          # 只随偏航对齐（不随俯仰/翻滚）
    # 这个 offset 是射线的几何起点，与观测项的 offset=0.5 无关
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.6)),  # 289 点
)
```

- **`ray_alignment="yaw"`**：射线网格随机器人**朝向**转，但**不随身体前倾/侧倾转**。
  这样即使机器人爬坡身体前倾，扫描网格仍是水平铺在地面上测高度，读数才有意义。
  若跟着俯仰转，前倾时网格会歪向斜前方，测的不再是"正下方地形"。

### ② 策略观测项（TODO3 policy，`:61`）

```python
height_scanner = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
    noise=Unoise(n_min=-0.02, n_max=0.02),        # policy 加噪
    clip=(-1.0, 5.0),
    history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,  # = 8
)
```

- **观测维度** = 289 × 8 = **2312**。接完打印实际维度对一下，不等于 2312 就有错。
- `clip=(-1,5)`：把 height_scan 夹在 −1~5m。地形高差正常都在这个范围，clip 掉的是
  RayCaster 偶发的 miss（打空返回极端值）这类异常。

### ③ critic 观测项（TODO3 critic，`:97`）——特权、无噪

```python
height_scanner = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
    # 这里若跟着加 Unoise，特权观测就白设了
    clip=(-1.0, 5.0),
    history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
)
```

- 和 policy 项**唯一的区别就是没有 `noise=`**。高度语义、clip、history 全部一致，
  保证 critic 看到的是"同一张图的干净版"。

### ④ 地形 metadata 加载（TODO1，`hoi_height_scan.py:40`）

TODO1 是两个纯函数，为 mesh 合并提供箱体几何：

```python
def _normalize_box_record(entry, idx):
    # 1. 读 pos / quat / half_size
    # 2. 若只给 full_size，按 half = full/2 换算（整边长的一半）
    half_size = entry.get("half_size")
    if half_size is None:
        full_size = entry.get("full_size")
        if full_size is not None:
            half_size = [float(v) / 2.0 for v in full_size]
    # 3. 逐项校验长度：pos=3, quat=4(wxyz), half_size=3，错就带下标报错
    if len(quat) != 4:
        raise ValueError(f"box[{idx}]: quat must have length 4 (wxyz), got {len(quat)}")
    ...

def _load_boxes(metadata_file, device):
    # 读 JSON → 取 mjcf_boxes（必须是非空 list）→ 逐个 normalize → 堆成张量
    # 按 (metadata_file, device) 缓存，避免重复读盘
```

- **`mjcf_boxes` 必须非空**：`climb_00` 那份 metadata 的 `mjcf_boxes` 真的是空的，
  正好撞上"非空校验"。它不是默认地形（默认 `climb_15`），所以严格抛错是对的。
- **quat 保持 wxyz**：注意注释写 `quat must have length 4 (wxyz)`——这里**不转成 xyzw**，
  与实践 7 的 npz 要求（xyzw）正好相反。约定跟着各自的下游走，不能想当然统一。
- **缓存**：同一 metadata 第二次加载走缓存（实测二次命中 2.3µs），避免每次重置都读盘。

**TODO1 实测**（真实 metadata）：
```
✅ climb_15 的 2 个 box：pos(2,3) quat(2,4) half(2,3)  dtype=float32
✅ climb_00 的 mjcf_boxes 为空 → 正确抛 ValueError
✅ full_size[2,4,6] → half_size[1,2,3]
✅ pos 长度错 / quat 长度错 / 缺 half_size → 报错都带下标
```

---

## 5. 我踩过的坑（面试重点素材）

### 坑：把 `OffsetCfg(z=20)` 判成了与 `clip=(-1,5)` 冲突

这个坑的价值不在"错了"，而在"**数值算得干干净净，结论却是错的**"——面试很爱听
这种"如何识别自己看似严密的推理其实建立在错误前提上"的故事。

```
现象：写 TODO2 加了参考答案要求的 OffsetCfg(pos=(0,0,20)) 后，
      我怀疑它和作业要求的 clip=(-1,5) 冲突。
第一反应：推理链看着很实：
         height_scan = pos_w.z − hit_z − offset
         若 pos_w.z 含 +20 → 平地 20.25 / 台阶 19.95 / 障碍 19.65
         clip 到 (-1,5) 后 → 全部变成 5.00，三种地形完全一样，高度场信息全丢！
         数值一算三种地形 clip 后完全相同，看着像实锤。
为什么这么想是错的：错在前提——我以为 OffsetCfg 的 (0,0,20) 会加进 pos_w。
                    实际上它只加到射线起点 ray_starts，根本不进 height_scan 公式。
怎么找到真因：救命的不是重算数值，而是一个外部矛盾——
            OffsetCfg(pos=(0,0,20)) 是 IsaacLab 的标准惯例：
            h1/go2/g1 的 velocity_env_cfg、官方 interactive_scene_cfg 全这么写，
            且都配 clip=(-1,1)。如果我的推理成立，官方所有地形任务的
            height scan 都是废的——这不可能。于是回去读代码。
根因：我把两个东西搞混了：
      · ray_caster.py:243 的 self._offset —— 是「mesh 相对物理刚体的位姿修正」
        （来自 _obtain_trackable_prim_view），与配置项无关；
      · ray_caster.py:224 的 cfg.offset —— self.ray_starts += offset_pos，
        只抬高射线起点，不进 pos_w。
      我误把前者当成了后者，才推出"20 会进 pos_w"。
教训：① 数值算得再干净，也只是在验证「假设 A 成立时会怎样」，
        不能反过来证明 A 成立。链条对，前提错，照样满盘皆错。
      ② 遇到"我发现官方/参考答案错了"的时刻，先假设是自己读错了——
        这是第二次栽在"只读了声明没读实现"（上一次是实践 11 误判"必须另建 20GB 环境"）。
      ③ 同名变量是重灾区：cfg.offset 与 self._offset 名字像、语义天差地别，
        读代码要看它真正被赋值/使用的那一行，不能凭名字脑补。
```

修正后三种地形 height_scan 分别是 0.25 / −0.05 / −0.35，clip 后各不相同，信息完整。

---

## 6. 面试可能怎么问

**概念题**

1. *为什么给 HOI 策略加 RayCaster？直接读仿真地形高度不行吗？*
   直接读是作弊——真机上没有上帝视角地形库，只有传感器实测。RayCaster 打在烘焙的
   地形 mesh 上模拟实测，训出来的策略才能部署。盲跟踪一旦地形与参考动作不齐就踩空。

2. *GridPattern 为什么是 289 个点？*
   size 1.6 / resolution 0.1 = 16 个间隔，含首尾边界是 17 个点，17×17=289。
   这是"栅栏柱"问题，16 个间隔对应 17 根柱子。漏掉 +1 会变 256，静默错位。

3. *height_scan 公式每一项？offset 去掉会怎样？*
   `pos_w.z − hit_z − offset`：躯干高度 − 命中点高度 = 离地高度，再减 offset(基准)
   让平地读数落在 0 附近。去掉 offset 只是整体平移一个常数，网络得自己学掉这个偏置。

**深挖题**

4. *代码里有两个 offset，分别是什么？*（高频，本实践核心）
   `RayCasterCfg.OffsetCfg(pos=(0,0,20))` 是射线的**几何起点**，只加到 ray_starts，
   把射线抬到躯干上方 20m 垂直向下打，避免自碰/内嵌；`ObsTerm` 里的 `offset=0.5` 是
   **观测高度基准**，进 height_scan 公式。同名但完全无关，一个影响"从哪打射线"，
   一个影响"读数减多少"。

5. *（追问）你怎么确认 z=20 不会破坏 height_scan？*
   我一度误以为它会进 pos_w、把读数顶到被 clip 成 5。真因是 `cfg.offset` 只落在
   `ray_starts += offset_pos`（ray_caster.py:224），不进 pos_w；而我混淆的
   `self._offset`（:243）是 mesh 相对物理刚体的位姿修正，和配置无关。识破靠的是
   "官方所有任务都这么写却都没坏"这个外部矛盾。

6. *为什么 policy 的 height_scan 加噪、critic 不加？*
   非对称 actor-critic：policy 练"图像有噪也能走"（对齐真机传感器噪声），critic 用
   干净的特权观测把价值估得准。critic 也加噪会让价值方差变大、训练变慢；policy 不加噪
   则真机一有噪声就崩。

7. *TODO1 里 quat 为什么是 wxyz，不是实践 7 的 xyzw？*
   约定跟下游走。HOI 箱体的 mesh 合并按 wxyz 用；实践 7 的 npz 规范要求 xyzw。
   不能想当然统一，每个接口按它自己的契约来。

**状态题**

8. *这个训练跑出来了吗？*
   如实说：代码补完、离线审计 24/24（纯函数单测+配置校验）通过，但整机训练还没启动，
   GPU 正被实践 8 的 AMP 正式训练占用。所以我现在能给"实现正确"的证据，
   给不了收敛曲线——不会把没跑的当跑过的讲。

---

## 7. 延伸：这个技术在工业界怎么用

- **高度扫描是足式机器人感知地形的主流做法**。ANYmal、Unitree 等的
  perceptive locomotion 普遍用"机器人下方的高度网格"当地形观测，来源可以是
  仿真里的 RayCaster，也可以是真机上的深度相机/激光重建的高程图（elevation map）。
- **教师-学生蒸馏的天然搭配**：仿真里用 RayCaster 拿"干净特权高度图"训教师（就是这里
  critic 用的那种），真机上用带噪的重建高程图训学生——这正是实践 6 蒸馏的思路，
  本实践的非对称 actor-critic 是它的雏形。
- **mesh 烘焙 vs 实时高程图**：仿真里地形静态，可以一次烘焙成 mesh 反复求交，很快；
  真机上地形是相机实时重建的，噪声更大、还有盲区，这也是 policy 侧要加噪训练的原因——
  提前适应真机高程图的不完美。
- **在完整栈里的位置**：本实践给"动作跟踪"补上了"地形感知"，是从
  `盲跟踪 → 感知跟踪 → 真机部署`的中间一环。再往下就是实践 11 的深度感知 + sim2sim。
