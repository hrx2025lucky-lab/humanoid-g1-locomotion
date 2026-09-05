# 实践 8 — 基于 AMP 的拟人走跑策略复现

> 代码：`shenlan_hw/unitree_lab_amp`（作业 §3.1 给的 `github.com/HeYee03/unitree_lab_amp`）
> 环境：复用 `envs/isaaclab`（作业 §3.3 方案一），**不需要新建 Python 环境**
> 验证：`sim2sim/verify_practice8_amp.py` — 44/44 通过

---

## 1. AMP 在解决什么问题

纯速度跟踪任务只关心"有没有达到目标速度"，不关心"怎么达到的"。
于是策略会找出各种不像人的捷径：僵硬迈步、不摆臂、高速滑步。
这些在奖励函数里都不违规，因为奖励压根没描述"像不像人"。

AMP 的做法是不去手写"像人"的奖励，而是**让一个判别器去学**：

```
GMR 重定向的专家动作 ──┐
                      ├──→ 判别器 D ──→ 风格奖励 r_amp
策略自己产生的动作 ────┘

最终奖励 r = (1-α)·r_amp + α·r_task
```

判别器看的不是单帧，而是一段长度 H=3 的运动窗口 —— 单帧分不出"走"和"跑"，
必须有时间维度才能描述步态。

## 2. 九个 TODO 与实现要点

### TODO1 — AMP 单帧观测（80 维）

```
base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + base_height(1)
+ joint_pos(29) + joint_vel(29) + key_links_pos_b(4×3) = 80
判别器窗口 = 3 × 80 = 240
```

**最容易埋雷的一点**：顺序必须与 `motion_dataset.py` 顶部的 `AMP_FEATURE_ORDER` 逐项对齐。
顺序错位不会报任何错，判别器照样训练、loss 照样下降 ——
它只会学到"第 i 维在两个分布里含义不同"这种**伪差异**，
风格奖励因此变成噪声。这类故障没有报错信号，只能靠契约检查提前拦住。

全部用**基座坐标系**的量：专家数据来自 GMR 重定向，其世界系位置/朝向
与仿真里的机器人无关，只有基座系特征可比。

关键连杆取双脚踝 + 双腕（`G1_AMP_KEY_LINK_NAMES`）—— 这四个点决定了
步态和摆臂的观感。

`enable_corruption = False`：若策略侧加噪而专家侧无噪，
判别器可以直接靠"有没有噪声"区分两者，风格奖励退化成噪声检测器。

### TODO2 — 观测分组

```python
obs_groups = {"policy": ["policy"], "critic": ["critic"], "amp": ["amp"]}
```

三路读的是不同东西：policy 带噪含指令、critic 可含特权信息、amp 是无噪风格特征。
把 amp 混进 actor 等于把判别器输入直接泄给策略。
`amp` 键是必需的，`resolve_obs_groups(..., ["amp"])` 会强制校验。

### TODO3 — 风格奖励

```
r_amp = Δt · β · q(D(x))
q(s)  = clamp(1 - 0.25(s-1)², 0, 1)      # LSQ 模式
```

**为什么要乘 Δt**：判别器给的是"每秒的风格收益"，而 PPO 累加的是每步奖励。
不乘的话，同一策略在不同控制频率下拿到的风格奖励总量不同，
风格项与任务项的相对强度会随 decimation 悄悄漂移。

**为什么用有界的 q 而不是直接用 logit**：logit 无界，判别器一旦自信起来
风格奖励就会盖过任务奖励，策略变成"只管像人、不管走到哪"。

### TODO4 — 奖励混合

```
r = (1 - α)·r_amp + α·r_task
```

实现上避免原地运算：`style_rewards` 来自 `no_grad` 分支、`task_rewards`
是环境返回的 buffer，两者调用方都还要用于日志统计。

### TODO5 — 环境步

历史不足 `history_steps` 帧的环境必须**退回纯任务奖励**。
否则 episode 刚重置的头几步会被灌进一个"风格极差"的假信号 ——
那几步的窗口里混着上一条轨迹的残留帧。

### TODO6 — PPO 总 loss

```
L = L_policy + c_v·L_value − c_e·H(π)
```

熵项前的**负号是关键**：优化器最小化 loss，而我们希望最大化熵。
写成 `+` 会让策略迅速塌缩成确定性动作 —— 在 AMP 里表现为步态僵死、
判别器轻易识破。（这正是实践 5 里"探索塌缩"的同一类失效。）

### TODO7 — WalkToRun Runner

```python
experiment_name = "unitree_g1_29dof_amp_walk_to_run"   # 独立目录，否则 checkpoint 互相覆盖
amp_motion_profile = "walk_to_run"                      # 必须与环境侧、数据侧一致
task_reward_weight = 0.5                                # α
```

α 取 0.5 的理由：比 Walk 的 0.6 低（走跑切换更依赖专家示范的步态转换），
比基类的 0.4 高（必须真跟上高速指令才谈得上"跑"）。

### TODO8 — FullPlay 评估环境

```
lin_vel_x = (-0.7, 2.5)    # Walk(-0.7,1.0) ∪ Run(1.5,2.5)
ang_vel_z = (-0.4, 0.4)    # 左右转弯
```

普通 Play 的 `lin_vel_x` 只到 1.2，那是**训练时的指令范围**。
验收要看的是走↔跑切换，指令必须同时穿过低速行走区和高速奔跑区。
取并集后是连续区间，采样会自然经过 1.0~1.5 这段"既不算走也不算跑"的
过渡带 —— 切换自不自然正是在这一段看出来的。

关课程：课程会随训练进度改指令范围，评估必须用固定范围，
否则不同 checkpoint 之间不可比。

### TODO9 — 注册 FullPlay 任务

`Unitree-G1-29dof-AMP-WalkToRun-FullPlay`：
train cfg 仍用 WalkToRun 的训练配置（保证网络结构与 checkpoint 一致），
只把 play env cfg 换成 FullPlay，runner 沿用同一个以便加载同实验目录的 checkpoint。

---

## 3. 两处环境适配

### G1 资产：USD → URDF

课程默认 `~/unitree_rl_lab-main/unitree_model/.../g1_29dof_rev_1_0.usd`，
本机不存在（全盘无 G1 的 USD）。`unitree_ros` 里有官方 URDF，
IsaacLab 会在首次加载时自动转换。实践 2/5 的主仓库同样走 URDF，
两边一致才能复用低层策略与关节顺序约定。

> 这是我在实践 5 踩过的同一个坑（硬编码资产路径），第二次遇到时直接照搬了修法。

### 专家数据缺失开关

`UNITREE_AMP_ALLOW_MISSING_CLIPS=1`（默认关闭，仍严格报错）。
缺一段专家数据会悄悄改变风格分布，训练出的步态与作业要求不一致却无任何提示，
所以默认必须硬报错；只在片段不全时用于冒烟测试。

---

## 4. 验证结果

### 离线验证 44/44

`sim2sim/verify_practice8_amp.py`，全部是可解析验证 —— 给定输入能手算出唯一答案：

| 组 | 内容 |
|---|---|
| 特征维度契约 | 80 维、240 窗口、数据侧与 env 侧顺序一致、无噪、单向量 |
| 风格奖励 | q(score) 逐点手算、r=dt·β·q、非负有界、dt 加倍则奖励加倍 |
| 奖励混合 | α=0 纯风格、α=1 纯任务、α=0.4 手算、不原地改写、单调性 |
| PPO loss | 熵项符号为负、公式手算 = 2.499 |
| 环境步 | 掩码退回、传给 PPO、episode 结束重置历史 |
| 配置注册 | obs_groups 三路、独立实验名、名称不重复、速度区间覆盖走与跑 |

端点值 α=0/1 是强约束：α=1 时风格奖励必须**完全不起作用**，
这一条能直接抓出混合方向写反的错误。

### 实机冒烟

```
RESULT group amp: (80,)      ← 与作业要求 3+3+3+1+29+29+4×3 吻合
RESULT amp tensor: (4, 80)   全部有限值
FullPlay 任务注册成功，lin_vel_x=(-0.7, 2.5) wz=(-0.4, 0.4)
```

6 轮训练（512 环境）跑通，指标全部合理：

| 指标 | 实测 | 判断依据 |
|---|---|---|
| `style reward/step` | 0.0287 | 理论上界 dt·β = 0.02×5 = 0.1，落在 [0, 0.1] ✅ |
| `amp_policy_score` | **−0.7036** | 策略样本判为"假" |
| `amp_expert_score` | **+0.7491** | 专家样本判为"真" |
| `discriminator loss` | 0.1812 | 在学，未饱和 |
| `style_task_ratio` | 1.7347 | 风格与任务贡献同量级 |

**policy_score 与 expert_score 被推向相反方向**，是整条链路正确的决定性证据：
这只有在 AMP 观测与专家数据同构、且风格奖励接线正确时才会发生。
若 80 维特征顺序错位，两个 score 会纠缠在一起分不开。

---

## 5. 数据链路：实践 7 → 实践 8

作业开头那张流程图的第一步"GMR 重定向运动数据"就是**实践 7 的产物**。
实测确认了这条链路：

| 来源 | 格式 | 能否直接用 |
|---|---|---|
| AMP 自带 `B1_-_stand_to_walk_stageii.npz` | `dof_pos / fps / link_body_list / local_body_pos / root_pos / root_rot` | 基准 |
| **实践 7 Project7 的 `B12_-_walk_turn_right_(90)_stageii.npz`** | **完全同构**（6 键 / 38 连杆） | ✅ 文件名也与 `motion_cfg` 要求逐字一致 |
| HW6 的 `accad_*.npz` | `fps / joint_pos / root_pos / root_quat_w` | ❌ 缺 `local_body_pos` 与 `link_body_list`，算不出 `key_links_pos_b` |

**结论**：完整训练需要 ACCAD 的 B/C 系列走跑片段（`walk_to_run` profile 要 10 段），
这些必须由实践 7 的 GMR 重定向产出。目前手上只有 2 段兼容片段，
够冒烟测试但不够正式训练。

---

## 6. 当前状态与下一步

- [x] 9 个 TODO 全部实现
- [x] 44/44 离线验证
- [x] 实机冒烟：环境构建、任务注册、判别器行为、训练循环全部跑通
- [ ] 正式训练 —— **卡在专家数据**，需先完成实践 7 的 GMR 重定向（缺 SMPL-X 模型）

训练命令（数据补齐后）：

```bash
cd ~/workspace/Roxan_warmup/shenlan_hw/unitree_lab_amp
PYTHONPATH="$PWD/source/unitree_rl_lab:$PWD" \
~/workspace/Roxan_warmup/envs/isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-AMP-WalkToRun --num_envs 4096 --headless

# 完整能力评估（走 + 跑 + 转弯）
PYTHONPATH="$PWD/source/unitree_rl_lab:$PWD" \
~/workspace/Roxan_warmup/envs/isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-AMP-WalkToRun-FullPlay
```

---

## 7. 官方验收标准逐条对照（2026-09-05）

作业 §7.5 列了 8 个必须验证的场景。当前代码与验证已就绪，
但**正式训练缺专家数据**（需实践 7 的 GMR 产物），
所以这里标注的是"能否验证"而非"已验证"。

| # | 官方验收内容 | 支撑手段 | 状态 |
|---|---|---|---|
| 1 | 低速前进能稳定行走，不明显滑步 | `foot_slip` 惩罚 −0.1 + FullPlay 低速段 | 待训练 |
| 2 | 中速前进步频步幅随速度增加 | FullPlay `lin_vel_x` 连续区间 (−0.7, 2.5) | 待训练 |
| 3 | 高速能从走路过渡到跑步 | FullPlay 覆盖 1.0~1.5 过渡带 | 待训练 |
| 4 | 减速能从跑步回到走路或站立 | 同上（区间连续，双向可采样） | 待训练 |
| 5 | 左转跟随正向角速度 | FullPlay `ang_vel_z` (−0.4, 0.4) | 待训练 |
| 6 | 右转跟随负向角速度 | 同上 | 待训练 |
| 7 | 长时间运行不频繁摔倒/关节发散 | 训练冒烟 6 轮无异常；判别器分数未饱和 | 部分 |
| 8 | 有无 AMP 对比：动作更接近专家 | `task_reward_weight` α 可调（0=纯风格，1=纯任务） | 可做 |

**第 8 条的对照设计**：α=1.0 时 `mix_rewards` 退化为纯任务奖励，
等价于"关掉 AMP"。所以只要跑两组（α=0.5 vs α=1.0），
就能得到干净的"有无 AMP"对照，不需要改代码。
这一点在验证脚本里已用端点值测过：α=1 时风格奖励**完全不起作用**。

### 官方还要求观察的细节（§7.5 末尾）

> 重点不只是"机器人有没有移动"，还应观察：脚是否明显打滑……

这与实践 5 的教训一致：**表面指标（有没有移动、reward 多少）
不等于任务质量**。AMP 特有的观察点：

| 观察项 | 对应指标 | 健康范围 |
|---|---|---|
| 风格奖励是否有效 | `style reward/step` | (0, dt·β] = (0, 0.1] |
| 判别器是否还在学 | `amp_discriminator loss` | 不为 0、不发散 |
| 判别器是否饱和 | `policy_score` vs `expert_score` | 两者分开但都不极端 |
| 风格与任务是否失衡 | `amp_style_task_ratio` | 同量级（~1） |

冒烟测试实测：`policy_score −0.7036` / `expert_score +0.7491` —— 
判别器把两个分布推向相反方向，且都未饱和，这是最健康的状态。
若两者绝对值都 >5，说明判别器赢得太彻底、风格梯度消失，
应降低 `discriminator_learning_rate` 或减少 `discriminator_updates`。
