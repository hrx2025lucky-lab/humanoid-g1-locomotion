# G1 Locomotion — Isaac Lab 人形运控实践

基于 Unitree G1 (29 DoF) 的强化学习运动控制，覆盖 **训练 → 验证 → 跨仿真器部署** 全链路。
仿真栈：Isaac Sim 5.1 / Isaac Lab 2.3.2 / rsl-rl 3.1.2 / MuJoCo 3.12。

---

## 内容

| 目录 | 内容 | 状态 |
|---|---|---|
| `g1_rough/` | 粗糙地形行走：地形课程 + 高度扫描感知 + 奖励重设计 | 进行中 |
| `sim2sim/` | Isaac Lab → MuJoCo 策略迁移，六项对齐验证 | 已完成 |

---

## `g1_rough/` — 粗糙地形行走

在六种程序化生成地形（金字塔楼梯 / 倒金字塔楼梯 / 随机方块 / 随机起伏 / 上坡 / 下坡）
上训练速度跟踪策略，10 级难度课程 × 20 列样本。

### 设计要点：零侵入挂载

策略配置不写进 `unitree_rl_lab` 的目录树，而是作为独立包通过 symlink 挂载：

```
g1_rough/  ──symlink──▶  unitree_rl_lab/.../g1/29dof/rough
```

Isaac Lab 的 `import_packages()` 会递归 import 所有**目录包**，因此挂载后任务自动注册，
**第三方仓库零改动**。两个前置假设在写代码前用最小实验验证过：

1. `pkgutil.iter_modules` 能否发现 symlink 目录 → 能，识别为 package
2. 相对导入能否穿过 `29dof` 这种**非法 Python 标识符**（数字开头）→ 能，
   相对导入做的是 `__package__` 字符串拼接，不走标识符校验

安装：

```bash
./g1_rough/install_overlay.sh
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

---

## 出处声明

- `sim2sim/sim2sim_flat.py` 参考课程示例实现的结构独立编写（502 行 vs 322 行，
  完全相同的有效行占 15.4%，且均为受功能约束的样板行：变量赋值、函数签名、字典键）。
  课程原始代码不包含在本仓库中。
- `unitree_rl_lab`、`IsaacLab` 为第三方项目，本仓库不包含其代码，
  也未对其做任何修改（见上文零侵入设计）。
