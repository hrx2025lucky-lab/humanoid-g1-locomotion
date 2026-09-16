# 实践 10 · PHP 的 HOI 人-物交互运动跟踪 — 我的改动清单

对照基准：`unitree_rl_lab` 框架 + 课程 HOI 代码

本实践没有可逐文件比对的课程原始包，因此按**目录归属**说明贡献。

## 我实现的部分（29 个 Python 文件）

全部位于 `source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/` 下：

| 子目录 | 内容 |
|---|---|
| `robots/g1_29dof/hoi_mimic_terrain/` | HOI 地形跟踪任务环境、课程学习 |
| `robots/g1_29dof/hoi_mimic_terrain_perceptive_raycast/` | 加地形感知射线的版本 |
| `mdp/` | HOI 奖励、地形高度扫描观测 |
| `sensors/hoi_merged_terrain_ray_caster.py` | 合并地形的射线投射传感器 |
| `utils/hoi/motion_loader.py` | HOI 运动数据加载（含物体位姿通道） |

其中 `hoi_merged_terrain_ray_caster.py` 是把 `*.terrain.json` 的 box primitive
生成 Warp mesh 供射线求交，这是实践 02 RayCaster 经验的直接延续。

## 上游原样保留

`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/`、`assets/`、
`rsl_rl/` 等目录来自 unitree_rl_lab 与 RSL-RL 上游。

## 验证范围

训练量只有参考配置的 0.025%，且**没有独立留出验证**：所选模型是从同一批
候选里挑的。世界锚点漂移等指标见展示页 `results.json`。
