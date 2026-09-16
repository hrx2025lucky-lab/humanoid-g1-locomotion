# 实践 08 · 基于 AMP 的拟人走跑 — 我的改动清单

对照基准：`HeYee03/unitree_lab_amp` 的 `main` 分支（AMP 框架骨架，不含算法实现）

骨架提供了环境、专家数据加载、判别器网络结构和 PPO 主体；下面 9 处是我补全的
核心实现。

## 核心实现位置

| # | 内容 | 文件 |
|---|---|---|
| 1 | AMP 单帧观测（80 维） | `source/.../amp/config/g1/amp_flat_env_cfg.py` `AMPObservationsCfg.AMPCfg` |
| 2 | actor / critic / amp 观测分组 | `source/.../amp/config/g1/agents/rsl_rl_ppo_cfg.py` `obs_groups` |
| 3 | AMP 风格奖励 | `rsl_rl_amp/algorithms/discriminator.py` `style_reward()` |
| 4 | 任务奖励与风格奖励混合 | 同上 `mix_rewards()` |
| 5 | 环境步内计算总奖励 | `rsl_rl_amp/algorithms/amp.py` `process_env_step()` |
| 6 | PPO 总 loss | `rsl_rl_amp/algorithms/ppo.py` |
| 7 | WalkToRun Runner 配置 | `source/.../agents/rsl_rl_ppo_cfg.py` `G1AMPWalkToRunRunnerCfg` |
| 8 | 完整走跑评估环境 | `source/.../amp_flat_env_cfg.py` `G1AMPWalkToRunFullPlayEnvCfg` |
| 9 | FullPlay 任务注册 | `source/.../amp/config/g1/__init__.py` |

单帧 AMP 特征 80 维 = 线速度 3 + 角速度 3 + 投影重力 3 + 基座高度 1
+ 关节位置 29 + 关节速度 29 + 4 个关键连杆位置 12。历史长度 3，
判别器窗口 240 维。特征顺序必须与 `motion_dataset.py` 中的专家数据一致。

## 我修改的文件（11 个）

- `rsl_rl_amp/algorithms/amp.py`
- `rsl_rl_amp/algorithms/discriminator.py`
- `rsl_rl_amp/algorithms/ppo.py`
- `scripts/rsl_rl/play.py` — 修上游把 `pretrained_checkpoint` 从
  `isaaclab.utils` 挪到 `isaaclab_rl.utils` 导致的 ImportError
- `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` — 改走 URDF 加载
- `source/.../amp/config/g1/__init__.py`
- `source/.../amp/config/g1/agents/rsl_rl_ppo_cfg.py`
- `source/.../amp/config/g1/amp_flat_env_cfg.py`
- `source/.../amp/config/g1/motion_cfg.py` — 增加 `p7_segments_20260911` profile
- `source/.../amp/motion_dataset.py`

## 我新增的文件

- `source/.../amp/data/p7_segments_20260911/provenance.json` — 实践 07 产出片段的来源记录

## 验证范围

自然步态未通过：双支撑占比 44%，正常人为 20%–25%；0.3 m/s 慢走受限于参考数据。
摆臂方向已修正并在新随机种子下复现。共八轮实验，记录见展示页。
