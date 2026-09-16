# 实践 01 · 仿真环境与行走策略部署 — 我的改动清单

对照基准：上游 `unitreerobotics/unitree_rl_lab`

任务书第 4.1 节明确写明「不要求深入修改代码」。本实践的任务是**跑通链路**并
定位关键入口文件，因此框架代码基本保持上游原样。

## 我的贡献

- `sim2sim_flat.py` — 自写的 MuJoCo 平地部署脚本：加载导出策略、对齐 29 关节
  顺序与默认姿态、按 50 Hz 策略周期驱动 PD 控制、记录 60 秒固定指令闭环。
- 环境搭建与训练运行（Isaac Sim 5.1 + Isaac Lab 2.2 二进制安装、4096 并行
  环境 PPO 训练至 11998 iteration）。
- 策略导出等价性核对：用 16 组观测比对训练权重与导出 `policy.pt` 的输出。

## 关键入口文件（任务书 4.1 要求能定位）

| 作用 | 文件 |
|---|---|
| 任务注册 | `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py` |
| 环境配置 | 同目录 `velocity_env_cfg.py` |
| PPO 配置 | `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py` |
| 训练入口 | `scripts/rsl_rl/train.py` |
| 回放与导出 | `scripts/rsl_rl/play.py` |

## 验证范围

同一权重的 Isaac Lab 与 MuJoCo 成对行为比较尚未完成；已有结果为策略导出
等价性与 60 秒 MuJoCo 固定指令记录。
