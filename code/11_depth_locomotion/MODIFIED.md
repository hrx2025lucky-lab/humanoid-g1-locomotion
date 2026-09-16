# 实践 11 · Project Instinct 跑酷与 Sim2Sim — 我的改动清单

对照基准：上游 `project-instinct/instinctlab`

训练侧使用 InstinctLab 框架（未收录上游全量代码）。本目录收录的是
**我自己从零写的 MuJoCo Sim2Sim 验证链路**。

## 我新增的文件

| 文件 | 作用 |
|---|---|
| `sim2sim/sim2sim.py` | Sim2Sim 主循环 |
| `sim2sim/onnx_inference.py` | ONNX 策略推理（actor + 深度编码器两段） |
| `sim2sim/mujoco_env.py` | MuJoCo 环境封装 |
| `sim2sim/config.py` | 关节顺序、PD 增益、控制频率配置 |
| `sim2sim/g1_29dof/rough_terrain_generator.py` | 粗糙地形生成 |
| `sim2sim/g1_29dof/rough_terrain_cfg.py` | 地形参数 |
| `sim2sim/g1_29dof/scene_flat.xml` / `scene_rough.xml` | MJCF 场景 |
| `record_sim2sim.py` | 录像 |
| `probe_command.py` | 指令扫描探针 |
| `test_pipeline.py` | 链路自检 |

> `g1_29dof/meshes_recv_1_0/` 网格文件未收录（Unitree 官方公开资产，约 58 MB），
> 从 `unitree_ros` 获取后放入该目录即可运行。

## 验证范围

自训练策略速度跟踪能力不足：XY RMSE 与均速几乎相等。指令扫描**推翻了我自己
先前公开发布的「迁移落差约 28%」说法**，更正记录见 `command_probe_outcome.json`。
