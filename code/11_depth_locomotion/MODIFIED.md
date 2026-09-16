# 实践 11 · Project Instinct 跑酷与 Sim2Sim — 我的改动清单

对照基准：训练侧为上游 `project-instinct/instinctlab`；Sim2Sim 侧为课程发放的
`sim2sim.zip`。下表逐文件与这两个来源做哈希比对得出。

## Sim2Sim 目录：哪些是我写的

`sim2sim/` 整个目录来自课程包，不是我从零写的。我的改动集中在一个函数里。

| 文件 | 与课程原版比对 | 说明 |
|---|---|---|
| `sim2sim/sim2sim.py` | **有改动，+10 −12 行** | 补全 `DepthImagePipeline.append` 的 5 处 TODO |
| `sim2sim/onnx_inference.py` | 哈希一致，未改动 | 课程包原样 |
| `sim2sim/mujoco_env.py` | 哈希一致，未改动 | 课程包原样 |
| `sim2sim/config.py` | 哈希一致，未改动 | 课程包原样 |
| `sim2sim/g1_29dof/rough_terrain_generator.py` | 哈希一致，未改动 | 课程包原样 |
| `sim2sim/g1_29dof/rough_terrain_cfg.py` | 哈希一致，未改动 | 课程包原样 |
| `sim2sim/g1_29dof/scene_flat.xml` | 哈希一致，未改动 | 课程包原样 |
| `sim2sim/g1_29dof/scene_rough.xml` | 哈希一致，未改动 | 课程包原样 |

### 那 10 行具体做了什么

`DepthImagePipeline.append` 的深度图预处理链，对照训练侧
`instinctlab/utils/noise/noise_model.py::crop_and_resize` 的定义实现：

1. `cv2.resize` 到 36×64，最近邻插值
2. 按边距 `(18, 0, 16, 16)` 裁剪为 18×32
3. `cv2.inpaint`，`INPAINT_NS`，半径 3
4. `cv2.GaussianBlur`，核 3×3，σ=1
5. 按 `[0, 2.5]` 米线性归一化到 `[0, 1]`

其中两处是实际会踩的坑：下边距为 0，写成 `image[y1:-y2]` 会返回空数组，
必须用 `size - margin`；`cv2.GaussianBlur` 第四个位置参数是 `dst` 而非 `sigmaY`，
必须用关键字传。

## 我自己写的文件

| 文件 | 作用 |
|---|---|
| `record_sim2sim.py` | 离屏录制入口。课程的 `sim2sim.py` 主循环依赖交互式 viewer，无法出文件；此入口只把 viewer 替换为离屏渲染，观测构造、ONNX 推理、动作解码、PD 控制与物理步进仍由未修改的原代码执行 |
| `probe_command.py` | 指令扫描探针，纯物理与推理、不渲染，用于比较不同恒定速度指令下的跟踪比 |
| `test_pipeline.py` | 深度处理链路自检，5 项断言 |

## 权重文件说明

`weights/` 下有两类，来源不同：

| 文件 | 来源 |
|---|---|
| `parkour_actor.onnx`、`parkour_depth_encoder.onnx`、`stand_actor.onnx`、`stand_depth_encoder.onnx` | **课程包 `sim2sim.zip` 自带的示例策略，不是我训练的**，收录用于复现对照实验 |
| `actor.onnx`、`0-depth_encoder.onnx` | 我自己训练的 model_5000 导出结果 |

> `g1_29dof/meshes_recv_1_0/` 网格文件未收录（Unitree 官方公开资产，约 58 MB），
> 从 `unitree_ros` 获取后放入该目录即可运行。

## 验证范围

补全后的链路是正确的：课程示例策略在 MuJoCo 平地连续走完 20 秒、13.87 米，
骨盆稳定在 0.71 米以上。

但我自己训练的策略迁移后几乎不前进。指令扫描**推翻了我先前公开发布的
「迁移落差约 28%」说法**：同一场景下课程策略跟踪比为 85%—114%，我的模型在
任何指令下只有 10%—21%，在训练允许的最大指令处落差依然存在。回看原生评估，
该模型 XY RMSE 0.2604 与均速 0.2572 几乎相等，说明它在原仿真器里本就没学会
跟踪速度。更正记录见 `command_probe_outcome.json`。
