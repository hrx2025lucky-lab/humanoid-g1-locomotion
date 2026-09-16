# 完整源代码

`practices/` 放的是每个实践的结论、指标与视频；本目录放**可读的完整源代码**。

每个实践目录下都有一份 `MODIFIED.md`，逐文件列出**哪些是我写的、哪些来自上游
框架或课程包**。先读它，再看代码。

| 实践 | 目录 | 框架 | 我的改动 |
|---|---|---|---|
| 01 仿真环境与行走策略部署 | `01_simulation_baseline/` | Isaac Lab + MuJoCo | 部署脚本 1 个（任务书不要求改框架代码） |
| 02 感知驱动的粗糙地形行走 | `02_rough_terrain/` | Isaac Lab + MuJoCo | 新增 6 / 修改 1 |
| 03 HoST 起身与增量动作部署 | `03_host_standup/` | MuJoCo | 修改 1 |
| 04 蹲姿行走：速度 + 骨盆高度 | `04_velocity_height/` | mjlab | 新增 2 / 修改 19 |
| 05 分层强化学习导航 | `05_hierarchical_navigation/` | Isaac Lab | 修改 9 |
| 06 教师-学生蒸馏 | `06_teacher_student/` | mjlab | 新增 4 / 修改 9 |
| 07 GMR 运动重定向 | `07_motion_retargeting/` | GMR | 新增 4 / 修改 4 |
| 08 AMP 拟人走跑 | `08_amp_locomotion/` | Isaac Lab + RSL-RL | 新增 1 / 修改 10 |
| 09 BeyondMimic 关键函数 | `09_motion_tracking/` | Isaac Lab / mjlab | 新增 3 / 修改 2 |
| 10 HOI 人-物交互跟踪 | `10_object_interaction/` | Isaac Lab | HOI 任务 29 个文件 |
| 11 Project Instinct 跑酷 | `11_depth_locomotion/` | InstinctLab + MuJoCo | Sim2Sim 链路 13 个文件 |

## 没有收录什么

为控制仓库体积，以下内容不在版本库中：

| 类型 | 体积 | 说明 |
|---|---|---|
| 训练日志与中间 checkpoint | 约 8.6 GB | TensorBoard events 与每 50 轮的存档 |
| Python 虚拟环境 `.venv/` | 约 3 GB | 第三方依赖，按 `pyproject.toml` 重装即可 |
| 机器人网格资产 `.dae/.usd/.STL` | 约 560 MB | Unitree 官方公开文件，从 `unitree_ros` / `unitree_model` 获取 |
| 编译产物 `build/` | 约 17 MB | C++ 控制器的 `.o`/`.a`/可执行文件 |
| ONNX Runtime 预编译库 | 约 63 MB | 第三方二进制 |
| 课程发放的作业要求文档 | — | 版权属深蓝学院，不转载 |

最终策略权重见各实践目录下的 `weights/`。

## 路径约定

代码里不含任何本机绝对路径，外部资源一律读环境变量：

```bash
export UNITREE_ROS_DIR=~/unitree_ros            # 机器人 URDF
export UNITREE_MODEL_DIR=~/unitree_model        # 机器人 USD
export MUJOCO_RAYCASTER_PLUGIN=~/mujoco_ray_caster/build/lib/libsensor_raycaster.so
export UNITREE_G1_LOW_LEVEL_POLICY_PATH=...     # 实践 05 的低层策略
export AMASS_ACCAD_DIR=~/datasets/AMASS/ACCAD/Female1Running_c3d  # 实践 07
```

## 许可证

各实践目录保留了对应上游框架的 `LICENSE`。我的改动遵循同一许可证。
