# 11 · 深度感知运动与Sim2Sim验证

Project Instinct · 深度历史 · PPO/风格奖励 · MuJoCo Sim2Sim

接入深度历史和运动风格训练，补全深度图处理链路，导出ONNX并在MuJoCo中完成跨仿真运行。

## 已实现

- 核对深度帧历史、特征编码、策略输入与ONNX导出一致性；完成风格系数配对与后续训练，比较5000与7500模型。
- 补全`sim2sim/sim2sim.py`中`DepthImagePipeline.append`的深度图处理：缩放到36×64、按(18,0,16,16)边距裁剪为18×32、INPAINT_NS修补、3×3高斯模糊、按[0,2.5]米线性归一化。实现对照训练侧`noise_model.crop_and_resize`的定义，通过5项单元测试（输出形状18×32、远裁剪映射到1.000000、1.25米映射到0.500000、深度观测(1,8,18,32)、左右顺序保持）。
- 完成MuJoCo跨仿真运行与录制：四组20秒连续记录，覆盖课程示例策略与自训练5000模型、平地与粗糙地形。

## 实验结果

保留5000模型；Isaac原生20秒协议中前进统计段均速0.2572m/s、XY速度RMSE 0.2604m/s，7500模型速度波动变大、XY误差约增加40.3%。

跨仿真运行结果（同一份代码、同一场景，仅更换策略）：

| 策略 | 场景 | 20秒前进距离 | 均速 | 最低骨盆高度 |
|---|---|---:|---:|---:|
| 课程示例 | 平地 | 13.87 m | 0.693 m/s | 0.713 m |
| 课程示例 | 粗糙 | 4.98 m | 0.249 m/s | 0.603 m |
| 自训练5000 | 平地 | 1.43 m | 0.072 m/s | 0.729 m |
| 自训练5000 | 粗糙 | 1.24 m | 0.062 m/s | 0.729 m |

**验证范围：**课程示例策略在平地连续行走20秒且骨盆稳定在0.71米以上，说明补全后的深度处理与整条执行链路正确；该策略在粗糙地形爬上台阶后蹲住，未摔倒但未完成跑酷。自训练5000模型在两种场景均几乎不前进。

进一步的指令扫描否定了"跨仿真损失"这一解释。该任务使用`PoseVelocityCommand`，观测到的指令不是设定值，而是随目标距离衰减的P控制输出（刚度2.0），并被钳在[0, 1.0]。因此在同一平地场景下扫描恒定指令：课程策略的跟踪比为85%—114%（指令0.6走0.682m/s、1.0走0.962、1.5走1.274），而自训练模型在任何指令下只有10%—21%（0.078、0.098、0.310）。在训练允许的最大值1.0处落差依然存在，说明问题不在指令幅度。

由此更正此前的表述：自训练模型在Isaac原生评估中前向均速0.2572m/s，而XY速度RMSE为0.2604m/s，误差与均速几乎相等，说明它在原生环境里本就几乎没有跟踪指令的能力。MuJoCo中的低速不是单纯的跨仿真损失，而是一个本就薄弱的策略在另一个仿真器里进一步退化。已排除的原因包括：观测缩放（两侧均为角速度0.25、关节速度0.05）、观测维度与8帧历史、深度处理链路（课程策略用同一条链路正常行走）、指令幅度。仍待查的是训练侧的奖励与课程设置，以及训练地形为perlin_rough而sim2sim提供平地与台阶方块这一分布差异。因此本页不声称跨仿真部署成功，也未达成完整跑酷。

录制使用独立入口以离屏渲染替代交互式viewer，观测构造、ONNX推理、动作解码、PD控制与物理步进均由未修改的原代码执行。

[查看数值结果](results.json) · [训练记录与实现文件](training/README.md)

## 视频与动画

### MuJoCo Sim2Sim：课程示例策略，平地20秒

[![深度感知策略MuJoCo跨仿真运行，平地20秒](media/preview.gif)](media/p11_sim2sim_course_flat_20s.mp4)

预览为早期model3000的Isaac部署录像，保留作为历史记录；下方为本次MuJoCo跨仿真的四组完整记录。

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [Sim2Sim · 课程示例策略 · 平地](media/p11_sim2sim_course_flat_20s.mp4) | 20.00秒 | 连续行走13.87米，骨盆稳定；用于验证深度处理与执行链路正确 |
| [Sim2Sim · 课程示例策略 · 粗糙地形](media/p11_sim2sim_course_rough_20s.mp4) | 20.00秒 | 前进4.98米，爬上台阶后蹲住，未摔倒但未完成跑酷 |
| [Sim2Sim · 自训练5000模型 · 平地](media/p11_sim2sim_ours5000_flat_20s.mp4) | 20.00秒 | 仅前进1.43米，站立稳定但几乎不移动 |
| [Sim2Sim · 自训练5000模型 · 粗糙地形](media/p11_sim2sim_ours5000_rough_20s.mp4) | 20.00秒 | 仅前进1.24米；与平地差异很小，说明瓶颈在迁移而非地形 |
| [深度感知策略部署回放（早期model3000）](media/early_depth_policy_3000.mp4) | 19.98秒 | Isaac中早期model3000录像；不是当前保留的5000模型 |

![深度感知运动结果图](media/p11_style_pair.png)

<details>
<summary>全部结果图</summary>

- [p11_3500_comparison](media/p11_3500_comparison.png)
- [p11_boundary_diagnostic](media/p11_boundary_diagnostic.png)
- [p11_camera_pair](media/p11_camera_pair.png)
- [p11_native_command_pair](media/p11_native_command_pair.png)
- [p11_native_diagnostic](media/p11_native_diagnostic.png)
- [p11_reward_components](media/p11_reward_components.png)
- [p11_sampling_reward](media/p11_sampling_reward.png)
- [p11_style_pair](media/p11_style_pair.png)

</details>

[返回项目首页](../../README.md) · [全部实践状态](../../PROJECT_STATUS.md)
