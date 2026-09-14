# 11 · 深度感知运动与跨仿真部署

Project Instinct · 深度历史 · PPO/风格奖励

接入深度历史和运动风格训练，核对策略导出、观测编码与执行链路，并比较继续训练的实际收益。

## 已实现

- 核对深度帧历史、特征编码、策略输入与ONNX导出一致性。
- 完成风格系数配对和后续训练，比较5000与7500模型。
- 逐项统计身体速度、世界位移、终止和奖励组成。

## 实验结果

保留5000模型；20秒协议中前进统计段均速0.2572m/s、XY速度RMSE0.2604m/s。7500模型速度波动变大，XY误差约增加40.3%。

**验证范围：**尚未达到完整跑酷和跨障碍可靠性。现有视频是早期3000模型的部署展示，不代表5000模型或最终跑酷能力。

[查看数值结果](results.json) · [训练记录与实现文件](training/README.md)

## 视频与动画

| 内容与条件 | 时长 | 文件 |
|---|---:|---|
| 早期model3000深度策略部署录像；不是当前保留的5000模型 | 19.98s | [播放 / 下载](media/early_depth_policy_3000.mp4) |

![深度感知运动与跨仿真部署结果图](media/p11_style_pair.png)

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
