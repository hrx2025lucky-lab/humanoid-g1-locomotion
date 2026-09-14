# 04 · 速度与骨盆高度联合控制

MJLab · PPO · 双指令控制

构建速度与骨盆高度双指令任务，将高度采样、策略观测和高度跟踪奖励连接为完整控制链路。

## 已实现

- 实现随机高度指令、100维Actor输入与29维关节动作。
- 完成主模型20,000次训练，以及无高度奖励、无高度观测等3,000次同预算消融。
- 完成固定蹲走、三组动态指令切换及额外朝向反馈对照。

## 实验结果

高度0.5m、前向速度0.5m/s指令下连续蹲走22秒；统计后20秒的高度MAE为0.6116cm、前向均速0.5357m/s。三组动态指令均完成24秒窗口。

**验证范围：**单种子平地结果仍有朝向与横向漂移。朝向反馈改善角度，但没有实现横向位置闭环；速度误差并非所有指标都改善。

[查看数值结果](results.json) · [训练记录与实现文件](training/README.md)

## 视频与动画

[![6秒动态预览](media/preview.gif)](media/velocity_height_20k_22s.mp4)

*前6秒节选，完整视频及条件如下。*

| 内容与条件 | 时长 | 文件 |
|---|---:|---|
| 20k主模型：高度0.5m、前向0.5m/s、零转向命令，22秒 | 22.00s | [播放 / 下载](media/velocity_height_20k_22s.mp4) |
| 同一20k模型增加外部朝向反馈；角度控制不等于横向位置控制 | 22.00s | [播放 / 下载](media/heading_feedback_20k_22s.mp4) |
| 3k消融：同时保留高度观测与奖励的基线 | 20.00s | [播放 / 下载](media/ablation_baseline_3k.mp4) |
| 3k消融：移除高度跟踪奖励 | 20.00s | [播放 / 下载](media/ablation_no_height_reward_3k.mp4) |
| 3k消融：Actor不接收高度指令 | 20.00s | [播放 / 下载](media/ablation_no_height_observation_3k.mp4) |

![速度与骨盆高度联合控制结果图](media/p4_3k_vs_20k.png)

<details>
<summary>全部结果图</summary>

- [p4_3k_vs_20k](media/p4_3k_vs_20k.png)
- [p4_direction_diagnosis](media/p4_direction_diagnosis.png)
- [p4_heading_comparison](media/p4_heading_comparison.png)

</details>

[返回项目首页](../../README.md) · [全部实践状态](../../PROJECT_STATUS.md)
