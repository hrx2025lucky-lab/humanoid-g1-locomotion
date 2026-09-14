# G1运动控制：视频索引

按实践编号浏览，模型阶段与播放条件列在每段旁。MP4提供原文件，可在GitHub预览或下载；动画和结果图同时放在对应实践目录。

| 实践 | 内容与条件 | 时长 | 视频 |
|---|---|---:|---|
| 02 | 15k适配策略的20秒台阶轨迹，后段包含方块区域 | 20.00s | [视频](practices/02_rough_terrain/media/p2_final_stairs_20s.mp4) |
| 02 | 旧策略与15k适配策略：平地和台阶配对，28秒剪辑 | 28.00s | [视频](practices/02_rough_terrain/media/p2_old_final_comparison_28s.mp4) |
| 03 | HoST动作接口对比动画 | 8.00s | [视频](practices/03_host_standup/media/action_space_ablation.gif) |
| 03 | HoST预训练起身动画 | 5.01s | [视频](practices/03_host_standup/media/host_standup.gif) |
| 03 | 已有HoST预训练策略；默认仰躺起身，60秒完整部署回放 | 60.00s | [视频](practices/03_host_standup/media/host_standup_60s.mp4) |
| 04 | 20k主模型：高度0.5m、前向0.5m/s、零转向命令，22秒 | 22.00s | [视频](practices/04_velocity_height/media/velocity_height_20k_22s.mp4) |
| 04 | 同一20k模型增加外部朝向反馈；角度控制不等于横向位置控制 | 22.00s | [视频](practices/04_velocity_height/media/heading_feedback_20k_22s.mp4) |
| 04 | 3k消融：同时保留高度观测与奖励的基线 | 20.00s | [视频](practices/04_velocity_height/media/ablation_baseline_3k.mp4) |
| 04 | 3k消融：移除高度跟踪奖励 | 20.00s | [视频](practices/04_velocity_height/media/ablation_no_height_reward_3k.mp4) |
| 04 | 3k消融：Actor不接收高度指令 | 20.00s | [视频](practices/04_velocity_height/media/ablation_no_height_observation_3k.mp4) |
| 05 | 固定布局训练模型的原生播放；含目标切换，与1536局统计协议单列 | 150.00s | [视频](practices/05_hierarchical_navigation/media/navigation_baseline_play.mp4) |
| 05 | 随机布局训练阶段模型的原生播放；不作为同预算统计对照 | 150.00s | [视频](practices/05_hierarchical_navigation/media/navigation_random_layout_play.mp4) |
| 06 | Action与KL学生的三段并排比较；各20k，训练库分别复位，23秒剪辑 | 23.00s | [视频](practices/06_teacher_student/media/p6_action_kl_3clips_23s.mp4) |
| 06 | 动作13：Action与KL学生同协议比较 | 6.22s | [视频](practices/06_teacher_student/media/p6_clip13_comparison.mp4) |
| 06 | 动作19：Action与KL学生同协议比较 | 8.12s | [视频](practices/06_teacher_student/media/p6_clip19_comparison.mp4) |
| 06 | 动作22：Action与KL学生同协议比较，含较弱场景 | 8.66s | [视频](practices/06_teacher_student/media/p6_clip22_comparison.mp4) |
| 07 | 行走片段：人体和G1运动学对照，不含力矩闭环 | 1.87s | [视频](practices/07_motion_retargeting/media/walk_human_g1.mp4) |
| 07 | 加速跑片段：人体和G1运动学对照，不含动态稳定结论 | 1.41s | [视频](practices/07_motion_retargeting/media/run_human_g1.mp4) |
| 07 | 右转片段：G1运动学参考回放 | 2.34s | [视频](practices/07_motion_retargeting/media/right_turn_g1.mp4) |
| 08 | 早期AMP模型播放；不是当前5759阶段模型，不用于证明稳定走跑 | 29.98s | [视频](practices/08_amp_locomotion/media/early_amp_play.mp4) |
| 09 | model29999完整单舞蹈播放；保留事件、失败条件旁路监测，与标称CPU误差统计分开 | 131.48s | [视频](practices/09_motion_tracking/media/motion_tracking_30k_full.mp4) |
| 10 | 对照model299的早期平台边缘路径；不是所选model399 | 6.18s | [视频](practices/10_object_interaction/media/p10_model299_edge_path.mp4) |
| 10 | 所选std0.3/model399：309控制帧、6.18秒完整平台路径 | 6.18s | [视频](practices/10_object_interaction/media/p10_std03_model399_full.mp4) |
| 11 | 早期model3000深度策略部署录像；不是当前保留的5000模型 | 19.98s | [视频](practices/11_depth_locomotion/media/early_depth_policy_3000.mp4) |

[项目首页](README.md) · [媒体文件校验清单](media_manifest.json)
