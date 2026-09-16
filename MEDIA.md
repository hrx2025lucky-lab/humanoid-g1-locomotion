# G1运动控制：视频与动画合集

已收录31段完整MP4和2段补充动画，按实践编号排列。每个文件都有独立标题，模型阶段、回放条件和时长在表中说明。首页提供实践2—11各一段动态预览。

## 实践01｜仿真环境与行走策略部署

当前未收录独立展示视频；已有部署记录见[实践01说明](practices/01_simulation_baseline/README.md)。

## 实践02｜感知驱动的粗糙地形行走

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [台阶与方块地形行走 · 跟随机位](practices/02_rough_terrain/media/p2_final_stairs_20s_follow.mp4) | 20.00秒 | 相机跟随机器人躯干，便于看清落脚与姿态 |
| [同一段状态 · 固定机位](practices/02_rough_terrain/media/p2_final_stairs_20s.mp4) | 20.00秒 | 固定世界相机，便于看清走过的距离与路径 |
| [粗糙地形行走：旧策略与适配策略对比](practices/02_rough_terrain/media/p2_old_final_comparison_28s.mp4) | 28.00秒 | 旧策略与15k适配策略：平地和台阶配对，28秒剪辑 |

## 实践03｜HoST仰躺起身（预训练策略部署）

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [HoST仰躺起身与站立：60秒完整部署回放](practices/03_host_standup/media/host_standup_60s.mp4) | 60.00秒 | 已有HoST预训练策略；默认仰躺起身，60秒完整部署回放 |
| [HoST动作接口对比（动画）](practices/03_host_standup/media/action_space_ablation.gif) | 8.00秒 | HoST动作接口对比动画 |
| [HoST预训练策略起身（动画）](practices/03_host_standup/media/host_standup.gif) | 5.01秒 | HoST预训练起身动画 |

## 实践04｜速度与骨盆高度联合控制

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [速度与骨盆高度联合控制：22秒蹲走](practices/04_velocity_height/media/velocity_height_20k_22s.mp4) | 22.00秒 | 20k主模型：高度0.5m、前向0.5m/s、零转向命令，22秒 |
| [蹲走朝向反馈：同一策略加入外部朝向控制](practices/04_velocity_height/media/heading_feedback_20k_22s.mp4) | 22.00秒 | 同一20k模型增加外部朝向反馈；角度控制不等于横向位置控制 |
| [高度控制消融：保留高度观测与奖励](practices/04_velocity_height/media/ablation_baseline_3k.mp4) | 20.00秒 | 3k消融：同时保留高度观测与奖励的基线 |
| [高度控制消融：移除高度跟踪奖励](practices/04_velocity_height/media/ablation_no_height_reward_3k.mp4) | 20.00秒 | 3k消融：移除高度跟踪奖励 |
| [高度控制消融：移除Actor高度指令观测](practices/04_velocity_height/media/ablation_no_height_observation_3k.mp4) | 20.00秒 | 3k消融：Actor不接收高度指令 |

## 实践05｜分层强化学习导航

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [分层导航：固定布局训练模型的目标跟随](practices/05_hierarchical_navigation/media/navigation_baseline_play.mp4) | 150.00秒 | 固定布局训练模型的原生播放；含目标切换，与1536局统计协议单列 |
| [分层导航：随机布局训练阶段模型回放](practices/05_hierarchical_navigation/media/navigation_random_layout_play.mp4) | 150.00秒 | 随机布局训练阶段模型的原生播放；不作为同预算统计对照 |

## 实践06｜教师学生蒸馏：Action与KL动作对比

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [Action与KL蒸馏：三段动作并排对比](practices/06_teacher_student/media/p6_action_kl_3clips_23s.mp4) | 23.00秒 | Action与KL学生的三段并排比较；各20k，训练库分别复位，23秒剪辑 |
| [Action与KL蒸馏对比：动作13](practices/06_teacher_student/media/p6_clip13_comparison.mp4) | 6.22秒 | 动作13：Action与KL学生同协议比较 |
| [Action与KL蒸馏对比：动作19](practices/06_teacher_student/media/p6_clip19_comparison.mp4) | 8.12秒 | 动作19：Action与KL学生同协议比较 |
| [Action与KL蒸馏对比：动作22](practices/06_teacher_student/media/p6_clip22_comparison.mp4) | 8.66秒 | 动作22：Action与KL学生同协议比较，含较弱场景 |

## 实践07｜人体到G1的运动重定向（运动学回放）

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [GMR行走重定向：人体与G1运动学对照](practices/07_motion_retargeting/media/walk_human_g1.mp4) | 1.87秒 | 行走片段：人体和G1运动学对照，不含力矩闭环 |
| [GMR加速跑重定向：人体与G1运动学对照](practices/07_motion_retargeting/media/run_human_g1.mp4) | 1.41秒 | 加速跑片段：人体和G1运动学对照，不含动态稳定结论 |
| [GMR右转重定向：G1运动学回放](practices/07_motion_retargeting/media/right_turn_g1.mp4) | 2.34秒 | 右转片段：G1运动学参考回放 |

## 实践08｜AMP阶段回放与问题对照

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [AMP · 摆臂方向已修正 · 侧面](practices/08_amp_locomotion/media/p8_armswing_fixed_side_20s.mp4) | 20.00秒 | 相位−0.70/−0.84反相摆臂，通过前进全部数值门槛 |
| [AMP · 摆臂方向已修正 · 正面](practices/08_amp_locomotion/media/p8_armswing_fixed_front_20s.mp4) | 20.00秒 | 同段轨迹正面机位 |
| [AMP候选 · 正面完整20秒](practices/08_amp_locomotion/media/p8_gate_pass_front_20s.mp4) | 20.00秒 | 通过0.5m/s全部数值门槛的两个模型，正面机位 |
| [AMP候选 · 侧面完整20秒](practices/08_amp_locomotion/media/p8_gate_pass_side_20s.mp4) | 20.00秒 | 同两个模型的侧面机位 |
| [6200阶段回放：步态未通过](practices/08_amp_locomotion/media/amp_selected_20s.mp4) | 20.00秒 | 6200阶段模型，0.5m/s命令的连续20秒回放；左右步态不协调、短促摆动，保留为问题记录。 |
| [转向奖励调整：慢走并排对照](practices/08_amp_locomotion/media/yaw_reward_comparison_20s.mp4) | 20.00秒 | 原保留5759与首轮6000候选；相同0.3m/s指令；兼顾航向改善与速度降低的结果 |

## 实践09｜全身舞蹈轨迹跟踪

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [全身舞蹈轨迹跟踪：131.48秒完整回放](practices/09_motion_tracking/media/motion_tracking_30k_full.mp4) | 131.48秒 | model29999完整单舞蹈播放；保留事件、失败条件旁路监测，与标称CPU误差统计分开 |

## 实践10｜平台地形动作跟踪（HOI）

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [平台路径完整回放（所选model399）](practices/10_object_interaction/media/p10_std03_model399_full.mp4) | 6.18秒 | 所选std0.3/model399：309控制帧、6.18秒完整平台路径 |
| [平台边缘路径回放（对照model299）](practices/10_object_interaction/media/p10_model299_edge_path.mp4) | 6.18秒 | 对照model299的早期平台边缘路径；不是所选model399 |

## 实践11｜深度感知策略部署（早期3000模型录像）

| 视频 / 动画标题 | 时长 | 内容与条件 |
|---|---:|---|
| [Sim2Sim · 课程示例 · 平地](practices/11_depth_locomotion/media/p11_sim2sim_course_flat_20s.mp4) | 20.00秒 | MuJoCo跨仿真，连续走13.87米 |
| [Sim2Sim · 课程示例 · 粗糙](practices/11_depth_locomotion/media/p11_sim2sim_course_rough_20s.mp4) | 20.00秒 | 前进4.98米，台阶后蹲住 |
| [Sim2Sim · 自训练5000 · 平地](practices/11_depth_locomotion/media/p11_sim2sim_ours5000_flat_20s.mp4) | 20.00秒 | 仅前进1.43米 |
| [Sim2Sim · 自训练5000 · 粗糙](practices/11_depth_locomotion/media/p11_sim2sim_ours5000_rough_20s.mp4) | 20.00秒 | 仅前进1.24米 |
| [深度感知策略部署回放（早期model3000）](practices/11_depth_locomotion/media/early_depth_policy_3000.mp4) | 19.98秒 | 早期model3000深度策略部署录像；不是当前保留的5000模型 |

[项目首页](README.md) · [媒体文件校验清单](media_manifest.json)
