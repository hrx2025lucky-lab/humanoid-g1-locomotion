# G1人形机器人：强化学习与运动控制

基于Unitree G1开展速度与高度控制、地形行走、分层导航、全身动作跟踪、教师学生蒸馏和深度感知运动。项目包含算法组件、训练配置、量化实验和仿真视频。

**更新：2026-09-14。全部结果来自仿真或离线运动学验证。**

[视频合集](MEDIA.md) · [当前状态](PROJECT_STATUS.md) · [算法与框架来源](REFERENCES.md)

## 动态预览与完整视频

已收录实践2—11的23段完整MP4，以及实践3的2段补充动画。实践08保留问题回放与量化记录；其余实践可从动画预览或带标题的链接打开完整文件。

### 实践01｜仿真环境与行走策略部署

当前未收录独立展示视频。已有策略导出等价性与60秒MuJoCo闭环记录，见[实践01说明](practices/01_simulation_baseline/README.md)。

### 实践02｜感知驱动的粗糙地形行走

预览是15k适配策略走完整20秒台阶与方块地形，相机跟随机器人；另附旧策略与适配策略的同场景对比。

[![实践02：台阶与方块地形行走（15k适配策略），跟随机位完整20秒](practices/02_rough_terrain/media/preview_follow.gif)](practices/02_rough_terrain/media/p2_final_stairs_20s_follow.mp4)

- [台阶与方块地形行走（15k适配策略）· 跟随机位](practices/02_rough_terrain/media/p2_final_stairs_20s_follow.mp4) · 20.00秒
- [同一段状态的固定机位版本](practices/02_rough_terrain/media/p2_final_stairs_20s.mp4) · 20.00秒
- [粗糙地形行走：旧策略与适配策略对比](practices/02_rough_terrain/media/p2_old_final_comparison_28s.mp4) · 28.00秒

[本实践的结果与条件](practices/02_rough_terrain/README.md)

### 实践03｜HoST仰躺起身（预训练策略部署）

预览来自默认仰躺初态的60秒部署录像；使用已有预训练策略。

[![实践03：HoST仰躺起身与站立：60秒完整部署回放，动态预览](practices/03_host_standup/media/preview.gif)](practices/03_host_standup/media/host_standup_60s.mp4)

- [HoST仰躺起身与站立：60秒完整部署回放](practices/03_host_standup/media/host_standup_60s.mp4) · 60.00秒
- [HoST动作接口对比（动画）](practices/03_host_standup/media/action_space_ablation.gif) · 8.00秒
- [HoST预训练策略起身（动画）](practices/03_host_standup/media/host_standup.gif) · 5.01秒

[本实践的结果与条件](practices/03_host_standup/README.md)

### 实践04｜速度与骨盆高度联合控制

预览展示20k主模型的蹲走；另附朝向反馈和三组3k高度控制消融。

[![实践04：速度与骨盆高度联合控制：22秒蹲走，动态预览](practices/04_velocity_height/media/preview.gif)](practices/04_velocity_height/media/velocity_height_20k_22s.mp4)

- [速度与骨盆高度联合控制：22秒蹲走](practices/04_velocity_height/media/velocity_height_20k_22s.mp4) · 22.00秒
- [蹲走朝向反馈：同一策略加入外部朝向控制](practices/04_velocity_height/media/heading_feedback_20k_22s.mp4) · 22.00秒
- [高度控制消融：保留高度观测与奖励](practices/04_velocity_height/media/ablation_baseline_3k.mp4) · 20.00秒
- [高度控制消融：移除高度跟踪奖励](practices/04_velocity_height/media/ablation_no_height_reward_3k.mp4) · 20.00秒
- [高度控制消融：移除Actor高度指令观测](practices/04_velocity_height/media/ablation_no_height_observation_3k.mp4) · 20.00秒

[本实践的结果与条件](practices/04_velocity_height/README.md)

### 实践05｜分层强化学习导航

预览为固定布局模型的目标跟随；下方两段150秒录像与配对统计测试分别说明。

[![实践05：分层导航：固定布局训练模型的目标跟随，动态预览](practices/05_hierarchical_navigation/media/preview.gif)](practices/05_hierarchical_navigation/media/navigation_baseline_play.mp4)

- [分层导航：固定布局训练模型的目标跟随](practices/05_hierarchical_navigation/media/navigation_baseline_play.mp4) · 150.00秒
- [分层导航：随机布局训练阶段模型回放](practices/05_hierarchical_navigation/media/navigation_random_layout_play.mp4) · 150.00秒

[本实践的结果与条件](practices/05_hierarchical_navigation/README.md)

### 实践06｜教师学生蒸馏：Action与KL动作对比

预览展示两个学生策略的并排回放；每段动作分别复位，另附三段完整对比。

[![实践06：Action与KL蒸馏：三段动作并排对比，动态预览](practices/06_teacher_student/media/preview.gif)](practices/06_teacher_student/media/p6_action_kl_3clips_23s.mp4)

- [Action与KL蒸馏：三段动作并排对比](practices/06_teacher_student/media/p6_action_kl_3clips_23s.mp4) · 23.00秒
- [Action与KL蒸馏对比：动作13](practices/06_teacher_student/media/p6_clip13_comparison.mp4) · 6.22秒
- [Action与KL蒸馏对比：动作19](practices/06_teacher_student/media/p6_clip19_comparison.mp4) · 8.12秒
- [Action与KL蒸馏对比：动作22](practices/06_teacher_student/media/p6_clip22_comparison.mp4) · 8.66秒

[本实践的结果与条件](practices/06_teacher_student/README.md)

### 实践07｜人体到G1的运动重定向（运动学回放）

预览为约1.9秒行走片段；下方另附加速跑和右转。这些是姿态映射回放，尚不代表动力学平衡。

[![实践07：GMR行走重定向：人体与G1运动学对照，动态预览](practices/07_motion_retargeting/media/preview.gif)](practices/07_motion_retargeting/media/walk_human_g1.mp4)

- [GMR行走重定向：人体与G1运动学对照](practices/07_motion_retargeting/media/walk_human_g1.mp4) · 1.87秒
- [GMR加速跑重定向：人体与G1运动学对照](practices/07_motion_retargeting/media/run_human_g1.mp4) · 1.41秒
- [GMR右转重定向：G1运动学回放](practices/07_motion_retargeting/media/right_turn_g1.mp4) · 2.34秒

[本实践的结果与条件](practices/07_motion_retargeting/README.md)

### 实践08｜AMP速度跟踪与步态评估

已定位并解除AMP判别器的饱和暂停：原设置下判别器全程只发生4次优化步，解除后升至2000，两脚摆腿抬升由2.13cm提高到5.01cm，**拖腿与左右不协调已解决**，0.5m/s前进已有3个模型通过全部数值门槛。但人工复核**判定不批准**：摆臂幅度虽由专家的9%—16%提高到32%—41%，与同侧髋的相位相关系数却是+0.85，即同手同脚；0.3m/s慢走则受限于参考动作库中最长连续慢走只有0.40秒。首页动态预览保持撤下，实践页保留正面／侧面完整20秒复核录像。

[查看根因分析、步态诊断与完整结果](practices/08_amp_locomotion/README.md)

### 实践09｜全身舞蹈轨迹跟踪

完整录像长131.48秒；失败条件旁路监测，与标称CPU误差统计所用条件分列。

[![实践09：全身舞蹈轨迹跟踪：131.48秒完整回放，动态预览](practices/09_motion_tracking/media/preview.gif)](practices/09_motion_tracking/media/motion_tracking_30k_full.mp4)

- [全身舞蹈轨迹跟踪：131.48秒完整回放](practices/09_motion_tracking/media/motion_tracking_30k_full.mp4) · 131.48秒

[本实践的结果与条件](practices/09_motion_tracking/README.md)

### 实践10｜平台地形动作跟踪（HOI）

预览来自所选model399的6.18秒平台路径；另附model299对照录像。

[![实践10：平台路径完整回放（所选model399），动态预览](practices/10_object_interaction/media/preview.gif)](practices/10_object_interaction/media/p10_std03_model399_full.mp4)

- [平台路径完整回放（所选model399）](practices/10_object_interaction/media/p10_std03_model399_full.mp4) · 6.18秒
- [平台边缘路径回放（对照model299）](practices/10_object_interaction/media/p10_model299_edge_path.mp4) · 6.18秒

[本实践的结果与条件](practices/10_object_interaction/README.md)

### 实践11｜深度感知策略部署（早期3000模型录像）

录像展示早期model3000的部署；当前保留5000模型，完整跑酷效果仍待完善。

[![实践11：深度感知策略部署回放（早期model3000），动态预览](practices/11_depth_locomotion/media/preview.gif)](practices/11_depth_locomotion/media/early_depth_policy_3000.mp4)

- [深度感知策略部署回放（早期model3000）](practices/11_depth_locomotion/media/early_depth_policy_3000.mp4) · 19.98秒

[本实践的结果与条件](practices/11_depth_locomotion/README.md)

## 代表性结果

| 方向 | 已完成的结果 | 展示 |
|---|---|---|
| 速度与骨盆高度联合控制 | 高度0.5m、前向速度0.5m/s指令下连续蹲走22秒；统计后20秒的高度MAE为0.6116cm、前向均速0.5357m/s。三组动态指令均完成24秒窗口。 | [结果与视频](practices/04_velocity_height/README.md) |
| 教师学生蒸馏与全身动作 | Action和KL学生均有25/25动作到达参考末尾；对齐身体平均距离分别为2.580cm、2.537cm，世界身体平均距离为10.90cm、12.86cm。 | [结果与视频](practices/06_teacher_student/README.md) |
| 自适应采样与全身轨迹跟踪 | 标称CPU协议下学习残差连续跟完131.48秒单舞蹈，对齐身体平均距离3.633cm、世界锚点平均距离0.581m；零残差参考PD在0.28秒触发末端条件。 | [结果与视频](practices/09_motion_tracking/README.md) |
| 感知驱动的粗糙地形行走 | 新增15,000次更新；8组同初态、各14秒场景中，新策略的XY速度与转速RMSE均低于旧策略。另完成20秒台阶穿越轨迹。 | [结果与视频](practices/02_rough_terrain/README.md) |
| 地形感知的人物交互动作跟踪 | 所选model399走完309控制帧、6.18秒参考；世界锚点RMSE为0.1297m。两个指定初态平移也到达参考末尾。 | [结果与视频](practices/10_object_interaction/README.md) |

## 按实践浏览

每个目录统一放置成果说明、`training/`训练记录与算法组件、`media/`视频和结果图。训练和评估的条件在对应页面说明。

| 编号 | 内容 | 主要技术 |
|---|---|---|
| 01 | [仿真环境与行走策略部署](practices/01_simulation_baseline/README.md) | Isaac Lab · PPO · MuJoCo |
| 02 | [感知驱动的粗糙地形行走](practices/02_rough_terrain/README.md) | Isaac Lab · PPO · 高度扫描 · MuJoCo |
| 03 | [HoST起身与增量动作部署](practices/03_host_standup/README.md) | HoST · PD · MuJoCo |
| 04 | [速度与骨盆高度联合控制](practices/04_velocity_height/README.md) | MJLab · PPO · 双指令控制 |
| 05 | [分层强化学习导航](practices/05_hierarchical_navigation/README.md) | Isaac Lab · 高层PPO · 冻结低层策略 |
| 06 | [教师学生蒸馏与全身动作](practices/06_teacher_student/README.md) | MJLab · PPO · Action Matching · KL Matching |
| 07 | [人体到G1的运动重定向](practices/07_motion_retargeting/README.md) | GMR · SMPL-X · 逆运动学 |
| 08 | [AMP拟人运动与风格奖励](practices/08_amp_locomotion/README.md) | Isaac Lab · PPO · AMP |
| 09 | [自适应采样与全身轨迹跟踪](practices/09_motion_tracking/README.md) | BeyondMimic方法 · MJLab · PPO |
| 10 | [地形感知的人物交互动作跟踪](practices/10_object_interaction/README.md) | Isaac Lab · PPO · 高度扫描 |
| 11 | [深度感知运动与策略导出](practices/11_depth_locomotion/README.md) | Project Instinct · 深度历史 · PPO/风格奖励 |

## 当前进展

已完成双指令控制、两类蒸馏、长动作跟踪、地形配对和平台路径的分项验证。分层导航已有1,536局配对结果，尚未得到随机布局训练的总体优势；AMP已解决拖腿与左右不协调，但自然摆臂与低速慢走仍未通过，拟人走跑整体未通过验证；深度策略尚未达到完整跑酷表现，跨仿真运行尚未开始。

各项结果保留模型、统计窗口、坐标和初态限制。预训练部署、运动学回放与自主训练分别说明，具体数值和未覆盖场景见对应实践页。
