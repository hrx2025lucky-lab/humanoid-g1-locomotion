# G1人形机器人：强化学习与运动控制

基于Unitree G1开展速度与高度控制、地形行走、分层导航、全身动作跟踪、教师学生蒸馏和深度感知运动。项目包含算法组件、训练配置、量化实验和仿真视频。

**更新：2026-09-16。全部结果来自仿真或离线运动学验证。**

[源代码](code/) · [视频合集](MEDIA.md) · [当前状态](PROJECT_STATUS.md) · [算法与框架来源](REFERENCES.md)

## 源代码

11 个实践的完整源代码与最终策略权重在 [`code/`](code/)。每个实践目录下的 `MODIFIED.md` 逐文件列出哪些是我写的、哪些来自上游框架。

## 各实践概览

下面每项给出一段代表视频，点击动图打开完整 MP4。全部 32 段视频及回放条件见[视频合集](MEDIA.md)，数据、模型阶段与验证范围见各实践页。

### 实践01｜仿真环境与行走策略部署

**Isaac Lab（训练）+ MuJoCo（跨仿真验证）· PPO**

把训练好的策略搬到另一个仿真器要对齐一串东西：关节排列顺序、动作尺度与默认关节角、PD 增益、控制周期。逐项核对通过，并用 60 秒闭环回放确认两侧行为一致。

尚无展示视频。[实践详情](practices/01_simulation_baseline/README.md)

### 实践02｜感知驱动的粗糙地形行走

**Isaac Lab · PPO · 高度扫描 · MuJoCo**

在机器人脚下布一圈射线量地面高度接进观测，让它踩上台阶前就知道前方高低。高度要换算成相对机身的值，射线打空返回的非有限值也必须处理。与旧策略在 8 组相同初态下对照，速度与转速误差都更低。

[![实践02：台阶与方块地形行走](practices/02_rough_terrain/media/preview_follow.gif)](practices/02_rough_terrain/media/p2_final_stairs_20s_follow.mp4)

[实践详情](practices/02_rough_terrain/README.md)

### 实践03｜HoST仰躺起身（预训练策略部署）

**HoST 预训练策略 · PD 控制 · MuJoCo**

用 HoST 已训练好的起身策略，重点是把接口接对：它输出的是关节角增量而非绝对角度，需自行累加，观测还要拼进多帧历史。在 23 关节 G1 上从仰躺完成起身到站立。

[![实践03：HoST仰躺起身与站立](practices/03_host_standup/media/preview.gif)](practices/03_host_standup/media/host_standup_60s.mp4)

[实践详情](practices/03_host_standup/README.md)

### 实践04｜速度与骨盆高度联合控制

**MJLab · PPO · 双指令控制**

让机器人同时听两个指令：走多快、骨盆保持多高。高度指令随机采样后拼进观测，再配一项高度跟踪奖励，做成蹲着走。另用移除高度奖励、移除高度观测两组消融确认效果来自这条链路。

[![实践04：速度与骨盆高度联合控制](practices/04_velocity_height/media/preview.gif)](practices/04_velocity_height/media/velocity_height_20k_22s.mp4)

[实践详情](practices/04_velocity_height/README.md)

### 实践05｜分层强化学习导航

**Isaac Lab · 高层 PPO + 冻结的低层行走策略**

把训好的行走策略冻住当执行层，只训高层网络：它看目标点输出速度指令，低层负责迈腿。原本假设随机布局训练的高层泛化更好，1536 局同预算配对下来并没有优势，按实际结果记为未达成预期。

[![实践05：分层导航目标跟随](practices/05_hierarchical_navigation/media/preview.gif)](practices/05_hierarchical_navigation/media/navigation_baseline_play.mp4)

[实践详情](practices/05_hierarchical_navigation/README.md)

### 实践06｜教师学生蒸馏：Action与KL对比

**MJLab · PPO · Action Matching / KL Matching**

教师能看到仿真里的特权信息，学生只能看到真实可得的观测。对比两种转移方式：直接回归教师输出的动作，或匹配两者输出分布的 KL 散度。两个学生在同一动作库上并排回放。

[![实践06：Action与KL蒸馏并排对比](practices/06_teacher_student/media/preview.gif)](practices/06_teacher_student/media/p6_action_kl_3clips_23s.mp4)

[实践详情](practices/06_teacher_student/README.md)

### 实践07｜人体到G1的运动重定向

**GMR · SMPL-X · 逆运动学，不含强化学习**

人的骨架比例、关节数量和转轴都与 G1 不同，动捕数据不能直接用。这一项做坐标系朝向、四元数分量顺序、关节命名与排列、帧率重采样的转换，产出可供后续训练使用的参考动作。视频是姿态映射回放，**不含动力学，机器人并没有真的在平衡**。

[![实践07：GMR行走重定向](practices/07_motion_retargeting/media/preview.gif)](practices/07_motion_retargeting/media/walk_human_g1.mp4)

[实践详情](practices/07_motion_retargeting/README.md)

### 实践08｜AMP拟人走跑与步态评估

**Isaac Lab · PPO + AMP 对抗式动作先验 · G1 29 自由度**

走得准靠速度跟踪奖励，走得像人靠判别器：它同时看人类参考动作和机器人动作并尽力分辨，机器人骗过它的程度折算成风格奖励。

**拟人走跑整体未通过验证。**十轮单因素实验后，拖腿与摆臂方向已修正，双支撑占比由 44% 降到 22%，但航向与左右对称性仍无法同时达标。根因分析与逐轮记录见实践页。

[![实践08：摆臂方向已修正的候选](practices/08_amp_locomotion/media/preview_armswing_fixed.gif)](practices/08_amp_locomotion/media/p8_armswing_fixed_side_20s.mp4)

[实践详情](practices/08_amp_locomotion/README.md)

### 实践09｜全身舞蹈轨迹跟踪

**MJLab · PPO · BeyondMimic 方法**

跟着一段两分钟的舞蹈走完全程。长序列里一处跟丢就会一路崩掉，所以按失败统计调整参考片段的采样概率，让练得差的片段被更多抽到。能连续跟完 131 秒，但世界坐标下位置会逐渐漂移。

[![实践09：全身舞蹈轨迹跟踪](practices/09_motion_tracking/media/preview.gif)](practices/09_motion_tracking/media/motion_tracking_30k_full.mp4)

[实践详情](practices/09_motion_tracking/README.md)

### 实践10｜平台地形动作跟踪（HOI）

**Isaac Lab · PPO · 高度扫描**

在有平台的场景里跟踪全身参考动作，机器人要与平台发生实际接触而不只是复现姿态，因此用高度扫描感知地形。

扩容训练后世界锚点 RMSE 由 0.1297 m 降到 0.0501 m，确认此前的短板是训练预算。**仍未做独立留出验证**，训练量也只有参考配置的 1.5%。

[![实践10：平台路径回放](practices/10_object_interaction/media/preview.gif)](practices/10_object_interaction/media/p10_std03_model399_full.mp4)

[实践详情](practices/10_object_interaction/README.md)

### 实践11｜深度感知与MuJoCo跨仿真验证

**Project Instinct + MuJoCo（Sim2Sim）· PPO + 运动风格奖励**

靠一个深度相机看路通过障碍。深度图维护历史队列并抽帧以模拟传感器延迟，再经缩放、裁剪、空洞修补、模糊和归一化进网络。策略导出 ONNX 后放到 MuJoCo 跨仿真运行。

补全深度处理链路后，参考策略在 MuJoCo 平地连续走完 20 秒、13.87 米，说明链路正确。**自训练的 5000 模型几乎不前进**——指令扫描后确认它在 Isaac 里本来就没学会跟踪速度（XY 速度 RMSE 0.2604 与均速 0.2572 几乎相等），不是跨仿真把它弄坏的。

[![实践11：深度感知策略MuJoCo跨仿真运行](practices/11_depth_locomotion/media/preview.gif)](practices/11_depth_locomotion/media/p11_sim2sim_course_flat_20s.mp4)

[实践详情](practices/11_depth_locomotion/README.md)

## 代表性结果

| 方向 | 已完成的结果 | 展示 |
|---|---|---|
| 速度与骨盆高度联合控制 | 连续蹲走22秒，后20秒高度MAE 0.6116cm、前向均速0.5357m/s | [详情](practices/04_velocity_height/README.md) |
| 教师学生蒸馏与全身动作 | Action与KL学生均25/25动作到达参考末尾，对齐身体平均距离2.580cm / 2.537cm | [详情](practices/06_teacher_student/README.md) |
| 自适应采样与全身轨迹跟踪 | 连续跟完131.48秒单舞蹈，对齐身体平均距离3.633cm、世界锚点0.581m | [详情](practices/09_motion_tracking/README.md) |
| 感知驱动的粗糙地形行走 | 8组同初态场景中，新策略XY速度与转速RMSE均低于旧策略 | [详情](practices/02_rough_terrain/README.md) |
| 地形感知的人物交互动作跟踪 | 扩容训练后世界锚点RMSE 0.0501m，较此前降低61% | [详情](practices/10_object_interaction/README.md) |

## 按实践浏览

每个目录统一放置成果说明、`training/` 训练记录与算法组件、`media/` 视频和结果图。

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

已完成双指令控制、两类蒸馏、长动作跟踪、地形配对和平台路径的分项验证。

未达成的三项：分层导航的随机布局训练没有显出总体优势；AMP 拟人走跑的自然摆臂与低速慢走仍未通过；自训练的深度策略迁移后几乎不前进，跨仿真部署不算成功。

各项结果保留模型、统计窗口、坐标和初态限制。预训练部署、运动学回放与自主训练分别说明，具体数值与未覆盖场景见对应实践页。
