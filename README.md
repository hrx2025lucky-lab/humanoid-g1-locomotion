# G1人形机器人：强化学习与运动控制

用 Unitree G1 做的一组运动控制项目：速度与高度控制、地形行走、分层导航、全身动作跟踪、教师学生蒸馏、深度感知。仓库里有完整代码、训练配置和仿真视频。

更新：2026-09-16。所有结果都来自仿真或离线回放，没有上过真机。

[源代码](code/) · [视频合集](MEDIA.md) · [当前状态](PROJECT_STATUS.md) · [算法与框架来源](REFERENCES.md)

代码在 [`code/`](code/)，每个目录里的 `MODIFIED.md` 写明哪些文件是我写的、哪些来自上游框架。具体数值在各项目页里。

## 做了什么

### 实践01｜仿真环境与行走策略部署

Isaac Lab（训练）+ MuJoCo（跨仿真验证）· PPO

在 Isaac Lab 上训好 G1 的速度跟踪策略，导出后接进 MuJoCo 跑起来。对齐了关节顺序、动作缩放、PD 参数和控制频率，核对过导出前后的输出一致，并跑了一段闭环回放。

两个仿真器的逐帧轨迹比对还没做。

尚无展示视频。[项目详情](practices/01_simulation_baseline/README.md)

### 实践02｜感知驱动的粗糙地形行走

Isaac Lab · PPO · 高度扫描 · MuJoCo

给机器人加了地形感知：脚下布一圈射线量地面高度，接进观测，让它上台阶前就知道前面的起伏。做了接触条件适配训练，和旧策略在相同初态下配对比较，速度和转速都更准。

[![实践02：台阶与方块地形行走](practices/02_rough_terrain/media/preview_follow.gif)](practices/02_rough_terrain/media/p2_final_stairs_20s_follow.mp4)

[项目详情](practices/02_rough_terrain/README.md)

### 实践03｜HoST仰躺起身（预训练策略部署）

HoST 预训练策略 · PD 控制 · MuJoCo

把 HoST 的起身策略接进 MuJoCo。它的动作是增量式的，要自己累加成关节角再送 PD，观测是多帧历史拼接。接好后机器人能从仰躺自己起身站住。

用的是现成的预训练策略，没有自己训练，也只测了默认仰躺这一种初态。

[![实践03：HoST仰躺起身与站立](practices/03_host_standup/media/preview.gif)](practices/03_host_standup/media/host_standup_60s.mp4)

[项目详情](practices/03_host_standup/README.md)

### 实践04｜速度与骨盆高度联合控制

MJLab · PPO · 双指令控制

在速度指令之外加了骨盆高度指令，两个一起给，机器人能压低身子往前走。另外做了两组消融，分别去掉高度奖励和 Actor 的高度观测，确认效果来自这条链路而不是训练时长。

[![实践04：速度与骨盆高度联合控制](practices/04_velocity_height/media/preview.gif)](practices/04_velocity_height/media/velocity_height_20k_22s.mp4)

[项目详情](practices/04_velocity_height/README.md)

### 实践05｜分层强化学习导航

Isaac Lab · 高层 PPO + 冻结的低层行走策略

把训好的行走策略冻住当执行层，只训上层：上层看目标点输出速度指令，下层负责迈腿。用两个模型、两类场景、三个种子做了成对评估。

原本想验证随机布局训练的上层泛化更好，结果两者成功率基本一样，没有得到支持。

[![实践05：分层导航目标跟随](practices/05_hierarchical_navigation/media/preview.gif)](practices/05_hierarchical_navigation/media/navigation_baseline_play.mp4)

[项目详情](practices/05_hierarchical_navigation/README.md)

### 实践06｜教师学生蒸馏：Action与KL对比

MJLab · PPO · Action Matching / KL Matching

教师用仿真里的特权信息训练，学生只用真机能拿到的观测。在同样预算下比了两种蒸馏方式：直接回归教师动作，和匹配两者输出分布的 KL 散度。两个学生都能跟完整个动作库。

测试动作全部来自训练库，不是没见过的动作，所以这不算泛化验证。

[![实践06：Action与KL蒸馏并排对比](practices/06_teacher_student/media/preview.gif)](practices/06_teacher_student/media/p6_action_kl_3clips_23s.mp4)

[项目详情](practices/06_teacher_student/README.md)

### 实践07｜人体到G1的运动重定向

GMR · SMPL-X · 逆运动学，不含强化学习

把人体动捕数据转成 G1 能用的参考动作：对齐坐标系朝向、四元数分量顺序、关节命名与排列，重采样帧率。覆盖行走、加速跑、右转三类，产出的文件能被实践 08 的数据加载器直接读。

视频是把关节角直接写进仿真的回放，不经过力矩控制，说明不了平衡能力。

[![实践07：GMR行走重定向](practices/07_motion_retargeting/media/preview.gif)](practices/07_motion_retargeting/media/walk_human_g1.mp4)

[项目详情](practices/07_motion_retargeting/README.md)

### 实践08｜AMP拟人走跑与步态评估

Isaac Lab · PPO + AMP 对抗式动作先验 · G1 29 自由度

在速度奖励之外加了 AMP 判别器，用人类参考动作约束步态风格。做了十轮单因素实验，判定标准都在开训前写死。解决了拖腿和左右不协调，修正了摆臂方向，双支撑占比也降进了正常区间。

整体还没通过验收：门槛要求慢走和常速前进同时达标，目前航向和左右对称性无法兼顾。

[![实践08：摆臂方向已修正的候选](practices/08_amp_locomotion/media/preview_armswing_fixed.gif)](practices/08_amp_locomotion/media/p8_armswing_fixed_side_20s.mp4)

[项目详情](practices/08_amp_locomotion/README.md)

### 实践09｜全身舞蹈轨迹跟踪

MJLab · PPO · BeyondMimic 方法

让机器人跟完一段两分钟的舞蹈。按失败统计调整片段采样概率，练得差的多练；动作接口用参考角度加策略修正量。对照组用纯参考 PD，撑不过一秒。

姿态跟住了，但世界坐标下位置会慢慢漂。

[![实践09：全身舞蹈轨迹跟踪](practices/09_motion_tracking/media/preview.gif)](practices/09_motion_tracking/media/motion_tracking_30k_full.mp4)

[项目详情](practices/09_motion_tracking/README.md)

### 实践10｜平台地形动作跟踪（HOI）

Isaac Lab · PPO · 高度扫描

在带平台的场景里跟踪全身动作，机器人要真的踩到平台上，所以加了地形感知。后来补做扩容训练，误差明显下降，而且位置、朝向、关节三项一起降，确认之前的短板就是训练量不够。

评估存档在开训前就定好了，但模型仍是从同一批候选里挑的，没有独立留出验证。

[![实践10：平台路径回放](practices/10_object_interaction/media/preview.gif)](practices/10_object_interaction/media/p10_std03_model399_full.mp4)

[项目详情](practices/10_object_interaction/README.md)

### 实践11｜深度感知与MuJoCo跨仿真验证

Project Instinct + MuJoCo（Sim2Sim）· PPO + 运动风格奖励

机器人靠深度相机看路。补全了深度图的处理链路（历史队列抽帧、缩放裁剪、补空洞、模糊、归一化），导出 ONNX 后放进 MuJoCo，录了四组对照。参考策略能在平地走完全程，说明链路是对的。

我自己训的策略几乎不动。原以为是跨仿真损失，扫描指令后发现它在 Isaac 里本来就没学会跟踪速度，之前公开写的「迁移落差约 28%」已更正。

[![实践11：深度感知策略MuJoCo跨仿真运行](practices/11_depth_locomotion/media/preview.gif)](practices/11_depth_locomotion/media/p11_sim2sim_course_flat_20s.mp4)

[项目详情](practices/11_depth_locomotion/README.md)

## 项目一览

| 编号 | 内容 | 主要技术 | 状态 |
|---|---|---|---|
| 01 | [仿真环境与行走策略部署](practices/01_simulation_baseline/README.md) | Isaac Lab · PPO · MuJoCo | 链路通过 |
| 02 | [感知驱动的粗糙地形行走](practices/02_rough_terrain/README.md) | Isaac Lab · PPO · 高度扫描 · MuJoCo | 优于旧策略 |
| 03 | [HoST起身与增量动作部署](practices/03_host_standup/README.md) | HoST · PD · MuJoCo | 部署复现 |
| 04 | [速度与骨盆高度联合控制](practices/04_velocity_height/README.md) | MJLab · PPO · 双指令控制 | 达成，含消融 |
| 05 | [分层强化学习导航](practices/05_hierarchical_navigation/README.md) | Isaac Lab · 高层PPO · 冻结低层策略 | 未达成预期 |
| 06 | [教师学生蒸馏与全身动作](practices/06_teacher_student/README.md) | MJLab · PPO · Action / KL Matching | 达成，非泛化验证 |
| 07 | [人体到G1的运动重定向](practices/07_motion_retargeting/README.md) | GMR · SMPL-X · 逆运动学 | 运动学回放 |
| 08 | [AMP拟人运动与风格奖励](practices/08_amp_locomotion/README.md) | Isaac Lab · PPO · AMP | 未通过验收 |
| 09 | [自适应采样与全身轨迹跟踪](practices/09_motion_tracking/README.md) | BeyondMimic方法 · MJLab · PPO | 达成，有世界漂移 |
| 10 | [地形感知的人物交互动作跟踪](practices/10_object_interaction/README.md) | Isaac Lab · PPO · 高度扫描 | 扩容后改善 |
| 11 | [深度感知运动与策略导出](practices/11_depth_locomotion/README.md) | Project Instinct · 深度历史 · PPO/风格奖励 | 自训练策略不足 |

三项没做成，按原样记在这里：分层导航的随机布局训练没显出优势；AMP 的自然摆臂和低速慢走过不了同一道门槛；自己训的深度策略搬到 MuJoCo 后几乎不动。

每项都留着当时的模型阶段、统计窗口和初始状态限制，预训练部署、运动学回放和自主训练分开写。
