# G1人形机器人：强化学习与运动控制

用 Unitree G1 做的一组运动控制项目：速度与高度控制、地形行走、分层导航、全身动作跟踪、教师学生蒸馏、深度感知。仓库里有完整代码、训练配置和仿真视频。

更新：2026-09-16。所有结果都来自仿真或离线回放，没有上过真机。

[源代码](code/) · [视频合集](MEDIA.md) · [当前状态](PROJECT_STATUS.md) · [算法与框架来源](REFERENCES.md)

代码在 [`code/`](code/)，每个目录里的 `MODIFIED.md` 写明哪些文件是我写的、哪些来自上游框架。具体数值在各项目页里。

## 做了什么

### 实践01｜仿真环境与行走策略部署

Isaac Lab（训练）+ MuJoCo（跨仿真验证）· PPO

在 Isaac Lab 上用 PPO 训 G1 的速度跟踪策略，导出后接进 MuJoCo 跑起来。跨仿真这一步要对齐四样东西：两边的关节排列顺序、网络输出到关节角的换算（乘动作缩放再加默认姿势）、PD 增益、策略与物理的频率。

验证分两步：先用同一批观测比对训练权重和导出模型的输出，确认导出没丢东西；再在 MuJoCo 里跑固定指令的闭环回放看行为。

同一份权重在两个仿真器里逐帧比轨迹这一步还没做，所以只能说链路通了。

尚无展示视频。[项目详情](practices/01_simulation_baseline/README.md)

### 实践02｜感知驱动的粗糙地形行走

Isaac Lab · PPO · 高度扫描 · MuJoCo

给机器人加地形感知：脚下布一圈射线量地面高度，接进观测，让它上台阶前就知道前面的起伏。

实现上有两处要注意。高度要用相对机身的值，用绝对高度的话机器人一抬腿整张高度图就跟着漂；射线打空时返回的不是有效数值，得先筛掉再进网络。观测这边策略侧加噪、价值网络不加噪，两条路分开配。

做了接触条件适配训练，和旧策略在若干组相同初态下逐一配对，新策略的速度和转速都更准，也走通了一段台阶。结果来自固定场景，转向还欠跟踪。

[![实践02：台阶与方块地形行走](practices/02_rough_terrain/media/preview_follow.gif)](practices/02_rough_terrain/media/p2_final_stairs_20s_follow.mp4)

[项目详情](practices/02_rough_terrain/README.md)

### 实践03｜HoST仰躺起身（预训练策略部署）

HoST 预训练策略 · PD 控制 · MuJoCo

把 HoST 的起身策略接进 MuJoCo。它的动作空间是增量式的：网络输出的不是关节目标角，而是相对上一帧的增量，要自己累加再送 PD。观测是多帧历史拼成的一长串，关节顺序和仿真时钟的推进节奏都得逐项核对。

接好后机器人能从仰躺自己翻起来站住，另外录了一段对比动画，展示按普通绝对角度接口去接会是什么结果。

用的是现成的预训练策略，没有自己训练，也只测了默认仰躺这一种初态。

[![实践03：HoST仰躺起身与站立](practices/03_host_standup/media/preview.gif)](practices/03_host_standup/media/host_standup_60s.mp4)

[项目详情](practices/03_host_standup/README.md)

### 实践04｜速度与骨盆高度联合控制

MJLab · PPO · 双指令控制

在速度指令之外加了骨盆高度指令。训练时高度随机采样，和速度一起拼进 Actor 观测，再配一项高度跟踪奖励，机器人就能压低身子往前走。除固定指令外还测了三组动态指令切换。

为了确认效果来自这条链路而不是训练时长，做了两组同预算消融：一组去掉高度跟踪奖励，一组去掉 Actor 的高度观测，其余完全相同。两组都做不到按指令改高度。

这是单种子的平地结果，朝向和横向都还会漂。后来加的朝向反馈稳住了角度，横向位置没有做闭环。

[![实践04：速度与骨盆高度联合控制](practices/04_velocity_height/media/preview.gif)](practices/04_velocity_height/media/velocity_height_20k_22s.mp4)

[项目详情](practices/04_velocity_height/README.md)

### 实践05｜分层强化学习导航

Isaac Lab · 高层 PPO + 冻结的低层行走策略

把训好的行走策略冻住当执行层，只训上层。上层看目标点的相对位姿，输出速度指令，下层负责怎么迈腿。层间接口是关键：下层的观测分组必须从它自己的训练配置里深拷贝过来，手写重建会让噪声设置和历史长度对不上。

原本想验证随机布局训练的上层泛化更好。用两个模型、两类场景、三个随机种子做了成对评估，场景清单可复读。

结果两者的成功局数基本一样，没有得到支持。这一项按原样记为未达成预期。

[![实践05：分层导航目标跟随](practices/05_hierarchical_navigation/media/preview.gif)](practices/05_hierarchical_navigation/media/navigation_baseline_play.mp4)

[项目详情](practices/05_hierarchical_navigation/README.md)

### 实践06｜教师学生蒸馏：Action与KL对比

MJLab · PPO · Action Matching / KL Matching

仿真里有些信息真机上拿不到，比如机身真实速度。教师可以用这些特权信息学得更好，但没法部署，所以要再训一个只看真机可得观测的学生去模仿它。

在同样预算下比了两种模仿方式：一种直接回归教师输出的动作，一种匹配两者输出分布之间的对角高斯 KL 散度。实现时要处理梯度的边界条件，蒸馏系数还要随训练退火。两个学生都能把动作库跟完，姿态误差接近，放到世界坐标下看 KL 那个反而略差。

测试动作全部来自训练库，不是没见过的动作，所以这不算泛化验证。

[![实践06：Action与KL蒸馏并排对比](practices/06_teacher_student/media/preview.gif)](practices/06_teacher_student/media/p6_action_kl_3clips_23s.mp4)

[项目详情](practices/06_teacher_student/README.md)

### 实践07｜人体到G1的运动重定向

GMR · SMPL-X · 逆运动学，不含强化学习

人的骨架比例、关节数量和转轴都和 G1 不同，动捕数据不能直接用。这一项做格式和坐标的转换：世界坐标与根节点坐标的朝向约定、四元数的分量顺序、关节与身体的命名排列、帧率重采样，以及运动库的存储格式。覆盖行走、加速跑、右转三类。

做完之后反过来验了下游：把产出文件丢给实践 08 的 AMP 数据加载器，确认能直接读进去，顺序没错位。

视频是把关节角直接写进仿真的回放，不经过力矩控制，说明不了平衡能力。片段偏短，脚地接触还有质量问题。

[![实践07：GMR行走重定向](practices/07_motion_retargeting/media/preview.gif)](practices/07_motion_retargeting/media/walk_human_g1.mp4)

[项目详情](practices/07_motion_retargeting/README.md)

### 实践08｜AMP拟人走跑与步态评估

Isaac Lab · PPO + AMP 对抗式动作先验 · G1 29 自由度

只用速度奖励训出来的步态很怪：僵硬、滑步、手臂乱甩。AMP 的做法是加一个判别器，让它同时看人类参考动作和机器人动作并努力分辨，机器人骗过它的程度折算成风格奖励，和任务奖励按权重混合。

一开始机器人拖着腿走，查下来是判别器被稳定性保护全程掐住了，整轮训练里真正更新的次数只有个位数。解除之后抬脚高度立刻上来。后面做了十轮单因素实验，每轮的判定标准都在开训前写死存档。有效的改动是两个：提高判别器的梯度惩罚，摆臂从同手同脚变成正常的反向摆动，换种子能复现；加一项直接惩罚双支撑时长的奖励，占比从四成多降到两成出头，同时每只脚的摆动次数不降反升，说明不是靠单腿站着凑出来的。

整体还没通过验收。门槛要求慢走和常速前进两个场景同时达标，目前航向和左右对称性无法兼顾：双支撑压下去航向就歪，航向提上来双支撑又涨回去。

[![实践08：摆臂方向已修正的候选](practices/08_amp_locomotion/media/preview_armswing_fixed.gif)](practices/08_amp_locomotion/media/p8_armswing_fixed_side_20s.mp4)

[项目详情](practices/08_amp_locomotion/README.md)

### 实践09｜全身舞蹈轨迹跟踪

MJLab · PPO · BeyondMimic 方法

让机器人跟完一段两分钟的舞蹈。长动作的麻烦是中间跟丢一处后面就一路崩，所以采样概率按失败统计调整，练得差的片段多抽几次；相位推进和参考坐标对齐也要一并处理。

动作接口用参考关节角加策略修正量，这样能把姿态误差和世界轨迹误差分开看。对照组用纯参考 PD、不给修正量，撑不过一秒就倒。

姿态是跟住了，但世界坐标下位置会慢慢漂。没见过的动作和外力扰动都没测。

[![实践09：全身舞蹈轨迹跟踪](practices/09_motion_tracking/media/preview.gif)](practices/09_motion_tracking/media/motion_tracking_30k_full.mp4)

[项目详情](practices/09_motion_tracking/README.md)

### 实践10｜平台地形动作跟踪（HOI）

Isaac Lab · PPO · 高度扫描

在带平台的场景里跟踪全身动作。和纯动作跟踪的区别是机器人要真的踩到平台上，而不是在空中把姿势比划对，所以加了地形感知，射线要和场景描述里的方块求交。另外比较了世界锚点奖励的两种尺度。

之前最大的问题是训练量，所选模型的规模离参考配置差得很远。后来补做扩容训练，环境数和更新次数都提上去，评估存档在开训前就按固定间隔定好，避免看完结果再挑。扩容后误差明显下降，而且世界位置、朝向、关节三项是一起降的，这才能说明之前就是训练量不够。

受本机显存限制，环境数仍低于参考配置。模型也是从同一批候选里挑的，没有独立留出验证。

[![实践10：平台路径回放](practices/10_object_interaction/media/preview.gif)](practices/10_object_interaction/media/p10_std03_model399_full.mp4)

[项目详情](practices/10_object_interaction/README.md)

### 实践11｜深度感知与MuJoCo跨仿真验证

Project Instinct + MuJoCo（Sim2Sim）· PPO + 运动风格奖励

机器人靠深度相机看路过障碍。深度图不直接用最新一帧，而是维护历史队列再抽帧，模拟真实相机的延迟；图像要走完缩放、按边距裁剪、补空洞、高斯模糊、线性归一化这一整套。我照着训练侧的定义补全，并写了单元测试卡住输出形状和数值映射。

策略导出成 ONNX 放进 MuJoCo，录了四组对照：参考策略和我自己训的，各跑平地和粗糙地形。参考策略在平地走完全程、骨盆高度稳定，说明整条链路是对的。

我自己训的策略两种地形都几乎不动。原以为是跨仿真损失，扫描指令后发现它在 Isaac 里本来就没学会跟踪速度，之前公开写的「迁移落差约 28%」已更正。

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
