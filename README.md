# G1人形机器人：强化学习与运动控制

本项目以 Unitree G1 为平台，围绕速度与高度联合控制、感知驱动的地形行走、分层导航、全身运动跟踪、教师学生蒸馏与深度感知运动展开，涵盖算法实现、训练配置、对照实验与仿真验证。

更新：2026-09-16。全部结果来自仿真环境或离线运动学回放，未进行实机部署。

[源代码](code/) · [视频合集](MEDIA.md) · [当前状态](PROJECT_STATUS.md) · [算法与框架来源](REFERENCES.md)

完整源代码与策略权重见 [`code/`](code/)，各子目录下的 `MODIFIED.md` 标注了自研文件与上游框架文件的分界。定量结果见各项目页。

## 项目概述

### 实践01｜仿真环境与行走策略部署

Isaac Lab（训练）+ MuJoCo（跨仿真验证）· PPO

在 Isaac Lab 中基于 PPO 训练 G1 的速度跟踪策略，并将导出策略部署至 MuJoCo，构建跨仿真验证链路。

跨仿真部署需在四个层面建立一致性：两侧仿真器的关节索引映射、网络输出到关节目标角的变换（动作缩放与默认姿态偏置）、PD 增益配置，以及策略频率与物理步长的匹配。验证分为两级：首先以同一批观测比对训练权重与导出模型的前向输出，确认序列化过程无信息损失；其次在 MuJoCo 中执行固定指令的闭环回放，考察实际行为。

局限性：尚未完成同一权重在两个仿真器下的逐帧轨迹配对比较，因此本项目仅能支持"部署链路正确"这一结论，不足以论证两侧动力学的一致性。

尚无展示视频。[项目详情](practices/01_simulation_baseline/README.md)

### 实践02｜感知驱动的粗糙地形行走

Isaac Lab · PPO · 高度扫描 · MuJoCo

引入高度扫描传感器构建地形感知通道：在机器人足端周围布置射线阵列采样地面高度，并将采样结果接入策略观测，使策略在接触地形前即可获得前方起伏信息。

实现中有两个关键约束。其一，高度值须转换至机身相对坐标，若直接使用世界系绝对高度，机体升降会引入与地形无关的整体偏移，破坏观测的平稳性。其二，射线未命中时返回非有限值，需先行掩码处理，否则将污染整个观测向量。观测配置上，策略组启用观测噪声、价值网络组不启用，二者分别定义。

在原策略基础上执行接触条件适配训练，并在若干组相同初始状态下与原策略逐一配对评估，新策略在线速度与角速度跟踪误差上均优于原策略，另完成一段台阶地形穿越。结果取自固定场景，转向跟踪仍存在欠调。

[![实践02：台阶与方块地形行走](practices/02_rough_terrain/media/preview_follow.gif)](practices/02_rough_terrain/media/p2_final_stairs_20s_follow.mp4)

[项目详情](practices/02_rough_terrain/README.md)

### 实践03｜HoST仰躺起身（预训练策略部署）

HoST 预训练策略 · PD 控制 · MuJoCo

将 HoST 预训练起身策略部署至 MuJoCo，重点在于动作空间与观测接口的正确复现。

该策略采用增量动作空间：网络输出为相对上一时刻的关节角增量而非绝对目标角，需在部署侧完成累加后送入 PD 控制器。观测侧为多帧历史拼接，关节顺序与仿真时钟推进节奏均须与原实现逐项对齐。完成接口适配后，机器人可在默认仰躺初态下完成起身并维持站立；另提供一组对照动画，展示按绝对角度接口接入时的行为差异。

局限性：采用既有预训练策略，未进行 HoST 自主训练；验证仅覆盖默认仰躺初态，未评估任意跌倒姿态下的恢复能力。

[![实践03：HoST仰躺起身与站立](practices/03_host_standup/media/preview.gif)](practices/03_host_standup/media/host_standup_60s.mp4)

[项目详情](practices/03_host_standup/README.md)

### 实践04｜速度与骨盆高度联合控制

MJLab · PPO · 双指令控制

在速度跟踪任务基础上引入骨盆高度指令，构成双指令 MDP。训练阶段对高度指令随机采样，与速度指令共同接入 Actor 观测，并配置高度跟踪奖励项，使策略获得在降低躯干高度状态下持续行进的能力。除固定指令外，另评估三组动态指令切换场景。

为确认性能增益来源于该指令通道而非训练时长，设计两组同预算消融：分别移除高度跟踪奖励项与 Actor 侧高度指令观测，其余配置保持一致。两组均无法按指令调节骨盆高度，支持了上述归因。

局限性：结果为单随机种子的平地条件，航向与横向位置仍存在漂移。后续引入的航向反馈改善了角度跟踪，但未构成横向位置闭环。

[![实践04：速度与骨盆高度联合控制](practices/04_velocity_height/media/preview.gif)](practices/04_velocity_height/media/velocity_height_20k_22s.mp4)

[项目详情](practices/04_velocity_height/README.md)

### 实践05｜分层强化学习导航

Isaac Lab · 高层 PPO + 冻结的低层行走策略

采用分层架构：冻结已训练的低层行走策略作为执行层，仅训练高层网络。高层以目标点的机体系相对位姿为输入，输出速度指令；低层负责具体步态生成。

层间接口是该架构的关键环节。低层观测分组须从其自身的训练配置中深拷贝获得，手工重建虽在结构上等价，但噪声配置与历史长度等细节易出现偏差，导致低层接收到的输入分布偏离其训练分布。

本项目旨在检验"随机布局训练可提升高层泛化能力"这一假设，采用两模型、两类场景族、三随机种子的成对评估设计，场景清单支持复读。评估结果显示两者的到达成功局数基本相当，未获得支持该假设的证据。本项目据此记为未达成预期。

[![实践05：分层导航目标跟随](practices/05_hierarchical_navigation/media/preview.gif)](practices/05_hierarchical_navigation/media/navigation_baseline_play.mp4)

[项目详情](practices/05_hierarchical_navigation/README.md)

### 实践06｜教师学生蒸馏：Action与KL对比

MJLab · PPO · Action Matching / KL Matching

仿真环境可提供实机不可观测的特权信息（如机体真实线速度）。教师策略利用该类信息可获得更优性能但无法部署，故需训练仅依赖实机可得观测的学生策略进行能力迁移。

本项目在同一训练预算下对比两种蒸馏目标：一为直接回归教师输出动作的 Action Matching，二为最小化师生输出分布间对角高斯 KL 散度的 KL Matching。实现中需处理梯度的边界条件，并对蒸馏系数施加训练过程中的退火调度。两种学生均可完整跟踪动作库全部条目，对齐坐标系下的姿态误差接近；在世界坐标系下，KL Matching 的表现略逊。

局限性：评估动作全部取自训练动作库，非留出集合，因此该结果不构成泛化能力验证。两种方案的退火调度亦不完全一致。

[![实践06：Action与KL蒸馏并排对比](practices/06_teacher_student/media/preview.gif)](practices/06_teacher_student/media/p6_action_kl_3clips_23s.mp4)

[项目详情](practices/06_teacher_student/README.md)

### 实践07｜人体到G1的运动重定向

GMR · SMPL-X · 逆运动学，不含强化学习

人体与 G1 在骨架比例、关节数量与转轴定义上存在系统性差异，动作捕捉数据无法直接驱动机器人。本项目完成数据格式与坐标系的映射：世界系与根节点系的朝向约定、四元数分量顺序、关节与刚体的命名及排列、帧率重采样，以及运动库的存储格式规范。数据覆盖行走、加速跑与右转三类。

为验证下游可用性，将产出文件接入实践 08 的 AMP 数据加载器进行读取测试，确认特征顺序与关节排列无错位。

局限性：视频为直接写入关节角的运动学回放，不经力矩控制环节，因此不能作为动态平衡能力的证据。片段时长较短，足地接触关系仍存在质量问题。

[![实践07：GMR行走重定向](practices/07_motion_retargeting/media/preview.gif)](practices/07_motion_retargeting/media/walk_human_g1.mp4)

[项目详情](practices/07_motion_retargeting/README.md)

### 实践08｜AMP拟人走跑与步态评估

Isaac Lab · PPO + AMP 对抗式动作先验 · G1 29 自由度

仅以速度跟踪奖励训练所得策略虽能满足指令跟踪，但步态存在僵硬、滑步与摆臂异常等问题。AMP 引入判别器对策略运动与人类参考运动进行区分，并将判别器得分转换为有界的风格奖励，与任务奖励按权重混合，从而在不手工设计步态约束的前提下约束动作风格。

诊断阶段定位到判别器在稳定性保护机制下被持续暂停，单轮训练中实际优化步数极少，风格奖励长期处于饱和状态，为拖腿步态的直接成因；解除该限制后摆腿抬升显著提升。此后开展十轮单因素对照实验，各轮判定门槛均于训练启动前写入冻结方案。其中两项改动被证实有效：提高判别器梯度惩罚系数后，肩关节与同侧髋关节由同相摆动转为反相摆动，并在更换随机种子后复现；引入直接惩罚双支撑时长的奖励项后，双支撑占比由四成余降至两成余，同时单脚摆动次数不降反升，排除了通过延长单脚支撑期获取奖励的套利路径，该防套利判据于训练前预先登记。

局限性：冻结门槛要求慢速与常速两个场景同时达标，当前无模型满足。主要矛盾在于航向跟踪与左右对称性的耦合——降低双支撑占比将延长单脚支撑期并导致航向退化，提高航向权重则使双支撑占比回升。本项目整体未通过验收。

[![实践08：摆臂方向已修正的候选](practices/08_amp_locomotion/media/preview_armswing_fixed.gif)](practices/08_amp_locomotion/media/p8_armswing_fixed_side_20s.mp4)

[项目详情](practices/08_amp_locomotion/README.md)

### 实践09｜全身舞蹈轨迹跟踪

MJLab · PPO · BeyondMimic 方法

针对时长约两分钟的舞蹈序列进行全身运动跟踪。长序列跟踪的主要困难在于误差累积：任一片段跟踪失败将导致后续整段发散。本项目采用基于失败统计的自适应采样，提高失败率较高片段的抽样概率，并同步处理参考相位推进与参考坐标对齐。

动作接口采用参考关节角叠加策略残差的形式，使局部姿态误差与世界轨迹误差可分别度量。对照组采用零残差的纯参考 PD 控制，在极短时间内即触发终止条件。

局限性：姿态跟踪可维持全程，但世界坐标系下存在持续位置漂移；未评估未见动作与外部扰动下的泛化性能。

[![实践09：全身舞蹈轨迹跟踪](practices/09_motion_tracking/media/preview.gif)](practices/09_motion_tracking/media/motion_tracking_30k_full.mp4)

[项目详情](practices/09_motion_tracking/README.md)

### 实践10｜平台地形动作跟踪（HOI）

Isaac Lab · PPO · 高度扫描

在含平台的场景中执行全身运动跟踪。与纯运动跟踪的区别在于机器人须与平台产生实际接触，而非仅复现参考姿态，因此引入高度扫描通道，射线需与地形描述中的 box primitive 求交。另比较了世界锚点奖励的两种尺度设置。

此前的主要约束是训练规模，所选模型的训练量远低于参考配置，不足以支撑方法层面的结论。后续补充扩容训练，并行环境数与更新次数同步提升，评估存档于训练启动前按等间隔预先登记，以规避事后择优。扩容后跟踪误差显著下降，且世界位置、朝向与关节三项误差同步改善，从而将"训练预算为主要约束"由推测确认为结论。

局限性：受本机显存限制，并行环境数仍低于参考配置；所选模型取自同一批候选，未设置独立留出验证。

[![实践10：平台路径回放](practices/10_object_interaction/media/preview.gif)](practices/10_object_interaction/media/p10_std03_model399_full.mp4)

[项目详情](practices/10_object_interaction/README.md)

### 实践11｜深度感知与MuJoCo跨仿真验证

Project Instinct + MuJoCo（Sim2Sim）· PPO + 运动风格奖励

基于深度相机的视觉感知运动策略，并完成 MuJoCo 跨仿真验证。深度观测不直接取用最新帧，而是维护历史队列并从中抽帧，以模拟真实传感器的采集延迟；图像预处理依次包含缩放、按边距裁剪、空洞修补、高斯模糊与线性归一化。该实现严格对照训练侧定义编写，并以单元测试对输出形状与数值映射逐项断言。

策略导出为 ONNX 后接入 MuJoCo，录制四组对照记录，覆盖参考策略与自训练策略在平地与粗糙地形下的表现。参考策略在平地可完成全程行进且骨盆高度稳定，验证了预处理链路与执行链路的正确性。

自训练策略在两种地形下均近乎静止。该现象最初被归因于跨仿真迁移损失，经指令扫描后修正：该策略在 Isaac 原生评估中的速度跟踪误差与平均速度量级相当，表明其在训练环境中即未习得速度跟踪能力。此前公开表述的"迁移落差约 28%"不成立，已作更正。

[![实践11：深度感知策略MuJoCo跨仿真运行](practices/11_depth_locomotion/media/preview.gif)](practices/11_depth_locomotion/media/p11_sim2sim_course_flat_20s.mp4)

[项目详情](practices/11_depth_locomotion/README.md)

## 项目一览

| 编号 | 内容 | 主要技术 | 状态 |
|---|---|---|---|
| 01 | [仿真环境与行走策略部署](practices/01_simulation_baseline/README.md) | Isaac Lab · PPO · MuJoCo | 链路通过 |
| 02 | [感知驱动的粗糙地形行走](practices/02_rough_terrain/README.md) | Isaac Lab · PPO · 高度扫描 · MuJoCo | 优于原策略 |
| 03 | [HoST起身与增量动作部署](practices/03_host_standup/README.md) | HoST · PD · MuJoCo | 部署复现 |
| 04 | [速度与骨盆高度联合控制](practices/04_velocity_height/README.md) | MJLab · PPO · 双指令控制 | 达成，含消融 |
| 05 | [分层强化学习导航](practices/05_hierarchical_navigation/README.md) | Isaac Lab · 高层PPO · 冻结低层策略 | 未达成预期 |
| 06 | [教师学生蒸馏与全身动作](practices/06_teacher_student/README.md) | MJLab · PPO · Action / KL Matching | 达成，非泛化验证 |
| 07 | [人体到G1的运动重定向](practices/07_motion_retargeting/README.md) | GMR · SMPL-X · 逆运动学 | 运动学回放 |
| 08 | [AMP拟人运动与风格奖励](practices/08_amp_locomotion/README.md) | Isaac Lab · PPO · AMP | 未通过验收 |
| 09 | [自适应采样与全身轨迹跟踪](practices/09_motion_tracking/README.md) | BeyondMimic方法 · MJLab · PPO | 达成，有世界漂移 |
| 10 | [地形感知的人物交互动作跟踪](practices/10_object_interaction/README.md) | Isaac Lab · PPO · 高度扫描 | 扩容后改善 |
| 11 | [深度感知运动与策略导出](practices/11_depth_locomotion/README.md) | Project Instinct · 深度历史 · PPO/风格奖励 | 自训练策略不足 |

三项未达成目标，按实际结果记录：分层导航中随机布局训练未显示出总体优势；AMP 拟人走跑的自然摆臂与低速慢走未能同时通过冻结门槛；自训练深度策略在跨仿真部署后近乎静止。

各项结果均保留了模型阶段、统计窗口、坐标系与初态限制。预训练策略部署、运动学回放与自主训练三类工作分别说明，未合并表述。
