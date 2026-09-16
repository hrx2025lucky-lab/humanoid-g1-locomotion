# G1人形机器人：强化学习与运动控制

基于Unitree G1开展速度与高度控制、地形行走、分层导航、全身动作跟踪、教师学生蒸馏和深度感知运动。项目包含算法组件、训练配置、量化实验和仿真视频。

**更新：2026-09-16。全部结果来自仿真或离线运动学验证。**

[源代码](code/) · [视频合集](MEDIA.md) · [当前状态](PROJECT_STATUS.md) · [算法与框架来源](REFERENCES.md)

## 源代码

11 个实践的完整源代码与最终策略权重在 [`code/`](code/)。每个实践目录下的 `MODIFIED.md` 逐文件列出哪些是我写的、哪些来自上游框架。

## 各实践概览

每项给出一段代表视频，点击动图打开完整 MP4。全部 32 段视频及回放条件见[视频合集](MEDIA.md)，完整数据、模型阶段与验证范围见各实践页。

### 实践01｜仿真环境与行走策略部署

**Isaac Lab（训练）+ MuJoCo（跨仿真验证）· PPO**

把训练好的策略搬到另一个仿真器要对齐一串东西：两边关节排列顺序不同需要建映射；策略输出的是归一化动作，要乘尺度再加默认关节角才是目标角度；PD 增益和 50Hz 控制周期也必须一致。用 16 组观测核对训练权重与导出策略的输出等价，再做 60 秒固定指令闭环回放。

**验证范围：**同一权重的 Isaac Lab 与 MuJoCo 成对行为比较尚未完成。

尚无展示视频。[实践详情](practices/01_simulation_baseline/README.md)

### 实践02｜感知驱动的粗糙地形行走

**Isaac Lab · PPO · 高度扫描 · MuJoCo**

在机器人脚下布一圈射线量地面高度，把 187 维高度值接进观测，它才能在踩上台阶前知道前方高低。有两处必须处理：高度要换算成相对机身的值，否则一抬腿整张高度图都会跟着变；射线打空返回的非有限值直接进网络会污染整个观测。策略侧加噪、价值网络不加噪，分别配置。

新增 15,000 次更新后，与旧策略在 8 组相同初态、各 14 秒的场景中对照，XY 速度与转速 RMSE 均更低。**验证范围：**固定场景结果，仍存在转向欠跟踪与侧向漂移；适配同时包含额外训练，不能单独归因于某一项改动。

[![实践02：台阶与方块地形行走](practices/02_rough_terrain/media/preview_follow.gif)](practices/02_rough_terrain/media/p2_final_stairs_20s_follow.mp4)

[实践详情](practices/02_rough_terrain/README.md)

### 实践03｜HoST仰躺起身（预训练策略部署）

**HoST 预训练策略 · PD 控制 · MuJoCo**

用 HoST 已训练好的起身策略，重点不在训练而在把接口接对：它输出的是关节角增量而非绝对角度，需要自行累加；观测是 456 维的多帧历史拼接，23 个关节的顺序和仿真时钟都要逐项核对。接好后在 G1 上从仰躺完成起身到站立，录 60 秒回放。

**验证范围：**使用已有预训练策略，未做 HoST 自主训练；只验了默认仰躺初态，未验证任意跌倒姿态恢复。

[![实践03：HoST仰躺起身与站立](practices/03_host_standup/media/preview.gif)](practices/03_host_standup/media/host_standup_60s.mp4)

[实践详情](practices/03_host_standup/README.md)

### 实践04｜速度与骨盆高度联合控制

**MJLab · PPO · 双指令控制**

让机器人同时听两个指令：走多快、骨盆保持多高。高度指令在训练中随机采样后拼进 100 维 Actor 观测，再配一项高度跟踪奖励，输出 29 维关节动作。主模型训练 20,000 次更新，另做移除高度奖励、移除 Actor 高度观测两组同预算 3,000 次消融，确认效果确实来自这条链路。

高度 0.5m、前向 0.5m/s 指令下连续蹲走 22 秒，后 20 秒高度 MAE 0.6116cm、前向均速 0.5357m/s，三组动态指令切换均完成 24 秒窗口。**验证范围：**单种子平地结果，仍有朝向与横向漂移；朝向反馈改善了角度但没做横向位置闭环。

[![实践04：速度与骨盆高度联合控制](practices/04_velocity_height/media/preview.gif)](practices/04_velocity_height/media/velocity_height_20k_22s.mp4)

[实践详情](practices/04_velocity_height/README.md)

### 实践05｜分层强化学习导航

**Isaac Lab · 高层 PPO + 冻结的低层行走策略**

把训好的行走策略冻住当执行层，只训一个高层网络：它看目标点输出 3 维速度指令，低层负责怎么迈腿。低层的观测契约必须从它自己的训练配置里深拷贝，手写重建会错。原本假设随机布局训练的高层泛化更好，做了 2 模型 × 2 场景族 × 3 种子 × 128 局共 1,536 局配对。

随机场景下固定布局训练模型 376/384、随机布局训练模型 374/384，**没有看出随机布局的优势，按实际结果记为未达成预期**。验证范围：成功条件含距离缓存边界，128 环境短步进不代表 4096 环境或完整导航验证。

[![实践05：分层导航目标跟随](practices/05_hierarchical_navigation/media/preview.gif)](practices/05_hierarchical_navigation/media/navigation_baseline_play.mp4)

[实践详情](practices/05_hierarchical_navigation/README.md)

### 实践06｜教师学生蒸馏：Action与KL对比

**MJLab · PPO · Action Matching / KL Matching**

教师能看到仿真里的特权信息，学生只能看到真实可得的观测，要把能力转移过去。对比两种做法：直接回归教师输出的动作，或匹配两者输出分布的对角高斯 KL 散度。两种学生各训 20,000 次更新，对 25 条动作逐条复位、等权统计。

两者均 25/25 动作到达参考末尾，对齐身体平均距离 2.580cm（Action）与 2.537cm（KL），世界身体平均距离 10.90cm 与 12.86cm。**验证范围：**评估动作来自训练库，**不是留出动作的泛化验证**；两种方案的退火日程也不同，KL 并非在所有指标上更优。

[![实践06：Action与KL蒸馏并排对比](practices/06_teacher_student/media/preview.gif)](practices/06_teacher_student/media/p6_action_kl_3clips_23s.mp4)

[实践详情](practices/06_teacher_student/README.md)

### 实践07｜人体到G1的运动重定向

**GMR · SMPL-X · 逆运动学，不含强化学习**

人的骨架比例、关节数量和转轴都与 G1 不同，动捕数据不能直接喂给机器人。这一项做格式与坐标的转换：世界/根节点坐标系朝向、wxyz 四元数分量顺序、关节与身体的命名排列、帧率重采样，产出可供后续训练使用的参考动作，并检查了 AMP 加载链路。覆盖行走、加速跑、右转三类。

**验证范围：**运动学回放直接驱动姿态，**不含动力学，机器人并没有真的在平衡**；片段较短，脚地关系仍有质量限制。

[![实践07：GMR行走重定向](practices/07_motion_retargeting/media/preview.gif)](practices/07_motion_retargeting/media/walk_human_g1.mp4)

[实践详情](practices/07_motion_retargeting/README.md)

### 实践08｜AMP拟人走跑与步态评估

**Isaac Lab · PPO + AMP 对抗式动作先验 · G1 29 自由度**

走得准靠速度跟踪奖励，走得像人靠判别器：它同时看人类参考动作和机器人动作并尽力分辨，机器人骗过它的程度折算成风格奖励。

先定位到判别器被稳定性保护全程暂停——500 次更新里判定饱和 500 次，只真正优化 4 次，拖腿由此而来；解除后摆腿抬升由 2.13cm 提到 5.01cm。随后用十轮单因素实验找到两个有效改动：**判别器梯度惩罚由 10 提到 50—100**，摆臂方向修正，肩髋相位相关由 +0.85（同手同脚）变为 −0.70/−0.84 并在新种子下复现；**新增一项直接惩罚双支撑时长的奖励**，该占比由 44.6% 降到 22.3%。没有采用框架自带的接触时序奖励，因为实测它可被"一直双脚着地"套利，启用后双支撑反而升到 49.8%。

**拟人走跑整体未通过验证。**冻结门槛要求 0.3m/s 慢走与 0.5m/s 前进两个场景同时通过，无模型达成。当前阻碍是航向与左右对称性互相耦合：降低双支撑会让单脚支撑期变长、航向退化（十个模型上相关系数 −0.849），提高航向权重又让双支撑回升到 31%—36%。

[![实践08：摆臂方向已修正的候选](practices/08_amp_locomotion/media/preview_armswing_fixed.gif)](practices/08_amp_locomotion/media/p8_armswing_fixed_side_20s.mp4)

[实践详情](practices/08_amp_locomotion/README.md)

### 实践09｜全身舞蹈轨迹跟踪

**MJLab · PPO · BeyondMimic 方法**

跟着一段两分钟的舞蹈走完全程。长序列里一处跟丢就会一路崩掉，所以按失败统计调整参考片段的采样概率，让练得差的片段被更多抽到，同时处理相位推进和参考坐标对齐。动作接口是参考关节角加策略残差，这样能把局部动作误差与世界轨迹误差分开看。训练约 30,000 次更新。

学习残差连续跟完 131.48 秒，对齐身体平均距离 3.633cm；作为对照，零残差的参考 PD 在 0.28 秒就触发末端条件。**验证范围：**世界锚点平均距离 0.581m，存在持续漂移；未验证未见动作或外部扰动的泛化。

[![实践09：全身舞蹈轨迹跟踪](practices/09_motion_tracking/media/preview.gif)](practices/09_motion_tracking/media/motion_tracking_30k_full.mp4)

[实践详情](practices/09_motion_tracking/README.md)

### 实践10｜平台地形动作跟踪（HOI）

**Isaac Lab · PPO · 高度扫描**

在有平台的场景里跟踪全身参考动作，机器人要与平台发生实际接触而不只是复现姿态，因此用高度扫描感知地形，并比较了世界锚点奖励的不同尺度。

此前最大的短板是训练量：所选模型只有参考配置的 0.025%。补做扩容训练（1536 并行环境 × 2000 次更新，约 7373 万条转换，是原来的 60 倍），五个**在训练开始前按等间隔登记**的存档逐个评估，全部跑完 309 帧参考无提前终止。世界锚点 RMSE 由 0.1297m 降到 0.0501m，降幅 61%，朝向与关节误差同步下降。**这确认了预算就是此前的约束**，该结论此前只是推测。

**验证范围：**4096 环境在本机显存跑不起来，这轮是 1536，训练量仍只有参考配置的 1.5%；模型**没有独立留出验证**。

[![实践10：平台路径回放](practices/10_object_interaction/media/preview.gif)](practices/10_object_interaction/media/p10_std03_model399_full.mp4)

[实践详情](practices/10_object_interaction/README.md)

### 实践11｜深度感知与MuJoCo跨仿真验证

**Project Instinct + MuJoCo（Sim2Sim）· PPO + 运动风格奖励**

靠一个深度相机看路通过障碍。深度图不是直接用最新一帧，而是维护历史队列并从中抽帧以模拟真实传感器延迟；图像要经缩放到 36×64、裁剪为 18×32、INPAINT_NS 空洞修补、3×3 高斯模糊、按 [0, 2.5] 米线性归一化才能进网络。这套实现对照训练侧定义写，过了 5 项单元测试。策略导出 ONNX 后放到 MuJoCo 跨仿真运行，录了四组 20 秒对照。

参考策略在 MuJoCo 平地连续走完 20 秒、13.87 米，骨盆稳定在 0.71m 以上，说明整条链路正确；**自训练的 5000 模型几乎不前进**，平地 1.43m、粗糙地形 1.24m。我一开始以为这是跨仿真损失，扫描指令后发现不是：它在 Isaac 原生评估里 XY 速度 RMSE 0.2604 与均速 0.2572 几乎相等，**这个策略本来就没学会跟踪速度**，不是迁移把它弄坏的。

[![实践11：深度感知策略MuJoCo跨仿真运行](practices/11_depth_locomotion/media/preview.gif)](practices/11_depth_locomotion/media/p11_sim2sim_course_flat_20s.mp4)

[实践详情](practices/11_depth_locomotion/README.md)

## 代表性结果

| 方向 | 已完成的结果 | 展示 |
|---|---|---|
| 速度与骨盆高度联合控制 | 连续蹲走22秒，后20秒高度MAE 0.6116cm、前向均速0.5357m/s；三组动态指令均完成24秒窗口 | [详情](practices/04_velocity_height/README.md) |
| 教师学生蒸馏与全身动作 | Action与KL学生均25/25动作到达参考末尾，对齐身体平均距离2.580cm / 2.537cm | [详情](practices/06_teacher_student/README.md) |
| 自适应采样与全身轨迹跟踪 | 学习残差连续跟完131.48秒，对齐身体平均距离3.633cm；零残差PD在0.28秒失败 | [详情](practices/09_motion_tracking/README.md) |
| 感知驱动的粗糙地形行走 | 新增15,000次更新；8组同初态、各14秒场景中XY速度与转速RMSE均低于旧策略 | [详情](practices/02_rough_terrain/README.md) |
| 地形感知的人物交互动作跟踪 | 扩容60倍后世界锚点RMSE由0.1297m降至0.0501m，三项误差同步下降 | [详情](practices/10_object_interaction/README.md) |

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

未达成的三项：分层导航的随机布局训练没有显出总体优势；AMP 拟人走跑的自然摆臂与低速慢走未同时通过冻结门槛；自训练的深度策略迁移后几乎不前进，跨仿真部署不算成功。

各项结果保留模型、统计窗口、坐标和初态限制。预训练部署、运动学回放与自主训练分别说明，具体数值与未覆盖场景见对应实践页。
