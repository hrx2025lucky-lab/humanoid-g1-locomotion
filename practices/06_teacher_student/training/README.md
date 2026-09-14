# 06 · 教师学生蒸馏与全身动作：训练与实现

在全身参考动作跟踪中实现动作回归和高斯分布KL两类蒸馏，比较学生策略在相同动作库中的表现。

[实际训练与模型记录](record.json)按本实验口径记录；模型编号、连续更新数和独立分支分别计数。配置文件中的最大迭代数是相应启动段的设置，不能直接替代累计训练量。

本目录保存算法组件与配置快照；组件依赖相应框架中的父类、任务注册、资产和数据。通用框架、完整运动库与全部模型存档不在此目录打包，代码中的占位路径需按运行环境配置。算法组件保留原接口，不能直接当作独立程序执行。

| 文件 | 用途 |
|---|---|
| [compare_distillation.py](../evaluation/compare_distillation.py) | 算法、配置或评估组件 |
| [agent_action.yaml](../training/agent_action.yaml) | 保存的Agent配置 |
| [agent_kl.yaml](../training/agent_kl.yaml) | 保存的Agent配置 |
| [action_matching.py](../training/components/action_matching.py) | 算法、配置或评估组件 |
| [distillation_utils.py](../training/components/distillation_utils.py) | 算法、配置或评估组件 |
| [kl_matching.py](../training/components/kl_matching.py) | 算法、配置或评估组件 |

训练启动参数可通过仓库中的[训练入口](../../../tools/train.py)检查。默认仅显示命令；需要明确指定框架目录、解释器、环境数量、更新次数以及执行开关才会启动。该入口不自动安装依赖或选择模型。
