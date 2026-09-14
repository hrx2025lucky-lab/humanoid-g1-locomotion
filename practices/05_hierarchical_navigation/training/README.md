# 05 · 分层强化学习导航：训练与实现

以高层速度决策连接冻结的低层行走策略，对固定布局和随机布局模型进行相同预算的交叉评估。

[实际训练与模型记录](record.json)按本实验口径记录；模型编号、连续更新数和独立分支分别计数。配置文件中的最大迭代数是相应启动段的设置，不能直接替代累计训练量。

本目录保存算法组件与配置快照；组件依赖相应框架中的父类、任务注册、资产和数据。通用框架、完整运动库与全部模型存档不在此目录打包，代码中的占位路径需按运行环境配置。算法组件保留原接口，不能直接当作独立程序执行。

| 文件 | 用途 |
|---|---|
| [evaluate_p5_paired.py](../evaluation/evaluate_p5_paired.py) | 算法、配置或评估组件 |
| [p5_scenario_manifest.py](../evaluation/p5_scenario_manifest.py) | 算法、配置或评估组件 |
| [agent_paired_baseline.yaml](../training/agent_paired_baseline.yaml) | 保存的Agent配置 |
| [pre_trained_policy_action.py](../training/components/pre_trained_policy_action.py) | 算法、配置或评估组件 |

训练启动参数可通过仓库中的[训练入口](../../../tools/train.py)检查。默认仅显示命令；需要明确指定框架目录、解释器、环境数量、更新次数以及执行开关才会启动。该入口不自动安装依赖或选择模型。
