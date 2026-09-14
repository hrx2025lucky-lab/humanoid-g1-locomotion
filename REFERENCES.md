# 算法、框架与资源

项目在现有研究方法与参考实现上完成任务接入、算法组件实现、训练比较和仿真验证。PPO、AMP、HoST、GMR与BeyondMimic的原始方法不作为本项目原创算法。

| 框架或方法 | 在项目中的用途 | 来源 |
|---|---|---|
| Isaac Lab | GPU并行仿真、任务管理与训练接口 | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) |
| Unitree RL Lab | G1模型、运动控制任务与部署接口 | [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) |
| Unitree Lab AMP | AMP判别器、风格奖励与G1走跑任务参考实现 | [Unitree Lab AMP](https://github.com/HeYee03/unitree_lab_amp) |
| MJLab | MuJoCo/Warp并行训练与动作跟踪 | [MJLab](https://github.com/mujocolab/mjlab) |
| MuJoCo | 刚体仿真、PD执行与策略迁移 | [MuJoCo](https://github.com/google-deepmind/mujoco) |
| GMR | 人体到机器人运动重定向 | [GMR](https://github.com/YanjieZe/GMR) |
| Project Instinct | 深度感知、运动风格与运动控制 | [InstinctLab](https://github.com/project-instinct/instinctlab) |

参考实现与本项目扩展各自保留来源。实践2的地形扩展、实践4的指令接口、实践5的配对评估，以及实践6/9的算法组件按对应目录说明。实践3使用已有HoST预训练策略，实践7的直接姿态回放仅用于运动学检查。

算法组件依赖相应框架与版本，仓库不打包整个上游框架、第三方资产、人体原始运动库和全部训练存档。训练参数、已有结果及展示视频均按实践编号组织。外部资源遵循各自授权；实践11配置来自InstinctLab，其原许可证随组件保留，不对全部第三方内容统一改授许可。

[返回首页](README.md)
