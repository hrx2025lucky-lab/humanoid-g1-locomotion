# 实践 04 · 蹲姿行走：速度 + 骨盆高度 MDP — 我的改动清单

对照基准：上游 mjlab 速度任务基线包

- 新增文件 **2**
- 修改文件 **19**
- 原样保留 **222**

## 我修改的文件

- `pyproject.toml`
- `src/mjlab/entity/variants.py`
- `src/mjlab/envs/manager_based_rl_env.py`
- `src/mjlab/envs/mdp/dr/material.py`
- `src/mjlab/envs/mdp/events.py`
- `src/mjlab/managers/manager_base.py`
- `src/mjlab/managers/recorder_manager.py`
- `src/mjlab/rl/runner.py`
- `src/mjlab/sim/sim.py`
- `src/mjlab/tasks/velocity/config/g1/__init__.py`
- `src/mjlab/tasks/velocity/config/g1/env_cfgs.py`
- `src/mjlab/tasks/velocity/mdp/height_command.py`
- `src/mjlab/tasks/velocity/mdp/observations.py`
- `src/mjlab/tasks/velocity/mdp/rewards.py`
- `src/mjlab/tasks/velocity/velocity_env_cfg.py`
- `src/mjlab/terrains/terrain_entity.py`
- `src/mjlab/terrains/terrain_generator.py`
- `src/mjlab/utils/buffers/delay_buffer.py`
- `tests/test_velocity_task.py`

## 我新增的文件

- `README.md`
- `tests/test_resume_learning_rate.py`

---

其余文件来自上游框架，未改动。
本清单由脚本对照上游基线逐文件哈希比对生成。
