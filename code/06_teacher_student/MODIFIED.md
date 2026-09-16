# 实践 06 · 教师-学生蒸馏的全身运动跟踪 — 我的改动清单

对照基准：课程 HW6 包

- 新增文件 **4**
- 修改文件 **9**
- 原样保留 **51**

## 我修改的文件

- `src/humanoid_hw6/config/g1/env_cfgs.py`
- `src/humanoid_hw6/config/g1/rl_cfg.py`
- `src/humanoid_hw6/rl/algorithms/action_matching.py`
- `src/humanoid_hw6/rl/algorithms/distillation_ppo.py`
- `src/humanoid_hw6/rl/algorithms/distillation_utils.py`
- `src/humanoid_hw6/rl/algorithms/kl_matching.py`
- `src/humanoid_hw6/scripts/data/visualize_motion_curate_viser.py`
- `src/humanoid_hw6/teacher_env_cfg.py`
- `tests/test_distillation_reference.py`

## 我新增的文件

- `README.md`
- `src/humanoid_hw6/config/g1/motion_data_cfg_hw9_dance.yaml`
- `src/humanoid_hw6/config/g1/motion_data_cfg_hw9_dance_fixed.yaml`
- `tests/test_distillation_resume.py`

---

其余文件来自上游框架/课程包，未改动。
本清单由脚本对照原始包逐文件哈希比对生成。
