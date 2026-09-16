# HOI Mimic 项目运行说明

- 训练脚本在 `scripts/rsl_rl/`
- 环境源码在 `source/unitree_rl_lab/`
- HOI 数据和已转换 mimic 数据在 `datasets/`
- 运行环境变量脚本是仓库根目录下的 `set_project_root.sh`

本文只覆盖以下两个实际使用任务：

- `Unitree-G1-29dof-Mimic-HOI_terrain`
- `Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast`

## 1. 前置环境

训练机器需要先准备：

1. Ubuntu + NVIDIA GPU 驱动
2. Isaac Sim 5.1 和 Isaac Lab 2.3
3. 一个已经安装 Isaac Lab 的 Python 环境，例如 `env_isaaclab`
4. 可选：Weights & Biases

```bash
conda activate env_isaaclab
```

如果要使用 W&B：

```bash
pip install wandb
wandb login
# 离线机器可用：
# export WANDB_MODE=offline
```

> **安装项目自带的 rsl_rl 库**
>
> 本项目包含一个定制的 `rsl_rl` 子模块（位于仓库根目录 `rsl_rl/`），带有自定义 logger 等扩展。
> 默认 Isaac Lab 环境中的 `rsl-rl-lib`（PyPI 版本）**不包含这些修改**，必须手动安装本项目版本。
> 激活 `env_isaaclab` 后执行：
>
> ```bash
> cd /path/to/HOI_Minic        # 替换为项目实际路径
> pip install -e rsl_rl/
> ```

## 2. 进入项目并设置路径

每次打开新 shell 后，都建议切换到项目目录，然后先执行：

```bash
conda activate env_isaaclab
source set_project_root.sh     # 注意 blind task 和 perceptive task 默认 URDF 不同
```

`set_project_root.sh` 会设置当前项目运行依赖的关键路径：

| 变量 | 默认值 |
|------|--------|
| `PROJECT_ROOT` | 当前仓库根目录 |
| `HOI_ROOT` | `$PROJECT_ROOT/datasets` |
| `HOI_MIMIC_DATA_DIR` | `$PROJECT_ROOT/datasets/hoi_mimic_data` |
| `HOI_MIMIC_TERRAIN_MOTION_FILE` | `datasets/hoi_mimic_data/climb_15_z_scale_1.0_mimic.npz` |
| `HOI_MIMIC_TERRAIN_META_FILE` | `datasets/hoi_mimic_data/climb_15_z_scale_1.0_mimic.terrain.json` |
| `HOI_MIMIC_TERRAIN_URDF` | `datasets/models/terrain/climb_15/multi_boxes_z_scale_1.0_isaac_world.urdf` |

如果只跑 blind 任务，也可以改用非 `isaac_world` 版本地形：

```bash
export HOI_MIMIC_TERRAIN_URDF="${HOI_ROOT}/models/terrain/climb_15/multi_boxes_z_scale_1.0.urdf"
```

## 3. 安装当前项目

首次在某台机器上运行时，安装 `unitree_rl_lab` 到 Isaac Lab 环境：

```bash
cd "${PROJECT_ROOT}"
./unitree_rl_lab.sh -i
```

安装完成后重启 shell，或重新激活 conda 环境，然后再次设置路径：

```bash
conda activate env_isaaclab
cd <project_path>
source set_project_root.sh
```

检查任务是否注册成功：

```bash
./unitree_rl_lab.sh -l | grep -E '\| Unitree-G1-29dof-Mimic-HOI_terrain[[:space:]]+\||\| Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast[[:space:]]+\|'
```

本文使用的任务 ID：

- `Unitree-G1-29dof-Mimic-HOI_terrain`
- `Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast`

## 4. 确认默认数据

当前仓库已经包含 `climb_15` 的默认训练数据：

```bash
ls -lh "${HOI_MIMIC_TERRAIN_MOTION_FILE}"
ls -lh "${HOI_MIMIC_TERRAIN_META_FILE}"
ls -lh "${HOI_MIMIC_TERRAIN_URDF}"
```

感知任务需要 `*.terrain.json` 中包含 `mjcf_boxes`：

```bash
python3 -c "import json, os; d=json.load(open(os.environ['HOI_MIMIC_TERRAIN_META_FILE'])); assert 'mjcf_boxes' in d; print('mjcf_boxes OK:', len(d['mjcf_boxes']))"
```

当前 `datasets/hoi_mimic_data/climb_15_z_scale_1.0_mimic.terrain.json`
已经包含 `mjcf_boxes`，默认不需要重新合并。

## 5. 快速 smoke train

> **新 shell 中跑任何训练/播放命令前**
>
> 下面各节的命令使用 `cd "${PROJECT_ROOT}"`，前提是当前 shell 已经通过
> Section 2 设置了 `PROJECT_ROOT`。在新 shell 中请先执行：
> ```bash
> cd /path/to/HOI_Minic        # 替换为项目实际路径
> conda activate env_isaaclab
> source set_project_root.sh
> ```
> 之后 `$PROJECT_ROOT` 在当前 shell 中就一直可用了。

先跑一个短训练，确认 Isaac、任务注册、数据路径和日志都正常。

> **注意**：首次运行前，确保已经完成了 [Section 1](#1-前置环境) 的 rsl_rl 安装和 [Section 3](#3-安装当前项目) 的 `./unitree_rl_lab.sh -i`。

```bash
cd "${PROJECT_ROOT}"
source set_project_root.sh

python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain \
  --num_envs 64 \
  --max_iterations 10 \
  --headless \
  --logger tensorboard
```

如果显存不足，继续降低 `--num_envs`。

## 6. 训练 blind HOI terrain mimic

```bash
cd "${PROJECT_ROOT}"
source set_project_root.sh
conda activate env_isaaclab

python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless \
  --logger wandb \
  --log_project_name unitree_hoi_mimic
```

不用 W&B 时：

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless \
  --logger tensorboard
```


## 7. 训练 perceptive Raycast 任务

Perceptive Raycast 任务使用 `RayCaster` height scanner，需要 `mjcf_boxes`。
当前默认 `set_project_root.sh` 已把地形 URDF 指向 `*_isaac_world.urdf`。
如果使用 `home_work` 分支，请先完成

跑训练之前，建议先确认 URDF 路径：

```bash
echo "URDF: ${HOI_MIMIC_TERRAIN_URDF}"
# 预期输出末尾为 multi_boxes_z_scale_1.0_isaac_world.urdf
# 如果输出的是非 isaac_world 版本，说明之前手动 export 过，需要重新 source set_project_root.sh
```

确信 URDF 指向正确后开始训练：

```bash
cd "${PROJECT_ROOT}"
conda activate env_isaaclab
source set_project_root.sh

python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast \
  --num_envs 1024 \
  --max_iterations 50000 \
  --headless \
  --logger wandb \
  --log_project_name unitree_hoi_mimic_terrain_perceptive_raycast
```

如果需要打开 Isaac GUI 可视化调试 RayCaster 扫描点，可以用下面的命令：

```bash
cd "${PROJECT_ROOT}"
conda activate env_isaaclab
source set_project_root.sh

python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast \
  --num_envs 1 \
  --height-scan-vis \
  --height-scan-print
```


## 8. 播放 checkpoint

播放 blind checkpoint：

```bash
cd "${PROJECT_ROOT}"
conda activate env_isaaclab
source set_project_root.sh

python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain \
  --checkpoint <checkpoint_path> \
  --num_envs 1 \
  --hoi-play-no-curriculum \
  --hoi-play-no-dr
```

播放 RayCaster perceptive checkpoint：

```bash
cd "${PROJECT_ROOT}"
conda activate env_isaaclab
source set_project_root.sh

python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast \
  --checkpoint <checkpoint_path> \
  --num_envs 1 \
  --hoi-play-no-curriculum \
  --hoi-play-no-dr \
  --height-scan-vis \
  --height-scan-print
```

其中 `<checkpoint_path>` 替换为实际训练得到的模型文件路径，例如
`logs/rsl_rl/<experiment_name>/<timestamp>/model_<iteration>.pt`。

常用 play 参数：

| 参数 | 作用 |
|------|------|
| `--hoi-play-no-curriculum` | 播放时关闭 curriculum level-up |
| `--hoi-play-no-dr` | 播放时冻结 startup domain randomization |
| `--hoi-play-no-obs-noise` | 关闭 policy observation corruption |
| `--hoi-play-exact-motion-joints` | reset 时强制关节精确对齐参考 motion |
| `--hoi-play-deterministic` | 等价于 no curriculum、no DR、no obs noise |
| `--height-scan-vis` | 显示 RayCaster height scan marker |
| `--height-scan-print` | 打印 RayCaster height scan 统计 |

## 9. 转换新的 HOI motion

原始 HOI motion 文件位于 `datasets/robot-terrain/` 目录下，以 `climb_*` 命名。
如果要换用其他 clip（例如 `datasets/robot-terrain/climb_15_z_scale_1.0.npz` 之外的 clip）：

```bash
cd "${PROJECT_ROOT}"
source set_project_root.sh

python scripts/mimic/hoi_to_mimic_npz.py \
  --file datasets/robot-terrain/climb_15_z_scale_1.0.npz \
  --output-dir datasets/hoi_mimic_data \
  --headless
```

批量转换 terrain 子集可用：

```bash
python scripts/mimic/hoi_to_mimic_npz.py \
  --task terrain \
  --filter climb_15 \
  --output-dir datasets/hoi_mimic_data \
  --headless
```

RayCaster perceptive 任务需要把 terrain URDF 的 box primitive 合并进
`*.terrain.json`：

```bash
PYTHONPATH=tools python -m pyHOIsim2sim.export_terrain_boxes \
  --urdf "${HOI_MIMIC_TERRAIN_URDF}" \
  --out-json "${HOI_MIMIC_DATA_DIR}/climb_15_z_scale_1.0_mimic.terrain.json" \
  --merge
```

换 clip 后，需要同步设置三类路径：

```bash
export HOI_MIMIC_TERRAIN_MOTION_FILE="${HOI_MIMIC_DATA_DIR}/<clip>_mimic.npz"
export HOI_MIMIC_TERRAIN_META_FILE="${HOI_MIMIC_DATA_DIR}/<clip>_mimic.terrain.json"
export HOI_MIMIC_TERRAIN_URDF="${HOI_ROOT}/models/terrain/<terrain>/multi_boxes_z_scale_1.0_isaac_world.urdf"
```

## 10. 辅助导出

导出 MuJoCo debug keypoints CSV：

```bash
python scripts/mimic/mimic_npz_to_mujoco_keypoints_csv.py \
  --input datasets/hoi_mimic_data/climb_00_z_scale_1.0_mimic.npz \
  --output datasets/hoi_mimic_data_mujoco/climb_00_z_scale_1.0_mimic_mujoco_keypoints.csv \
  --fps 30.0
```

导出 `g1_ctrl` mimic CSV：

```bash
python scripts/mimic/mimic_npz_to_g1_ctrl_csv.py \
  --input datasets/hoi_mimic_data/climb_15_z_scale_1.0_mimic.npz \
  --output datasets/hoi_mimic_data_g1_ctrl/climb_15_z_scale_1.0_mimic.npz_g1_ctrl.csv
```

## 11. 常见问题

| 现象 | 处理 |
|------|------|
| 找不到 motion/meta/URDF | 重新执行 `source set_project_root.sh`，并确认文件在 `datasets/` 下 |
| 目标任务没有列出 | 在 `env_isaaclab` 中重新执行 `./unitree_rl_lab.sh -i` |
| RayCaster perceptive 报缺少 `mjcf_boxes` | 运行 `PYTHONPATH=tools python -m pyHOIsim2sim.export_terrain_boxes ... --merge` |
| CUDA OOM | 降低 `--num_envs`，例如 `4096 -> 1024 -> 512` |
| W&B 登录失败或不能联网 | 使用 `--logger tensorboard`，或设置 `WANDB_MODE=offline` |
| 播放行为抖动较大 | 先试 `--hoi-play-no-curriculum --hoi-play-no-dr`，调试时可加 `--hoi-play-exact-motion-joints` |
| 训练报错与 rsl_rl 相关（如 logger 找不到类） | 检查 `pip show rsl-rl-lib` 的 Location 是否指向项目内的 `rsl_rl/`，否则重新执行 `pip install -e rsl_rl/` |

## 12. 最小验证清单

- `conda activate env_isaaclab` 成功
- `source set_project_root.sh` 打印的路径都在当前仓库下
- `./unitree_rl_lab.sh -l | grep -E '\| Unitree-G1-29dof-Mimic-HOI_terrain[[:space:]]+\||\| Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast[[:space:]]+\|'` 能列出两个目标任务
- `datasets/hoi_mimic_data/climb_15_z_scale_1.0_mimic.npz` 存在
- `datasets/hoi_mimic_data/climb_15_z_scale_1.0_mimic.terrain.json` 包含 `mjcf_boxes`
- `--num_envs 64 --max_iterations 10 --headless` 的 smoke train 能完成
