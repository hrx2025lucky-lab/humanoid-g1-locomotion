#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 实践 5 / 实践 6 对照训练
# ═══════════════════════════════════════════════════════════════════════════
#
# 【实践 5】Part 2 单因素对照
#   baseline      Unitree-G1-29dof-Navigation-HRL-Baseline      固定竞技场
#   random_arena  Unitree-G1-29dof-Navigation-HRL-RandomArena   每 episode 随机重排障碍
#
#   ⚠️ 不用课程自带的 HRL-Extension 做对照——它与 Baseline 相差五处
#      （布局/课程/观测维度 273 vs 1092/目标模式/终止条件），
#      观测维度不同意味着网络输入层都不一样，无法归因到单一因素。
#      RandomArena 继承 Baseline 只替换 events，并共用同一份 PPORunnerCfg。
#
# 【实践 6】两种蒸馏方式对照
#   action_matching  只对齐动作均值
#   kl_matching      对齐完整高斯分布（均值 + 标准差）
#   两者共享同一个 Teacher checkpoint 与同一份受限 Student 观测。
#
# 用法：
#   ./run_p5p6_compare.sh p5              # 跑实践 5 两组
#   ./run_p5p6_compare.sh p6              # 跑实践 6 两组
#   ./run_p5p6_compare.sh p5 baseline     # 只跑其中一组
#   ITERS=6000 ./run_p5p6_compare.sh p6
#
# ⚠️ 必须等 GPU 空闲。单个 4096 环境训练会把 3090 打到 97% 利用率。
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=log_paths.sh
source "$HERE/log_paths.sh"

ISAACLAB_PY="/home/limx/workspace/Roxan_warmup/envs/isaaclab/bin/python"
HW5="/home/limx/workspace/Roxan_warmup/shenlan_hw/hw5_navigation/unitree_rl_lab"
HW6="/home/limx/workspace/Roxan_warmup/shenlan_hw/hw6_distill"

SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-4096}"
ITERS="${ITERS:-8000}"

run_p5() {
  local key="$1" task
  # 探索类修复组共用 Baseline 任务，只改环境变量 —— 这样地形、目标采样、
  # 奖励权重全部与已跑完的 baseline 严格一致，差异可归因到单一因子。
  case "$key" in
    baseline)     task="Unitree-G1-29dof-Navigation-HRL-Baseline" ;;
    random_arena) task="Unitree-G1-29dof-Navigation-HRL-RandomArena" ;;
    fix_smoothing)
      task="Unitree-G1-29dof-Navigation-HRL-Baseline"
      # 单因子：只关 EMA 平滑。实测平滑把探索速度压到 1/5
      # （0.045 vs 0.223 m/s，理论 sqrt(0.1/1.9)=0.229）。
      export NAV_COMMAND_SMOOTHING=1.0 NAV_RUN_NAME="$key" ;;
    fix_both)
      task="Unitree-G1-29dof-Navigation-HRL-Baseline"
      # 在 fix_smoothing 基础上再抬熵系数，对冲 std 0.20→0.07 的过早塌缩
      export NAV_COMMAND_SMOOTHING=1.0 NAV_ENTROPY_COEF=0.02 NAV_RUN_NAME="$key" ;;
    *) echo "实践5 未知组别 '$key'，可选: baseline random_arena fix_smoothing fix_both" >&2
       return 2 ;;
  esac
  local log; log="$(hp_log p5 "$key")"
  echo "──── 实践5 / $key ────  task=$task  envs=$NUM_ENVS  iters=$ITERS  seed=$SEED"
  echo "     平滑=${NAV_COMMAND_SMOOTHING:-0.1(默认)}  熵系数=${NAV_ENTROPY_COEF:-0.005(默认)}"
  echo "     日志: $log"
  cd "$HW5"
  # 关键：Python 环境里装的 unitree_rl_lab 指向主仓库 repos/unitree_rl_lab，
  # 而实践 5 用的是自己那份 fork（多了 navigation 任务与 training_logger）。
  # 不重装包，改用 PYTHONPATH 让 hw5 的源码目录优先，避免影响实践 2。
  PYTHONPATH="$HW5/source/unitree_rl_lab:${PYTHONPATH:-}" \
  "$ISAACLAB_PY" scripts/rsl_rl/train.py \
    --task "$task" --num_envs "$NUM_ENVS" --max_iterations "$ITERS" \
    --seed "$SEED" --headless > "$log" 2>&1
  echo "✅ 实践5 / $key 结束"
}

run_p6() {
  local key="$1" task
  case "$key" in
    action_matching) task="Mjlab-Humanoid-HW6-Student-Action-Matching-G1" ;;
    kl_matching)     task="Mjlab-Humanoid-HW6-Student-KL-Matching-G1" ;;
    teacher)         task="Mjlab-Humanoid-HW6-Teacher-G1" ;;
    *) echo "实践6 未知组别 '$key'，可选: action_matching kl_matching teacher" >&2; return 2 ;;
  esac
  local log; log="$(hp_log p6 "$key")"
  echo "──── 实践6 / $key ────  task=$task  envs=$NUM_ENVS  iters=$ITERS  seed=$SEED"
  echo "     日志: $log"
  cd "$HW6"
  # mjlab 的 --agent.logger 默认是 wandb，本机未配置，必须显式指定
  ./.venv/bin/train "$task" \
    --env.scene.num-envs="$NUM_ENVS" \
    --agent.max-iterations="$ITERS" \
    --agent.seed="$SEED" \
    --agent.logger=tensorboard \
    --agent.run-name="compare_${key}" > "$log" 2>&1
  echo "✅ 实践6 / $key 结束"
}

WHICH="${1:-}"
GROUP="${2:-}"
case "$WHICH" in
  p5)
    if [[ -n "$GROUP" ]]; then run_p5 "$GROUP"
    else for k in baseline random_arena; do run_p5 "$k"; done; fi
    echo; echo "对比曲线: tensorboard --logdir $HW5/logs/rsl_rl"
    echo "关键指标（作业讲解指定，不能只看 reward）："
    echo "  成功率 / 目标距离误差 / 跌倒率 / 超时比例 / 动作饱和 / 命令变化率"
    ;;
  p6)
    if [[ -n "$GROUP" ]]; then run_p6 "$GROUP"
    else for k in action_matching kl_matching; do run_p6 "$k"; done; fi
    echo; echo "对比曲线: tensorboard --logdir $HW6/logs"
    echo "关键指标："
    echo "  Action Matching → Loss/bc, Loss/action_mae, Loss/action_rmse"
    echo "  KL Matching     → Loss/kl, Loss/mean_rmse, Loss/std_rmse"
    ;;
  *)
    echo "用法: $0 {p5|p6} [组别]" >&2
    echo "  p5 组别: baseline random_arena"
    echo "  p6 组别: action_matching kl_matching teacher"
    exit 2
    ;;
esac
