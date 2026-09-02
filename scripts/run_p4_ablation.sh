#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 实践 4 消融实验 —— 三组对照训练
# ═══════════════════════════════════════════════════════════════════════════
#
# 验证蹲姿行走的三段闭环「缺一不可」：
#     ① 采样高度指令  →  ② Actor 观测到指令  →  ③ 奖励高度跟踪
#
#   baseline     三段齐全
#   NoHeightRew  切断③：track_base_height.weight = 0
#   BlindActor   切断②：Actor 移除 height_command（Critic 保留）
#
# 公平对照：三组共用同一份 rl_cfg，且显式固定 seed / num_envs /
# max_iterations，唯一变量就是上面那一处配置差异。
#
# 用法：
#   ./run_p4_ablation.sh              # 顺序跑三组
#   ./run_p4_ablation.sh baseline     # 只跑其中一组
#   ITERS=4000 ./run_p4_ablation.sh   # 缩短迭代数快速验证
#
# ⚠️ 必须等 GPU 空闲后再跑。实测单个 4096 环境的训练会把 3090 打到
#    97% 利用率，并行跑第二个只会互相拖慢（详见 docs/00_实践总览.md）。
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HW4="/home/limx/workspace/Roxan_warmup/shenlan_hw/hw4_mjlab"
TRAIN="$HW4/.venv/bin/train"

# 三组共用的超参数——公平对照的前提
NUM_ENVS="${NUM_ENVS:-4096}"
ITERS="${ITERS:-8000}"      # 默认 30000 太长；实践 2 的经验是看奖励斜率而非跑满
SEED="${SEED:-42}"

# --agent.logger 默认是 wandb，本机没配，必须显式改成 tensorboard
COMMON=(
  --env.scene.num-envs="$NUM_ENVS"
  --agent.max-iterations="$ITERS"
  --agent.seed="$SEED"
  --agent.logger=tensorboard
)

declare -A TASKS=(
  [baseline]="Mjlab-VelocityHeight-Flat-Unitree-G1"
  [no_height_rew]="Mjlab-VelocityHeight-Flat-Unitree-G1-NoHeightRew"
  [blind_actor]="Mjlab-VelocityHeight-Flat-Unitree-G1-BlindActor"
)
ORDER=(baseline no_height_rew blind_actor)

run_one() {
  local key="$1" task="${TASKS[$1]}"
  local log="$HOME/p4_${key}.log"
  echo "════════════════════════════════════════════════════════════"
  echo " 组别   : $key"
  echo " 任务   : $task"
  echo " 参数   : num_envs=$NUM_ENVS  iters=$ITERS  seed=$SEED"
  echo " 日志   : $log"
  echo "════════════════════════════════════════════════════════════"
  cd "$HW4"
  nohup "$TRAIN" "$task" "${COMMON[@]}" --agent.run-name="ablation_${key}" \
    > "$log" 2>&1
  echo "✅ $key 训练结束"
}

if [[ $# -ge 1 ]]; then
  [[ -v "TASKS[$1]" ]] || { echo "未知组别 '$1'，可选: ${ORDER[*]}" >&2; exit 2; }
  run_one "$1"
else
  echo "顺序跑三组，预计总耗时 = 单组时长 × 3"
  echo "（4096 env / 8000 iter 单组约 6~7 小时，三组需跨夜）"
  echo
  for k in "${ORDER[@]}"; do run_one "$k"; done
fi

echo
echo "对比三组曲线："
echo "  tensorboard --logdir $HW4/logs"
echo "关键指标（作业 §7 指定）："
echo "  Metrics/base_velocity/error_height   ← 主判据"
echo "  Episode_Reward/track_base_height"
echo "  Episode_Reward/track_linear_velocity ← 确认速度任务未受影响"
