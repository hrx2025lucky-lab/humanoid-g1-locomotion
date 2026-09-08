#!/usr/bin/env bash
# 实践 10 续训：从已有的 model_2999.pt 接着跑，不从头开始。
#
#   nohup bash scripts/resume_p10_train.sh > /dev/null 2>&1 &
#
# 为什么要续训：3000 轮跑完时误差**还在快速下降**
#   error_joint_pos  1.678 → 1.169  （末段降 30%）
#   error_body_pos   0.233 → 0.124  （降 47%）
#   error_anchor_pos 0.575 → 0.357  （降 38%）
# 这不是"练到头了"，是"练到一半被叫停"。
# 直观表现就是回放里机器人上台阶时小腿先撞到台沿、脚没能跨上去——
# 平均关节误差 1.169 rad ≈ 67°，抬腿高度和时机都还差得远。
#
# 注意这与实践 9 是两种完全不同的情况：
#   实践 10：误差在**降**（还没练够）      → 加轮数有效
#   实践 9 ：误差在**涨**且奖励饱和 4e-18  → 加轮数无效，得先修 std
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
HOI="$ROOT/shenlan_hw/HOI_Mimic"
PY="$ROOT/envs/isaaclab/bin/python"
TASK="Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast"
EXP="unitree_g1_29dof_mimic_hoi_terrain_perceptive_raycast"
LOAD_RUN="${P10_LOAD_RUN:-2026-09-07_21-38-30}"
# 注意 max_iterations 在 rsl_rl 里是**增量**不是总数：
#   on_policy_runner.py:97  tot_iter = start_iter + num_learning_iterations
# 从 model_2999 续训、传 6000，实际跑到 8999 轮（总量约官方 30000 的 30%）。
ITERS="${P10_ITERS:-6000}"
NUM_ENVS="${P10_NUM_ENVS:-2048}"   # 4096 会 CUDA OOM，见 docs/实践10 第九节

LOG_DIR="$HOME/humanoid_logs/p10_hoi"
PIPE="$HOME/humanoid_logs/pipeline/p10_resume.log"
mkdir -p "$LOG_DIR" "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════ 实践 10 续训 ════"

# 先确认要接续的 checkpoint 真的存在，别跑了半天才发现 load_run 写错
src="$HOI/logs/rsl_rl/$EXP/$LOAD_RUN"
best=$(ls "$src" 2>/dev/null | grep -oE "^model_[0-9]+\.pt$" | sed 's/model_//;s/\.pt//' | sort -n | tail -1)
if [ -z "$best" ]; then
    log "❌ $LOAD_RUN 里没有 checkpoint，先确认 run 名"
    exit 1
fi
log "从 $LOAD_RUN/model_${best}.pt 续训，再跑 $ITERS 轮（到约 $((best + ITERS)) 轮）"

GPU_LOCK="/tmp/humanoid_gpu.lock"
exec 9>"$GPU_LOCK"
log "等待 GPU 锁…"
flock 9
log "已获得 GPU 锁"

while : ; do
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
    [ -z "$busy" ] && break
    log "  等 GPU 上的进程退出（$busy）"
    sleep 300
done
sleep 60
log "GPU 空闲：$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)"

T_LOG="$LOG_DIR/p10_resume_train.log"
cd "$HOI" || exit 1
[ -f set_project_root.sh ] && . ./set_project_root.sh >/dev/null 2>&1
export PYTHONPATH="$HOI/source/unitree_rl_lab:$HOI:${PYTHONPATH:-}"

log "训练 → $T_LOG"
"$PY" scripts/rsl_rl/train.py \
    --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$ITERS" \
    --resume --load_run "$LOAD_RUN" \
    --headless --logger tensorboard \
    > "$T_LOG" 2>&1
rc=$?
last=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$T_LOG" 2>/dev/null | tail -1)
log "训练结束 rc=$rc  最后：${last:-未知}"
[ "$rc" -ne 0 ] && { log "最后 25 行："; tail -25 "$T_LOG" | tee -a "$PIPE"; }

log "── 续训后效果验收（看误差有没有继续降）──"
cd "$ROOT/motion control/humanoid_practice/g1_locomotion" 2>/dev/null \
  && "$PY" scripts/verify_training_outcome.py --practice 10 2>&1 | tee -a "$PIPE"

log "════ 结束 ════"
