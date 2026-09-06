#!/usr/bin/env bash
# 实践 10 感知任务 smoke train，排在所有 GPU 任务之后。
#
# 三组 TODO 已实现（审计 24/24），这一步验证官方那 10 分：
# 「能连续运行且无 NaN/Inf」。跑 10 轮即可，不是正式训练。
#
# 独立成脚本而不是并进 overnight_p7p8.sh，因为后者已经在跑了，
# 改文件对已启动的 bash 进程无效（bash 是边读边执行的，
# 中途改动只会让它读到错位的字节）。
#
# 用法：nohup bash scripts/run_p10_smoke.sh > /dev/null 2>&1 &
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
HOI="$ROOT/shenlan_hw/HOI_Mimic"
PY_LAB="$ROOT/envs/isaaclab/bin/python"
TASK="Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast"

LOG_DIR="$HOME/humanoid_logs/p10_hoi"
PIPE="$HOME/humanoid_logs/pipeline/p10_smoke.log"
mkdir -p "$LOG_DIR" "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════ 实践 10 smoke train 接续 ════"
[ -d "$HOI" ] || { log "❌ 找不到 $HOI"; exit 1; }

log "等所有 GPU 任务让出显存…"
while pgrep -f "Navigation-HRL-RandomArena" >/dev/null 2>&1 \
   || pgrep -f "Instinct-Parkour" >/dev/null 2>&1 \
   || pgrep -f "AMP-WalkToRun" >/dev/null 2>&1; do
    sleep 300
done
sleep 90    # IsaacSim 退出后显存回收有延迟
log "GPU 空闲：$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)"

S_LOG="$LOG_DIR/p10_smoke.log"
cd "$HOI" || exit 1
[ -f set_project_root.sh ] && . ./set_project_root.sh 2>/dev/null
export PYTHONPATH="$HOI/source/unitree_rl_lab:$HOI:${PYTHONPATH:-}"

log "smoke: --num_envs 64 --max_iterations 10 → $S_LOG"
timeout 5400 "$PY_LAB" scripts/rsl_rl/train.py \
    --task "$TASK" --num_envs 64 --max_iterations 10 \
    --headless --logger tensorboard \
    > "$S_LOG" 2>&1
rc=$?

if grep -qE "Learning iteration [0-9]+/" "$S_LOG"; then
    it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$S_LOG" | tail -1)
    log "✅ 训练循环已启动（$it）"

    # 官方 10 分项明确要求"无 NaN/Inf"。只匹配独立词，
    # 避免 "info"/"inference" 这类词里的 inf 造成误报。
    if grep -qiE "(^|[^a-z])(nan|inf)([^a-z]|$)" "$S_LOG"; then
        log "⚠️ 疑似 NaN/Inf，相关行："
        grep -inE "(^|[^a-z])(nan|inf)([^a-z]|$)" "$S_LOG" | head -5 | tee -a "$PIPE"
    else
        log "✅ 无 NaN/Inf"
    fi

    # 观测维自检：参考答案给了确定值 289×8=2312
    dim=$(grep -oE "policy[^0-9]*([0-9]{3,5})" "$S_LOG" | grep -oE "[0-9]{3,5}" | head -1)
    [ -n "$dim" ] && log "   观测维线索: $dim（height_scan 部分应为 289×8=2312）"
else
    log "❌ smoke 失败 rc=$rc，最后 30 行："
    tail -30 "$S_LOG" | tee -a "$PIPE"
    exit 1
fi

log "════ 结束 ════"
