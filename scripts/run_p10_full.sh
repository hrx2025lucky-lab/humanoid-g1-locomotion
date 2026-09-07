#!/usr/bin/env bash
# 实践 10 正式训练 —— 补 check_deliverables.py 里缺的「正式训练 checkpoint」。
#
# 为什么要单独一个脚本：run_p10_smoke.sh 只跑 10 轮验证「能连续运行且无 NaN/Inf」，
# 跑完就 exit，队列到那里就断了，正式训练一直没人接。
#
# 轮数选 3000 而不是配置默认的 30000：单卡 3090，其它实践实测 3000 轮
# 已能看出收敛趋势；30000 轮按实践 11 的 16.8 s/轮折算要 5.8 天，不现实。
# 已核实 BasePPORunnerCfg 的 save_interval=500 < 3000，
# 不会重演实践 11 那次「save_interval 大于总轮数、全程不落存档」的坑
# （见 docs/实践11_跑酷与深度感知.md 第九节）。
#
# 用法：nohup bash scripts/run_p10_full.sh > /dev/null 2>&1 &
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
HOI="$ROOT/shenlan_hw/HOI_Mimic"
PY_LAB="$ROOT/envs/isaaclab/bin/python"
TASK="Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast"
ITERS=3000
# 为什么不用配置默认的 4096：这个任务的观测特别大——
# policy 组 3120 维（其中 height_scanner 就占 2312 = 289×8），critic 3252 维。
# rollout 缓冲区是 num_steps_per_env × num_envs × obs_dim：
#   24 × 4096 × (3120+3252) × 4 B ≈ 2.5 GB，PPO 多轮更新还要再翻几倍。
# 实测 4096 在第 0 轮结束时 CUDA OOM（只差 24 MiB，说明就是刚好超）。
# 差得这么少更要留足余量：后面 PPO 更新阶段的峰值比第 0 轮更高，
# 卡着上限跑等于把几小时的训练押在一次侥幸上。
NUM_ENVS="${P10_NUM_ENVS:-2048}"

LOG_DIR="$HOME/humanoid_logs/p10_hoi"
PIPE="$HOME/humanoid_logs/pipeline/p10_full.log"
mkdir -p "$LOG_DIR" "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════ 实践 10 正式训练 ════"

# ── GPU 互斥锁 ─────────────────────────────────────────────
GPU_LOCK="/tmp/humanoid_gpu.lock"
exec 9>"$GPU_LOCK"
log "等待 GPU 锁…"
flock 9
log "已获得 GPU 锁"

# 拿到锁不等于显存已释放：别的训练可能没走锁（实践 11 那次就是），
# 所以再按任务名确认一次，并等显存真的回收。
log "等所有 GPU 任务让出显存…"
while pgrep -f "Navigation-HRL-RandomArena" >/dev/null 2>&1 \
   || pgrep -f "Instinct-Parkour" >/dev/null 2>&1 \
   || pgrep -f "AMP-WalkToRun" >/dev/null 2>&1; do
    sleep 300
done
sleep 90
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
log "GPU 空闲显存：${free_mb:-?} MiB"
# 显存被别的东西占着就再降一档。真正的 OOM 兜底在下面的重试循环里，
# 这里只是提前一步，省掉一次要跑几十秒才失败的尝试。
if [ -n "$free_mb" ] && [ "$free_mb" -lt 12000 ] && [ "$NUM_ENVS" -gt 1024 ]; then
    log "⚠️ 空闲显存不足 12 GB，$NUM_ENVS 环境风险高，先降到 1024"
    NUM_ENVS=1024
fi

# ── 确认 smoke 过了 ────────────────────────────────────────
# smoke 没过就跑正式训练是纯浪费 GPU：同样的错会在第 10 轮再炸一次。
#
# 这一步必须放在拿到锁**之后**：本脚本是排在 run_p10_smoke.sh 后面启动的，
# 启动那一刻 smoke 还没重跑，日志里留的是上一次的失败记录
# （KeyError: '/World/ground'）。若在拿锁前检查，会拿旧日志判定失败并直接退出，
# 正式训练永远轮不到。
S_LOG="$LOG_DIR/p10_smoke.log"
if [ ! -f "$S_LOG" ]; then
    log "❌ 找不到 smoke 日志 $S_LOG，先跑 scripts/run_p10_smoke.sh"
    exit 1
fi
if ! grep -qE "Learning iteration [0-9]+/" "$S_LOG"; then
    log "❌ smoke 日志里没有训练循环，说明 smoke 没跑起来，最后 20 行："
    tail -20 "$S_LOG" | tee -a "$PIPE"
    exit 1
fi
log "✅ smoke 已通过：$(grep -oE 'Learning iteration [0-9]+/[0-9]+' "$S_LOG" | tail -1)"

# ── 正式训练 ────────────────────────────────────────────────
T_LOG="$LOG_DIR/p10_full_train.log"
cd "$HOI" || exit 1
[ -f set_project_root.sh ] && . ./set_project_root.sh 2>/dev/null
export PYTHONPATH="$HOI/source/unitree_rl_lab:$HOI:${PYTHONPATH:-}"

log "训练：--num_envs $NUM_ENVS --max_iterations $ITERS → $T_LOG"
run_train() {   # run_train <num_envs>
    "$PY_LAB" scripts/rsl_rl/train.py \
        --task "$TASK" --num_envs "$1" --max_iterations "$ITERS" \
        --headless --logger tensorboard \
        > "$T_LOG" 2>&1
}
run_train "$NUM_ENVS"
rc=$?

# 显存不够就自动减半重试，而不是把整夜浪费在一次 OOM 上。
# 这个任务的显存占用主要由 num_envs 线性决定（rollout 缓冲区），
# 减半基本就能过；最多退到 512，再小就没有训练意义了。
while [ "$rc" -ne 0 ] && grep -q "OutOfMemoryError" "$T_LOG" 2>/dev/null && [ "$NUM_ENVS" -gt 512 ]; do
    NUM_ENVS=$(( NUM_ENVS / 2 ))
    log "⚠️ CUDA OOM，降到 --num_envs $NUM_ENVS 重试"
    sleep 30    # 等上一个进程把显存真正还回来
    run_train "$NUM_ENVS"
    rc=$?
done

last=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$T_LOG" 2>/dev/null | tail -1)
log "训练结束 rc=$rc  最后：${last:-未知}  (num_envs=$NUM_ENVS)"
if [ "$rc" -ne 0 ]; then
    log "最后 25 行："
    tail -25 "$T_LOG" | tee -a "$PIPE"
fi

# ── 验收：checkpoint 必须是真训练出来的 ──────────────────────
# 只判断「有没有 .pt」会被冒烟存档骗过去（实践 11 的 model_3.pt 就是），
# 所以按文件名里的轮数卡一道。
run=$(ls -td "$HOI"/logs/rsl_rl/unitree_g1_29dof_mimic_hoi_terrain_perceptive_raycast/*/ 2>/dev/null | head -1)
if [ -n "$run" ]; then
    log "run 目录: $run"
    best=$(find "$run" -name "model_*.pt" 2>/dev/null \
           | grep -oE "model_[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
    if [ -n "$best" ] && [ "$best" -ge 100 ]; then
        log "✅ 有效 checkpoint：model_${best}.pt"
    else
        log "❌ 只有 model_${best:-无}.pt，轮数 < 100，算不上正式训练成果"
    fi
else
    log "❌ 找不到 run 目录"
fi

log "接着跑效果验收（看真实任务指标，不看 reward）"
cd "$ROOT/motion control/humanoid_practice/g1_locomotion" 2>/dev/null \
  && "$PY_LAB" scripts/verify_training_outcome.py --practice 10 2>&1 | tee -a "$PIPE"

log "════ 实践 10 正式训练结束 ════"
