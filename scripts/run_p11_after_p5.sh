#!/usr/bin/env bash
# 实践 5 Part 2 跑完后，自动接上实践 11-C 跑酷训练。
#
# 单张 3090，训练必须串行。这个脚本守在 RandomArena 后面，
# 它一结束就先做冒烟测试探显存，通过了再进正式训练。
#
# 用法：nohup bash scripts/run_p11_after_p5.sh > /dev/null 2>&1 &
set -uo pipefail

REPO_ROOT="/home/limx/workspace/Roxan_warmup"
LAB="$REPO_ROOT/repos/instinctlab"
PY="$REPO_ROOT/envs/isaaclab/bin/python"
export PYTHONPATH="$REPO_ROOT/repos/instinct_rl:$LAB/source/instinctlab:${PYTHONPATH:-}"

TASK="Instinct-Parkour-Target-Amp-G1-v0"
LOG_DIR="$HOME/humanoid_logs/p11_parkour"
PIPE_LOG="$HOME/humanoid_logs/pipeline/p11_chain.log"
mkdir -p "$LOG_DIR" "$(dirname "$PIPE_LOG")"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE_LOG"; }

log "════ 实践 11-C 接续任务启动 ════"
log "等待实践 5 Part 2 (RandomArena) 让出 GPU…"

while pgrep -f "Navigation-HRL-RandomArena" >/dev/null 2>&1 \
   || pgrep -f "AMP-WalkToRun" >/dev/null 2>&1; do
  sleep 300
done
log "GPU 已空闲"

# IsaacSim 退出后显存回收有延迟，等一会儿再开下一个，否则冒烟测试会误判显存不足
sleep 90
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
log "当前空闲显存: ${free_mb:-未知} MB"

# ── 冒烟测试：先用小规模探路，别一上来就长跑 ──────────────────
# 显存需求未知（跑酷带 height scan，可能比导航吃得多），
# 官方给的降级档是 512，就从这里试。
SMOKE_LOG="$LOG_DIR/p11_smoke.log"
log "冒烟测试：--num_envs 512 --max_iterations 5"
cd "$LAB" || { log "❌ 进不去 $LAB"; exit 1; }

timeout 1800 "$PY" scripts/instinct_rl/train.py \
  --headless --task="$TASK" --num_envs 512 --max_iterations 5 \
  > "$SMOKE_LOG" 2>&1
smoke_rc=$?

if grep -qE "Learning iteration [0-9]+/" "$SMOKE_LOG"; then
  log "✅ 冒烟通过（训练循环已启动）"
elif grep -qiE "out of memory|CUDA error" "$SMOKE_LOG"; then
  log "⚠️ 显存不足，降到 --num_envs 256 重试"
  timeout 1800 "$PY" scripts/instinct_rl/train.py \
    --headless --task="$TASK" --num_envs 256 --max_iterations 5 \
    > "$SMOKE_LOG" 2>&1
  if grep -qE "Learning iteration [0-9]+/" "$SMOKE_LOG"; then
    log "✅ 256 env 冒烟通过"
    NUM_ENVS=256
  else
    log "❌ 256 env 仍失败，最后 25 行："
    tail -25 "$SMOKE_LOG" | tee -a "$PIPE_LOG"
    exit 1
  fi
else
  log "❌ 冒烟失败（rc=$smoke_rc），最后 25 行："
  tail -25 "$SMOKE_LOG" | tee -a "$PIPE_LOG"
  exit 1
fi
NUM_ENVS="${NUM_ENVS:-512}"

# ── 正式训练 ────────────────────────────────────────────────
# 配置默认 max_iterations=30000，按其它实践的经验那远超收敛所需，
# 也远超一晚能跑完的量。先跑 3000 轮看曲线，不够再续。
ITERS=3000
# 配置里 save_interval=5000 是配 max_iterations=30000 的。
# 一旦用 --max_iterations 把总轮数压到 3000，save_interval 就大于总轮数，
# 训练全程一个 checkpoint 都不落，只在结束时存一次——
# 等于十几个小时“全有或全无”，中途崩溃或想提前收工都颗粒无收。
# train.py 没有 --save_interval，但它走 hydra，可以用 agent.xxx=yyy 覆盖。
SAVE_INTERVAL=200
TRAIN_LOG="$LOG_DIR/p11_parkour_train.log"
log "正式训练：--num_envs $NUM_ENVS --max_iterations $ITERS save_interval=$SAVE_INTERVAL"
log "日志 → $TRAIN_LOG"

"$PY" scripts/instinct_rl/train.py \
  --headless --task="$TASK" --num_envs "$NUM_ENVS" --max_iterations "$ITERS" \
  agent.save_interval="$SAVE_INTERVAL" \
  > "$TRAIN_LOG" 2>&1
train_rc=$?

last_it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$TRAIN_LOG" 2>/dev/null | tail -1)
log "训练结束 rc=$train_rc  最后：${last_it:-未知}"
if [ "$train_rc" -ne 0 ]; then
  log "最后 25 行："
  tail -25 "$TRAIN_LOG" | tee -a "$PIPE_LOG"
fi

run_dir=$(ls -td "$LAB"/logs/instinct_rl/*/*/ 2>/dev/null | head -1)
[ -n "$run_dir" ] && log "run 目录: $run_dir"
log "════ 实践 11-C 接续任务结束 ════"
