#!/usr/bin/env bash
# 过夜队列的接续版：P10 训练已经在跑，接着把剩下的事做完。
#
#   nohup bash scripts/resume_overnight.sh > /dev/null 2>&1 &
#
# 为什么单独写一个而不是重启 run_overnight_queue.sh：
# 那个脚本会从头再来一遍，而 P10 已经跑到 350/3000（约 35 分钟）。
# 录像的三处 bug（play 不退出、上游 API 变更 ×2）是在队列走过录像阶段
# 之后才修好的，所以录像要重来，但 P10 的训练进度不该跟着一起丢。
#
# 顺序：等 P10 训完 → 重录 4/5/8 视频 → 跑 P11 → 收尾核对
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/limx/workspace/Roxan_warmup"
PY="$ROOT/envs/isaaclab/bin/python"
PIPE="$HOME/humanoid_logs/pipeline/overnight_queue.log"
mkdir -p "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

step() {   # step <说明> <超时秒> <命令...>
    local desc="$1" tmo="$2"; shift 2
    log "── $desc（预算 $((tmo / 60)) 分钟）──"
    local t0=$SECONDS
    timeout "$tmo" "$@" >> "$PIPE" 2>&1
    local rc=$?
    local used=$(( (SECONDS - t0) / 60 ))
    case $rc in
        0)   log "   ✅ 完成，用时 ${used} 分钟" ;;
        124) log "   ⏱ 到预算上限停止，用时 ${used} 分钟（checkpoint 已按间隔保存）" ;;
        *)   log "   ⚠️ 退出码 $rc，用时 ${used} 分钟" ;;
    esac
}

log "════════ 接续队列启动 ════════"

# ── 1. 等 P10 训练结束 ────────────────────────────────────────
# 它是被上一轮队列启动的，不归本脚本管，只能等。
#
# 不用 pgrep -f "Perceptive-Raycast" 判存活：那个模式会匹配到任何
# 命令行里含该字符串的进程，包括查询命令自己，于是永远等不到"结束"
# （这个坑在本项目里已经踩过好几次）。
# 改成一开始就锁定训练进程的 PID，之后只问"这个 PID 还在不在"。
P10_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | head -1)
if [ -z "$P10_PID" ]; then
    log "⚠️ GPU 上没有计算进程，实践 10 可能已经结束"
else
    log "实践 10 训练进程 PID=$P10_PID，等它结束"
fi

P10_DEADLINE=$(( SECONDS + 12600 ))     # 最多再等 3.5 小时
waited=0
while [ -n "$P10_PID" ] && kill -0 "$P10_PID" 2>/dev/null; do
    if [ "$SECONDS" -ge "$P10_DEADLINE" ]; then
        log "⏱ 实践 10 到 3.5 小时上限，主动结束以腾出时间"
        kill "$P10_PID" 2>/dev/null
        sleep 60
        kill -9 "$P10_PID" 2>/dev/null
        sleep 30
        break
    fi
    it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" \
         "$HOME/humanoid_logs/p10_hoi/p10_full_train.log" 2>/dev/null | tail -1)
    log "   实践 10 进行中：${it:-初始化}"
    sleep 900
    waited=$((waited + 15))
done
log "实践 10 训练结束（等了约 ${waited} 分钟）"

best=$(find "$ROOT/shenlan_hw/HOI_Mimic/logs" -name "model_*.pt" 2>/dev/null \
       | grep -oE "model_[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
if [ -n "$best" ] && [ "$best" -ge 100 ]; then
    log "   ✅ 实践 10 有效 checkpoint：model_${best}.pt"
else
    log "   ⚠️ 实践 10 最大 checkpoint 只有 model_${best:-无}.pt"
fi

sleep 90    # 等显存回收

# ── 2. 重录三段视频（录像的三个 bug 已修）──────────────────────
step "重录实践 4/5/8 回放视频" 5400 bash "$HERE/record_all_videos.sh" p4 p5 p8

# ── 3. 实践 11 跑酷训练，拿剩下的时间 ──────────────────────────
step "实践 11 跑酷训练" 21600 bash "$HERE/run_p11_after_p5.sh"

# ── 4. 收尾 ──────────────────────────────────────────────────
log "── 材料清单 ──"
"$PY" "$HERE/check_deliverables.py" >> "$PIPE" 2>&1
log "── 效果验收（看任务指标，不看 reward）──"
"$PY" "$HERE/verify_training_outcome.py" >> "$PIPE" 2>&1

log "════════ 接续队列结束 ════════"
