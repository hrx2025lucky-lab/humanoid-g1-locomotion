#!/usr/bin/env bash
# 过夜队列（串行执行，单卡装不下两个 IsaacSim）。
#
#   nohup bash scripts/run_overnight_queue.sh > /dev/null 2>&1 &
#   想中途收回显卡：bash scripts/stop_all_gpu_jobs.sh
#
# 排序原则：便宜且能闭环的先做，长训练按时间预算切分。
#
# 为什么不按"实践编号顺序"排：实践 11 一个人就要 14 h，排在前面
# 会把整夜吃光，后面几项一项都轮不到。而录像每段只要几分钟，
# 却各自能补齐一份缺失材料。先把便宜的做完，剩下时间再分给训练。
#
# 为什么长训练敢设超时：save_interval 已修好（实践 11 补了
# agent.save_interval=200，实践 10 本来就是 500），中途停下也有 checkpoint。
# 在此之前跑 14 h 是"全有或全无"，超时等于全丢。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/limx/workspace/Roxan_warmup"
PY="$ROOT/envs/isaaclab/bin/python"
PIPE="$HOME/humanoid_logs/pipeline/overnight_queue.log"
mkdir -p "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
if [ -n "$busy" ]; then
    log "❌ GPU 上还有计算进程（$busy），先跑 scripts/stop_all_gpu_jobs.sh"
    exit 1
fi

log "════════ 过夜队列启动 ════════"
log "空闲显存 $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)"

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

cd "$HERE/.." || exit 1

# ── 1. 补录三段回放视频（每段几分钟，各补一份缺失材料）──────────
step "录制实践 4/5/8 回放视频" 5400 bash "$HERE/record_all_videos.sh" p4 p5 p8

# ── 2. 实践 10 正式训练（从未成功训练过，风险最高，先跑）────────
# 放在长训练的第一位是因为它此前一次都没跑通（先后卡在 RayCaster
# 类属性遮蔽、rsl_rl policy 键两个 bug 上），未知问题最多。
# 早点暴露还有半个晚上可以补救；排最后出问题就只能等明天。
step "实践 10 正式训练" 18000 bash "$HERE/run_p10_full.sh"

# ── 3. 实践 11 跑酷训练（拿剩下的时间）────────────────────────
# 这个任务已验证健康：616 轮时 terrain_levels 0.02→1.79、
# episode_length 74→727，曲线没问题，只是需要时间。
step "实践 11 跑酷训练" 25200 bash "$HERE/run_p11_after_p5.sh"

# ── 4. 收尾：核对材料 + 按真实任务指标验收 ────────────────────
log "── 材料清单 ──"
"$PY" "$HERE/check_deliverables.py" >> "$PIPE" 2>&1
log "── 效果验收（看任务指标，不看 reward）──"
"$PY" "$HERE/verify_training_outcome.py" >> "$PIPE" 2>&1

log "════════ 过夜队列结束 ════════"
log "看结果：tail -100 $PIPE"
