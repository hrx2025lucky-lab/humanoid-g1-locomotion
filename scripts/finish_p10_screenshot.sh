#!/usr/bin/env bash
# 过夜流程的最后一步：抓实践 10 的 RayCaster 截图。
#
#   nohup bash scripts/finish_p10_screenshot.sh > /dev/null 2>&1 &
#
# 为什么单独一个脚本：resume_overnight.sh 已经在跑了，
# 而 bash 是边读边执行的——改一个正在运行的脚本会让它读到错位的字节。
# 所以新需求只能另起一个，靠"等 GPU 空闲"和它串起来。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/limx/workspace/Roxan_warmup"
PY="$ROOT/envs/isaaclab/bin/python"
PIPE="$HOME/humanoid_logs/pipeline/overnight_queue.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] [截图] $*" | tee -a "$PIPE"; }

log "等过夜队列结束后再抓截图"

# 等到「没有别的排队脚本 且 GPU 上没有计算进程」为止。
# 两个条件都要：只看 GPU 会在两个任务交接的空档里误判为空闲，
# 一头扎进去就和下一个训练抢显存了。
me=$$
# 数一数还有几个排队/训练脚本在跑，排除自己和父进程。
# pgrep -f 匹配整条命令行，所以脚本自身、以及包含该模式的调用
# 都会被数进去；不排除就会永远等不到"队列结束"，截图这一步永不执行。
count_running() {
    local n=0 p
    for p in $(pgrep -f "resume_overnight|run_p1[01]|record_all_videos|retrain_p9" 2>/dev/null); do
        [ "$p" = "$me" ] || [ "$p" = "$PPID" ] || n=$((n + 1))
    done
    echo "$n"
}

while : ; do
    running=$(count_running)
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
    if [ "${running:-0}" -le 0 ] && [ -z "$busy" ]; then
        break
    fi
    sleep 600
done

log "队列已结束，GPU 空闲，开始抓图"
sleep 60    # 显存回收有延迟

if bash "$HERE/capture_p10_raycast.sh" 2>&1 | tee -a "$PIPE" | tail -3; then
    log "✅ 截图完成"
else
    log "⚠️ 截图失败，明早可手动重跑：bash scripts/capture_p10_raycast.sh"
fi

log "── 最终材料清单 ──"
"$PY" "$HERE/check_deliverables.py" >> "$PIPE" 2>&1
log "── 最终效果验收 ──"
"$PY" "$HERE/verify_training_outcome.py" >> "$PIPE" 2>&1
log "════ 全部结束 ════"
