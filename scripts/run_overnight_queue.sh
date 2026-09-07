#!/usr/bin/env bash
# 过夜队列总入口 —— 不用电脑时跑这一条就行。
#
#   bash scripts/run_overnight_queue.sh
#
# 四个任务靠 /tmp/humanoid_gpu.lock 串行（单卡 3090 装不下两个 IsaacSim）：
#   1. 实践 11 跑酷训练      3000 轮，约 14 h
#   2. 实践 10 smoke        10 轮，约 10 min
#   3. 实践 10 正式训练      3000 轮
#   4. 实践 9  重训          std=1.0
#
# 想中途停下把显卡还给自己：bash scripts/stop_all_gpu_jobs.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/limx/workspace/Roxan_warmup"
PIPE="$HOME/humanoid_logs/pipeline/overnight_queue.log"
mkdir -p "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

# 已经有任务在跑就不要再叠一层，否则两个 IsaacSim 一起抢卡必然 OOM。
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
if [ -n "$busy" ]; then
    log "❌ GPU 上还有计算进程（PID: $busy），先跑 scripts/stop_all_gpu_jobs.sh"
    exit 1
fi

free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
log "════ 过夜队列启动，空闲显存 ${free_mb:-?} MiB ════"

start() {   # start <脚本名> <说明>
    if [ ! -f "$HERE/$1" ]; then
        log "⚠️ 找不到 $1，跳过"
        return
    fi
    nohup bash "$HERE/$1" > /dev/null 2>&1 &
    log "  已排入 [$!] $2  ($1)"
    sleep 2
}

start run_p11_after_p5.sh   "实践 11 跑酷训练（save_interval=200，可随时中断）"
start run_p10_smoke.sh      "实践 10 smoke（验证 RayCaster 修复）"
start run_p10_full.sh       "实践 10 正式训练"
start retrain_p9_fixed_std.sh "实践 9 重训 std=1.0"

log "════ 已全部排入，靠文件锁串行执行 ════"
log "看进度：tail -f $PIPE"
log "看训练：tail -f \$HOME/humanoid_logs/*/*.log"
log "想停下：bash scripts/stop_all_gpu_jobs.sh"
