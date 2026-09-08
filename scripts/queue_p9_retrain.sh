#!/usr/bin/env bash
# 等 GPU 空闲后重训实践 9（关节顺序已修正）
set -uo pipefail
HERE="/home/limx/workspace/Roxan_warmup/motion control/humanoid_practice/g1_locomotion/scripts"
PIPE="$HOME/humanoid_logs/pipeline/p9_queue.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }
me=$$
count(){ local n=0 p; for p in $(pgrep -f "resume_p10_train|queue_p11_after_p10|run_p11_after_p5" 2>/dev/null); do
  [ "$p" = "$me" ] || [ "$p" = "$PPID" ] || n=$((n+1)); done; echo "$n"; }
log "等实践 10 / 11 跑完再训练实践 9"
while : ; do
  r=$(count); b=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [ "$r" -le 0 ] && [ -z "$b" ] && break
  sleep 900
done
log "GPU 空闲，开始实践 9 重训（修正后的关节顺序）"
sleep 60
bash "$HERE/retrain_p9_fixed_std.sh" >> "$PIPE" 2>&1
log "实践 9 重训结束，验收："
/home/limx/workspace/Roxan_warmup/envs/isaaclab/bin/python "$HERE/verify_training_outcome.py" --practice 9 2>&1 | tee -a "$PIPE"
