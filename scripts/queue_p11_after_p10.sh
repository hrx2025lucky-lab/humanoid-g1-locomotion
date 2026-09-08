#!/usr/bin/env bash
# 实践 10 续训跑完后，自动用 4096 环境重跑实践 11。
#
#   nohup bash scripts/queue_p11_after_p10.sh > /dev/null 2>&1 &
#
# 为什么要重跑实践 11：上一次正式训练误用了冒烟规模的 512 环境
# （run_p11_after_p5.sh 里 ${NUM_ENVS:-512}），按课程给的折算
#   Env × Iter ≈ 4096 × 10000，  4096→10k  2048→20k  1024→40k  512→80k
# 512×3000 只相当于官方基准的 3.75%，地形等级才爬到 1.28/9 就停了。
# 当时还误以为"3000 轮跑得真快"，其实是环境数少了 8 倍。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 允许用 ROXAN_ROOT 覆盖，方便换机器时不用改脚本
ROOT="${ROXAN_ROOT:-/home/limx/workspace/Roxan_warmup}"
PY="$ROOT/envs/isaaclab/bin/python"
PIPE="$HOME/humanoid_logs/pipeline/p11_rerun.log"
mkdir -p "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════ 等实践 10 续训结束，再重跑实践 11 ════"

# 按 PID 判存活，不用 pgrep -f 匹配任务名——那会匹配到查询命令自己，
# 导致永远等不到"结束"（这个坑在本项目踩过多次）。
P10_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | head -1)
if [ -n "$P10_PID" ]; then
    log "实践 10 训练 PID=$P10_PID，等它结束"
    while kill -0 "$P10_PID" 2>/dev/null; do
        it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" \
             "$HOME/humanoid_logs/p10_hoi/p10_resume_train.log" 2>/dev/null | tail -1)
        log "   实践 10：${it:-初始化}"
        sleep 900
    done
fi
log "实践 10 已结束"

best=$(find "$ROOT/shenlan_hw/HOI_Mimic/logs" -name "model_*.pt" 2>/dev/null \
       | grep -oE "model_[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
log "实践 10 最大 checkpoint：model_${best:-无}.pt"
log "── 实践 10 效果验收 ──"
"$PY" "$HERE/verify_training_outcome.py" --practice 10 2>&1 | tee -a "$PIPE"

sleep 120   # 等显存回收

# ── 重跑实践 11 ──────────────────────────────────────────────
# 4096 环境已实测可行：9-7 那次用 4096 跑了 616 轮全程无 OOM。
# 轮数取 3000（= 官方基准 30%，与实践 4/5/6/8 同口径，那几个都收敛了）。
log "── 实践 11 重跑（4096 环境）──"
NUM_ENVS=4096 P11_ITERS=3000 timeout 43200 bash "$HERE/run_p11_after_p5.sh" >> "$PIPE" 2>&1
rc=$?
case $rc in
    0)   log "   ✅ 实践 11 完成" ;;
    124) log "   ⏱ 到 12 小时上限（checkpoint 每 200 轮已存）" ;;
    *)   log "   ⚠️ 退出码 $rc" ;;
esac

log "── 最终验收 ──"
"$PY" "$HERE/verify_training_outcome.py" 2>&1 | tee -a "$PIPE"
"$PY" "$HERE/check_deliverables.py" 2>&1 | tail -20 | tee -a "$PIPE"
log "════ 全部结束 ════"
