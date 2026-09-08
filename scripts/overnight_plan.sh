#!/usr/bin/env bash
# 过夜训练总调度：把 13.5 小时按"边际收益"分配给实践 11 和实践 9。
#
#   nohup bash scripts/overnight_plan.sh > /dev/null 2>&1 &
#
# ── 时间账（2026-09-08 19:35 实测）──────────────────────────
#   可用时间          13.5 h（到明早 9:00）
#   实践 11  9.53 s/轮 @4096 envs  →  3000 轮 7.9 h，6000 轮 15.9 h
#   实践 9   需要至少 4 h 才能看出曲线走向
#
# ── 为什么不把时间全给实践 11 ──────────────────────────────
# 实践 11 当前 713 轮已到 terrain_levels 1.869/9，末段斜率 +0.00169/轮，
# 外推 3000 轮可达 5.74/9 —— 这个成绩已经能说明"课程在推进、方法有效"。
# 而实践 9 是刚修好关节顺序（FK 误差 0.178m→0.0011m）后的**首次验证**，
# 一次都还没跑过。
#
#   已知有效的东西再多跑 3000 轮，边际收益是 5.74 → 7 左右；
#   完全未验证的东西跑起来，边际收益是 0 → 1（从"不知道对不对"到"知道"）。
#
# 后者明显更值。所以：实践 11 按原计划 3000 轮收工，剩下的时间给实践 9。
#
# ── 分配 ────────────────────────────────────────────────
#   实践 11 跑完 3000 轮        约 6.1 h（已跑 704）
#   实践 9  拿剩下的            约 6.5 h
#   两者都有 checkpoint 保护，超时停下也有产出
set -uo pipefail

ROOT="${ROXAN_ROOT:-/home/limx/workspace/Roxan_warmup}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/envs/isaaclab/bin/python"
PIPE="$HOME/humanoid_logs/pipeline/overnight_plan.log"
mkdir -p "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════════ 过夜计划启动 ════════"
log "实践 11 按 3000 轮收工，剩余时间全给实践 9（首次验证关节顺序修复）"

# ── 1. 等实践 11 结束 ────────────────────────────────────
# 按 PID 判存活，不用 pgrep 匹配任务名——那会匹配到查询命令自己。
P11_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | head -1)
if [ -n "$P11_PID" ]; then
    log "实践 11 训练 PID=$P11_PID，等它跑完 3000 轮"
    while kill -0 "$P11_PID" 2>/dev/null; do
        it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" \
             "$HOME/humanoid_logs/p11_parkour/p11_parkour_train.log" 2>/dev/null | tail -1)
        lv=$(ls -td "$ROOT"/repos/instinctlab/logs/instinct_rl/g1_parkour/*/ 2>/dev/null | head -1)
        log "   实践 11：${it:-初始化}"
        sleep 1800
    done
fi
log "实践 11 已结束"

# 记录最终成绩
"$PY" - <<'PYEOF' 2>&1 | tee -a "$PIPE"
import glob, os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ds = sorted(glob.glob(os.path.expanduser(
    "~/workspace/Roxan_warmup/repos/instinctlab/logs/instinct_rl/g1_parkour/*/")),
    key=os.path.getmtime)
if ds:
    fs = glob.glob(ds[-1] + "events.out.*")
    if fs:
        ea = EventAccumulator(fs[0], size_guidance={
            "scalars": 4000, "histograms": 1, "images": 1, "audio": 1, "tensors": 1})
        ea.Reload()
        t = [x for x in ea.Tags()["scalars"] if x.endswith("terrain_levels")]
        if t:
            v = np.array([x.value for x in ea.Scalars(t[0])])
            print(f"  实践 11 最终 terrain_levels {v[-1]:.2f}/9")
            print("  五段:", [round(float(x.mean()), 2) for x in np.array_split(v, 5)])
PYEOF

sleep 120   # 等显存回收

# ── 2. 实践 9 拿剩下的时间 ───────────────────────────────
# 这是关节顺序修复后的首次训练，最需要的是"能不能看出误差转为下降"。
# 8000 轮在 mjlab 上约 6~7 h，正好匹配剩余时间。
log "── 实践 9 重训（关节顺序已修正）──"
P9_ITERS="${P9_ITERS:-8000}" timeout 25200 bash "$HERE/retrain_p9_fixed_std.sh" >> "$PIPE" 2>&1
rc=$?
case $rc in
    0)   log "   ✅ 实践 9 完成" ;;
    124) log "   ⏱ 到 7 小时上限（checkpoint 已按间隔保存）" ;;
    *)   log "   ⚠️ 退出码 $rc" ;;
esac

# ── 3. 收尾：全量验收 ────────────────────────────────────
log "── 全量效果验收 ──"
"$PY" "$HERE/verify_training_outcome.py" >> "$PIPE" 2>&1
log "── 材料清单 ──"
"$PY" "$HERE/check_deliverables.py" 2>&1 | tail -12 | tee -a "$PIPE"
log "── 训练预算评估 ──"
"$PY" "$HERE/assess_training_budget.py" >> "$PIPE" 2>&1

log "════════ 过夜计划结束 ════════"
log "看结果：tail -120 $PIPE"
