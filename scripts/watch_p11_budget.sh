#!/usr/bin/env bash
# 实践 11 训练看护：跑到中途用**真实曲线**判断 3000 轮够不够，
# 而不是等跑完才发现不够。
#
#   nohup bash scripts/watch_p11_budget.sh > /dev/null 2>&1 &
#
# 为什么需要：上一轮用 512 环境跑 3000 轮，terrain_levels 在约 1800 轮
# 就卡在 1.3 不动了——那不是"练到头"，是样本量不足导致课程升不上去。
# 这次换成 4096，但 3000 轮到底够不够仍是未知数，得看曲线说话。
#
# 判断时机选 1500 轮（一半）：这时课程已经启动、趋势明朗，
# 又还剩一半时间可以做决定（续训 or 收工）。
set -uo pipefail

ROOT="${ROXAN_ROOT:-/home/limx/workspace/Roxan_warmup}"
PY="$ROOT/envs/isaaclab/bin/python"
LOG="$HOME/humanoid_logs/p11_parkour/p11_parkour_train.log"
PIPE="$HOME/humanoid_logs/pipeline/p11_budget.log"
mkdir -p "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

CHECK_AT="${P11_CHECK_AT:-1500}"
log "════ 实践 11 训练预算看护（到 $CHECK_AT 轮时评估）════"

while : ; do
    it=$(grep -oE "Learning iteration [0-9]+/" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    [ -z "$it" ] && { sleep 300; continue; }
    [ "$it" -ge "$CHECK_AT" ] && break
    # 训练已经退出就不用再等了
    pgrep -f "Instinct-Parkour" >/dev/null 2>&1 || {
        log "训练进程已退出（当前 $it 轮），提前评估"
        break
    }
    log "  当前 $it/$CHECK_AT"
    sleep 900
done

log "── 用真实曲线评估 ──"
run=$(ls -td "$ROOT"/repos/instinctlab/logs/instinct_rl/g1_parkour/*/ 2>/dev/null | head -1)
"$PY" - "$run" <<'PYEOF' 2>&1 | tee -a "$PIPE"
import sys, glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

files = glob.glob(sys.argv[1] + "/events.out.*")
if not files:
    print("  没有 tfevents"); raise SystemExit
ea = EventAccumulator(files[0], size_guidance={
    "scalars": 4000, "histograms": 1, "images": 1, "audio": 1, "tensors": 1})
ea.Reload()

def get(sfx):
    t = [x for x in ea.Tags()["scalars"] if x.endswith(sfx)]
    return np.array([x.value for x in ea.Scalars(t[0])]) if t else None

lv = get("terrain_levels")
if lv is None or len(lv) < 50:
    print("  数据不足，稍后再看"); raise SystemExit

n = len(lv)
print(f"  terrain_levels 当前 {lv[-1]:.2f} / 上限 9   （{n} 个记录点）")
print("  十等分:", [round(float(x.mean()), 2) for x in np.array_split(lv, 10)])

# 用最后 25% 的斜率外推到 3000 轮
tail = lv[int(n * 0.75):]
k = np.polyfit(np.arange(len(tail)), tail, 1)[0]
per_pt = 3000 / max(n, 1)          # 一个记录点约等于几轮
k_per_iter = k / max(per_pt, 1e-9)
proj = lv[-1] + k_per_iter * (3000 - n * per_pt)
print(f"  末段斜率 {k_per_iter:+.5f}/轮  →  外推到 3000 轮 ≈ {min(proj, 9):.2f}/9")

# 卡住判定：斜率接近零 = 样本量不足或已到能力上限
print()
if k_per_iter < 1e-5:
    print("  ⚠️ 课程已经不再推进（斜率≈0）")
    print("     上一轮 512 环境时就是这样卡在 1.3——那是样本量不足。")
    print("     这次已经是 4096，若仍卡住，说明是任务难度上限而非预算问题。")
elif proj >= 5:
    print("  ✅ 按当前速率，3000 轮能到中高难度地形，预算够用")
else:
    need = (5 - lv[-1]) / k_per_iter + n * per_pt
    print(f"  🔶 3000 轮只能到 {min(proj,9):.1f}/9。若想到 5.0，约需 {need:,.0f} 轮")
    print("     决策依据：这是'还能再练'而不是'练错了'——")
    print("     曲线在涨说明方向对，只是时间不够。")
PYEOF

log "════ 评估结束 ════"
