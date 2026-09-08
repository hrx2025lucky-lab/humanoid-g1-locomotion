#!/usr/bin/env bash
# 看护实践 11 训练，跑完自动按**真实任务指标**验证。
#
# 为什么不只看 reward：实践 9 的 reward 由负转正（-0.75→6.98），
# 但真实跟踪误差在恶化——策略学的是"靠不摔倒混时长"而不是"跟得准"。
# 那次是靠录像看到画面才发现的，代价是 20000 轮白跑。
#
# 所以这里跑完就查：
#   1. tracking_exp_* 跟踪奖励是否在涨（真的在学跟踪）
#   2. error_* 涨了多少，能否被课程难度解释
#   3. bad_orientation 摔倒率是否下降
#
# 用法：nohup bash scripts/watch_p11_train.sh > /dev/null 2>&1 &
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
LOG="$HOME/humanoid_logs/p11_parkour/p11_train.log"
PIPE="$HOME/humanoid_logs/pipeline/p11_watch.log"
mkdir -p "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════ 看护实践 11 训练 ════"

while pgrep -f "Instinct-Parkour" >/dev/null 2>&1; do
    it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1)
    log "  ${it:-初始化}"
    sleep 1800
done

it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1)
log "训练结束：${it:-未知}"

run=$(ls -td "$ROOT"/repos/instinctlab/logs/instinct_rl/g1_parkour/*/ 2>/dev/null | head -1)
[ -z "$run" ] && { log "❌ 找不到 run 目录"; exit 1; }
log "run: $run"

"$ROOT/envs/isaaclab/bin/python" - <<PY 2>&1 | tee -a "$PIPE"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os
fs = sorted(glob.glob(os.path.join("$run", "events.out.tfevents.*")))
if not fs:
    print("  没有 tfevents"); raise SystemExit
ea = EventAccumulator(fs[0], size_guidance={"scalars": 0}); ea.Reload()
tags = ea.Tags()["scalars"]

def seg5(tag):
    if tag not in tags: return None
    v = [p.value for p in ea.Scalars(tag)]
    n = len(v)
    if n < 10: return None
    return [sum(v[i*n//5:(i+1)*n//5]) / max(1, len(v[i*n//5:(i+1)*n//5])) for i in range(5)]

def show(label, tag, lower_better=False):
    s = seg5(tag)
    if s is None:
        print(f"  {label:<28} 无数据"); return None
    good = (s[-1] < s[0]) if lower_better else (s[-1] > s[0])
    print(f"  {label:<28} " + "→".join(f"{x:.3f}" for x in s) + ("  ✅" if good else "  ⚠️"))
    return s

print("\n── 是否真的在学跟踪（奖励项应上涨）──")
tx = show("tracking_exp_vel_xy", next((t for t in tags if t.endswith("tracking_exp_vel_xy")), ""))
ty = show("tracking_exp_vel_yaw", next((t for t in tags if t.endswith("tracking_exp_vel_yaw")), ""))

print("\n── 稳定性（摔倒率应下降）──")
show("bad_orientation 终止", next((t for t in tags if t.endswith("bad_orientation")), ""), lower_better=True)

print("\n── 跟踪误差 vs 课程难度 ──")
ex = show("error_vel_xy", next((t for t in tags if t.endswith("error_vel_xy")), ""), lower_better=True)
# 实践 11 的前缀是 Episode/Curriculum/ 而不是 Curriculum/，两种都认
cur = [t for t in tags if "Curriculum/" in t]
cl = None
for t in cur[:2]:
    cl = show(t.split("/")[-1], t)

# ★ 关键判据：误差涨幅能否被课程难度解释 ★
if ex and cl and cl[0] > 0:
    err_ratio = ex[-1] / max(ex[0], 1e-9)
    cur_ratio = cl[-1] / cl[0]
    print(f"\n  误差涨 {err_ratio:.2f}× · 课程难度涨 {cur_ratio:.2f}×")
    if err_ratio <= cur_ratio * 1.3:
        print("  ✅ 误差涨幅在课程难度范围内，属正常")
    else:
        print("  ⚠️ 误差涨幅超出课程解释范围，按实践 9 的方法查奖励是否饱和：")
        print("     看 exp(-err²/std²) 在当前 err 量级上还有没有梯度")

print("\n── 结论 ──")
if tx and tx[-1] > tx[0] and ex:
    print("  跟踪奖励在涨 = 策略确实在学跟踪，不是靠混时长")
else:
    print("  ⚠️ 跟踪奖励没涨，需要人工查看")
PY

ck=$(find "$run" -name "*.pt" 2>/dev/null | grep -oE "model_[0-9]+" | sed 's/model_//' | sort -n | tail -1)
log "最大 checkpoint: model_${ck:-无}"
log "════ 看护结束 ════"
