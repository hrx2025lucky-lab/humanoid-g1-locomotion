#!/usr/bin/env bash
# 明早收尾用：一条命令看全今晚的结果 + 检查交付完整性。
#
#   bash scripts/morning_report.sh
#
# 把分散在几个地方的信息汇总到一屏，省得明早到处翻日志。
set -uo pipefail

ROOT="${ROXAN_ROOT:-/home/limx/workspace/Roxan_warmup}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/envs/isaaclab/bin/python"

B="\033[1m"; G="\033[32m"; Y="\033[33m"; D="\033[2m"; N="\033[0m"

echo -e "\n${B}══════ 过夜训练晨报 $(date '+%m-%d %H:%M') ══════${N}\n"

# ── 1. 还在跑吗 ────────────────────────────────────────
echo -e "${B}【当前状态】${N}"
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
if [ -n "$busy" ]; then
    echo -e "  ${Y}训练仍在进行${N}（PID $busy）"
else
    echo -e "  ${G}GPU 已空闲，训练全部结束${N}"
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'

# ── 2. 实践 11 结果 ────────────────────────────────────
echo -e "\n${B}【实践 11 · 跑酷】${N}"
it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" \
     "$HOME/humanoid_logs/p11_parkour/p11_parkour_train.log" 2>/dev/null | tail -1)
echo "  进度：${it:-未知}"
"$PY" - <<'PYEOF'
import glob, os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ds = sorted(glob.glob(os.path.expanduser(
    "~/workspace/Roxan_warmup/repos/instinctlab/logs/instinct_rl/g1_parkour/*/")),
    key=os.path.getmtime)
if not ds:
    print("  没有日志"); raise SystemExit
fs = glob.glob(ds[-1] + "events.out.*")
if not fs:
    print("  没有 tfevents"); raise SystemExit
ea = EventAccumulator(fs[0], size_guidance={
    "scalars": 4000, "histograms": 1, "images": 1, "audio": 1, "tensors": 1})
ea.Reload()
t = [x for x in ea.Tags()["scalars"] if x.endswith("terrain_levels")]
if t:
    v = np.array([x.value for x in ea.Scalars(t[0])])
    print(f"  terrain_levels 最终 {v[-1]:.2f} / 上限 9")
    print("  五段:", [round(float(x.mean()), 2) for x in np.array_split(v, 5)])
    # 上一轮 512 环境的对照
    print(f"  {'':2}对照：上一轮用 512 环境跑满 3000 轮只到 1.35")
ck = [f for f in os.listdir(ds[-1]) if f.startswith("model_")]
if ck:
    best = max(int(f[6:-3]) for f in ck if f[6:-3].isdigit())
    print(f"  checkpoint：{len(ck)} 个，最大 model_{best}.pt")
PYEOF

# ── 3. 实践 9 结果（今晚重点）────────────────────────────
echo -e "\n${B}【实践 9 · BeyondMimic 轨迹跟踪】${N}"
echo -e "  ${D}关节顺序修复后的首次验证——重点看误差有没有转为下降${D}${N}"
"$PY" - <<'PYEOF'
import glob, os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ds = sorted(glob.glob(os.path.expanduser(
    "~/workspace/Roxan_warmup/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_teacher/*/")),
    key=os.path.getmtime)
if not ds:
    print("  还没有训练日志"); raise SystemExit
newest = ds[-1]
name = os.path.basename(newest.rstrip("/"))
# 关键：区分"修复后的新 run"和"修复前的旧 run"。
# 旧 run 名里有 hw9_dance_20k，是 9-05 那次用错误关节顺序训的。
# 不区分的话，万一今晚实践 9 没跑起来，会拿旧数据误判成"修复无效"。
import datetime
mtime = datetime.datetime.fromtimestamp(os.path.getmtime(newest))
is_new = mtime > datetime.datetime(2026, 9, 8, 19, 0)
print(f"  run: {name}   ({mtime:%m-%d %H:%M})")
if not is_new:
    print("  ⚠️ 这是**修复前**的旧 run——今晚的实践 9 训练没跑起来或还没开始")
    print("     下面的数字不能用来判断关节顺序修复的效果")
fs = glob.glob(newest + "events.out.*")
if not fs:
    print("  没有 tfevents（可能刚启动）"); raise SystemExit
ea = EventAccumulator(fs[0], size_guidance={
    "scalars": 9000, "histograms": 1, "images": 1, "audio": 1, "tensors": 1})
ea.Reload()
for sfx, label in [("error_joint_pos", "关节位置误差"),
                   ("error_body_pos", "刚体位置误差"),
                   ("mean_episode_length", "存活时长")]:
    t = [x for x in ea.Tags()["scalars"] if x.endswith(sfx)]
    if not t:
        continue
    v = np.array([x.value for x in ea.Scalars(t[0])])
    if len(v) < 5:
        continue
    seg = [round(float(x.mean()), 3) for x in np.array_split(v, 5)]
    trend = "↓ 下降" if seg[-1] < seg[0] else "↑ 上升"
    flag = ""
    if sfx == "error_joint_pos":
        # 修复前那次是 0.932 → 1.834（恶化 97%）
        flag = "  ← 修复前是 0.932→1.834（恶化）"
    print(f"  {label:<12} {seg[0]:>7.3f} → {seg[-1]:>7.3f}  {trend}{flag}")
PYEOF

# ── 4. 全量验收 ────────────────────────────────────────
echo -e "\n${B}【效果验收】${N}"
"$PY" "$HERE/verify_training_outcome.py" 2>/dev/null | sed -n '/^2 /,/^====/p' | head -20

echo -e "\n${B}【材料清单】${N}"
"$PY" "$HERE/check_deliverables.py" 2>/dev/null | tail -8

echo -e "\n${B}【今日待办】${N}"
echo "  1. 用真实数据补全简历条目里的【待填】"
echo "     → ../私密_求职材料/简历_人形项目条目.md"
echo "  2. 把条目插入 黄若轩_简历_运控方向.md 的「项目经历」第一位"
echo "  3. 更新 teaching/11 里的训练结果（现在写的是待重跑）"
echo "  4. git push（如有新提交）"
echo ""
