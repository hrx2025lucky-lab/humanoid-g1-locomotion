#!/usr/bin/env bash
# 实践 9 重训：用**关节顺序已修正**的参考动作重新训练。
#
# ⚠️ 脚本名里的 "fixed_std" 是历史遗留。最早以为根因是指数核 std，
#    后来证明那个诊断是错的（见下），真正要修的是动作数据的关节列序。
#    保留文件名是为了不打断已经排好的队列脚本引用。
#
# 为什么要重训（完整诊断见 docs/实践9_自适应采样与轨迹跟踪.md 附五）：
#   参考动作的 joint_pos 是**广度优先**列序（逐层：左髋、右髋、腰…），
#   而 MuJoCo 模型是深度优先（逐链：左腿到底，再右腿，再腰，再双臂）。
#   loader 按 meta.json 的 body_names 重排了刚体（library.py:129-136），
#   但 meta 里没有 joint_names，关节是原样传递的（:143）——于是错位。
#
#   用正运动学交叉验证（拿 joint_pos 驱动模型，比对 body_pos_w）：
#     修正前 FK 误差 0.178 m（还不如随机排列）
#     修正后 FK 误差 0.0011 m   ← 163 倍
#   关节超限 11/29 → 1/29。修正前 right_ankle_pitch 被要求转 153°，
#   而限位只有 ±30°——机器人物理上做不到，跟踪误差当然降不下来。
#
#   这解释了那组矛盾曲线：error_body_pos ↓75%（刚体目标是对的）
#   而 error_joint_pos ↑97%（关节目标是错的，越练越偏）。
#
# ── 关于 std：曾经的误诊，记录在此避免重蹈 ──────────────────
#   我一度判断"std=0.3 让奖励饱和到 4e-18、梯度消失"。那是错的。
#   错因：把**指标** error_joint_pos = torch.norm(diff)（L2 范数）
#   代进了用**均值**的奖励公式 exp(-mean(diff²)/std²)，
#   漏除 29 个关节，等于把指数放大 29 倍。
#   实测 Reward_per_Sec/motion_joint_pos 一直是 0.643 → 0.469，从未饱和。
#   官方参考答案用的也是 std=0.3（teacher_env_cfg.py:344），保持不变。
#
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
HW6="$ROOT/shenlan_hw/hw6_distill"
LOG_DIR="$HOME/humanoid_logs/p9_retrain"
PIPE="$HOME/humanoid_logs/pipeline/p9_retrain.log"
mkdir -p "$LOG_DIR" "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

# 这次重训的理由：**跟踪目标本来就是错的**，不是超参问题。
#
# 原始 npz 的 joint_pos 是广度优先（BFS）列序，而模型是深度优先（DFS）。
# loader 只按 meta.json 的 body_names 重排了刚体（library.py:129-136），
# 关节是 `joint_pos=payload.joint_pos` 原样传的（:143），于是错位。
#
# 用正运动学验证（拿 joint_pos 驱动模型，比对 body_pos_w）：
#   原始顺序 FK 误差 0.178 m（还不如随机排列）
#   修正顺序 FK 误差 0.0011 m   ← 163 倍
# 超限关节 11/29 → 1/29。之前 right_ankle_pitch 被要求转 153°，
# 而限位只有 ±30°——机器人物理上做不到，误差当然降不下来。
#
# 这解释了那组矛盾曲线：error_body_pos ↓75% 而 error_joint_pos ↑97%。
# 刚体目标是对的所以跟得上，关节目标是错的所以越练越偏。
#
# std 保持课程默认的 0.3——官方参考答案就是这个值
#（course_code/shanlan_HW6/.../teacher_env_cfg.py:344）。
# 此前判断"std=0.3 让奖励饱和到 4e-18"是错的：那是把 L2 范数
#（指标 error_joint_pos = torch.norm）代进了用均值的奖励公式
#（exp(-mean(diff²)/std²)），漏除 29 个关节。实测奖励一直有 0.47。
STD="${HW9_JOINT_POS_STD:-0.3}"
ITERS="${P9_ITERS:-8000}"

log "════ 实践 9 重训（std=$STD）════"

# mjlab 走 MuJoCo Warp，与 IsaacSim 抢同一张卡
# ── GPU 互斥锁 ─────────────────────────────────────────────
# 逐个 pgrep 列举对方的任务名不可靠：新增任务时要改所有脚本，
# 漏一个就会两个训练同时抢卡。改用文件锁，谁先拿到谁跑。
GPU_LOCK="/tmp/humanoid_gpu.lock"
exec 9>"$GPU_LOCK"
log "等待 GPU 锁…"
flock 9
log "已获得 GPU 锁"

log "等 GPU 空闲…"
while pgrep -f "Instinct-Parkour" >/dev/null 2>&1 \
   || pgrep -f "AMP-WalkToRun" >/dev/null 2>&1 \
   || pgrep -f "Perceptive-Raycast" >/dev/null 2>&1; do
    sleep 300
done
sleep 60
log "GPU 空闲：$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)"

cd "$HW6" || { log "❌ 进不去 $HW6"; exit 1; }
T_LOG="$LOG_DIR/p9_std${STD}.log"
log "训练 $ITERS 轮 → $T_LOG"

# 命令形式与首次训练 20k 时一致（见 docs/实践9 §训练命令）：
# 入口是 ./.venv/bin/train，动作文件走 HW6_MOTION_SOURCE 环境变量，
# 其余参数是 tyro 的点号路径形式
HW6_MOTION_SOURCE=${HW9_MOTION:-motion_data_cfg_hw9_dance_fixed.yaml} \
HW9_JOINT_POS_STD="$STD" \
./.venv/bin/train Mjlab-Humanoid-HW6-Teacher-G1 \
    --env.scene.num-envs=4096 \
    --agent.max-iterations="$ITERS" \
    --agent.seed=42 \
    --agent.logger=tensorboard \
    --agent.run-name="hw9_dance_std${STD}" \
    > "$T_LOG" 2>&1
rc=$?

last=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$T_LOG" 2>/dev/null | tail -1)
log "训练结束 rc=$rc · ${last:-未知}"
if [ "$rc" -ne 0 ]; then
    log "最后 25 行："
    tail -25 "$T_LOG" | tee -a "$PIPE"
    exit 1
fi

# ★ 验收判据是 error_joint_pos 转为下降，不是 reward 涨了 ★
run=$(ls -td "$HW6"/logs/rsl_rl/g1_hw6_teacher/*std${STD}* 2>/dev/null | head -1)
if [ -n "$run" ]; then
    log "run 目录：$run"
    "$ROOT/envs/isaaclab/bin/python" - <<PY 2>&1 | tee -a "$PIPE"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os
f = sorted(glob.glob(os.path.join("$run", "events.out.tfevents.*")))[0]
ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
t = "Metrics/motion/error_joint_pos"
if t in ea.Tags()["scalars"]:
    v = [p.value for p in ea.Scalars(t)]; n = len(v)
    seg = [sum(v[i*n//5:(i+1)*n//5])/max(1, len(v[i*n//5:(i+1)*n//5])) for i in range(5)]
    print("  error_joint_pos: " + " → ".join(f"{s:.3f}" for s in seg))
    print(f"  旧 run（std=0.3）对照: 1.161 → 1.900（恶化）")
    print("  ✅ 修复有效" if seg[-1] < seg[0] else "  ⚠️ 仍未改善，需要查关节顺序")
else:
    print("  找不到 error_joint_pos")
PY
fi
log "════ 结束 ════"
