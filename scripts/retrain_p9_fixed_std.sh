#!/usr/bin/env bash
# 实践 9 重训：放宽 motion_joint_pos 的指数核 std，解决梯度消失。
#
# 为什么要重训（诊断见 docs/实践9_自适应采样与轨迹跟踪.md 附二）：
#   奖励 = exp(-mean(err²)/std²)，课程默认 std=0.3 假定 err 在 0.3 量级。
#   我们的 err 从训练第一步就是 1.16，代入得 3e-07 —— 这一项从头到尾
#   饱和为零，梯度消失。策略转去优化还有梯度的 body_lin_vel，
#   关节角越跑越偏：err 20000 轮里从 1.161 涨到 1.900，
#   比"完全静止不动"的 0.395 还差 5 倍。
#
#   std=1.0 时 err=1.16 → 奖励 0.26，梯度恢复。
#
# 这不是调参碰运气：改前先算过各阶段的奖励值，确认了饱和；
# 改后要验证 error_joint_pos 转为下降，而不是只看 reward 涨没涨。
#
# 用法：nohup bash scripts/retrain_p9_fixed_std.sh > /dev/null 2>&1 &
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
HW6="$ROOT/shenlan_hw/hw6_distill"
LOG_DIR="$HOME/humanoid_logs/p9_retrain"
PIPE="$HOME/humanoid_logs/pipeline/p9_retrain.log"
mkdir -p "$LOG_DIR" "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

# ⚠️ std 保持课程默认的 0.3——这是官方参考答案的值
#   （course_code/shanlan_HW6/.../teacher_env_cfg.py:344 params={"std": 0.3}）
#
# 曾经我判断"std=0.3 让奖励饱和到 4e-18、梯度消失"，那个结论是**错的**。
# 错因是把两个不同的量当成了同一个：
#   指标 error_joint_pos = torch.norm(diff)        ← L2 范数（base.py:782）
#   奖励               = exp(-mean(diff²)/std²)   ← 均值（rewards.py:25）
# norm² = N × mean(diff²)，N=29 个关节。我直接把 norm 代进奖励公式，
# 等于把指数项放大了 29 倍，于是 0.28 被算成了 5.9e-17。
#
# 实测奖励值 Reward_per_Sec/motion_joint_pos = 0.643 → 0.469，
# 梯度充足，根本没有饱和。用正确公式复算也得 0.72 → 0.28，与实测吻合。
#
# 所以这个脚本不改 std。真正的问题另有原因（见 docs/实践9 附三）。
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
HW6_MOTION_SOURCE=motion_data_cfg_hw9_dance.yaml \
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
