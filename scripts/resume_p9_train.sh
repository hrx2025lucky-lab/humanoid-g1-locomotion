#!/usr/bin/env bash
# 实践 9 续训到接近官方规模。
#
#   nohup bash scripts/resume_p9_train.sh > /dev/null 2>&1 &
#
# ── 为什么要续训 ────────────────────────────────────────
# 参考答案给的 Teacher 训练规模是 max_iterations=30_000
# （course_code/shanlan_HW6/.../config/g1/rl_cfg.py:58），
# 而我们上一轮只跑到 7063 轮就被 7 小时超时掐断了 —— 只有 23.5%。
#
# 掐断时所有指标**都还在下降**：
#   error_joint_pos   0.735  斜率 -0.000031/轮
#   error_anchor_pos  0.643  斜率 -0.000055/轮   ← 降得最快的一项
#   error_body_rot    0.217  斜率 -0.000002/轮
# 这是"练到一半被叫停"，不是"练到头了"。
#
# ── 一个差点走错的方向 ──────────────────────────────────
# 回放截图里绿色参考骨架和白色机器人隔着好几米，我一度判断是
# motion_global_root_pos 的 std=0.3 对世界系绝对距离太严（误差 0.696 m
# 时奖励只剩 0.0046，几乎没梯度），准备把 std 放宽到 0.8。
#
# 但对照参考答案后发现：奖励配置**逐字一致**（weight=0.5, std=0.3），
# env_cfgs / rl_cfg 也只差我自己加的环境变量开关。
# 既然配置没错、指标还在降，那就是轮数不够，不该改 std 去掩盖。
#
# 改 std 会让曲线立刻好看（奖励从 0.0046 跳到 0.47），
# 但那是把"没练够"伪装成"练好了"——而且偏离了官方基准，
# 面试时说不清为什么和参考答案不一样。
set -uo pipefail

ROOT="${ROXAN_ROOT:-/home/limx/workspace/Roxan_warmup}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW6="$ROOT/shenlan_hw/hw6_distill"
PIPE="$HOME/humanoid_logs/pipeline/p9_resume.log"
LOG_DIR="$HOME/humanoid_logs/p9_retrain"
mkdir -p "$(dirname "$PIPE")" "$LOG_DIR"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

# max_iterations 是**增量**不是总数（日志里显示的 "7125/20000" 是
# start_iter + max_iterations 算出来的总数）。
# 从 7000 轮续训、传 23000，实际跑到 30000 —— **对齐官方规模**
# （course_code/shanlan_HW6/.../config/g1/rl_cfg.py:58  max_iterations=30_000）。
#
# 一开始我设的是 13000（到 20000，官方的 2/3），理由是"折中"。
# 但既然已经确认根因就是训练量不足、且掐断时所有指标都还在降，
# 折中就没有道理——那只会得到一个"还是没练够"的结果，
# 而且说不清为什么不跑到官方基准。实测 2.58 s/轮，到 30000 需 16.4 h。
ITERS="${P9_RESUME_ITERS:-23000}"

log "════ 实践 9 续训 ════"

cd "$HW6" || exit 1
# 只认**真正有 checkpoint** 的 run。失败的启动（比如 wandb 没配 key）
# 也会留下一个空目录，按时间取最新就会取到它，然后报"没有 checkpoint"。
RUN=""; CK=""
for d in $(ls -td logs/rsl_rl/g1_hw6_teacher/2026-09-09_*/ 2>/dev/null); do
    c=$(ls "$d"model_*.pt 2>/dev/null | grep -oE "model_[0-9]+" \
        | grep -oE "[0-9]+" | sort -n | tail -1)
    [ -n "$c" ] && { RUN="$d"; CK="$c"; break; }
done
[ -z "$RUN" ] && { log "❌ 找不到带 checkpoint 的 2026-09-09 run"; exit 1; }
# load-run / load-checkpoint 都是**正则**（mjlab/utils/os.py:52 get_checkpoint_path）。
# checkpoint 名里的 . 在正则里是通配符，要转义成 \. 才是精确匹配，
# 否则 model_7000.pt 也能匹配到 model_7000Xpt 这类名字。
log "从 $(basename "${RUN%/}")/model_${CK}.pt 续训 $ITERS 轮（到约 $((CK + ITERS))）"

# ── GPU 互斥锁 ─────────────────────────────────────────
GPU_LOCK="/tmp/humanoid_gpu.lock"
exec 9>"$GPU_LOCK"
log "等待 GPU 锁…"
flock 9
log "已获得 GPU 锁"
while : ; do
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
    [ -z "$busy" ] && break
    log "  等 GPU 上的进程退出（$busy）"
    sleep 300
done
sleep 60

T_LOG="$LOG_DIR/p9_resume.log"
log "训练 → $T_LOG"
HW6_MOTION_SOURCE=motion_data_cfg_hw9_dance_fixed.yaml \
./.venv/bin/train Mjlab-Humanoid-HW6-Teacher-G1 \
    --env.scene.num-envs=4096 \
    --agent.max-iterations="$ITERS" \
    --agent.seed=42 \
    --agent.logger=tensorboard \
    --agent.resume=True \
    --agent.load-run="$(basename "${RUN%/}")" \
    --agent.load-checkpoint="model_${CK}\.pt" \
    > "$T_LOG" 2>&1
rc=$?
last=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$T_LOG" 2>/dev/null | tail -1)
log "训练结束 rc=$rc  最后：${last:-未知}"
[ "$rc" -ne 0 ] && { log "最后 25 行："; tail -25 "$T_LOG" | tee -a "$PIPE"; }

log "── 续训后验收 ──"
cd "$ROOT/motion control/humanoid_practice/g1_locomotion" 2>/dev/null \
  && "$ROOT/envs/isaaclab/bin/python" scripts/verify_training_outcome.py --practice 9 2>&1 | tee -a "$PIPE"
log "════ 结束 ════"
