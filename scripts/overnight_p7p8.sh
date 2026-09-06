#!/usr/bin/env bash
# 过夜流水线：实践 7 批量重定向 → 补齐实践 8 的专家数据 → 实践 8 正式训练
#
# 实践 7 走 CPU（我们自建的脚本默认 device="cpu"），所以它可以和
# 实践 5 / 11 的 GPU 训练并行；实践 8 要用 GPU，必须排在它们后面。
#
# 用法：nohup bash scripts/overnight_p7p8.sh > /dev/null 2>&1 &
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
GMR="$ROOT/repos/GMR"
PY_GMR="$ROOT/envs/gmr/bin/python"
PY_LAB="$ROOT/envs/isaaclab/bin/python"
AMP="$ROOT/shenlan_hw/unitree_lab_amp"
ACCAD="$ROOT/datasets/AMASS/ACCAD"
OUT="$ROOT/datasets/g1_amp_npz"

LOG_DIR="$HOME/humanoid_logs"
PIPE="$LOG_DIR/pipeline/overnight_p7p8.log"
mkdir -p "$(dirname "$PIPE")" "$LOG_DIR/p7_retarget" "$LOG_DIR/p8_amp" "$OUT"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════════ 实践 7 → 8 过夜流水线 ════════"

# ─────────────────────────────────────────────────────────
# 阶段 1：批量重定向（CPU，与 GPU 训练并行，不用等）
# ─────────────────────────────────────────────────────────
R_LOG="$LOG_DIR/p7_retarget/batch_retarget.log"
log "阶段1 · 批量重定向 ACCAD → G1"
log "  源 $ACCAD"
log "  出 $OUT"
log "  日志 $R_LOG"

cd "$GMR" || { log "❌ 进不去 $GMR"; exit 1; }
# 参数名以 --help 为准：是 --overwrite 不是 --override，且没有 --num_cpus
#（照搬作业 PDF 里的写法会报 unrecognized arguments —— 那是上游原版脚本的参数）
# --profile walk_to_run 正好挑出实践 8 这个任务需要的片段
timeout 36000 "$PY_GMR" scripts/smplx_to_robot_dataset_npz.py \
    --src_folder "$ACCAD" --tgt_folder "$OUT" \
    --robot unitree_g1 --profile walk_to_run \
    > "$R_LOG" 2>&1
rc=$?

n_out=$(find "$OUT" -name "*.npz" 2>/dev/null | wc -l)
log "阶段1 结束 rc=$rc · 产出 $n_out 个 npz"
if [ "$n_out" -eq 0 ]; then
    log "❌ 没有产出，最后 25 行："
    tail -25 "$R_LOG" | tee -a "$PIPE"
    exit 1
fi

# 官方 §7 要求覆盖三类动作，逐类点名而不是只数总数
for cat in walk run turn; do
    c=$(find "$OUT" -iname "*${cat}*.npz" 2>/dev/null | wc -l)
    log "  类别 $cat: $c 条"
done

# ─────────────────────────────────────────────────────────
# 阶段 2：把专家数据接进实践 8
# ─────────────────────────────────────────────────────────
DATA="$AMP/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp/data"
log "阶段2 · 接入实践 8 的 motion 目录"
log "  $DATA"

before=$(find "$DATA" -name "*.npz" 2>/dev/null | wc -l)
# 挑基础 locomotion，避开武术/踢腿——作业 §3.2 明确说那些容易重定向失败
for f in $(find "$OUT" -name "*.npz" \
           ! -iname "*martial*" ! -iname "*kick*" ! -iname "*punch*" \
           ! -iname "*stance*" 2>/dev/null | head -40); do
    cp -n "$f" "$DATA/" 2>/dev/null
done
after=$(find "$DATA" -name "*.npz" 2>/dev/null | wc -l)
log "  专家数据 $before → $after 条"

has_run=$(find "$DATA" -iname "*run*.npz" -o -iname "*jog*.npz" 2>/dev/null | wc -l)
log "  其中跑步类 $has_run 条（此前为 0，是实践 8 唯一的缺口）"
if [ "$has_run" -eq 0 ]; then
    log "⚠️ 仍无跑步动作，实践 8 的三类覆盖不达标，但仍继续训练"
fi

# ─────────────────────────────────────────────────────────
# 阶段 3：等 GPU，然后训练实践 8
# ─────────────────────────────────────────────────────────
log "阶段3 · 等 GPU 空闲"
while pgrep -f "Navigation-HRL-RandomArena" >/dev/null 2>&1 \
   || pgrep -f "Instinct-Parkour" >/dev/null 2>&1; do
    sleep 300
done
sleep 90   # IsaacSim 退出后显存回收有延迟
log "  GPU 已空闲（$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)）"

T_LOG="$LOG_DIR/p8_amp/p8_train.log"
log "阶段3 · 实践 8 AMP 训练 → $T_LOG"
cd "$AMP" || { log "❌ 进不去 $AMP"; exit 1; }
# rsl_rl_amp 没被 pip 安装，靠 cwd 进 sys.path —— 必须在项目根目录跑
export PYTHONPATH="$AMP/source/unitree_rl_lab:$AMP:${PYTHONPATH:-}"

"$PY_LAB" scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-AMP-WalkToRun \
    --num_envs 4096 --max_iterations 3000 --headless \
    > "$T_LOG" 2>&1
rc=$?

last=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$T_LOG" 2>/dev/null | tail -1)
log "阶段3 结束 rc=$rc · ${last:-未知}"
[ "$rc" -ne 0 ] && { log "最后 25 行："; tail -25 "$T_LOG" | tee -a "$PIPE"; }

ck=$(find "$AMP/logs" -name "model_*.pt" -newermt "-12 hours" 2>/dev/null \
     | grep -oE "model_([0-9]+)" | sed 's/model_//' | sort -n | tail -1)
log "最大 checkpoint: model_${ck:-无}"

log "════════ 流水线结束 ════════"
