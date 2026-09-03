#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 训练流水线 —— 等 GPU 空闲后按顺序跑下一批训练
# ═══════════════════════════════════════════════════════════════════════════
#
# 为什么需要串行排队：单个 4096 环境的训练会把 3090 打到 85~97% 利用率，
# 并行跑第二个只会互相拖慢，总时长不降反升（实践 2 期间实测）。
#
# 为什么不用 `wait`：训练进程是上一个会话用 nohup 启动的，不是本脚本的子进程，
# wait 拿不到它。改为轮询「有没有 train.py 在跑」。
#
# 用法：
#   ./run_pipeline.sh                  # 等当前训练结束 → 实践2收尾 → 实践4三组
#   ./run_pipeline.sh --now            # 不等待，立即开始（GPU 已空闲时用）
#   ITERS=3000 ./run_pipeline.sh       # 缩短每组迭代数
#   ./run_pipeline.sh --skip-finish    # 跳过实践2收尾，直接排实践4
#   ONLY=p6aligned ./run_pipeline.sh   # 只跑实践6的超参对齐重跑
#   ONLY=p6aligned_then_p5 ./run_pipeline.sh   # 先实践6对齐，再实践5
#
# 全程日志：~/pipeline.log
# 中断：kill 掉本脚本不会停掉已启动的训练，需另外 kill 对应 PID
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELOG="$HOME/pipeline.log"
ITERS="${ITERS:-3000}"

WAIT_FIRST=1
DO_FINISH=1
for a in "$@"; do
  case "$a" in
    --now)          WAIT_FIRST=0 ;;
    --skip-finish)  DO_FINISH=0 ;;
    *) echo "未知参数 $a" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$PIPELOG"; }

train_running() {
  pgrep -f "scripts/rsl_rl/train.py" > /dev/null 2>&1 || \
  pgrep -f "\.venv/bin/train" > /dev/null 2>&1
}

wait_for_gpu() {
  local waited=0
  while train_running; do
    if (( waited % 300 == 0 )); then
      # 从当前最新的训练日志里取进度。不能写死某一个文件 ——
      # 之前写死 /tmp/g1_resume.log，实践 2 结束后它就不再更新，
      # 等待信息会一直显示过期的 "9999/10000"，看着像卡住了。
      local latest it=""
      latest=$(ls -1t "$HOME"/p[456]_*.log /tmp/g1_resume.log 2>/dev/null | head -1)
      [[ -n "$latest" ]] && it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$latest" 2>/dev/null | tail -1)
      log "训练进行中${it:+（$it）}，已等待 $((waited / 60)) 分钟…"
    fi
    sleep 60
    waited=$((waited + 60))
  done
  log "GPU 已空闲"
  sleep 20   # 给显存释放留出余量
}

# ── 实践 2 收尾：指标 + 录像 + 导出 policy.pt ────────────────────────────
step_finish_p2() {
  log "════ 实践 2 收尾 ════"
  log "① 提取最终指标"
  "$HERE/finish_p2.sh" metrics 2>&1 | tee -a "$PIPELOG"

  log "② 录制 play 视频并导出 policy.pt / policy.onnx"
  if "$HERE/record_play.sh" >> "$PIPELOG" 2>&1; then
    log "   录像完成"
  else
    log "   ⚠️ 录像失败，见 $PIPELOG"
  fi

  log "③ 用自训策略跑 sim2sim 无头量化评估"
  local run
  run=$(ls -1d /home/limx/workspace/Roxan_warmup/repos/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_rough/*/ \
        2>/dev/null | sort | tail -1)
  if [[ -f "${run}exported/policy.pt" ]]; then
    /home/limx/workspace/Roxan_warmup/envs/isaaclab/bin/python \
      "$HERE/../sim2sim/eval_rough_headless.py" --run-dir "${run%/}" 2>&1 | tee -a "$PIPELOG"
  else
    log "   ⚠️ 没有 exported/policy.pt，跳过（录像步骤会生成它）"
  fi
}

# ── 实践 4：三组消融 ─────────────────────────────────────────────────────
step_p4() {
  log "════ 实践 4 消融实验（每组 $ITERS iter）════"
  for key in baseline no_height_rew blind_actor; do
    wait_for_gpu
    log "开始 实践4/$key"
    if ITERS="$ITERS" "$HERE/run_p4_ablation.sh" "$key" >> "$PIPELOG" 2>&1; then
      log "✅ 实践4/$key 完成"
    else
      log "❌ 实践4/$key 失败，见 ~/p4_${key}.log"
    fi
  done
}

# ── 实践 5：分层导航两组对照 ─────────────────────────────────────────────
step_p5() {
  log "════ 实践 5 分层导航对照（每组 $ITERS iter）════"
  for key in baseline random_arena; do
    wait_for_gpu
    log "开始 实践5/$key"
    if ITERS="$ITERS" "$HERE/run_p5p6_compare.sh" p5 "$key" >> "$PIPELOG" 2>&1; then
      log "✅ 实践5/$key 完成"
    else
      log "❌ 实践5/$key 失败，见 ~/p5_${key}.log"
    fi
  done
}

# ── 实践 6：两种蒸馏方式对照 ─────────────────────────────────────────────
step_p6() {
  log "════ 实践 6 蒸馏对照（每组 $ITERS iter）════"
  for key in action_matching kl_matching; do
    wait_for_gpu
    log "开始 实践6/$key"
    if ITERS="$ITERS" "$HERE/run_p5p6_compare.sh" p6 "$key" >> "$PIPELOG" 2>&1; then
      log "✅ 实践6/$key 完成"
    else
      log "❌ 实践6/$key 失败，见 ~/p6_${key}.log"
    fi
  done
}

# ── 实践 6 补充：超参对齐后的严格单因素对照 ──────────────────────────────
# 首轮两组除蒸馏目标外还差三处超参（lr / entropy_coef / desired_kl），
# 实测 KL 组全面更优，但那个差距无法归因到蒸馏目标本身。
# HW6_ALIGN_HPARAMS=1 让两组共用同一套超参，唯一变量才只剩蒸馏目标；
# 结果写到 *_aligned 目录，不覆盖首轮数据，两轮可对照。
step_p6_aligned() {
  log "════ 实践 6 超参对齐重跑（每组 $ITERS iter）════"
  for key in action_matching kl_matching; do
    wait_for_gpu
    log "开始 实践6-aligned/$key"
    if HW6_ALIGN_HPARAMS=1 ITERS="$ITERS" \
       "$HERE/run_p5p6_compare.sh" p6 "$key" >> "$PIPELOG" 2>&1; then
      log "✅ 实践6-aligned/$key 完成"
    else
      log "❌ 实践6-aligned/$key 失败，见 ~/p6_${key}.log"
    fi
  done
}

log "════════════════════════════════════════════"
log "流水线启动  ITERS=$ITERS  日志=$PIPELOG"
log "════════════════════════════════════════════"

if (( WAIT_FIRST )); then
  log "等待当前训练结束…"
  wait_for_gpu
fi

case "${ONLY:-}" in
  p6aligned) step_p6_aligned ;;
  p6aligned_then_p5) step_p6_aligned; step_p5 ;;
  p4)        step_p4 ;;
  p5)        step_p5 ;;
  p6)        step_p6 ;;
  *)
    (( DO_FINISH )) && step_finish_p2
    step_p4
    step_p5
    step_p6
    ;;
esac

log "════ 全部完成 ════"
log "实践4 曲线: tensorboard --logdir /home/limx/workspace/Roxan_warmup/shenlan_hw/hw4_mjlab/logs"
log "实践5 曲线: tensorboard --logdir /home/limx/workspace/Roxan_warmup/shenlan_hw/hw5_navigation/unitree_rl_lab/logs/rsl_rl"
log "实践6 曲线: tensorboard --logdir /home/limx/workspace/Roxan_warmup/shenlan_hw/hw6_distill/logs"
