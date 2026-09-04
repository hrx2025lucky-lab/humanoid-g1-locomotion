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
#   ONLY=p5_then_p9 ./run_pipeline.sh          # 先实践5，再实践9 P2
#   ONLY=p9 ./run_pipeline.sh                  # 只跑实践9 P2
#
# 全程日志：~/pipeline.log
# 中断：kill 掉本脚本不会停掉已启动的训练，需另外 kill 对应 PID
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=log_paths.sh
source "$HERE/log_paths.sh"
PIPELOG="$(hp_log_raw pipeline pipeline.log)"
HW6_DIR="/home/limx/workspace/Roxan_warmup/shenlan_hw/hw6_distill"
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
      # 日志集中到 $HP_LOG_DIR 后改成递归扫全部子目录，新增实践不用再改这里。
      local latest it=""
      latest=$(ls -1t "$HP_LOG_DIR"/*/p[0-9]_*.log /tmp/g1_resume.log 2>/dev/null | head -1)
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
      log "❌ 实践4/$key 失败，见 $(hp_log p4 "$key")"
    fi
  done
}

# ── 实践 5：分层导航两组对照 ─────────────────────────────────────────────
step_p5() {
  log "════ 实践 5 分层导航对照（每组 $ITERS iter）════"
  for key in baseline random_arena; do
    wait_for_gpu
    log "开始 实践5/$key"
    # 同时起健康巡检：实践 5 有"站着不动"这个局部最优，
    # 只看 episode_length / reward 会把它误判为健康（详见 docs 第九节）。
    P5_LOG="$(hp_log p5 "$key")" nohup "$HERE/watch_p5.sh" --loop 600 \
      > "$(hp_log p5 "watch_$key")" 2>&1 &
    local watch_pid=$!
    if ITERS="$ITERS" "$HERE/run_p5p6_compare.sh" p5 "$key" >> "$PIPELOG" 2>&1; then
      log "✅ 实践5/$key 完成"
    else
      log "❌ 实践5/$key 失败，见 $(hp_log p5 "$key")"
    fi
    kill "$watch_pid" 2>/dev/null || true
    log "   巡检结果见 $(hp_log p5 "watch_$key")"
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
      log "❌ 实践6/$key 失败，见 $(hp_log p6 "$key")"
    fi
  done
}

# 判断某个 aligned 组是否已经跑到目标迭代数。
# 依据是 checkpoint：instinct/mjlab 每 save_interval 存一次 model_<iter>.pt，
# 最大编号达到 ITERS-1 即视为跑完。只看目录存在是不够的 ——
# 中途被杀的 run 目录也在，那种要重跑。
p6_aligned_done() {
  local key="$1"
  local base="/home/limx/workspace/Roxan_warmup/shenlan_hw/hw6_distill/logs/rsl_rl"
  local d
  d=$(ls -1dt "$base/g1_hw6_student_${key}_aligned"/*/ 2>/dev/null | head -1)
  [[ -n "$d" ]] || return 1
  local last
  last=$(ls -1 "$d"model_*.pt 2>/dev/null \
         | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
  [[ -n "$last" ]] || return 1
  (( last >= ITERS - 1 ))
}

# ── 实践 6 补充：超参对齐后的严格单因素对照 ──────────────────────────────
# 首轮两组除蒸馏目标外还差三处超参（lr / entropy_coef / desired_kl），
# 实测 KL 组全面更优，但那个差距无法归因到蒸馏目标本身。
# HW6_ALIGN_HPARAMS=1 让两组共用同一套超参，唯一变量才只剩蒸馏目标；
# 结果写到 *_aligned 目录，不覆盖首轮数据，两轮可对照。
step_p6_aligned() {
  log "════ 实践 6 超参对齐重跑（每组 $ITERS iter）════"
  for key in action_matching kl_matching; do
    # 跳过已经跑完的组。流水线可能被重启（改脚本时），
    # 而上一实例启动的训练仍在跑；不检查的话会把同一组重跑一遍，
    # 每组 2.5 小时，白白浪费 GPU。
    if p6_aligned_done "$key"; then
      log "⏭  实践6-aligned/$key 已完成（$ITERS iter），跳过"
      continue
    fi
    wait_for_gpu
    log "开始 实践6-aligned/$key"
    if HW6_ALIGN_HPARAMS=1 ITERS="$ITERS" \
       "$HERE/run_p5p6_compare.sh" p6 "$key" >> "$PIPELOG" 2>&1; then
      log "✅ 实践6-aligned/$key 完成"
    else
      log "❌ 实践6-aligned/$key 失败，见 $(hp_log p6 "$key")"
    fi
  done
}

# ── 实践 9 P2：轨迹跟踪训练（舞蹈动作）──────────────────────────────────
# 实践 9 的作业包只有 MotionCommand 与动作数据，没有训练环境。
# 但实践 6 的 hw6_distill 正是一套完整的 BeyondMimic 跟踪框架，
# 把 dance1_subject2.npz 接进去即可完成 P2。
#
# 该 npz 不带刚体名，靠同目录 meta.json 提供；名字顺序是广度优先
#（IsaacLab articulation 的排列），已用物理量验证过：
# 脚是最低刚体 z≈0.10、肩最高 z≈1.03、躯干 0.79 在骨盆 0.75 之上。
step_p9() {
  log "════ 实践 9 P2 轨迹跟踪训练（$ITERS iter）════"
  wait_for_gpu
  log "开始 实践9/dance_tracking"
  local log_file; log_file="$(hp_log p9 dance)"
  cd "$HW6_DIR" || { log "❌ 找不到 $HW6_DIR"; return 1; }
  if HW6_MOTION_SOURCE=motion_data_cfg_hw9_dance.yaml \
     ./.venv/bin/train Mjlab-Humanoid-HW6-Teacher-G1 \
       --env.scene.num-envs="${NUM_ENVS:-4096}" \
       --agent.max-iterations="$ITERS" \
       --agent.seed=42 --agent.logger=tensorboard \
       --agent.run-name=hw9_dance_tracking > "$log_file" 2>&1; then
    log "✅ 实践9/dance_tracking 完成"
  else
    log "❌ 实践9/dance_tracking 失败，见 $log_file"
  fi
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
  p9)        step_p9 ;;
  p5_then_p9) step_p5; step_p9 ;;
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
