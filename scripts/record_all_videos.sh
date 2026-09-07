#!/usr/bin/env bash
# 批量录制各实践的验收视频。
#
# 作业普遍要求提交回放视频。逐个手敲 play 命令容易出错——
# 每套仿真器的参数形式不同（IsaacLab 用 --task 和下划线选项，
# mjlab 用位置参数和中划线，实践 6/9 用 uv run play），
# video_length 还要按各自的 episode 长度算。都固化在这里。
#
# 用法：
#   bash scripts/record_all_videos.sh              # 录全部（等 GPU 空闲）
#   bash scripts/record_all_videos.sh p5 p8        # 只录指定实践
#   bash scripts/record_all_videos.sh --list       # 看有哪些可录
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
PY_LAB="$ROOT/envs/isaaclab/bin/python"
LOG_DIR="$HOME/humanoid_logs/videos"
PIPE="$HOME/humanoid_logs/pipeline/record_all.log"
mkdir -p "$LOG_DIR" "$(dirname "$PIPE")"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

# ── 各实践的录制方式 ───────────────────────────────────────
# video_length = episode_length_s / step_dt，给小了视频会中途断
declare -A DESC=(
  [p2]="粗糙地形行走"
  [p4]="蹲姿行走（三组消融）"
  [p5]="分层导航（Baseline + RandomArena）"
  [p6]="蒸馏对照（KL vs Action）"
  [p8]="AMP 拟人走跑（FullPlay 全速度）"
  [p9]="BeyondMimic 轨迹跟踪"
  [p10]="HOI 感知跟踪"
  [p11]="跑酷"
)

record_p2() {
  cd "$ROOT/repos/unitree_rl_lab" || return 1
  PYTHONPATH="$PWD/source/unitree_rl_lab:$PWD" \
  "$PY_LAB" scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity-Rough \
      --num_envs 1 --video --video_length 1000 --headless
}

record_p5() {
  cd "$ROOT/shenlan_hw/hw5_navigation/unitree_rl_lab" || return 1
  export PYTHONPATH="$PWD/source/unitree_rl_lab:$PWD"
  # 两个任务共用 experiment_name，靠 --load_run 的时间戳区分
  "$PY_LAB" scripts/rsl_rl/play.py --task Unitree-G1-29dof-Navigation-HRL-Baseline \
      --num_envs 1 --video --video_length 750 --headless \
      --load_run 2026-09-05_20-10-32 || true
  "$PY_LAB" scripts/rsl_rl/play.py --task Unitree-G1-29dof-Navigation-HRL-RandomArena \
      --num_envs 1 --video --video_length 750 --headless \
      --load_run 2026-09-06_16-37-04
}

record_p8() {
  cd "$ROOT/shenlan_hw/unitree_lab_amp" || return 1
  # rsl_rl_amp 没被 pip 装，靠 cwd 进 sys.path —— 必须在项目根目录跑
  export PYTHONPATH="$PWD/source/unitree_rl_lab:$PWD"
  "$PY_LAB" scripts/rsl_rl/play.py --task Unitree-G1-29dof-AMP-WalkToRun-FullPlay \
      --num_envs 1 --video --video_length 1500 --headless
}

record_p10() {
  cd "$ROOT/shenlan_hw/HOI_Mimic" || return 1
  [ -f set_project_root.sh ] && . ./set_project_root.sh 2>/dev/null
  export PYTHONPATH="$PWD/source/unitree_rl_lab:$PWD"
  "$PY_LAB" scripts/rsl_rl/play.py \
      --task Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast \
      --num_envs 1 --video --video_length 1000 --headless
}

record_p11() {
  cd "$ROOT/repos/instinctlab" || return 1
  export PYTHONPATH="$ROOT/repos/instinct_rl:$PWD/source/instinctlab"
  "$PY_LAB" scripts/instinct_rl/play.py \
      --task Instinct-Parkour-Target-Amp-G1-Play-v0 \
      --num_envs 1 --video --video_length 1000 --headless
}

record_p4() {
  cd "$ROOT/shenlan_hw/hw4_mjlab" || return 1
  # mjlab：任务名是位置参数，选项用中划线
  local base="logs/rsl_rl/g1_velocity_height"
  for run in 2026-09-02_20-59-11_ablation_baseline \
             2026-09-02_21-52-14_ablation_no_height_rew \
             2026-09-02_22-45-34_ablation_blind_actor; do
    [ -d "$base/$run" ] || continue
    .venv/bin/python -m mjlab.scripts.play Mjlab-VelocityHeight-Flat-Unitree-G1 \
        --checkpoint-file "$base/$run/model_latest.pt" \
        --video True --video-length 1000 --video-width 1280 --video-height 720 || true
  done
}

record_p6() {
  cd "$ROOT/shenlan_hw/hw6_distill" || return 1
  local motion="src/humanoid_hw6/config/g1/motion_data_cfg_g1_accad_walk.yaml"
  uv run play Mjlab-Humanoid-HW6-Student-KL-Matching-G1 \
      --checkpoint-file logs/rsl_rl/g1_hw6_student_kl_matching/2026-09-05_21-49-13_compare_kl_matching/model_latest.pt \
      --motion-file "$motion" --num-envs 1 --video True --video-length 1000 || true
  uv run play Mjlab-Humanoid-HW6-Student-Action-Matching-G1 \
      --checkpoint-file logs/rsl_rl/g1_hw6_student_action_matching_aligned/2026-09-03_11-15-33_compare_action_matching/model_latest.pt \
      --motion-file "$motion" --num-envs 1 --video True --video-length 1000
}

record_p9() {
  cd "$ROOT/shenlan_hw/hw6_distill" || return 1
  uv run play Mjlab-Humanoid-HW6-Teacher-G1 \
      --checkpoint-file logs/rsl_rl/g1_hw6_teacher/2026-09-05_23-41-39_hw9_dance_20k/model_latest.pt \
      --motion-file src/humanoid_hw6/config/g1/motion_data_cfg_g1_accad_walk.yaml \
      --num-envs 1 --video True --video-length 1000
}

# ── 主流程 ───────────────────────────────────────────────
# 被自己以 --run-one 拉起时，只执行指定的那一个录制函数
if [ "${1:-}" = "--run-one" ]; then
    fn="record_${2:?缺少目标}"
    declare -F "$fn" >/dev/null || { echo "未知目标 $2"; exit 2; }
    "$fn"
    exit $?
fi

if [ "${1:-}" = "--list" ]; then
    echo "可录制的实践："
    for k in p2 p4 p5 p6 p8 p9 p10 p11; do
        printf "  %-5s %s\n" "$k" "${DESC[$k]}"
    done
    echo -e "\n实践 7 的录像另有脚本：scripts/record_p7_videos.py"
    exit 0
fi

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(p2 p5 p8 p10 p11 p4 p6 p9)

log "════ 批量录像：${TARGETS[*]} ════"

# GPU 独占：IsaacSim 实例并存会争显存
log "等 GPU 空闲…"
while pgrep -f "Instinct-Parkour" >/dev/null 2>&1 \
   || pgrep -f "AMP-WalkToRun" >/dev/null 2>&1 \
   || pgrep -f "Navigation-HRL" >/dev/null 2>&1 \
   || pgrep -f "Perceptive-Raycast" >/dev/null 2>&1; do
    sleep 300
done
sleep 90   # IsaacSim 退出后显存回收有延迟
log "GPU 空闲：$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)"

ok=0; fail=0
for t in "${TARGETS[@]}"; do
    fn="record_$t"
    if ! declare -F "$fn" >/dev/null; then
        log "⚠️ 未知目标 $t，跳过"; continue
    fi
    out="$LOG_DIR/${t}_record.log"
    log "▸ ${DESC[$t]:-$t} → $out"
    # timeout 只能跑外部程序，不能直接调 shell 函数。
    # 用 timeout 重新拉起本脚本、走 --run-one 分支，
    # 这样超时控制交给 timeout 自己，不用手写 watchdog
    #（第一版用后台 sleep 做 watchdog，那个 sleep 会拖住整个脚本退出）
    if timeout 3600 bash "$0" --run-one "$t" > "$out" 2>&1; then
        log "  ✅ 完成"; ok=$((ok+1))
    else
        rc=$?
        [ "$rc" -eq 124 ] && log "  ⏱ 超时 1 小时" \
                          || log "  ❌ 失败（rc=$rc），最后 8 行："
        tail -8 "$out" | sed 's/^/      /' | tee -a "$PIPE"
        fail=$((fail+1))
    fi
done

log "════ 结束：成功 $ok · 失败 $fail ════"
log "视频位置：各实践的 logs/**/videos/play/"
log "提交前压缩：ffmpeg -i in.mp4 -vcodec libx264 -crf 28 out.mp4"
