#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 实践 5 训练健康巡检 —— 专抓"站着不动"这类局部最优
# ═══════════════════════════════════════════════════════════════════════════
#
# 为什么需要它：实践 2 已经栽过一次 —— episode_length 与 reward 全线上涨，
# 录像里机器人却在原地踏步。导航任务里这个陷阱更强：
# 摔倒罚 -400，而"站着不动"稳拿 0，理性策略当然不动。
#
# 所以不能只看 episode_length / time_out / reward 这三个"表面指标"，
# 必须同时盯真实任务指标（净进展、到达率、距离误差）。
#
# 判据（三条同时成立才算健康）：
#   ① Mean episode length 不低于上限的 60%   —— 没在频繁摔倒
#   ② position_progress > 0                  —— 真的在朝目标移动
#   ③ error_pos_2d 相比训练早期在下降         —— 距离确实缩小了
#
# 用法：
#   ./watch_p5.sh              # 巡检一次
#   ./watch_p5.sh --loop 600   # 每 600 秒巡检一次，直到训练结束
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

LOG="${P5_LOG:-$HOME/p5_baseline.log}"
INTERVAL=0
[[ "${1:-}" == "--loop" ]] && INTERVAL="${2:-600}"

# 指标名可能含空格（如 "Mean episode length"），取 $NF 而不是 $2 ——
# 用 $2 会取到名字里的第二个词（"episode"）而不是数值。
last() { grep -oE "$1: [-0-9.]+" "$LOG" 2>/dev/null | tail -1 | awk '{print $NF}'; }
nth_from_start() { grep -oE "$1: [-0-9.]+" "$LOG" 2>/dev/null | head -"${2:-5}" | tail -1 | awk '{print $NF}'; }

check_once() {
  local iter ep_len prog goals err_now err_early term_out bad
  iter=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1)
  ep_len=$(last "Mean episode length")
  prog=$(last "Episode_Reward/position_progress")
  goals=$(last "Metrics/pose_command/goals_reached")
  err_now=$(last "Metrics/pose_command/error_pos_2d")
  err_early=$(nth_from_start "Metrics/pose_command/error_pos_2d" 5)
  term_out=$(last "Episode_Termination/time_out")
  bad=$(last "Episode_Termination/bad_orientation")

  echo "════════════════════════════════════════════════════════════"
  echo "  $(date +%H:%M:%S)   ${iter:-（尚无迭代记录）}"
  echo "════════════════════════════════════════════════════════════"
  echo "  ── 表面指标（好看不代表学会了）──"
  printf "     %-26s %s\n" "Mean episode length" "${ep_len:-—}"
  printf "     %-26s %s\n" "time_out 占比" "${term_out:-—}"
  printf "     %-26s %s\n" "bad_orientation 占比" "${bad:-—}"
  echo "  ── 真实任务指标 ──"
  printf "     %-26s %s\n" "position_progress" "${prog:-—}"
  printf "     %-26s %s\n" "goals_reached" "${goals:-—}"
  printf "     %-26s %s  (早期 ${err_early:-—})\n" "error_pos_2d" "${err_now:-—}"

  # 判据
  local verdict="✅ 健康" detail=""
  if [[ -n "${ep_len:-}" ]] && awk "BEGIN{exit !($ep_len < 90)}"; then
    verdict="⚠️  摔倒偏多"; detail="episode length $ep_len 低于上限 150 的 60%"
  elif [[ -n "${prog:-}" ]] && awk "BEGIN{exit !($prog <= 0)}"; then
    verdict="❌ 疑似「站着不动」局部最优"
    detail="position_progress=$prog ≤ 0，机器人没在朝目标移动——
       这与实践 2 的原地踏步是同一类问题：摔倒罚 -400，不动稳拿 0。
       若持续到 1000 iter 之后仍无进展，需要重新平衡
       position_progress 与 termination_penalty 的边际激励比。"
  elif [[ -n "${err_now:-}" && -n "${err_early:-}" ]] \
       && awk "BEGIN{exit !($err_now >= $err_early)}"; then
    verdict="⚠️  距离误差未下降"; detail="error_pos_2d $err_early → $err_now"
  fi
  echo
  echo "  判定：$verdict"
  [[ -n "$detail" ]] && echo "       $detail"
  echo
}

if (( INTERVAL > 0 )); then
  # 由流水线启动时训练可能还没起来（Isaac Sim 启动要 1~2 分钟），
  # 先等它出现，否则循环条件一上来就是假，巡检会立刻退出。
  for _ in $(seq 1 30); do
    pgrep -f "rsl_rl/train.py" > /dev/null && break
    sleep 10
  done
  while pgrep -f "rsl_rl/train.py" > /dev/null; do
    check_once
    sleep "$INTERVAL"
  done
  echo "训练已结束，最后一次巡检："
  check_once
else
  check_once
fi
