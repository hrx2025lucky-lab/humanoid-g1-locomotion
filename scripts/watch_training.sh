#!/usr/bin/env bash
# 训练实时看板 —— 一屏看清"跑到哪了 / 学会没有 / GPU 还剩多少"。
#
# 为什么不用 tail -f：
# rsl_rl 每轮刷几十行，滚动太快根本看不清；而且 reward 和 episode_length
# 这类"表面指标"在实践 5 里会骗人（站着不动也能拿高分、活得久）。
# 这里只挑真正能判断成败的指标，并保留最近几轮做趋势对比。
#
# 用法：
#   ./scripts/watch_training.sh                     # 自动跟最新的训练日志
#   ./scripts/watch_training.sh p5 fix_smoothing    # 指定某一组
#   ./scripts/watch_training.sh --once              # 只打一屏就退出

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=log_paths.sh
source "$HERE/log_paths.sh"

ONCE=0
[[ "${1:-}" == "--once" ]] && { ONCE=1; shift; }

if [[ $# -ge 2 ]]; then
  LOG="$(hp_log "$1" "$2")"
else
  # 挑"最近被写入过"的日志，而不是最近创建的 —— 训练结束后文件还在，
  # 但不再更新；按 mtime 排序能自动跟到当前真正在跑的那个。
  LOG=$(ls -1t "$HP_LOG_DIR"/*/p[0-9]_*.log 2>/dev/null \
        | grep -vE "watch|probe|runner" | head -1)
fi

[[ -z "${LOG:-}" || ! -f "$LOG" ]] && { echo "找不到训练日志（HP_LOG_DIR=$HP_LOG_DIR）"; exit 1; }

# 从日志里取某个指标的最后 N 个值。指标名含空格，所以取 $NF 而不是 $2。
metric() { grep -F "$1" "$LOG" 2>/dev/null | awk '{print $NF}' | tail -"${2:-1}" | tr '\n' ' '; }

draw() {
  clear 2>/dev/null
  local iter age
  iter=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$LOG" | tail -1)
  age=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))

  echo "════════════════════════════════════════════════════════════════"
  echo "  $(date '+%m-%d %H:%M:%S')   $(basename "$LOG")"
  echo "  ${iter:-（初始化中，IsaacSim 启动约需 3~5 分钟）}"
  echo "════════════════════════════════════════════════════════════════"

  # 超过 120 秒没有新输出，基本可以判定卡住或已结束
  if (( age > 120 )); then
    echo "  ⚠️  日志已 ${age}s 未更新（训练可能已结束或卡住）"
  fi

  echo
  echo "  ── 表面指标（好看≠学会了）──"
  printf "     %-24s %s\n" "Mean reward"        "$(metric 'Mean reward:' 3)"
  printf "     %-24s %s\n" "Mean episode length" "$(metric 'Mean episode length:' 3)"

  echo
  echo "  ── 真实任务指标 ──"
  # ★ 判据用 Episode_Termination/goal_reached 而不是 goals_reached。
  # 后者只在 update_goal_on_success=True（到达后重采新目标）时才累加，
  # 而 Baseline 用的是 SingleGoal 配置：到达即终止、不重采，
  # 所以 goals_reached **恒为 0**，是这个任务下的无效指标。
  # 实测踩过：修复 update_history 后 95.44% 的 episode 以到达目标结束，
  # 而 goals_reached 仍显示 0.0000，差点被误判为训练失败。
  printf "     %-24s %s\n" "到达终止占比" "$(metric 'Episode_Termination/goal_reached:' 3)"
  printf "     %-24s %s\n" "摔倒终止占比" "$(metric 'Episode_Termination/bad_orientation:' 3)"
  printf "     %-24s %s\n" "error_pos_2d"   "$(metric 'error_pos_2d:' 3)"
  printf "     %-24s %s\n" "position_progress" "$(metric 'position_progress:' 3)"

  echo
  echo "  ── 探索是否塌缩 ──"
  printf "     %-24s %s\n" "action noise std" "$(metric 'Mean action noise std:' 3)"

  echo
  echo "  ── 速度 / 资源 ──"
  printf "     %-24s %s\n" "Iteration time" "$(metric 'Iteration time:' 3)"
  printf "     %-24s %s\n" "GPU" \
    "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)"

  # 预估剩余时间：用最近几轮的平均耗时，比用总平均更贴近当前状态
  if [[ -n "$iter" ]]; then
    local cur tot spi
    cur=$(echo "$iter" | grep -oE "[0-9]+/" | tr -d '/')
    tot=$(echo "$iter" | grep -oE "/[0-9]+" | tr -d '/')
    spi=$(grep -oE "Iteration time: [0-9.]+" "$LOG" | awk '{print $3}' | tail -20 \
          | awk '{s+=$1; n++} END {if(n) printf "%.2f", s/n}')
    if [[ -n "$spi" && -n "$cur" && -n "$tot" ]]; then
      awk -v c="$cur" -v t="$tot" -v s="$spi" 'BEGIN{
        r=(t-c)*s; printf "\n     预计剩余  %.1f 小时（剩 %d 轮 × %.2fs）\n", r/3600, t-c, s}'
    fi
  fi

  echo
  echo "  日志: $LOG"
  (( ONCE == 0 )) && echo "  每 15 秒刷新，Ctrl-C 退出"
}

if (( ONCE )); then
  draw
else
  while true; do draw; sleep 15; done
fi
