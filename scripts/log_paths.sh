#!/usr/bin/env bash
# 训练日志路径的**单一真源**。所有会写日志的脚本都 source 这个文件。
#
# 为什么要单独抽出来：
# 之前每个脚本各自写死 "$HOME/p5_baseline.log"，日志全堆在主目录里，
# 既难和实践对应，也没法整体搬走。改一次路径要动四个脚本，容易漏。
#
# 覆盖方式（不改代码）：
#   HP_LOG_DIR=/data/logs ./scripts/run_pipeline.sh
#
# 目录按实践编号分组，与 docs/ 下的实践文档一一对应，
# 排障时可以顺着 humanoid_logs/README.md 找到对应的分析文档和诊断脚本。

HP_LOG_DIR="${HP_LOG_DIR:-$HOME/humanoid_logs}"

# 实践编号 → 子目录名。新增实践时只改这里。
hp_log_subdir() {
  case "$1" in
    p2) echo "p2_rough_terrain" ;;
    p4) echo "p4_dual_command" ;;
    p5) echo "p5_navigation" ;;
    p6) echo "p6_distill" ;;
    p9) echo "p9_beyondmimic" ;;
    pipeline) echo "pipeline" ;;
    *) echo "misc" ;;
  esac
}

# 用法: hp_log p5 baseline  ->  /home/limx/humanoid_logs/p5_navigation/p5_baseline.log
# 顺带建目录，调用方不必各自 mkdir。
hp_log() {
  local practice="$1" name="$2"
  local dir="$HP_LOG_DIR/$(hp_log_subdir "$practice")"
  mkdir -p "$dir"
  echo "$dir/${practice}_${name}.log"
}

# 给已有完整文件名的日志用（如 pipeline.log、g1_rough_train2.log）
hp_log_raw() {
  local practice="$1" filename="$2"
  local dir="$HP_LOG_DIR/$(hp_log_subdir "$practice")"
  mkdir -p "$dir"
  echo "$dir/$filename"
}
