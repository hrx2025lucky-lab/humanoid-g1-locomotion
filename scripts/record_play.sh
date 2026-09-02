#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 录制 G1 粗糙地形策略的 play 视频 —— 固定录满一整个 episode
# ═══════════════════════════════════════════════════════════════════════════
#
# 为什么需要这个脚本，而不是每次手敲 play.py：
#
# 1) --video_length 必须是 1000，不能用更小的值。
#    episode_length_s = 20.0 s（velocity_env_cfg.py:377）
#    step_dt          = 0.02 s（decimation 4 × sim dt 0.005）
#    → 一个完整 episode = 20.0 / 0.02 = 1000 步
#    第一次录像用了 600，只拍到 60%，视频在机器人刚要走出出生平台时就断了。
#    录满一整个 episode 才能看到"重置 → 起步 → 巡航 → 超时"的完整过程，
#    也才能和 TensorBoard 里的 Mean episode length 直接对上。
#
# 2) --checkpoint 必须传绝对路径。
#    play.py:80-93 走 retrieve_file_path()，传相对路径会直接 FileNotFoundError。
#
# 用法：
#   ./record_play.sh                      # 自动选最新 run 的最新 checkpoint
#   ./record_play.sh /abs/path/model.pt   # 指定 checkpoint
#
# 产物（都在 checkpoint 所在 run 目录下）：
#   videos/play/rl-video-step-0.mp4       录像
#   exported/policy.onnx  policy.pt       导出的策略，实践 3 的 sim2sim 要用
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="/home/limx/workspace/Roxan_warmup/repos/unitree_rl_lab"
PY="/home/limx/workspace/Roxan_warmup/envs/isaaclab/bin/python"
TASK="Unitree-G1-29dof-Velocity-Rough"
EXP="unitree_g1_29dof_velocity_rough"

# 一个完整 episode 的步数：episode_length_s(20.0) / step_dt(0.02)
EPISODE_STEPS=1000

CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
  RUN=$(ls -1d "$REPO/logs/rsl_rl/$EXP"/*/ 2>/dev/null | sort | tail -1)
  [[ -n "$RUN" ]] || { echo "找不到任何训练 run，先跑训练" >&2; exit 1; }
  # 按 model_<数字>.pt 的数字排序取最大。不能用字典序：model_9999 会排在 model_10000 后面
  CKPT=$(ls -1 "$RUN"model_*.pt 2>/dev/null \
         | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
  [[ -n "$CKPT" ]] || { echo "run 目录里没有 checkpoint: $RUN" >&2; exit 1; }
fi
CKPT=$(readlink -f "$CKPT")

echo "checkpoint : $CKPT"
echo "录制步数   : $EPISODE_STEPS 步 = 20 秒（一个完整 episode）"

cd "$REPO"
"$PY" scripts/rsl_rl/play.py \
  --task "$TASK" \
  --num_envs 32 \
  --headless \
  --video \
  --video_length "$EPISODE_STEPS" \
  --checkpoint "$CKPT"

OUT="$(dirname "$CKPT")/videos/play/rl-video-step-0.mp4"
if [[ -f "$OUT" ]]; then
  echo
  echo "录像完成: $OUT"
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 "$OUT" 2>/dev/null || true
  echo
  echo "本机 Totem 缺 H.264 解码器（gstreamer1.0-libav 未装），用 Chrome 看："
  echo "  google-chrome --new-window \"file://$OUT\""
fi
