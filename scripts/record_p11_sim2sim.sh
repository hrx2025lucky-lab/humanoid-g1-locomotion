#!/usr/bin/env bash
# 录制实践 11 的 sim2sim 运行画面。
#
# course_code/sim2sim/sim2sim.py 没有 --video 参数，它开的是
# MuJoCo viewer + 两个 cv2 深度图窗口。所以用 ffmpeg 抓 X display 录屏 ——
# 这样一次能把三个窗口（机器人 + 原始深度图 + 归一化深度图）都录进去，
# 正好对应作业要求展示的"深度感知 + 部署闭环"。
#
# 注意这不占 GPU 的 CUDA 上下文（MuJoCo 走 CPU 物理 + 软件渲染），
# 所以能和 IsaacSim 训练并行跑。
#
# 用法：
#   bash scripts/record_p11_sim2sim.sh            # 录 parkour
#   bash scripts/record_p11_sim2sim.sh stand 25   # 录 stand，25 秒
set -uo pipefail

TASK="${1:-parkour}"
SECONDS_LEN="${2:-30}"
DISPLAY_ID="${DISPLAY:-:1}"

ROOT="/home/limx/workspace/Roxan_warmup"
SIM="$ROOT/motion control/humanoid_practice/course_code/sim2sim/sim2sim"
PY="$ROOT/envs/isaaclab/bin/python"
OUT_DIR="$HOME/humanoid_logs/p11_parkour/videos"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/p11_sim2sim_${TASK}.mp4"

echo "录制 sim2sim --task $TASK，${SECONDS_LEN} 秒 → $OUT"

[ -d "$SIM" ] || { echo "❌ 找不到 $SIM"; exit 1; }

# 先确认 X display 能用，否则 ffmpeg 会录出全黑
if ! DISPLAY="$DISPLAY_ID" xdpyinfo >/dev/null 2>&1; then
    echo "❌ DISPLAY=$DISPLAY_ID 不可用"
    exit 1
fi
SCREEN=$(DISPLAY="$DISPLAY_ID" xdpyinfo 2>/dev/null \
         | awk '/dimensions:/{print $2; exit}')
echo "  屏幕 $SCREEN"

cd "$SIM" || exit 1

# 启动 sim2sim（后台），给它时间把窗口画出来
DISPLAY="$DISPLAY_ID" nohup "$PY" sim2sim.py --task "$TASK" \
    > /tmp/p11_sim2sim_$TASK.log 2>&1 &
SIM_PID=$!
sleep 12

if ! kill -0 "$SIM_PID" 2>/dev/null; then
    echo "❌ sim2sim 启动失败，日志末尾："
    tail -12 /tmp/p11_sim2sim_$TASK.log
    exit 1
fi
echo "  sim2sim 运行中 (PID $SIM_PID)"

# 只抓 MuJoCo 窗口，不录整个桌面 ——
# 第一版录全屏，把浏览器和聊天窗口都录进去了，不能当提交材料。
# 没有 xdotool/wmctrl，用 xwininfo 按窗口名取几何位置。
WIN_INFO=$(DISPLAY="$DISPLAY_ID" xwininfo -name "MuJoCo" 2>/dev/null) || \
WIN_INFO=$(DISPLAY="$DISPLAY_ID" xwininfo -root -tree 2>/dev/null \
           | grep -i mujoco | head -1 | grep -oE '0x[0-9a-f]+' \
           | head -1 | xargs -r -I{} sh -c 'DISPLAY='"$DISPLAY_ID"' xwininfo -id {}')

if [ -n "$WIN_INFO" ]; then
    WX=$(echo "$WIN_INFO" | awk '/Absolute upper-left X/{print $NF}')
    WY=$(echo "$WIN_INFO" | awk '/Absolute upper-left Y/{print $NF}')
    WW=$(echo "$WIN_INFO" | awk '/^  Width:/{print $NF}')
    WH=$(echo "$WIN_INFO" | awk '/^  Height:/{print $NF}')
    # ffmpeg 的 x264 要求宽高为偶数
    WW=$((WW / 2 * 2)); WH=$((WH / 2 * 2))
    GEOM="${WW}x${WH}"
    GRAB="${DISPLAY_ID}+${WX},${WY}"
    echo "  只抓 MuJoCo 窗口 ${GEOM} @ (${WX},${WY})"
else
    GEOM="$SCREEN"; GRAB="$DISPLAY_ID"
    echo "  ⚠️ 找不到 MuJoCo 窗口，退回全屏录制"
fi

echo "  开始录屏…"

# -draw_mouse 0 去掉鼠标指针；crf 28 控制体积
ffmpeg -y -loglevel error \
    -f x11grab -framerate 25 -video_size "$GEOM" -draw_mouse 0 \
    -i "$GRAB" -t "$SECONDS_LEN" \
    -vcodec libx264 -crf 28 -preset fast -pix_fmt yuv420p \
    "$OUT" 2>&1 | tail -3

kill "$SIM_PID" 2>/dev/null
sleep 2

if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" -gt 20000 ]; then
    echo "✅ $(basename "$OUT")  $(du -h "$OUT" | cut -f1)"
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,nb_frames -of csv=p=0 "$OUT" 2>/dev/null \
        | xargs echo "   "
else
    echo "❌ 视频没生成或太小"
    tail -8 /tmp/p11_sim2sim_$TASK.log
    exit 1
fi
