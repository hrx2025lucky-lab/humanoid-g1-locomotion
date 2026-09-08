#!/usr/bin/env bash
# 自动抓实践 10 的 RayCaster 命中点截图。
#
#   bash scripts/capture_p10_raycast.sh
#
# 为什么要脚本化：作业要一张"能看到射线网格与命中点"的截图，
# 原本的做法是开窗口然后人工按截图键。过夜跑的时候没人在，
# 手动那一步就会把整条流水线卡住。
#
# 做法：开有窗口的 play（不加 --headless），等 IsaacSim 把场景渲染出来，
# 再用 ImageMagick 的 import 抓那个窗口，最后结束进程。
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
HOI="$ROOT/shenlan_hw/HOI_Mimic"
PY="$ROOT/envs/isaaclab/bin/python"
TASK="Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast"
OUT_DIR="$HOME/humanoid_logs/p10_hoi"
LOG="$OUT_DIR/p10_raycast_capture.log"
DISP="${DISPLAY:-:1}"
mkdir -p "$OUT_DIR"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 这台机器上没有 ImageMagick(import) 和 xdotool，但有 ffmpeg + xwininfo。
# 不为截图去装新包（可能要 sudo，过夜跑时没人输密码），用已有的组合：
#   xwininfo 找窗口几何 → ffmpeg 按坐标从 X11 抓帧。
for c in ffmpeg xwininfo; do
    command -v "$c" >/dev/null 2>&1 || { log "❌ 缺少 $c"; exit 1; }
done

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
[ -n "$busy" ] && { log "❌ GPU 上还有计算进程（$busy），等训练结束再跑"; exit 1; }

cd "$HOI" || exit 1
[ -f set_project_root.sh ] && . ./set_project_root.sh >/dev/null 2>&1

log "启动带窗口的 play（HOI_RAYCAST_DEBUG_VIS=1）"
HOI_RAYCAST_DEBUG_VIS=1 \
PYTHONPATH="$HOI/source/unitree_rl_lab:$HOI:${PYTHONPATH:-}" \
DISPLAY="$DISP" "$PY" scripts/rsl_rl/play.py \
    --task "$TASK" --num_envs 1 > "$LOG" 2>&1 &
pid=$!

# 等窗口出现。IsaacSim 冷启动要一两分钟，别急着抓。
win=""
for i in $(seq 1 40); do
    sleep 15
    kill -0 "$pid" 2>/dev/null || { log "❌ play 进程已退出，最后 15 行："; tail -15 "$LOG"; exit 1; }
    # 找 Isaac 主窗口的 id。-root -tree 里同一个应用会有多个条目
    # （工具窗、菜单等），只认尺寸够大的那个才是主窗口——
    # 小条目的几何常是 1x1（还没 map 完），拿去裁剪会得到空图。
    WIN_ID=""
    for cand in $(DISPLAY="$DISP" xwininfo -root -tree 2>/dev/null \
                  | grep -iE "isaac|omniverse" | grep -oE "0x[0-9a-f]+"); do
        big=$(DISPLAY="$DISP" xwininfo -id "$cand" 2>/dev/null | awk '
            /^  Width:/ {w=$NF} /^  Height:/ {h=$NF}
            END {if (w>600 && h>400) print "yes"}')
        [ "$big" = "yes" ] && { WIN_ID="$cand"; break; }
    done
    [ -n "$WIN_ID" ] && { log "Isaac 主窗口 id=$WIN_ID（等了 $((i * 15))s）"; break; }
done
[ -z "$WIN_ID" ] && { log "❌ 10 分钟没等到 Isaac 窗口"; kill "$pid" 2>/dev/null; exit 1; }

# 屏幕尺寸从 xwininfo -root 的 Width/Height 取——已实测这两个字段可靠
SCR_W=$(DISPLAY="$DISP" xwininfo -root 2>/dev/null | awk '/Width:/{print $2}')
SCR_H=$(DISPLAY="$DISP" xwininfo -root 2>/dev/null | awk '/Height:/{print $2}')
[ -z "$SCR_W" ] && { log "❌ 取不到屏幕尺寸"; kill "$pid" 2>/dev/null; exit 1; }
log "屏幕 ${SCR_W}x${SCR_H}"

# 窗口刚出来时往往还是黑的或在加载资产，多等一会儿再抓，
# 并连抓三张——万一某一帧正好在切换，还有备份。
log "等场景渲染完成…"
sleep 90
ok=0
for n in 1 2 3; do
    # 每次抓之前确认 play 还活着。昨晚就是 play 早已崩掉
    # （KeyError: 'policy'），脚本却继续抓，拍下一张锁屏壁纸当"材料"。
    if ! kill -0 "$pid" 2>/dev/null; then
        log "  ❌ play 进程已退出，不再抓图。最后 15 行："
        tail -15 "$LOG"
        break
    fi
    shot="$OUT_DIR/p10_raycast_$(date +%H%M%S)_$n.png"
    full="$OUT_DIR/.full_$n.png"
    if ffmpeg -loglevel error -y -f x11grab -video_size "${SCR_W}x${SCR_H}" \
              -i "${DISP}+0,0" -frames:v 1 "$full" 2>>"$LOG"; then
        # 抓全屏再裁窗口，而不是让 ffmpeg 直接抓窗口区域：
        # 全屏抓取不会因为窗口几何暂时读成 1x1 而失败，
        # 裁剪又能保证交付材料里只有仿真画面，
        # 不会混进浏览器、聊天窗口这些无关内容。
        geo=$(DISPLAY="$DISP" xwininfo -id "$WIN_ID" 2>/dev/null | awk '
            /Absolute upper-left X/ {x=$NF}
            /Absolute upper-left Y/ {y=$NF}
            /^  Width:/  {w=$NF}
            /^  Height:/ {h=$NF}
            END {if (w>200 && h>200) print x, y, w, h}')
        if [ -n "$geo" ]; then
            set -- $geo
            "$PY" - "$full" "$shot" "$1" "$2" "$3" "$4" <<'PYEOF'
import sys
from PIL import Image
src, dst, x, y, w, h = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
Image.open(src).crop((x, y, x + w, y + h)).save(dst)
PYEOF
            rm -f "$full"
        else
            # 查不到几何就退回全屏，总比没有强
            mv "$full" "$shot"
        fi
        # 只看文件大小拦不住假图：昨晚那张锁屏也有 675 KB。
        # 改为验证图像内容——仿真画面色彩丰富，
        # 而锁屏/纯色背景的独立颜色数很少。
        n_colors=$("$PY" - "$shot" <<'PYEOF'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB")
im.thumbnail((640, 640))
colors = im.getcolors(maxcolors=1 << 24) or []
print(len(colors))
PYEOF
)
        if [ "${n_colors:-0}" -ge 3000 ]; then
            log "  ✅ $(basename "$shot")  独立颜色数 $n_colors"
            ok=$((ok + 1))
        else
            log "  ⚠️ $(basename "$shot") 颜色数只有 ${n_colors:-0}，像壁纸不像仿真画面，丢弃"
            rm -f "$shot"
        fi
    fi
    sleep 20
done

log "结束 play 进程"
kill "$pid" 2>/dev/null; sleep 5; kill -9 "$pid" 2>/dev/null
wait "$pid" 2>/dev/null

if [ "$ok" -gt 0 ]; then
    log "✅ 抓到 $ok 张截图 → $OUT_DIR"
    exit 0
fi
log "❌ 一张都没抓到"
exit 1
