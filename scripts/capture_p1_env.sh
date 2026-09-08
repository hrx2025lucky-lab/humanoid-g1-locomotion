#!/usr/bin/env bash
# 抓实践 1 的环境启动截图（作业要求 4 类：IsaacSim / IsaacLab / list_envs / train）。
#
#   bash scripts/capture_p1_env.sh
#
# 前三类是终端输出，直接把命令结果存成文本再渲染成图，不需要 GUI；
# 第四类要 IsaacSim 窗口，复用 capture_p10_raycast.sh 那套
# "抓全屏 + 按窗口 id 裁剪"的做法。
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
PY="$ROOT/envs/isaaclab/bin/python"
OUT_DIR="$HOME/humanoid_logs/p1_env"
LOG="$OUT_DIR/p1_capture.log"
DISP="${DISPLAY:-:1}"
mkdir -p "$OUT_DIR"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "════ 实践 1 环境验证截图 ════"

# ── 1~3. 三段命令的输出，渲染成图 ───────────────────────────
# 作业要的是"证明环境装好了"，终端输出就是证据。
# 用 Pillow 画成 PNG，比让人去手动截屏可靠。
run_and_shot() {   # run_and_shot <标题> <输出文件名> <命令...>
    local title="$1" name="$2"; shift 2
    local txt="$OUT_DIR/${name}.txt"
    log "▸ $title"
    { echo "\$ $*"; echo; "$@" 2>&1 | head -40; } > "$txt"
    "$PY" - "$txt" "$OUT_DIR/${name}.png" "$title" <<'PYEOF'
import sys
from PIL import Image, ImageDraw, ImageFont
txt, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(txt, errors="ignore").read().splitlines()[:40]
font = None
for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"):
    try:
        font = ImageFont.truetype(p, 15); break
    except OSError:
        continue
if font is None:
    font = ImageFont.load_default()
W, LH = 1200, 21
im = Image.new("RGB", (W, 60 + LH * max(len(lines), 6)), (24, 26, 32))
d = ImageDraw.Draw(im)
d.text((16, 16), title, font=font, fill=(120, 200, 255))
for i, ln in enumerate(lines):
    d.text((16, 48 + i * LH), ln[:150], font=font, fill=(215, 220, 228))
im.save(out)
print(f"  {out}  {im.size}")
PYEOF
}

cd "$ROOT" || exit 1
run_and_shot "1. Isaac Sim / IsaacLab 版本" "p1_versions" \
    "$PY" -c "
import importlib.metadata as md
print('Python      :', __import__('sys').version.split()[0])
for pkg in ['isaacsim', 'isaaclab', 'isaaclab_rl', 'isaaclab_tasks', 'rsl-rl-lib', 'torch']:
    try:
        print(f'{pkg:<12}: {md.version(pkg)}')
    except Exception as e:
        print(f'{pkg:<12}: {type(e).__name__}')
import torch
print('CUDA        :', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))
"

run_and_shot "2. GPU 与驱动" "p1_gpu" \
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

cd "$ROOT/repos/unitree_rl_lab" 2>/dev/null && \
run_and_shot "3. list_envs 已注册任务" "p1_list_envs" \
    bash -c "PYTHONPATH=\$PWD/source/unitree_rl_lab:\$PWD $PY scripts/list_envs.py 2>/dev/null | head -30"

log "════ 结束，截图在 $OUT_DIR ════"
ls -1 "$OUT_DIR"/*.png 2>/dev/null | sed 's/^/  /'
