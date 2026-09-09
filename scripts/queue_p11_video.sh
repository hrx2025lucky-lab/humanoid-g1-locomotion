#!/usr/bin/env bash
# 等实践 9 续训结束后，补录实践 11 的跑酷回放视频。
#
#   nohup bash scripts/queue_p11_video.sh > /dev/null 2>&1 &
#
# 为什么还缺这个视频：上一次补录时用的命令没给 --load_run / --checkpoint，
# 而 instinctlab 的 play.py 不像别的框架会自动挑最新的，直接报
# "No checkpoint specified and play.py resumes from a checkpoint by default"。
#
# 材料清单里实践 11 的必需项是 sim2sim 视频（已有），所以这个不影响"齐备"，
# 但跑酷训练成果没有回放视频，简历里少了最直观的一段。
set -uo pipefail

ROOT="${ROXAN_ROOT:-/home/limx/workspace/Roxan_warmup}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/envs/isaaclab/bin/python"
PIPE="$HOME/humanoid_logs/pipeline/p11_video.log"
mkdir -p "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════ 等实践 9 续训结束后补录跑酷视频 ════"

# 按 PID 判存活。pgrep -f 会匹配到查询命令自己，这个坑在本项目踩过多次。
PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | head -1)
if [ -n "$PID" ]; then
    log "等训练进程 $PID（实践 9 续训到 30000 轮，对齐官方规模）"
    while kill -0 "$PID" 2>/dev/null; do
        it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" \
             "$HOME/humanoid_logs/p9_retrain/p9_resume.log" 2>/dev/null | tail -1)
        log "   ${it:-进行中}"
        sleep 1800
    done
fi
log "GPU 已空闲"
sleep 90

# 视频写完就结束进程——play.py 录完会进交互式 viewer 永不退出。
# 判据用"文件大小连续两次不变"：mp4 边写边落盘，刚出现时还在写，
# 此时杀掉会得到截断的坏文件。
play_until_video() {   # <监视目录> <最长秒数> <命令...>
    local watch="$1" limit="$2"; shift 2
    local marker; marker="$(mktemp)"
    "$@" >> "$PIPE" 2>&1 &
    local pid=$! waited=0 vid="" size=0 prev=-1
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$limit" ]; do
        sleep 10; waited=$((waited + 10))
        vid=$(find "$watch" -name "*.mp4" -newer "$marker" 2>/dev/null | head -1)
        [ -z "$vid" ] && continue
        size=$(stat -c %s "$vid" 2>/dev/null || echo 0)
        if [ "$size" -gt 0 ] && [ "$size" = "$prev" ]; then
            log "   ✅ 视频写完 $((size / 1024)) KB → $vid"
            kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null; rm -f "$marker"; return 0
        fi
        prev="$size"
    done
    kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null; rm -f "$marker"
    [ -n "$vid" ] && { log "   ⚠️ 到上限但视频已存在"; return 0; }
    log "   ❌ ${limit}s 内没有视频"
    return 1
}

log "── 实践 11 · 跑酷回放 ──"
cd "$ROOT/repos/instinctlab" || exit 1

# 只认真正带 checkpoint 的 run —— 失败的启动会留下空目录
RUN=""; CK=""
for d in $(ls -td logs/instinct_rl/g1_parkour/*/ 2>/dev/null); do
    c=$(ls "$d"model_*.pt 2>/dev/null | grep -oE "model_[0-9]+" \
        | grep -oE "[0-9]+" | sort -n | tail -1)
    [ -n "$c" ] && { RUN="$d"; CK="$c"; break; }
done
[ -z "$RUN" ] && { log "❌ 找不到带 checkpoint 的 run"; exit 1; }
log "   run=$(basename "${RUN%/}")  checkpoint=model_${CK}.pt"

export PYTHONPATH="$ROOT/repos/instinct_rl:$ROOT/repos/instinctlab/source/instinctlab"
play_until_video "$ROOT/repos/instinctlab/logs" 1500 \
    "$PY" scripts/instinct_rl/play.py \
        --task Instinct-Parkour-Target-Amp-G1-Play-v0 \
        --load_run "$(basename "${RUN%/}")" \
        --checkpoint "model_${CK}.pt" \
        --num_envs 1 --video --video_length 1000 --headless

n=$(find "$ROOT/repos/instinctlab" -name "*.mp4" 2>/dev/null | wc -l)
log "跑酷视频总数：$n"

log "── 实践 9 续训后验收 ──"
"$PY" "$HERE/verify_training_outcome.py" --practice 9 2>&1 | tee -a "$PIPE"
log "════ 结束 ════"
