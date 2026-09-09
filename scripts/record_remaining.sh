#!/usr/bin/env bash
# 训练全部结束后，自动补齐还缺的回放视频。
#
#   nohup bash scripts/record_remaining.sh > /dev/null 2>&1 &
#
# 现状（2026-09-09）：
#   实践 9  只有 1 个视频，而且是 9-05 那次**关节列序错误**的 run 录的
#   实践 10 HOI 感知跟踪    0 个
#   实践 11 跑酷            0 个
#
# 这三个都是刚跑完最终训练的，视频要用新 checkpoint 重录。
set -uo pipefail

ROOT="${ROXAN_ROOT:-/home/limx/workspace/Roxan_warmup}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/envs/isaaclab/bin/python"
PIPE="$HOME/humanoid_logs/pipeline/record_remaining.log"
VID_LOG="$HOME/humanoid_logs/videos"
mkdir -p "$(dirname "$PIPE")" "$VID_LOG"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════ 补录剩余视频 ════"

# ── 等训练结束 ──────────────────────────────────────────
# 按 PID 判存活：pgrep -f 会匹配到查询命令自己，这个坑踩过多次。
PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | head -1)
if [ -n "$PID" ]; then
    log "等训练进程 $PID 结束"
    while kill -0 "$PID" 2>/dev/null; do
        it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" \
             "$HOME/humanoid_logs/p9_retrain"/*.log 2>/dev/null | tail -1)
        log "   ${it:-进行中}"
        sleep 600
    done
fi
log "训练已结束，等显存回收"
sleep 90

# 视频写完就结束进程——各框架的 play.py 录完会进交互式 viewer 永不退出。
# 判据用"文件大小连续两次不变"，因为 mp4 是边写边落盘的，
# 刚出现时还在写，此时杀掉会得到截断的坏文件。
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
            log "   ✅ 视频写完 $((size / 1024)) KB"
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

# ── 实践 9：用修正列序后重训的 checkpoint ──────────────────
log "── 实践 9 · BeyondMimic 轨迹跟踪 ──"
cd "$ROOT/shenlan_hw/hw6_distill" || exit 1
RUN=$(ls -td logs/rsl_rl/g1_hw6_teacher/2026-09-09_*/ 2>/dev/null | head -1)
if [ -n "$RUN" ]; then
    CK=$(ls "$RUN"/model_*.pt 2>/dev/null | grep -oE "model_[0-9]+" \
         | grep -oE "[0-9]+" | sort -n | tail -1)
    log "   run=$(basename "${RUN%/}")  checkpoint=model_${CK}.pt"
    # 动作文件必须是跳舞的修正版，不是走路——拿走路动作播跳舞策略画面对不上
    play_until_video "$RUN" 900 \
        uv run play Mjlab-Humanoid-HW6-Teacher-G1 \
            --checkpoint-file "${RUN}model_${CK}.pt" \
            --motion-file src/humanoid_hw6/config/g1/motion_data_cfg_hw9_dance_fixed.yaml \
            --num-envs 1 --video True --video-length 1000
else
    log "   ⚠️ 找不到 2026-09-09 的 run"
fi

# ── 实践 11：跑酷 ────────────────────────────────────────
log "── 实践 11 · 跑酷 ──"
cd "$ROOT/repos/instinctlab" || exit 1
# 用 export 而不是 `VAR=x play_until_video ...`：后者其实也能生效
# （实测过），但那个变量只在函数体内可见，而这里真正需要它的是
# 函数内部 fork 出去的 python 子进程，写成 export 语义更明确、不易出错。
export PYTHONPATH="$ROOT/repos/instinct_rl:$ROOT/repos/instinctlab/source/instinctlab"
# 这个 play.py 不会自动挑最新 checkpoint，不给就报
# "No checkpoint specified and play.py resumes from a checkpoint by default"
P11_RUN=$(ls -td "$ROOT"/repos/instinctlab/logs/instinct_rl/g1_parkour/*/ 2>/dev/null | head -1)
P11_CK=$(ls "$P11_RUN"model_*.pt 2>/dev/null | grep -oE "model_[0-9]+" \
         | grep -oE "[0-9]+" | sort -n | tail -1)
if [ -n "$P11_CK" ]; then
    log "   run=$(basename "${P11_RUN%/}")  checkpoint=model_${P11_CK}.pt"
    play_until_video "$ROOT/repos/instinctlab/logs" 1200 \
        "$PY" scripts/instinct_rl/play.py \
            --task Instinct-Parkour-Target-Amp-G1-Play-v0 \
            --load_run "$(basename "${P11_RUN%/}")" \
            --checkpoint "model_${P11_CK}.pt" \
            --num_envs 1 --video --video_length 1000 --headless
else
    log "   ⚠️ 找不到跑酷的 checkpoint"
fi

# ── 实践 10：HOI 感知跟踪 ────────────────────────────────
log "── 实践 10 · HOI 感知跟踪 ──"
cd "$ROOT/shenlan_hw/HOI_Mimic" || exit 1
[ -f set_project_root.sh ] && . ./set_project_root.sh >/dev/null 2>&1
export PYTHONPATH="$ROOT/shenlan_hw/HOI_Mimic/source/unitree_rl_lab:$ROOT/shenlan_hw/HOI_Mimic:${PYTHONPATH:-}"
play_until_video "$ROOT/shenlan_hw/HOI_Mimic/logs" 1200 \
    "$PY" scripts/rsl_rl/play.py \
        --task Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast \
        --num_envs 1 --video --video_length 1000 --headless

# ── 汇总 ────────────────────────────────────────────────
log "── 结果 ──"
for p in shenlan_hw/hw6_distill repos/instinctlab shenlan_hw/HOI_Mimic; do
    n=$(find "$ROOT/$p" -name "*.mp4" 2>/dev/null | wc -l)
    log "   $(basename "$p"): $n 个视频"
done
"$PY" "$HERE/check_deliverables.py" 2>&1 | tail -8 | tee -a "$PIPE"
log "════ 结束 ════"
