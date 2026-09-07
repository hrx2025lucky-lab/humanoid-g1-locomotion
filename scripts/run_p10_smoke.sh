#!/usr/bin/env bash
# 实践 10 感知任务 smoke train，排在所有 GPU 任务之后。
#
# 三组 TODO 已实现（审计 24/24），这一步验证官方那 10 分：
# 「能连续运行且无 NaN/Inf」。跑 10 轮即可，不是正式训练。
#
# 独立成脚本而不是并进 overnight_p7p8.sh，因为后者已经在跑了，
# 改文件对已启动的 bash 进程无效（bash 是边读边执行的，
# 中途改动只会让它读到错位的字节）。
#
# 用法：nohup bash scripts/run_p10_smoke.sh > /dev/null 2>&1 &
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
HOI="$ROOT/shenlan_hw/HOI_Mimic"
PY_LAB="$ROOT/envs/isaaclab/bin/python"
TASK="Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast"

LOG_DIR="$HOME/humanoid_logs/p10_hoi"
PIPE="$HOME/humanoid_logs/pipeline/p10_smoke.log"
mkdir -p "$LOG_DIR" "$(dirname "$PIPE")"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPE"; }

log "════ 实践 10 smoke train 接续 ════"
[ -d "$HOI" ] || { log "❌ 找不到 $HOI"; exit 1; }

# ── GPU 互斥锁 ─────────────────────────────────────────────
# 逐个 pgrep 列举对方的任务名不可靠：新增任务时要改所有脚本，
# 漏一个就会两个训练同时抢卡。改用文件锁，谁先拿到谁跑。
GPU_LOCK="/tmp/humanoid_gpu.lock"
exec 9>"$GPU_LOCK"
log "等待 GPU 锁…"
flock 9
log "已获得 GPU 锁"

log "等所有 GPU 任务让出显存…"
while pgrep -f "Navigation-HRL-RandomArena" >/dev/null 2>&1 \
   || pgrep -f "Instinct-Parkour" >/dev/null 2>&1 \
   || pgrep -f "AMP-WalkToRun" >/dev/null 2>&1; do
    sleep 300
done
sleep 90    # IsaacSim 退出后显存回收有延迟
log "GPU 空闲：$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)"

S_LOG="$LOG_DIR/p10_smoke.log"
cd "$HOI" || exit 1
[ -f set_project_root.sh ] && . ./set_project_root.sh 2>/dev/null
export PYTHONPATH="$HOI/source/unitree_rl_lab:$HOI:${PYTHONPATH:-}"

log "smoke: --num_envs 64 --max_iterations 10 → $S_LOG"
timeout 5400 "$PY_LAB" scripts/rsl_rl/train.py \
    --task "$TASK" --num_envs 64 --max_iterations 10 \
    --headless --logger tensorboard \
    > "$S_LOG" 2>&1
rc=$?

if grep -qE "Learning iteration [0-9]+/" "$S_LOG"; then
    it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$S_LOG" | tail -1)
    log "✅ 训练循环已启动（$it）"

    # 官方 10 分项明确要求"无 NaN/Inf"。但要分清两种 NaN：
    #
    #   1. 训练发散的 NaN —— reward / loss 变成 NaN，是真故障；
    #   2. "尚未评估"的哨兵 NaN —— curriculum_HOI.py:872-875 把
    #      last_agg_episode_length / last_time_out_ratio /
    #      last_length_ratio_to_max 显式初始化成 float("nan")，
    #      要等仿真步数超过 eval_steps（:896）才会被真实值替换。
    #      smoke 只跑 10 轮 × 24 步 = 240 步，远不到评估点，
    #      这三个必然是 NaN，属于设计如此，不是训练坏了。
    #
    # 一律报警会让真故障淹没在必然出现的噪声里，所以只查危险的那一类。
    danger=$(grep -inE "(reward|loss|value_function|surrogate)[^0-9-]*(nan|inf)([^a-z]|$)" "$S_LOG" | head -5)
    if [ -n "$danger" ]; then
        log "❌ reward/loss 出现 NaN/Inf —— 训练发散："
        echo "$danger" | tee -a "$PIPE"
    else
        log "✅ reward/loss 无 NaN/Inf"
        sentinel=$(grep -coE "Curriculum/(agg_episode_length|time_out_ratio|length_ratio_to_max): *nan" "$S_LOG")
        [ "${sentinel:-0}" -gt 0 ] && \
            log "   （另有 $sentinel 处 Curriculum 哨兵 NaN，是「未到评估步」的初值，正常）"
    fi

    # 观测维自检：参考答案给了确定值 289×8=2312。
    # 注意要取 height_scanner **这一项**，不是 policy 组的总维度——
    # 组总维还包含 motion_command / joint_pos_rel 等，
    # 之前抓成 3120 险些以为实现错了（3120 = 2312+58+6+24+24+232×3，其实是对的）。
    dim=$(grep -oE "height_scanner *\| *\(([0-9]+),\)" "$S_LOG" | grep -oE "[0-9]+" | head -1)
    if [ -n "$dim" ]; then
        if [ "$dim" = "2312" ]; then
            log "✅ height_scanner 观测维 $dim = 289×8，与规格一致"
        else
            log "❌ height_scanner 观测维 $dim，规格要求 289×8=2312"
        fi
    else
        log "⚠️ 日志里没找到 height_scanner 的维度"
    fi
else
    log "❌ smoke 失败 rc=$rc，最后 30 行："
    tail -30 "$S_LOG" | tee -a "$PIPE"
    exit 1
fi

log "════ 结束 ════"
