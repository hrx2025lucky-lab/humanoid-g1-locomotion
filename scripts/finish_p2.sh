#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 实践 2 训练收尾流程
# ═══════════════════════════════════════════════════════════════════════════
#
# 训练结束后一键完成三件事：
#   ① 提取最终训练指标，生成可直接贴进报告的数据表
#   ② 录制完整 episode 的 play 视频（1000 步）并导出 ONNX
#   ③ 检查 MuJoCo sim2sim 的前置条件（课程要求的验证方式）
#
# 用法：
#   ./finish_p2.sh              # 全部三步
#   ./finish_p2.sh metrics      # 只提取指标
#   ./finish_p2.sh video        # 只录像
#   ./finish_p2.sh sim2sim      # 只检查 sim2sim 前置
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="/home/limx/workspace/Roxan_warmup/repos/unitree_rl_lab"
PY="/home/limx/workspace/Roxan_warmup/envs/isaaclab/bin/python"
EXP="unitree_g1_29dof_velocity_rough"
# 实践 2 的 sim2sim 包来自 ch2_sim2sim_v1.zip（"ch2" = 第二章），
# 不是 course_code/sim2sim/——后者是实践 11 的 Instinct Parkour 包
# （里面是 parkour_actor.onnx / stand_depth_encoder.onnx）。
S2S="/home/limx/workspace/Roxan_warmup/shenlan_hw/hw2_sim2sim/sim2sim"
MJ_BUILD_LIB="/home/limx/workspace/Roxan_warmup/repos/mujoco_src/build/lib"
PLUGIN_REPO="/home/limx/workspace/Roxan_warmup/repos/mujoco_ray_caster"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

latest_run() {
  ls -1d "$REPO/logs/rsl_rl/$EXP"/*/ 2>/dev/null | sort | tail -1
}

all_runs() {
  ls -1d "$REPO/logs/rsl_rl/$EXP"/*/ 2>/dev/null | sort
}

step_metrics() {
  local run; run="$(latest_run)"
  [[ -n "$run" ]] || { echo "找不到训练 run" >&2; return 1; }
  echo "════ ① 提取最终训练指标 ════"
  echo
  # 传入全部 run，由 Python 侧按 step 区间自动串联续训链
  # shellcheck disable=SC2046
  "$PY" - $(all_runs) <<'PYEOF'
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── 续训会新建 run 目录，指标被切成多段 ────────────────────────────────
# rsl_rl 的 resume 保留全局迭代计数（tot_iter = start_iter + max_iterations，
# start_iter 从 checkpoint 的 "iter" 字段读出），所以各段的 step 区间首尾相接：
#   run A: step 0    → 9000
#   run B: step 9000 → 10000
# 于是可以只按 step 区间就把链条串起来，不需要解析日志或 git 元数据。
#
# 但同一 experiment 下还躺着若干早期废弃 run（改奖励权重之前的），
# 它们的 step 也从 0 开始，不能一股脑合并。做法是从最新 run 往回走：
# 取当前段的起始 step，找出「包含该 step 且不是自己」的那个 run 作为前驱，
# 沿链回溯到 step 0 为止。没有前驱就停，天然把废弃 run 排除在外。

def load(run):
    files = sorted(glob.glob(run.rstrip("/") + "/events.out.tfevents.*"))
    if not files:
        return None
    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()
    if not ea.Tags()["scalars"]:
        return None
    return ea

runs = {}
ordered = [r.rstrip("/") for r in sys.argv[1:]]   # bash 侧已按目录名排序
for r in ordered:
    ea = load(r)
    if ea is None:
        continue
    probe = ea.Tags()["scalars"][0]
    steps = [x.step for x in ea.Scalars(probe)]
    runs[r] = (ea, min(steps), max(steps))

if not runs:
    sys.exit("没有可用的 tfevents")

# 从最新的 run（目录名是时间戳，字典序 = 时间序）出发向前回溯。
# 注意锚点不能按「最大 step」选：早期废弃 run 也跑到过 9999，
# 会盖过当前只跑到 9100 的续训段。
chain = [r for r in ordered if r in runs][-1:]
if not chain:
    sys.exit("没有可用的 tfevents")
while True:
    lo = runs[chain[0]][1]
    if lo <= 1:
        break
    # 前驱必须同时满足：step 区间接得上，且时间上早于当前段。
    # 只按 step 区间会误命中早期废弃 run（它们也从 0 跑到过 9999）。
    prev = [r for r, (_, a, b) in runs.items()
            if r not in chain and a < lo <= b + 1 and r < chain[0]]
    if not prev:
        break
    chain.insert(0, max(prev))

print("── 训练链 ──")
for r in chain:
    _, lo, hi = runs[r]
    print(f"  {r.split('/')[-1]}   step {lo} → {hi}")
if len(chain) > 1:
    print("  （续训分段，以下指标已按 step 合并）")
print()

# 合并：后段覆盖前段的重叠 step
def series(tag):
    merged = {}
    for r in chain:
        ea = runs[r][0]
        if tag in ea.Tags()["scalars"]:
            for x in ea.Scalars(tag):
                merged[x.step] = x.value
    return sorted(merged.items())

all_tags = set()
for r in chain:
    all_tags |= set(runs[r][0].Tags()["scalars"])

def last(tag):
    s = series(tag)
    return s[-1][1] if s else None

print("── 报告用关键指标（最终值）──")
key = [
    ("Train/mean_episode_length", "Mean episode length"),
    ("Episode_Reward/track_lin_vel_xy", "track_lin_vel_xy"),
    ("Episode_Reward/track_ang_vel_z", "track_ang_vel_z"),
    ("Episode_Reward/feet_air_time", "feet_air_time"),
    ("Episode_Reward/stand_still", "stand_still"),
    ("Episode_Reward/gait", "gait（应为 0）"),
    ("Curriculum/terrain_levels", "terrain_levels"),
    ("Curriculum/lin_vel_cmd_levels", "lin_vel_cmd_levels"),
    ("Metrics/base_velocity/error_vel_xy", "error_vel_xy"),
    ("Metrics/base_velocity/error_vel_yaw", "error_vel_yaw"),
    ("Episode_Termination/time_out", "time_out 占比"),
    ("Episode_Termination/bad_orientation", "bad_orientation 占比"),
]
for tag, label in key:
    v = last(tag)
    print(f"  {label:<28}{'—' if v is None else f'{v:.4f}'}")

print("\n── 全部奖励项分解（按值排序，报告 §6 用）──")
rows = []
for t in sorted(all_tags):
    if t.startswith("Episode_Reward/"):
        v = last(t)
        if v is not None:
            rows.append((t.replace("Episode_Reward/", ""), v))
pos = sum(v for _, v in rows if v >= 0)
neg = sum(v for _, v in rows if v < 0)
for n, v in sorted(rows, key=lambda x: -x[1]):
    print(f"  {n:<26}{v:+9.4f}")
print(f"\n  正奖励合计 {pos:+.4f}   惩罚合计 {neg:+.4f}   净 {pos+neg:+.4f}")

print("\n── terrain_levels 演化（报告用曲线数据）──")
s = series("Curriculum/terrain_levels")
if s:
    last_step = s[-1][0]
    marks = [m for m in range(0, 10001, 1000) if m <= last_step]
    if last_step not in marks:
        marks.append(last_step)
    for it in marks:
        vals = [v for st, v in s if st <= it]
        if vals:
            print(f"  @{it:<6}{vals[-1]:.4f}")
PYEOF
}

step_video() {
  echo
  echo "════ ② 录制完整 episode 的 play 视频 ════"
  "$HERE/record_play.sh"
}

step_sim2sim() {
  echo
  echo "════ ③ sim2sim 前置条件检查 ════"
  local run ckpt ok=1
  run="$(latest_run)"

  echo "-- 资源文件 --"
  for f in "$S2S/sim2sim_raycaster.py" "$S2S/policy_inference.py" \
           "$S2S/mujoco_env.py" "$S2S/config.py" "$S2S/assets/scene_rough.xml"; do
    if [[ -e "$f" ]]; then echo "  ✅ $(basename "$f")"; else echo "  ❌ 缺失 $(basename "$f")"; ok=0; fi
  done

  echo "-- raycaster 插件（height scanner 的硬依赖）--"
  # 注意：set -euo pipefail 下 find 未命中会让整个脚本在此中断，
  # 后面的检查项全部跳过。用 || true 兜住。
  local so
  so=$(find "$MJ_BUILD_LIB" "$PLUGIN_REPO/lib" \
            -name "libsensor_raycaster.so" 2>/dev/null | head -1 || true)
  if [[ -n "$so" ]]; then
    echo "  ✅ $so"
  else
    echo "  ❌ 未编译。跑 ./scripts/build_raycaster_plugin.sh 即可"
    echo "     （它会处理 MuJoCo 3.12 的三处上游改动 + gcc 的 -Werror 误报）"
    echo "     报错特征: plugin mujoco.sensor.ray_caster not found"
    ok=0
  fi

  echo "-- config.py 是否已指向真实插件 --"
  if [[ -n "$so" ]] && grep -q "$so" "$S2S/config.py" 2>/dev/null; then
    echo "  ✅ RAYCASTER_PLUGIN_LIBRARY 已配置"
  else
    echo "  ⚠️  需把 RAYCASTER_PLUGIN_LIBRARY 设为上面的 .so 路径"
  fi

  echo "-- 训练产物 --"
  ckpt=$(ls -1 "$run"model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
  [[ -n "$ckpt" ]] && echo "  ✅ checkpoint: $(basename "$ckpt")" || { echo "  ❌ 无 checkpoint"; ok=0; }
  [[ -f "$run/params/deploy.yaml" ]] && echo "  ✅ deploy.yaml" || echo "  ⚠️  deploy.yaml 缺失"
  [[ -f "$run/exported/policy.pt" ]] && echo "  ✅ exported/policy.pt" || echo "  ⚠️  policy.pt 缺失（跑 ② 后生成）"

  echo "-- 课程自带参考 checkpoint（可先用它验证链路，不必等自训模型）--"
  if [[ -d "$S2S/policy/2026-06-12_10-36-30" ]]; then
    echo "  ✅ policy/2026-06-12_10-36-30（config.py 默认指向此处）"
  else
    echo "  ⚠️  未找到"
  fi

  echo
  if [[ $ok -eq 1 ]]; then
    echo "前置齐备。config.py 需要设置："
    echo "  RAYCASTER_PLUGIN_LIBRARY = \"$so\""
    echo "  TRAIN_RUN_DIR            = \"${run%/}\""
    echo "启动键盘控制 sim2sim："
    echo "  cd \"$S2S\" && $PY sim2sim_raycaster.py"
    echo "  方向键 ↑↓ 前后 · ←→ 转向 · Space 停 · R 重置"
    echo "验证要点（课程 §7.4）：红色 raycaster 点应落在地形而非机器人身上；"
    echo "  机器人能在粗糙地形保持站立并响应速度命令；policy 输入维度与训练一致。"
  else
    echo "⚠️  前置未齐（见上方 ❌）。可先用课程自带 checkpoint 验证除插件外的链路。"
  fi
}

case "${1:-all}" in
  metrics) step_metrics ;;
  video)   step_video ;;
  sim2sim) step_sim2sim ;;
  all)     step_metrics; step_video; step_sim2sim ;;
  *) echo "用法: $0 {all|metrics|video|sim2sim}" >&2; exit 2 ;;
esac
