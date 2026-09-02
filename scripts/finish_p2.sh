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
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

latest_run() {
  ls -1d "$REPO/logs/rsl_rl/$EXP"/*/ 2>/dev/null | sort | tail -1
}

step_metrics() {
  local run; run="$(latest_run)"
  [[ -n "$run" ]] || { echo "找不到训练 run" >&2; return 1; }
  echo "════ ① 提取最终训练指标 ════"
  echo "run: $run"
  echo
  "$PY" - "$run" <<'PYEOF'
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
files = sorted(glob.glob(run + "events.out.tfevents.*"))
if not files:
    sys.exit("该 run 下没有 tfevents 文件")
ea = EventAccumulator(files[0], size_guidance={"scalars": 0}); ea.Reload()
tags = ea.Tags()["scalars"]

def last(tag):
    return ea.Scalars(tag)[-1].value if tag in tags else None

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
for t in tags:
    if t.startswith("Episode_Reward/"):
        rows.append((t.replace("Episode_Reward/", ""), ea.Scalars(t)[-1].value))
pos = sum(v for _, v in rows if v >= 0)
neg = sum(v for _, v in rows if v < 0)
for n, v in sorted(rows, key=lambda x: -x[1]):
    print(f"  {n:<26}{v:+9.4f}")
print(f"\n  正奖励合计 {pos:+.4f}   惩罚合计 {neg:+.4f}   净 {pos+neg:+.4f}")

print("\n── terrain_levels 演化（报告用曲线数据）──")
if "Curriculum/terrain_levels" in tags:
    s = ea.Scalars("Curriculum/terrain_levels")
    for it in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]:
        vals = [x.value for x in s if x.step <= it]
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
  local so
  so=$(find /home/limx/workspace/Roxan_warmup/repos/mujoco_src/build/lib \
            /home/limx/workspace/Roxan_warmup/repos/mujoco_ray_caster/lib \
            -name "libsensor_raycaster.so" 2>/dev/null | head -1)
  if [[ -n "$so" ]]; then
    echo "  ✅ $so"
    echo "     → 填入 $S2S/config.py 的 RAYCASTER_PLUGIN_LIBRARY"
  else
    echo "  ❌ 未编译。构建方式（插件须在 MuJoCo 源码树内编译，版本必须与运行时一致）："
    echo "     1) git clone --branch 3.12.0 google-deepmind/mujoco  → repos/mujoco_src"
    echo "     2) 把 repos/mujoco_ray_caster 放到 mujoco_src/plugin/ 下"
    echo "     3) 在 mujoco_src/CMakeLists.txt 追加 add_subdirectory(plugin/mujoco_ray_caster)"
    echo "     4) cmake -B build && cmake --build build -j"
    echo "     报错特征: plugin mujoco.sensor.ray_caster not found"
    ok=0
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
