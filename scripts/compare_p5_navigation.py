#!/usr/bin/env python3
"""实践 5 两组对照的结果分析。

    baseline      固定竞技场
    random_arena  每 episode 随机重排障碍（继承 Baseline，只换 events）

两组共用同一份 PPORunnerCfg（seed / num_steps_per_env / 网络结构完全一致），
唯一变量是障碍布局是否随机 —— 这是严格的单因素对照。

> 对照组不用课程自带的 HRL-Extension：它与 Baseline 相差五处
> （布局/课程/观测维度 273 vs 1092/目标模式/终止条件），
> 观测维度不同意味着网络输入层都不一样，无法归因到单一因素。

★ 判据不能只看 reward 和 episode_length。
本实践踩过一次：修好发散后策略立刻学会"站着不动就不会摔"——
episode_length 满值 150、time_out 99.4%、bad_orientation 0.6%，
全部指标看着完美，但 position_progress 是负的、goals_reached 为 0。
所以下面把"表面指标"与"真实任务指标"分开列。

用法：
    python compare_p5_navigation.py
"""

from __future__ import annotations

import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.environ.get(
    "HW5_LOG_DIR",
    "/home/limx/workspace/Roxan_warmup/shenlan_hw/hw5_navigation/unitree_rl_lab"
    "/logs/rsl_rl",
)

# ⚠️ 两组共用同一份 agent cfg（NavigationV5CompactSingleGoalPPORunnerCfg），
# 因此 experiment_name 相同、日志全落在 ..._baseline 目录下，
# **靠目录名分不开两组**（一度把 RandomArena 当成 baseline 分析）。
# 可靠的判据是 run 自己存下来的 params/env.yaml：
# 只有 RandomArena 会注册 randomize_mixed_obstacle_layout 事件。
EXPERIMENT_DIR = "unitree_g1_29dof_navigation_hrl_baseline"
ARENA_MARK = "randomize_mixed_obstacle_layout"

GROUPS = [
    ("baseline", "固定竞技场"),
    ("random_arena", "每 episode 随机重排障碍"),
]

# 表面指标：好看不代表学会了
SURFACE = [
    ("Train/mean_episode_length", "平均 episode 长度", None),
    ("Episode_Termination/time_out", "time_out 占比", None),
    ("Episode_Termination/bad_orientation", "bad_orientation 占比", True),
    ("Train/mean_reward", "平均回报", False),
]

# 真实任务指标：导航到底做到没有
TASK = [
    # ★ 不要用 Metrics/pose_command/goals_reached ★
    # 它只在 update_goal_on_success=True 时累加，而两组都是 SingleGoal
    # （到达即终止、不重采目标），所以恒为 0.0000 —— 修复生效后 95% 的
    # episode 明明到达了目标，它仍然显示 0，差点被判成"没学会"。
    ("Episode_Termination/goal_reached", "到达率（终止原因）", False),
    ("Episode_Reward/position_progress", "位置进展奖励", False),
    ("Metrics/pose_command/error_pos_2d", "目标距离误差", True),
    ("Metrics/pose_command/error_heading", "朝向误差", True),
    ("Episode_Reward/success_bonus", "success_bonus", False),
    ("Episode_Reward/obstacle_soft_zone", "障碍软约束惩罚", False),
]


def load(run_dir: str) -> EventAccumulator | None:
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        return None
    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()
    return ea if ea.Tags()["scalars"] else None


def tail_mean(ea, tag: str, frac: float = 0.1) -> float | None:
    if tag not in ea.Tags()["scalars"]:
        return None
    vals = [x.value for x in ea.Scalars(tag)]
    if not vals:
        return None
    n = max(int(len(vals) * frac), 1)
    return sum(vals[-n:]) / n


def head_mean(ea, tag: str, n: int = 5) -> float | None:
    if tag not in ea.Tags()["scalars"]:
        return None
    vals = [x.value for x in ea.Scalars(tag)][:n]
    return sum(vals) / len(vals) if vals else None


def fmt(v: float | None, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def classify(run_dir: str) -> str | None:
    """读 run 自己存下的 env.yaml，判断它属于哪一组。

    不能靠目录名或时间戳：两组共用 agent cfg，experiment_name 相同，
    没设 NAV_RUN_NAME 时目录名只有时间戳，完全分不开。
    env.yaml 是训练启动时由该 run 实际生效的配置导出的，最可信。
    """
    cfg = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    return "random_arena" if ARENA_MARK in text else "baseline"


def main() -> int:
    runs = {}
    all_runs = [d for d in sorted(glob.glob(os.path.join(ROOT, EXPERIMENT_DIR, "*")))
                if os.path.isdir(d)]
    # 同组有多个 run 时取最新的（sorted 后在后面）
    for d in all_runs:
        key = classify(d)
        if key is None:
            continue
        ea = load(d)
        if ea is not None:
            runs[key] = (ea, os.path.basename(d))

    for key, _ in GROUPS:
        if key not in runs:
            print(f"⚠️  找不到 {key} 的 run（在 {EXPERIMENT_DIR}/ 下按 env.yaml 识别）",
                  file=sys.stderr)

    if not runs:
        sys.exit("没有可用的 run")

    print()
    print("═" * 76)
    print("实践 5 对照 —— 固定竞技场 vs 随机重排障碍")
    print("═" * 76)
    for key, desc in GROUPS:
        if key in runs:
            probe = runs[key][0].Tags()["scalars"][0]
            last_step = runs[key][0].Scalars(probe)[-1].step
            print(f"  {key:<15}{desc:<24}{runs[key][1]}  (至 {last_step} iter)")
    print()

    print("── 表面指标（好看不代表学会了）──")
    print(f"  {'指标':<22}" + "".join(f"{k:>18}" for k, _ in GROUPS if k in runs))
    for tag, label, _ in SURFACE:
        row = f"  {label:<22}"
        for key, _ in GROUPS:
            if key in runs:
                row += f"{fmt(tail_mean(runs[key][0], tag)):>18}"
        print(row)
    print()

    print("── 真实任务指标（导航到底做到没有）──")
    print(f"  {'指标':<22}" + "".join(f"{k:>18}" for k, _ in GROUPS if k in runs)
          + f"{'更优':>8}")
    for tag, label, lower_better in TASK:
        vals = {k: tail_mean(runs[k][0], tag) for k, _ in GROUPS if k in runs}
        row = f"  {label:<22}"
        for key, _ in GROUPS:
            if key in runs:
                row += f"{fmt(vals.get(key)):>18}"
        pair = [(k, v) for k, v in vals.items() if v is not None]
        if len(pair) == 2 and lower_better is not None:
            best = min(pair, key=lambda x: x[1]) if lower_better \
                else max(pair, key=lambda x: x[1])
            row += f"{best[0][:4]:>8}"
        print(row)
    print()

    # 「站着不动」自检 —— 本实践的特有陷阱
    print("── 「站着不动」自检 ──")
    healthy = True
    for key, _ in GROUPS:
        if key not in runs:
            continue
        ea = runs[key][0]
        prog = tail_mean(ea, "Episode_Reward/position_progress")
        goals = tail_mean(ea, "Episode_Termination/goal_reached")
        e0 = head_mean(ea, "Metrics/pose_command/error_pos_2d")
        e1 = tail_mean(ea, "Metrics/pose_command/error_pos_2d")
        issues = []
        if prog is not None and prog <= 0:
            issues.append(f"position_progress={prog:.5f} ≤ 0")
        if goals is not None and goals <= 0:
            issues.append("Episode_Termination/goal_reached=0（真的没到达过）")
        if e0 and e1 and e1 >= e0:
            issues.append(f"距离误差未下降 {e0:.3f}→{e1:.3f}")
        if issues:
            healthy = False
            print(f"  ❌ {key}: " + "；".join(issues))
        else:
            print(f"  ✅ {key}: 真的在朝目标移动"
                  f"（progress={prog:.5f}，距离 {e0:.3f}→{e1:.3f}）")
    print()

    print("═" * 76)
    if not healthy:
        print("结论：至少一组陷入「站着不动」局部最优 —— 摔倒罚分远大于移动收益。")
        print("      需要重新平衡 position_progress 与 termination_penalty 的边际激励比")
        print("      （见 docs/实践5_分层强化学习导航.md 第九节的算法）。")
    else:
        b = tail_mean(runs["baseline"][0], "Episode_Termination/goal_reached") \
            if "baseline" in runs else None
        r = tail_mean(runs["random_arena"][0], "Episode_Termination/goal_reached") \
            if "random_arena" in runs else None
        if b is not None and r is not None:
            b_it = runs["baseline"][0].Scalars(
                "Episode_Termination/goal_reached")[-1].step
            r_it = runs["random_arena"][0].Scalars(
                "Episode_Termination/goal_reached")[-1].step
            print(f"两组均在学习。到达率：")
            print(f"  baseline      {b:6.2%}   （固定布局，至 {b_it} iter）")
            print(f"  random_arena  {r:6.2%}   （每 episode 重排，至 {r_it} iter）")
            gap = b - r
            if abs(b_it - r_it) > 0.2 * max(b_it, r_it):
                # 训练量差一截时不能直接比高低，实践 6 在这上面栽过
                print(f"\n  ⚠️ 两组训练量相差较大（{b_it} vs {r_it} iter），")
                print(f"     现在比较高低不公平 —— 等两边都跑满再下结论。")
            elif abs(gap) < 0.02:
                print(f"\n  差距 {gap:+.2%}，在噪声范围内：随机重排障碍没有明显损害"
                      f"到达率，\n     说明策略学到的不是记住固定布局，而是真的会绕障。")
            else:
                worse = "random_arena" if gap > 0 else "baseline"
                print(f"\n  差距 {gap:+.2%}，{worse} 更低。随机重排通常收敛更慢但泛化更好，"
                      f"\n     判断时要看曲线趋势而非单点。")
    print("═" * 76)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
