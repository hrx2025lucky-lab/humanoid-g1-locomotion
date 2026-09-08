#!/usr/bin/env python3
"""统一评估各实践的训练量是否够 —— 判据是收敛，不是轮数。

课程给的推荐轮数是按"从零练到能用"估的上限，实际跑到平台期就够了。
反过来说，轮数占比高也不代表够：实践 9 跑到官方轮数的 15% 时核心指标
还在单调上升，那就是没跑够。

所以对每个 run 做三件事：

1. 取核心指标的曲线，比较**前 20% 与后 20%** 的均值 —— 涨幅还大就是没收敛
2. 看**后半段的斜率**：拟合一条直线，斜率接近 0 才算进了平台期
3. 用后 10% 的**标准差**衡量噪声，涨幅小于噪声就没有统计意义

用法：
    envs/isaaclab/bin/python scripts/assess_training_budget.py
    envs/isaaclab/bin/python scripts/assess_training_budget.py --json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, asdict

WS = "/home/limx/workspace/Roxan_warmup"

G, R, Y, D, B, N = ("\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")


@dataclass
class Spec:
    practice: str
    name: str
    pattern: str           # run 目录 glob
    metric: str            # 核心指标 tag
    official: int | None   # 官方推荐轮数
    higher_better: bool = True
    note: str = ""


SPECS = [
    # 实践 2 的日志在 repos/unitree_rl_lab 下，不在 shenlan_hw
    Spec("2", "粗糙地形行走",
         f"{WS}/repos/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_rough/*",
         "Train/mean_reward", 15000,
         note="track 2.236/3.0，官方参考 15000 轮是 2.263"),
    Spec("4", "蹲姿行走 · baseline",
         f"{WS}/shenlan_hw/hw4_mjlab/logs/rsl_rl/g1_velocity_height/*ablation_baseline",
         "Train/mean_reward", None, note="消融三组，看对照是否成立"),
    Spec("5", "分层导航 · Baseline",
         f"{WS}/shenlan_hw/hw5_navigation/**/2026-09-05_20-10-32",
         "Episode_Termination/goal_reached", 30000, note="固定竞技场"),
    Spec("5", "分层导航 · RandomArena",
         f"{WS}/shenlan_hw/hw5_navigation/**/2026-09-06_16-37-04",
         "Episode_Termination/goal_reached", 30000, note="Part2 难度扩展"),
    # 蒸馏也是跟踪类，同样看误差不看 reward（与实践 9 口径一致）
    Spec("6", "蒸馏 · KL",
         f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_student_kl_matching/2026-09-05*",
         "Metrics/motion/error_joint_pos", 5000, higher_better=False),
    Spec("6", "蒸馏 · Action",
         f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_student_action_matching_aligned/*",
         "Metrics/motion/error_joint_pos", 5000, higher_better=False),
    # ★ 模仿/跟踪类任务不能用 mean_reward 判收敛 ★
    # 实践 9 的 reward 涨 343%，但那是 episode 变长带来的累积，
    # 单步跟踪质量其实在变差（error_joint_pos 1.161→1.900）。
    # 参考答案的诊断表明确写着"奖励升但误差不降 → 查定义"。
    Spec("9", "轨迹跟踪 P2",
         f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_teacher/2026-09-05_23-41-39*",
         "Metrics/motion/error_joint_pos", 20000, higher_better=False),
    Spec("8", "AMP 拟人走跑",
         f"{WS}/shenlan_hw/unitree_lab_amp/logs/**/2026-09-07*",
         "Train/mean_reward", None, note="今日重跑"),
    # 实践 10/11 同为跟踪/课程类，一律看任务指标不看 reward
    Spec("10", "HOI 感知跟踪",
         f"{WS}/shenlan_hw/HOI_Mimic/logs/rsl_rl/unitree_g1_29dof_mimic_hoi_terrain_perceptive_raycast/*",
         "Metrics/motion/error_joint_pos", 30000, higher_better=False,
         note="官方默认 30000，单卡只跑 3000"),
    Spec("11", "跑酷",
         f"{WS}/repos/instinctlab/logs/instinct_rl/g1_parkour/*",
         "Episode/Curriculum/terrain_levels", 30000,
         note="地形等级上限 9（num_rows=10）"),
]


def load(pattern: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    dirs = [d for d in glob.glob(pattern, recursive=True) if os.path.isdir(d)]
    if not dirs:
        return None, None
    d = sorted(dirs)[-1]
    files = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
    if not files:
        return None, None
    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()
    return (ea, os.path.basename(d)) if ea.Tags()["scalars"] else (None, None)


def assess(ea, tag: str, higher_better: bool) -> dict | None:
    if tag not in ea.Tags()["scalars"]:
        # 换个常见别名再试
        for alt in ("Train/mean_reward", "Loss/value_function"):
            if alt in ea.Tags()["scalars"]:
                tag = alt
                break
        else:
            return None

    def five_seg(values: list[float]) -> list[float]:
        m = len(values)
        return [sum(values[i * m // 5:(i + 1) * m // 5])
                / max(len(values[i * m // 5:(i + 1) * m // 5]), 1) for i in range(5)]

    pts = ea.Scalars(tag)
    vals = [p.value for p in pts]
    n = len(vals)
    if n < 20:
        return None

    # ★ 有课程学习时，reward 下降往往是任务变难而不是训练退化。
    # 实践 2 就是这样：reward 38.9→28.3 看着像崩了，但同期
    # lin_vel_cmd_levels 0.21→1.00（拉满）、terrain_levels 2.21→3.92，
    # 是难度在升。只看 reward 会得出完全相反的结论。
    curr = {}
    for t in ea.Tags()["scalars"]:
        if t.startswith("Curriculum/"):
            cv = [p.value for p in ea.Scalars(t)]
            if len(cv) >= 20:
                cs = five_seg(cv)
                curr[t.split("/")[-1]] = {"seg": cs, "rise": cs[-1] - cs[0]}
    curriculum_rising = any(c["rise"] > 0.05 * max(abs(c["seg"][-1]), 1e-6)
                            for c in curr.values())

    seg = five_seg(vals)
    if not higher_better:
        seg = [-s for s in seg]

    first_gain = seg[1] - seg[0]        # 起步阶段的增量
    last_gain = seg[-1] - seg[-2]       # 末段的增量
    scale = max(abs(seg[-1]), 1e-6)

    rel_last = last_gain / scale                     # 末段相对变化率
    decay = last_gain / first_gain if abs(first_gain) > 1e-9 else 0.0

    # 后 10% 的标准差 = 噪声水平
    t = max(n // 10, 2)
    tv = vals[-t:]
    mu = sum(tv) / t
    sd = (sum((v - mu) ** 2 for v in tv) / t) ** 0.5

    # 三个判据满足任一即算收敛：
    #   a. 末段相对变化 < 3%           —— 涨幅已经无关紧要
    #   b. 增速衰减到起步的 10% 以下    —— 典型收敛曲线尾部
    #   c. 末段增量淹没在噪声里         —— 涨幅没有统计意义
    plateau = (abs(rel_last) < 0.03) or (0 <= decay < 0.10) or (abs(last_gain) < sd)

    # reward 在跌但课程难度在升 —— 单独归一类，别和"还在涨"混为一谈
    harder_task = last_gain < 0 and curriculum_rising

    # 继续训的性价比：按最后 20% 的实际斜率，还要多久才能再改善 10%。
    #
    # 为什么要这个数：光说"占官方基准 30%"没法决策——实践 5 只跑了 3%
    # 就收敛了，实践 9 跑到 200% 反而更差。真正该问的是
    # "再投入 N 小时能换来多少改善"，而不是"跑够比例了没有"。
    tail_n = max(n // 5, 5)
    tv2 = vals[-tail_n:]
    mx = sum(range(tail_n)) / tail_n
    my = sum(tv2) / tail_n
    den = sum((i - mx) ** 2 for i in range(tail_n))
    slope = (sum((i - mx) * (v - my) for i, v in enumerate(tv2)) / den) if den else 0.0
    if not higher_better:
        slope = -slope          # 统一成"正数=在变好"
    # 每个记录点对应多少轮
    per_pt = max(pts[-1].step / max(n - 1, 1), 1)
    want = abs(seg[-1]) * 0.10  # 想再改善 10%
    iters_needed = (want / (slope / per_pt)) if slope > 1e-12 else None

    return {
        "iters": pts[-1].step,
        "points": n,
        "seg": seg,
        "head": seg[0], "tail": seg[-1],
        "rel_last": rel_last,
        "decay": decay,
        "noise_sd": sd,
        "plateau": plateau,
        "curriculum": curr,
        "harder_task": harder_task,
        "iters_for_10pct": iters_needed,
        "tag": tag,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = []
    for s in SPECS:
        ea, run = load(s.pattern)
        if ea is None:
            rows.append({"spec": asdict(s), "status": "no_data"})
            continue
        a = assess(ea, s.metric, s.higher_better)
        rows.append({"spec": asdict(s), "run": run, "assess": a,
                     "status": "ok" if a else "too_short"})

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    print(f"\n{B}训练量评估 —— 判据是收敛而非轮数{N}")
    print("=" * 100)
    print(f"{'实践':<5}{'名称':<20}{'实际轮':>7}{'官方':>7}{'占比':>6}"
          f"  {'五段均值走势':<44}{'末段':>8}  结论")
    print("-" * 100)

    for r in rows:
        s = r["spec"]
        label = f"{s['practice']:<5}{s['name']:<20}"
        if r["status"] != "ok":
            print(f"{label}{D}（无数据）{N}")
            continue
        a = r["assess"]
        off = s["official"]
        pct = f"{a['iters']/off:.0%}" if off else "—"
        offs = str(off) if off else "—"

        # 数值跨度大，按量级选格式。
        # higher_better=False 时 seg 存的是取负后的值（为了统一"越大越好"的
        # 内部逻辑），显示时要还原成原始量纲，否则误差会显示成负数。
        sign = 1 if s.get("higher_better", True) else -1
        disp = [v * sign for v in a["seg"]]
        w = 3 if max(abs(x) for x in disp) < 10 else 1
        curve = "→".join(f"{v:.{w}f}" for v in disp)

        if a.get("harder_task"):
            v, color = "课程升难度", B
        elif a["plateau"]:
            v, color = "已收敛 ✅", G
        elif not s.get("higher_better", True):
            # 误差类指标没进平台 = 还在变差或还在改善，措辞要分开说
            v, color = ("误差仍在恶化 ⚠️" if a["rel_last"] < 0
                        else "误差仍在下降"), Y
        else:
            v, color = "还在涨", Y
        print(f"{label}{a['iters']:>7}{offs:>7}{pct:>6}  {curve:<44}"
              f"{a['rel_last']:>+7.1%}  {color}{v}{N}")
        # 没收敛的，给出"再练多久才值"的估算——比"占官方基准百分之几"更能决策
        if not a["plateau"] and not a.get("harder_task"):
            need = a.get("iters_for_10pct")
            if need and need > 0:
                print(f"{'':<32}  {D}按当前斜率，再改善 10% 约需 {need:,.0f} 轮{N}")
            else:
                print(f"{'':<32}  {D}当前斜率为零或反向，继续训不会更好{N}")
        if a.get("harder_task"):
            for cname, c in list(a["curriculum"].items())[:2]:
                cc = "→".join(f"{x:.2f}" for x in c["seg"])
                print(f"{'':<32}  {D}{cname}: {cc}{N}")

    print("=" * 100)
    print(f"\n{B}判据{N}（满足任一即算收敛）")
    print(f"  {D}a. 末段相对变化 < 3%        —— 再涨也无关紧要{N}")
    print(f"  {D}b. 增速衰减到起步的 10% 以下 —— 典型收敛曲线尾部{N}")
    print(f"  {D}c. 末段增量小于噪声标准差    —— 涨幅没有统计意义{N}")
    print(f"\n  {D}用五段均值而非单一斜率：斜率只答『还涨不涨』，"
          f"五段能看出『涨得还快不快』，{N}")
    print(f"  {D}而收敛的真正特征是增速衰减，不是斜率归零。{N}")
    print(f"  {D}轮数占比低不代表不够 —— 实践 5 只跑了官方的 3% 就到 98%。{N}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
