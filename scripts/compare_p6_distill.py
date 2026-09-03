#!/usr/bin/env python3
"""实践 6 两种蒸馏方式的结果对比。

    action_matching  只对齐动作均值      loss = MSE(a_student, a_teacher)
    kl_matching      对齐完整高斯分布    loss = KL(teacher ‖ student)

两组共享同一个 Teacher checkpoint 与同一份受限的 Student 观测。

⚠️ 这不是严格的单因素对照。除蒸馏目标外，两组还有三处超参不同：
        learning_rate   5e-4   vs  3e-4
        entropy_coef    0.0025 vs  0.005
        desired_kl      0.005  vs  0.008
   因此**不能**据此断言"某种蒸馏目标本质上更好"，只能说
   "各自在其调好的超参下能达到什么水平"。本脚本会显式打印这一点，
   避免把混杂因素读成因果。

判据不能只看蒸馏 loss —— 两组的 loss 定义不同（MSE vs KL），量纲不可比。
真正可比的是**学生自己的动作跟踪质量**：Metrics/motion/error_* 与存活率。

用法：
    python compare_p6_distill.py
"""

from __future__ import annotations

import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.environ.get(
    "HW6_LOG_DIR",
    "/home/limx/workspace/Roxan_warmup/shenlan_hw/hw6_distill/logs/rsl_rl",
)

GROUPS = [
    ("action_matching", "g1_hw6_student_action_matching", "对齐动作均值 (MSE)"),
    ("kl_matching", "g1_hw6_student_kl_matching", "对齐完整分布 (KL)"),
]

# 两组不同的超参 —— 显式列出，提醒读者这不是单因素对照
CONFOUNDS = {
    "action_matching": {"learning_rate": 5.0e-4, "entropy_coef": 0.0025, "desired_kl": 0.005},
    "kl_matching": {"learning_rate": 3.0e-4, "entropy_coef": 0.005, "desired_kl": 0.008},
}

# 可比的共同判据：学生自己的跟踪质量，与蒸馏 loss 的定义无关
COMMON_METRICS = [
    ("Metrics/motion/error_joint_pos", "关节位置误差", True),
    ("Metrics/motion/error_body_pos", "身体位置误差", True),
    ("Metrics/motion/error_body_rot", "身体姿态误差", True),
    ("Metrics/motion/error_anchor_pos", "anchor 位置误差", True),
    ("Train/mean_episode_length", "平均 episode 长度", False),
    ("Termination_Frac/time_out", "超时占比（越高越好）", False),
    ("Termination_Frac/fail", "失败占比（越低越好）", True),
    ("Policy/mean_std", "策略噪声 std", False),
]


def load(run_dir: str) -> EventAccumulator | None:
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        return None
    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()
    return ea if ea.Tags()["scalars"] else None


def tail_mean(ea: EventAccumulator, tag: str, frac: float = 0.1) -> float | None:
    if tag not in ea.Tags()["scalars"]:
        return None
    vals = [x.value for x in ea.Scalars(tag)]
    if not vals:
        return None
    n = max(int(len(vals) * frac), 1)
    return sum(vals[-n:]) / n


def series_at(ea: EventAccumulator, tag: str, steps: list[int]) -> list[float | None]:
    if tag not in ea.Tags()["scalars"]:
        return [None] * len(steps)
    pts = [(x.step, x.value) for x in ea.Scalars(tag)]
    return [([v for st, v in pts if st <= s] or [None])[-1] for s in steps]


def fmt(v: float | None, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> int:
    runs = {}
    for key, subdir, _ in GROUPS:
        matches = sorted(glob.glob(os.path.join(ROOT, subdir, "*")))
        matches = [m for m in matches if os.path.isdir(m)]
        if not matches:
            print(f"⚠️  找不到 {key} 的 run", file=sys.stderr)
            continue
        ea = load(matches[-1])
        if ea is None:
            print(f"⚠️  {key} 的 run 里没有 scalars", file=sys.stderr)
            continue
        runs[key] = (ea, os.path.basename(matches[-1]))

    if len(runs) < 2:
        sys.exit("需要两组 run 才能对比")

    print()
    print("═" * 76)
    print("实践 6 蒸馏对比 —— Action Matching vs KL Matching")
    print("═" * 76)
    for key, _, desc in GROUPS:
        if key in runs:
            print(f"  {key:<18}{desc:<24}{runs[key][1]}")
    print()

    print("── ⚠️ 混杂因素：两组超参不同，不是严格单因素对照 ──")
    print(f"  {'超参':<18}" + "".join(f"{k:>20}" for k, _, _ in GROUPS))
    for hp in ("learning_rate", "entropy_coef", "desired_kl"):
        row = f"  {hp:<18}"
        for k, _, _ in GROUPS:
            row += f"{CONFOUNDS[k][hp]:>20.4g}"
        print(row)
    print("  → 结论只能是「各自调好后能到什么水平」，不能归因到蒸馏目标本身。")
    print()

    print("── 蒸馏 loss（量纲不同，仅看各自收敛趋势，不可横向比大小）──")
    print(f"  {'组别':<18}{'loss 名':>10}{'首值':>14}{'末值':>14}{'降幅':>10}")
    for key, _, _ in GROUPS:
        if key not in runs:
            continue
        ea = runs[key][0]
        tag = "Loss/bc" if "Loss/bc" in ea.Tags()["scalars"] else "Loss/kl"
        if tag not in ea.Tags()["scalars"]:
            print(f"  {key:<18}{'—':>10}")
            continue
        vals = [x.value for x in ea.Scalars(tag)]
        first = sum(vals[:5]) / min(len(vals), 5)
        last = tail_mean(ea, tag)
        drop = f"{(1 - last / first) * 100:.0f}%" if first and last else "—"
        print(f"  {key:<18}{tag.split('/')[-1]:>10}{first:>14.4f}{last:>14.4f}{drop:>10}")
    print()

    print("── 可比判据：学生自己的跟踪质量与存活 ──")
    print(f"  {'指标':<22}" + "".join(f"{k:>20}" for k, _, _ in GROUPS) + f"{'更优':>8}")
    for tag, label, lower_better in COMMON_METRICS:
        vals = {}
        for key, _, _ in GROUPS:
            if key in runs:
                vals[key] = tail_mean(runs[key][0], tag)
        row = f"  {label:<22}"
        for key, _, _ in GROUPS:
            row += f"{fmt(vals.get(key)):>20}"
        pair = [(k, v) for k, v in vals.items() if v is not None]
        if len(pair) == 2 and lower_better is not None:
            best = min(pair, key=lambda x: x[1]) if lower_better else max(pair, key=lambda x: x[1])
            row += f"{best[0][:2].upper():>8}"
        print(row)
    print()

    marks = [0, 500, 1000, 1500, 2000, 2500, 2999]
    print("── error_joint_pos 收敛过程 ──")
    print("  " + f"{'iter':<8}" + "".join(f"{k:>20}" for k, _, _ in GROUPS if k in runs))
    series = {k: series_at(runs[k][0], "Metrics/motion/error_joint_pos", marks)
              for k, _, _ in GROUPS if k in runs}
    for i, m in enumerate(marks):
        row = f"  {m:<8}"
        for k, _, _ in GROUPS:
            if k in series:
                row += f"{fmt(series[k][i]):>20}"
        print(row)
    print()

    print("═" * 76)
    a = tail_mean(runs["action_matching"][0], "Metrics/motion/error_joint_pos")
    b = tail_mean(runs["kl_matching"][0], "Metrics/motion/error_joint_pos")
    if a is not None and b is not None:
        better = "action_matching" if a < b else "kl_matching"
        print(f"主判据 error_joint_pos：action={a:.4f}  kl={b:.4f}"
              f"  →  {better} 更优（差 {abs(a - b) / max(a, b) * 100:.1f}%）")
        print("但两组超参不同（见上表），该差距不能归因到蒸馏目标本身。")
        print("若要做因果结论，需把 lr / entropy_coef / desired_kl 对齐后重跑。")
    print("═" * 76)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
