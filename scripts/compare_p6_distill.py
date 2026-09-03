#!/usr/bin/env python3
"""实践 6 蒸馏对照 —— 2×2 分析（蒸馏目标 × 超参设置）。

    action_matching  只对齐动作均值      loss = MSE(a_student, a_teacher)
    kl_matching      对齐完整高斯分布    loss = KL(teacher ‖ student)

首轮两组除蒸馏目标外还差三处超参，实测 KL 全面更优，
但那个差距**无法归因到蒸馏目标本身**。于是补跑了超参对齐的 action_matching：

    组别                  learning_rate   entropy_coef   desired_kl
    AC 原始                   5e-4          0.0025         0.005
    AC 对齐                   3e-4          0.005          0.008   ← 补跑
    KL（原始即对齐值）          3e-4          0.005          0.008

★ 对齐值就取自 KL 组的超参，所以 KL 不需要重跑 ——
  「AC 对齐」与「KL 原始」的超参已经完全一致（实测两份 agent.yaml 逐项相同），
  它们之间的比较就是唯一变量为蒸馏目标的可归因对照。
  一开始我还真的排了一次 KL 重跑，跑到 1500 iter 才反应过来是无用功，
  停掉后把 1.5 小时 GPU 让给了实践 5。

于是能回答两个问题：
  · 纵向（AC 原始 vs AC 对齐）：超参本身影响多大？
  · 横向（AC 对齐 vs KL）：蒸馏目标的真实差距是多少？

判据不看蒸馏 loss —— 两组 loss 定义不同（MSE vs KL），量纲不可比。
可比的是**学生自己的动作跟踪质量**与存活率。

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

METHODS = [("action_matching", "AC 对齐动作均值"), ("kl_matching", "KL 对齐完整分布")]
VARIANTS = [("", "原始超参"), ("_aligned", "对齐超参")]
# KL 组不需要 _aligned 变体：它的原始超参就是对齐值
SKIP = {("kl_matching", "_aligned")}

HPARAMS = {
    ("action_matching", ""): "lr 5e-4 / ec 0.0025 / kl 0.005",
    ("kl_matching", ""): "lr 3e-4 / ec 0.005  / kl 0.008",
    ("action_matching", "_aligned"): "lr 3e-4 / ec 0.005  / kl 0.008",
    ("kl_matching", "_aligned"): "lr 3e-4 / ec 0.005  / kl 0.008",
}

# 可比判据：与蒸馏 loss 的定义无关
METRICS = [
    ("Metrics/motion/error_joint_pos", "关节位置误差", True),
    ("Metrics/motion/error_body_pos", "身体位置误差", True),
    ("Metrics/motion/error_body_rot", "身体姿态误差", True),
    ("Metrics/motion/error_anchor_pos", "anchor 位置误差", True),
    ("Termination_Frac/fail", "失败占比", True),
    ("Train/mean_episode_length", "episode 长度", False),
    ("Policy/mean_std", "策略噪声 std", None),
]

PRIMARY = "Metrics/motion/error_joint_pos"
TARGET_ITERS = 2999


def load(method: str, variant: str):
    pat = os.path.join(ROOT, f"g1_hw6_student_{method}{variant}", "*")
    dirs = [d for d in sorted(glob.glob(pat)) if os.path.isdir(d)]
    if not dirs:
        return None, None
    files = sorted(glob.glob(os.path.join(dirs[-1], "events.out.tfevents.*")))
    if not files:
        return None, None
    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()
    if not ea.Tags()["scalars"]:
        return None, None
    probe = ea.Tags()["scalars"][0]
    return ea, ea.Scalars(probe)[-1].step


def tail_mean(ea, tag: str, frac: float = 0.1) -> float | None:
    if ea is None or tag not in ea.Tags()["scalars"]:
        return None
    vals = [x.value for x in ea.Scalars(tag)]
    if not vals:
        return None
    n = max(int(len(vals) * frac), 1)
    return sum(vals[-n:]) / n


def fmt(v, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> int:
    runs = {}
    for m, _ in METHODS:
        for v, _ in VARIANTS:
            ea, step = load(m, v)
            if ea is not None:
                runs[(m, v)] = (ea, step)

    if not runs:
        sys.exit("没有可用的 run")

    print()
    print("═" * 80)
    print("实践 6 蒸馏对照 —— 2×2（蒸馏目标 × 超参设置）")
    print("═" * 80)
    print(f"  {'组别':<28}{'超参':<32}{'跑到':>8}")
    for m, ml in METHODS:
        for v, vl in VARIANTS:
            key = (m, v)
            if key in SKIP:
                print(f"  {ml + ' · ' + vl:<28}{'（无需重跑，原始超参即对齐值）':<32}")
                continue
            step = runs[key][1] if key in runs else None
            mark = "" if step is None else ("" if step >= TARGET_ITERS else "  ⏳未跑完")
            print(f"  {ml + ' · ' + vl:<28}{HPARAMS[key]:<32}"
                  f"{'—' if step is None else step:>8}{mark}")
    print()

    # ── 纵向：同方法换超参 ────────────────────────────────────────────────
    print("── 纵向对照：同一蒸馏方法，只换超参 ──")
    for m, ml in METHODS:
        if (m, "_aligned") in SKIP:
            print(f"  {ml:<18}原始超参即对齐值，无纵向差异")
            continue
        a, b = runs.get((m, "")), runs.get((m, "_aligned"))
        if not (a and b):
            print(f"  {ml:<18}数据不全，跳过")
            continue
        va, vb = tail_mean(a[0], PRIMARY), tail_mean(b[0], PRIMARY)
        if va and vb:
            d = (vb - va) / va * 100
            note = "  ⏳对齐组未跑完" if b[1] < TARGET_ITERS else ""
            print(f"  {ml:<18}关节误差 {va:.4f} → {vb:.4f}  ({d:+.1f}%)"
                  f"  {'对齐超参更好' if vb < va else '原始超参更好'}{note}")
    print()

    # ── 横向：同超参换方法（唯一可归因的对照）──────────────────────────────
    print("── 横向对照：同一超参，只换蒸馏目标 ★ 唯一可归因的比较 ──")
    ac = runs.get(("action_matching", "_aligned"))
    # KL 组的原始超参就是对齐值，无需重跑，直接用它做横向对照
    kl = runs.get(("kl_matching", ""))
    ready = (ac and kl and ac[1] >= TARGET_ITERS and kl[1] >= TARGET_ITERS)

    if ready:
        print(f"  {'指标':<20}{'AC(对齐)':>14}{'KL':>14}{'更优':>10}{'差距':>10}")
        for tag, label, lower in METRICS:
            va, vb = tail_mean(ac[0], tag), tail_mean(kl[0], tag)
            row = f"  {label:<20}{fmt(va):>14}{fmt(vb):>14}"
            if va is not None and vb is not None and lower is not None:
                better = "AC" if (va < vb) == lower else "KL"
                gap = abs(va - vb) / max(abs(va), abs(vb)) * 100
                row += f"{better:>10}{gap:>9.1f}%"
            print(row)
        print()
        va, vb = tail_mean(ac[0], PRIMARY), tail_mean(kl[0], PRIMARY)
        print("═" * 80)
        if va and vb:
            better = "action_matching" if va < vb else "kl_matching"
            gap = abs(va - vb) / max(va, vb) * 100
            print(f"结论（可归因）：唯一变量为蒸馏目标时，{better} 的关节跟踪误差更低。")
            print(f"                AC {va:.4f} vs KL {vb:.4f}，差距 {gap:.1f}%")
            # 与首轮的混杂结果对比，量化"超参效应"占了多少
            a0, k0 = runs.get(("action_matching", "")), runs.get(("kl_matching", ""))
            if a0 and k0:
                va0, vb0 = tail_mean(a0[0], PRIMARY), tail_mean(k0[0], PRIMARY)
                if va0 and vb0:
                    gap0 = abs(va0 - vb0) / max(va0, vb0) * 100
                    print()
                    print(f"对比首轮（超参混杂）的差距 {gap0:.1f}% —— "
                          f"对齐后变为 {gap:.1f}%，")
                    print(f"说明首轮那个差距里有 {abs(gap0 - gap):.1f} 个百分点"
                          f"其实来自超参而非蒸馏目标。")
        print("═" * 80)
    else:
        print("  ⏳ 对齐组尚未全部跑完，暂不做横向归因（避免拿未收敛的数据下结论）。")
        done = [f"{m}{v}" for (m, v), r in runs.items()
                if r[1] and r[1] >= TARGET_ITERS]
        print(f"     已完成：{', '.join(done) if done else '无'}")
        print()
        a0, k0 = runs.get(("action_matching", "")), runs.get(("kl_matching", ""))
        if a0 and k0:
            print("── 首轮结果（⚠️ 超参混杂，不可归因，仅供参考）──")
            print(f"  {'指标':<20}{'AC(原始)':>14}{'KL(原始)':>14}")
            for tag, label, _ in METRICS[:5]:
                print(f"  {label:<20}{fmt(tail_mean(a0[0], tag)):>14}"
                      f"{fmt(tail_mean(k0[0], tag)):>14}")
        print("═" * 80)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
