#!/usr/bin/env python3
"""按真实任务指标验收训练效果 —— 不看 reward。

为什么需要这个脚本
==================

实践 9 跑满 20000 轮，reward 从 1.88 涨到 8.34（+343%），
用"曲线进平台"判定为已收敛。**结论是错的** ——
真实跟踪误差从 1.161 恶化到 1.900，策略学的是"靠不摔倒混时长"。

那次是靠录验收视频**看到画面**才发现的。数值指标全程自洽地错着。

所以有了这个脚本：对每个训练 run，查的是**任务真正要求的指标**，
并回答三个问题：

  1. 任务指标在改善吗？（不是 reward）
  2. 若指标变差，能被课程难度上升解释吗？
  3. 若不能解释，是 bug 还是设计权衡？

第 3 问的判据（这是实践 8 vs 9 的关键区别）：
**看奖励函数在当前误差量级上还有没有梯度。**
  - 实践 9：exp(-1.9²/0.3²) = 3e-18，梯度消失 → bug
  - 实践 8：任务奖励占 40% 但确实在起作用（摔倒率 0.992→0.128）→ 权衡

用法：
    envs/isaaclab/bin/python scripts/verify_training_outcome.py
    envs/isaaclab/bin/python scripts/verify_training_outcome.py --practice 9
"""
from __future__ import annotations

import argparse
import glob
import math
import os

WS = "/home/limx/workspace/Roxan_warmup"
G, R, Y, D, B, N = ("\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")

# (实践号, 名称, run glob, 主指标, 越小越好, 稳定性指标, 奖励核的 std)
SPECS = [
    ("2", "粗糙地形行走",
     f"{WS}/repos/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_rough/*",
     "Metrics/base_velocity/error_vel_xy", True,
     "Episode_Termination/bad_orientation", None),
    ("4", "蹲姿行走",
     f"{WS}/shenlan_hw/hw4_mjlab/logs/rsl_rl/g1_velocity_height/*ablation_baseline",
     "Metrics/base_height/error_height", True, None, None),
    ("5", "分层导航",
     f"{WS}/shenlan_hw/hw5_navigation/**/2026-09-06_16-37-04",
     "Metrics/pose_command/error_pos_2d", True,
     "Episode_Termination/goal_reached", None),
    ("6", "蒸馏 KL",
     f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_student_kl_matching/2026-09-05*",
     "Metrics/motion/error_joint_pos", True, None, 0.3),
    ("8", "AMP 拟人走跑",
     f"{WS}/shenlan_hw/unitree_lab_amp/logs/rsl_rl_amp/*/2026-09-07*",
     "Metrics/base_velocity/error_vel_xy", True,
     "Episode_Termination/bad_orientation", None),
    ("9", "轨迹跟踪",
     f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_teacher/2026-09-05_23-41-39*",
     "Metrics/motion/error_joint_pos", True, None, 0.3),
    ("11", "跑酷",
     f"{WS}/repos/instinctlab/logs/instinct_rl/g1_parkour/*",
     "Metrics/base_velocity/error_vel_xy", True,
     "Episode_Termination/bad_orientation", None),
]


def load(pattern: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    dirs = [d for d in glob.glob(pattern, recursive=True) if os.path.isdir(d)]
    if not dirs:
        return None
    for d in sorted(dirs, reverse=True):
        files = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
        if files:
            ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
            ea.Reload()
            if ea.Tags()["scalars"]:
                return ea, os.path.basename(d.rstrip("/"))
    return None


def seg5(ea, tag: str):
    """五段均值。找不到 tag 时按后缀模糊匹配 —— 各框架前缀不统一。"""
    tags = ea.Tags()["scalars"]
    if tag not in tags:
        leaf = tag.split("/")[-1]
        cand = [t for t in tags if t.endswith(leaf)]
        if not cand:
            return None
        tag = cand[0]
    v = [p.value for p in ea.Scalars(tag)]
    n = len(v)
    if n < 10:
        return None
    return [sum(v[i * n // 5:(i + 1) * n // 5]) / max(1, len(v[i * n // 5:(i + 1) * n // 5]))
            for i in range(5)]


def curriculum_rise(ea):
    """课程难度涨了多少倍。前缀在各框架里不一致，用包含匹配。"""
    best = None
    for t in ea.Tags()["scalars"]:
        if "Curriculum/" not in t:
            continue
        s = seg5(ea, t)
        if s and s[0] > 1e-6:
            ratio = s[-1] / s[0]
            if best is None or ratio > best[1]:
                best = (t.split("/")[-1], ratio, s)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--practice", help="只查某个实践，如 9")
    args = ap.parse_args()

    print(f"\n{B}训练效果验收 —— 按真实任务指标，不看 reward{N}")
    print("=" * 88)

    problems = []
    for pid, name, pat, metric, lower_better, stab, std in SPECS:
        if args.practice and args.practice != pid:
            continue
        got = load(pat)
        label = f"{pid:<4}{name:<16}"
        if not got:
            print(f"{label}{D}（无数据）{N}")
            continue
        ea, run = got

        s = seg5(ea, metric)
        if s is None:
            print(f"{label}{D}找不到 {metric.split('/')[-1]}{N}")
            continue

        leaf = metric.split("/")[-1]
        curve = "→".join(f"{x:.3f}" for x in s)
        improved = (s[-1] < s[0]) if lower_better else (s[-1] > s[0])

        if improved:
            print(f"{label}{G}✅{N} {leaf} {curve}")
            continue

        # 指标变差 —— 先看能否用课程难度解释
        ratio = s[-1] / max(s[0], 1e-9)
        cur = curriculum_rise(ea)
        print(f"{label}{Y}⚠️{N}  {leaf} {curve}  ({ratio:.2f}×)")

        if cur and cur[1] > 1.0:
            cname, crise, _ = cur
            print(f"{'':<20}{D}课程 {cname} 涨 {crise:.2f}×{N}", end="")
            if ratio <= crise * 1.3:
                print(f"  {G}→ 误差涨幅在难度范围内，正常{N}")
                continue
            print(f"  {R}→ 超出难度可解释范围{N}")

        # 第 3 问：还有没有梯度
        if std is not None:
            grad = math.exp(-(s[-1] ** 2) / (std ** 2))
            print(f"{'':<20}{D}奖励 exp(-err²/std²) 当前 = {grad:.2e}{N}", end="")
            if grad < 1e-4:
                print(f"  {R}→ 梯度已消失，是 bug{N}")
                problems.append((pid, name, f"奖励饱和 {grad:.1e}，std={std} 与误差量级不匹配"))
                continue
            print(f"  {G}→ 仍有梯度{N}")

        # 稳定性还在改善 → 多半是设计权衡
        if stab:
            st = seg5(ea, stab)
            if st and st[-1] < st[0] * 0.5:
                print(f"{'':<20}{G}→ 但 {stab.split('/')[-1]} "
                      f"{st[0]:.3f}→{st[-1]:.3f} 大幅改善，属设计权衡{N}")
                continue
        problems.append((pid, name, "指标变差且无法归因"))

    print("=" * 88)
    if problems:
        print(f"\n{R}{B}需要处理{N}")
        for pid, name, why in problems:
            print(f"  实践 {pid} {name}：{why}")
    else:
        print(f"\n{G}全部通过 —— 每个实践的任务指标要么在改善，"
              f"要么变差可被课程难度或设计权衡解释{N}")

    print(f"\n{D}判据说明：{N}")
    print(f"  {D}① 任务指标在改善吗（不是 reward）{N}")
    print(f"  {D}② 若变差，涨幅 ≤ 课程难度涨幅×1.3 视为正常{N}")
    print(f"  {D}③ 若超出，看 exp(-err²/std²) 还有没有梯度：{N}")
    print(f"  {D}   < 1e-4 = 梯度消失是 bug；仍有梯度且稳定性在改善 = 设计权衡{N}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
