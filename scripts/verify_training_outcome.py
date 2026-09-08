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
     # std 与维度成对给出：指标是 L2 范数，奖励用均值，换算要除以维度
     "Metrics/motion/error_joint_pos", True, None, (0.3, 29)),
    ("8", "AMP 拟人走跑",
     f"{WS}/shenlan_hw/unitree_lab_amp/logs/rsl_rl_amp/*/2026-09-07*",
     "Metrics/base_velocity/error_vel_xy", True,
     "Episode_Termination/bad_orientation", None),
    ("9", "轨迹跟踪",
     # 不锁定某一次 run：重训会产生新目录，load() 取最新的那个。
     # 锁死旧 run 的话，验收永远在看那次已知有 bug 的训练。
     f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_teacher/*",
     "Metrics/motion/error_joint_pos", True,
     # 稳定性指标此前留空(None)，导致第四层判据被跳过，
     # 于是明明是"权衡"却报成"变差且无法归因"。
     # 这个任务的稳定性看失败率：1.000 → 0.684，同期
     # mean_episode_length 5.04 → 144.82（28.7 倍）。
     "Episode_Termination/fail",
     # std 与训练时同源（retrain_p9_fixed_std.sh 用同一个环境变量）。
     # 默认 0.3 是官方参考答案的值，此前误判它"饱和"是我算错了——
     # 见下方 n_dof 的注释。29 = G1 的关节数。
     (float(os.getenv("HW9_JOINT_POS_STD", "0.3")), 29)),
    ("10", "HOI 感知跟踪",
     # 日志根路径来自 HOI_Mimic/scripts/rsl_rl/train.py:188
     #   os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
     # experiment_name 见 mimic/agents/rsl_rl_ppo_cfg.py:68
     f"{WS}/shenlan_hw/HOI_Mimic/logs/rsl_rl/unitree_g1_29dof_mimic_hoi_terrain_perceptive_raycast/*",
     # error_joint_pos 已从源码核实存在（mimic 任务的 metrics 字典），
     # 与实践 9 同名；seg5 会按后缀模糊匹配，前缀不一致也能找到。
     "Metrics/motion/error_joint_pos", True,
     # 终止项名称与奖励 std 都没在源码里核实到，宁可留空也不写猜测值——
     # 填错会让第三层「梯度是否消失」判据给出看似有据实则无效的结论。
     None, None),
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
            # size_guidance 里 0 表示"全部保留"。原来写的是 {"scalars": 0}，
            # 一次把每个 tag 的全部标量点都读进内存；实践 11 一个 run 就有
            # 119 个 tag，多个 run 叠加会明显吃内存（实测能把机器拖卡）。
            # 这里改成有限上限：5000 点足够算五段均值，
            # 且当前最长的 run 也只有 3000 轮，不会触发降采样、结论不受影响。
            ea = EventAccumulator(files[0], size_guidance={
                "scalars": 5000, "histograms": 1, "compressedHistograms": 1,
                "images": 1, "audio": 1, "tensors": 1,
            })
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
    partial = []
    skipped = []
    passed = 0
    for pid, name, pat, metric, lower_better, stab, std in SPECS:
        if args.practice and args.practice != pid:
            continue
        got = load(pat)
        label = f"{pid:<4}{name:<16}"
        if not got:
            print(f"{label}{D}（无数据）{N}")
            skipped.append((pid, name, "没有训练日志"))
            continue
        ea, run = got

        s = seg5(ea, metric)
        if s is None:
            print(f"{label}{D}找不到 {metric.split('/')[-1]}{N}")
            skipped.append((pid, name, f"日志里没有 {metric.split('/')[-1]}"))
            continue

        leaf = metric.split("/")[-1]
        curve = "→".join(f"{x:.3f}" for x in s)
        improved = (s[-1] < s[0]) if lower_better else (s[-1] > s[0])

        if improved:
            print(f"{label}{G}✅{N} {leaf} {curve}")
            passed += 1
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
                passed += 1
                continue
            print(f"  {R}→ 超出难度可解释范围{N}")

        # 第 3 问：还有没有梯度
        if std is not None:
            # std 字段是 (std, n_dof) 二元组。为什么要带维度：
            #     指标 error_joint_pos = torch.norm(diff)      ← L2 范数
            #     奖励               = exp(-mean(diff²)/std²) ← 均值
            #     norm² = n_dof × mean(diff²)
            # 早先这里直接写 exp(-err²/std²)，等于把指数放大了 n_dof 倍，
            # 把实际 0.28 的奖励算成 5.9e-17，于是给出"梯度消失是 bug"的
            # 错误结论——而实测 Reward_per_Sec/motion_joint_pos 一直有 0.47。
            std_val, n_dof = std
            mean_sq = (s[-1] ** 2) / n_dof
            grad = math.exp(-mean_sq / (std_val ** 2))
            print(f"{'':<20}{D}奖励 exp(-mean(err²)/std²) = {grad:.4f}"
                  f"（指标为 {n_dof} 维 L2 范数，std={std_val}）{N}", end="")
            if grad < 1e-4:
                print(f"  {R}→ 梯度已消失，是 bug{N}")
                problems.append((pid, name,
                                 f"奖励饱和 {grad:.1e}，std={std_val} 与误差量级不匹配"))
                continue
            print(f"  {G}→ 仍有梯度{N}")

        # 稳定性还在改善 → 多半是设计权衡
        #
        # 分三档而不是一刀切。原来只有 ">50% 就算权衡"这一个门槛，
        # 于是实践 9（失败率降 32%、存活时长涨 28.7 倍）落进了
        # "无法归因"，报得比实情严重。但也不该为了让它通过就把门槛
        # 调松——那是把判据改成迎合结论。正确做法是让判据能表达中间态。
        if stab:
            st = seg5(ea, stab)
            if st and st[0] > 1e-9:
                drop = 1 - st[-1] / st[0]
                leaf_s = stab.split("/")[-1]
                if drop >= 0.5:
                    print(f"{'':<20}{G}→ 但 {leaf_s} {st[0]:.3f}→{st[-1]:.3f} "
                          f"（降 {drop:.0%}）大幅改善，属设计权衡{N}")
                    passed += 1
                    continue
                if drop >= 0.2:
                    print(f"{'':<20}{Y}→ {leaf_s} {st[0]:.3f}→{st[-1]:.3f} "
                          f"（降 {drop:.0%}）有改善但不充分{N}")
                    print(f"{'':<20}{D}   策略在学『活下去』，精度还没顾上；"
                          f"不是 bug，是训练未完成{N}")
                    partial.append((pid, name,
                                    f"{leaf_s} 降 {drop:.0%}，但主指标仍在变差"))
                    continue
                print(f"{'':<20}{R}→ {leaf_s} 仅降 {drop:.0%}，稳定性也没改善{N}")
        problems.append((pid, name, "指标变差且无法归因"))

    print("=" * 88)
    if problems:
        print(f"\n{R}{B}需要处理{N}")
        for pid, name, why in problems:
            print(f"  实践 {pid} {name}：{why}")
    elif passed:
        print(f"\n{G}已验证 {passed} 项 —— 任务指标要么在改善，"
              f"要么变差可被课程难度或设计权衡解释{N}")

    # 「无数据」不等于「通过」。之前这两种情况直接 continue 掉，
    # 于是一个还没开始训练的实践也会被计进"全部通过"——
    # 假阳性比漏报危险，必须单独列出来。
    if partial:
        print(f"\n{Y}部分达标 {len(partial)} 项（稳定性在改善，但主指标仍变差）{N}")
        for pid, name, why in partial:
            print(f"  实践 {pid} {name}：{why}")

    if skipped:
        print(f"\n{Y}未验证 {len(skipped)} 项（缺数据，不算通过）{N}")
        for pid, name, why in skipped:
            print(f"  实践 {pid} {name}：{why}")

    if not problems and not passed and not skipped:
        print(f"\n{D}没有匹配的实践{N}")

    print(f"\n{D}判据说明：{N}")
    print(f"  {D}① 任务指标在改善吗（不是 reward）{N}")
    print(f"  {D}② 若变差，涨幅 ≤ 课程难度涨幅×1.3 视为正常{N}")
    print(f"  {D}③ 若超出，看 exp(-err²/std²) 还有没有梯度：{N}")
    print(f"  {D}   < 1e-4 = 梯度消失是 bug；仍有梯度且稳定性在改善 = 设计权衡{N}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
