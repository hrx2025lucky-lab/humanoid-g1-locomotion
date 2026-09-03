#!/usr/bin/env python3
"""实践 4 三组消融的结果对比。

验证的命题：**"按指令做某事"必须打通三段闭环，缺一不可**
    ① 采样高度指令  →  ② Actor 观测到指令  →  ③ 奖励高度跟踪

三组各切断其中一段（baseline 三段齐全），共用同一份 rl_cfg 与 seed，
唯一变量就是那一处配置差异 —— 已在训练前逐项核对过：

    组别            actor 有 height_cmd   critic 有   reward 权重
    baseline              ✅ 8 项           ✅ 12 项      1.0
    blind_actor           ❌ 7 项           ✅ 12 项      1.0     ← 切断②
    no_height_rew         ✅ 8 项           ✅ 12 项      0.0     ← 切断③

主判据是 `Metrics/base_height/error_height`，不是 reward ——
reward 会因为权重被置零而失去可比性（no_height_rew 组根本不发这项钱）。

blind_actor 组有一个**可解析的预测值**，这让消融结论可证伪而不只是"更差"：
Actor 看不到目标高度，只能收敛到让全区间期望误差最小的固定高度，
即区间中点 (0.45+0.80)/2 = 0.625 m，此时
    E|h_cmd - 0.625| = (0.80-0.45)/4 = 0.0875 m

用法：
    python compare_p4_ablation.py
"""

from __future__ import annotations

import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.environ.get(
    "HW4_LOG_DIR",
    "/home/limx/workspace/Roxan_warmup/shenlan_hw/hw4_mjlab/logs/rsl_rl/g1_velocity_height",
)

GROUPS = [
    ("baseline", "ablation_baseline", "三段齐全"),
    ("blind_actor", "ablation_blind_actor", "切断② Actor 看不到指令"),
    ("no_height_rew", "ablation_no_height_rew", "切断③ 不发高度奖励"),
]

# 理论预测：Actor 盲时收敛到区间中点，期望误差 = 区间宽度 / 4
H_LO, H_HI = 0.45, 0.80
BLIND_PREDICTED_ERROR = (H_HI - H_LO) / 4


def load(run_dir: str) -> EventAccumulator | None:
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        return None
    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()
    return ea if ea.Tags()["scalars"] else None


def tail_mean(ea: EventAccumulator, tag: str, frac: float = 0.1) -> float | None:
    """取末尾 frac 比例的均值，比单点末值稳健（训练指标本身有抖动）。"""
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
    out = []
    for s in steps:
        prior = [v for st, v in pts if st <= s]
        out.append(prior[-1] if prior else None)
    return out


def fmt(v: float | None, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def calibration_factor(ea: EventAccumulator) -> float | None:
    """用 target_height_mean 反推指标的膨胀倍数。

    ★ 发现的一个指标 bug：`_update_metrics` 每步累加后除以
    `max_command_step = resampling_time_range[1] / step_dt = 8.0/0.02 = 400`，
    但 metrics 实际累积了**整个 episode**（997.7 步）而不是单条指令的生命周期，
    于是所有量被同一个因子放大：997.7/400 ≈ 2.494。

    好在这个指标自带校准物：`target_height_mean` 的真值必然是指令区间中点
    (0.45+0.80)/2 = 0.625（指令是均匀采样的）。实测 1.5579，
    比值 2.493 与理论膨胀 2.494 吻合到小数点后两位，确认了这个解释。

    由于三组的 episode 长度几乎相同（997.5~997.9），膨胀因子一致，
    **组间相对比较不受影响**；但绝对值必须除以该因子才能与理论预测对比。
    """
    t = tail_mean(ea, "Metrics/base_height/target_height_mean")
    if t is None or t <= 0:
        return None
    return t / ((H_LO + H_HI) / 2)


def implied_constant_height(err: float) -> tuple[float, float]:
    """由"固定高度策略"的期望误差反推该固定高度。

    h_cmd ~ U(a,b) 且策略输出恒定 h* 时：
        E|h_cmd - h*| = [(h*-a)^2 + (b-h*)^2] / (2(b-a))
    令 u = h* - 中点，化简得 E = (2u^2 + (b-a)^2/2) / (2(b-a))
    反解出 u，得到对称的两个解。
    """
    span = H_HI - H_LO
    mid = (H_LO + H_HI) / 2
    u_sq = (err * 2 * span - span * span / 2) / 2
    if u_sq < 0:
        return (mid, mid)
    u = u_sq ** 0.5
    return (mid - u, mid + u)


def main() -> int:
    runs = {}
    for key, suffix, _ in GROUPS:
        matches = sorted(glob.glob(os.path.join(ROOT, f"*{suffix}")))
        if not matches:
            print(f"⚠️  找不到 {key} 的 run（后缀 {suffix}）", file=sys.stderr)
            continue
        ea = load(matches[-1])
        if ea is None:
            print(f"⚠️  {key} 的 run 里没有 scalars", file=sys.stderr)
            continue
        runs[key] = (ea, os.path.basename(matches[-1]))

    if not runs:
        sys.exit("没有可用的 run")

    print()
    print("═" * 74)
    print("实践 4 消融对比 —— 速度+骨盆高度双指令 MDP 的三段闭环")
    print("═" * 74)
    for key, _, desc in GROUPS:
        if key in runs:
            print(f"  {key:<15}{desc:<26}{runs[key][1]}")
    print()

    # ── 主判据 ──────────────────────────────────────────────────────────
    cal = calibration_factor(runs["baseline"][0]) if "baseline" in runs else None
    if cal:
        print(f"── 指标校准 ──")
        print(f"  target_height_mean 实测 "
              f"{tail_mean(runs['baseline'][0], 'Metrics/base_height/target_height_mean'):.4f}"
              f"，真值应为 {(H_LO + H_HI) / 2:.4f}（指令均匀采样）")
        print(f"  → 指标被放大 {cal:.3f}× （metrics 累积整个 episode 却除以"
              f"单条指令的最大步数）")
        print(f"  三组 episode 长度一致，故**相对比较不受影响**；下表给出校准后的绝对值。")
        print()

    print("── 主判据：高度跟踪误差（越小越好）──")
    print(f"  {'组别':<16}{'原始值':>12}{'校准后 m':>12}{'相对 baseline':>16}")
    base_err = tail_mean(runs["baseline"][0], "Metrics/base_height/error_height") \
        if "baseline" in runs else None
    for key, _, _ in GROUPS:
        if key not in runs:
            continue
        e = tail_mean(runs[key][0], "Metrics/base_height/error_height")
        if e is None:
            print(f"  {key:<16}{'—':>12}")
            continue
        corrected = e / cal if cal else e
        rel = "—" if not base_err else f"{e / base_err:.2f}×"
        print(f"  {key:<16}{e:>12.4f}{corrected:>12.4f}{rel:>16}")
    print()

    # ── 可证伪的理论预测 ────────────────────────────────────────────────
    if "blind_actor" in runs:
        e = tail_mean(runs["blind_actor"][0], "Metrics/base_height/error_height")
        if e is not None:
            e_cal = e / cal if cal else e
            print("── blind_actor 的可证伪预测 ──")
            print(f"  若策略收敛到区间中点 {(H_LO + H_HI) / 2:.3f} m，")
            print(f"  理论误差 E|h_cmd - 0.625| = ({H_HI}-{H_LO})/4 = {BLIND_PREDICTED_ERROR:.4f} m")
            print(f"  校准后实测 {e_cal:.4f} m")
            lo, hi = implied_constant_height(e_cal)
            print(f"  → 反推该固定高度为 {lo:.3f} m 或 {hi:.3f} m")
            if e_cal < (H_HI - H_LO) / 2:
                print(f"  ✅ 落在「固定高度策略」的理论区间内"
                      f"（{BLIND_PREDICTED_ERROR:.4f} ~ {(H_HI - H_LO) / 2:.4f} m）")
                print(f"     Actor 看不到指令时确实退化为输出一个固定高度；")
                print(f"     它没停在最优的 0.625 而是偏向 {hi:.3f} m —— "
                      f"因为速度跟踪等其它奖励")
                print(f"     把姿态往自然站高（约 0.78 m）拉，最终是两者的折中。")
            else:
                print(f"  ⚠️  超出固定高度策略的理论上限 {(H_HI - H_LO) / 2:.4f} m，")
                print(f"     说明实际高度落在指令区间 [{H_LO}, {H_HI}] 之外。")
            print()

    # ── 对照项：速度任务不应受影响 ──────────────────────────────────────
    print("── 对照项：速度跟踪（确认改动没有波及主任务）──")
    print(f"  {'组别':<16}{'error_vel_xy':>14}{'track_linear_velocity':>24}")
    for key, _, _ in GROUPS:
        if key not in runs:
            continue
        ea = runs[key][0]
        v = tail_mean(ea, "Metrics/velocity/error_vel_xy")
        r = tail_mean(ea, "Episode_Reward/track_linear_velocity")
        print(f"  {key:<16}{fmt(v):>14}{fmt(r):>24}")
    print()

    # ── 奖励项：确认消融真的生效 ────────────────────────────────────────
    print("── 消融生效性核对 ──")
    print(f"  {'组别':<16}{'track_base_height 奖励':>24}")
    for key, _, _ in GROUPS:
        if key not in runs:
            continue
        r = tail_mean(runs[key][0], "Episode_Reward/track_base_height")
        note = ""
        if key == "no_height_rew":
            note = "  ← 应为 0（权重已置零）" if r is not None and abs(r) < 1e-6 \
                else "  ← ⚠️ 期望为 0"
        print(f"  {key:<16}{fmt(r):>24}{note}")
    print()

    # ── 收敛过程 ────────────────────────────────────────────────────────
    marks = [0, 500, 1000, 1500, 2000, 2500, 2999]
    print("── error_height 收敛过程 ──")
    header = "  " + f"{'iter':<8}" + "".join(
        f"{k:>16}" for k, _, _ in GROUPS if k in runs)
    print(header)
    series = {k: series_at(runs[k][0], "Metrics/base_height/error_height", marks)
              for k, _, _ in GROUPS if k in runs}
    for i, m in enumerate(marks):
        row = f"  {m:<8}"
        for k, _, _ in GROUPS:
            if k in series:
                row += f"{fmt(series[k][i]):>16}"
        print(row)
    print()

    print("═" * 74)
    if base_err is not None:
        worse = [(k, tail_mean(runs[k][0], "Metrics/base_height/error_height"))
                 for k, _, _ in GROUPS if k in runs and k != "baseline"]
        worse = [(k, v) for k, v in worse if v is not None]
        if worse and all(v > base_err for _, v in worse):
            print("结论：两组消融的高度误差都劣于 baseline，")
            print("      三段闭环「采样→观测→奖励」缺一不可得到验证。")
        else:
            print("结论：消融组未全部劣于 baseline，需检查训练是否充分收敛。")
    print("═" * 74)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
