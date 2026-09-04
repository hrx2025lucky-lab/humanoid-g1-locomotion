"""实践 5 探索能力探针 —— 量化 EMA 平滑对高层探索的抑制。

动机
----
baseline 跑满 3000 iter 仍然 `goals_reached = 0`、`error_pos_2d` 反而从 6.60
劣化到 7.65。日志里 `Mean action noise std` 在前 100 iter 就从 0.20 塌到 0.07。

一个可疑的机制是我自己在 §10 加的 EMA 平滑（alpha=0.1）：
EMA 对**时间上不相关**的信号衰减极强。若高层每步注入独立高斯噪声 eps~N(0,s^2)，
经 y_t = (1-a) y_{t-1} + a x_t 后，噪声分量的稳态方差是

    Var[y_noise] = s^2 * a / (2 - a)

alpha=0.1 时衰减系数 sqrt(0.1/1.9) = 0.229 —— 探索幅度只剩 23%。
而策略**均值**是时间相关的慢变量，几乎不被衰减。也就是说 EMA
选择性地掐掉了探索，却保留了「站着不动」的均值。

这与我之前用白噪声测低层鲁棒性时踩的坑是同一个机制（见 §10 附注）：
当时 EMA 把测试信号衰减掉，让我误判低层很鲁棒。

怎么测才有说服力
----------------
不能用已经训练过的 baseline 策略：它的均值已经塌到 0，关不关平滑都不动，
测不出差别。要测的是**学习能否 bootstrap**，所以必须用**随机初始化**策略
（std=0.20，与 init_noise_std 一致）问一个更本质的问题：

    在完全不会走的阶段，智能体有没有可能"瞎走"到目标附近，
    从而尝到 success_bonus，让 PPO 有梯度可跟？

若开平滑时位移显著小于关平滑，则 EMA 就是探索塌缩的原因之一。

用法
----
    python probe_p5_exploration.py                 # 默认对照 0.1 vs 1.0
    python probe_p5_exploration.py --steps 300
"""

from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=200, help="每组仿真步数")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--std", type=float, default=0.20,
                    help="随机策略的动作噪声 std（与 init_noise_std 对齐）")
parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0],
                    help="要对照的 EMA 平滑系数")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=str(pathlib.Path.home()
                    / "humanoid_logs" / "p5_navigation" / "exploration_probe.json"),
                    help="结果 JSON（多次调用会累积合并，便于一次只跑一个 alpha）")
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

NAV_TASK = "Unitree-G1-29dof-Navigation-HRL-Baseline"
_HL_DT = 0.2  # 高层步长（秒），构造环境后用 env.step_dt 覆盖为真值


def _root_xy(env) -> torch.Tensor:
    """机器人根节点的世界系 xy，取副本避免拿到会被原地改写的视图。"""
    return env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()


def run_one(alpha: float) -> dict:
    """在给定平滑系数下用随机策略滚一条轨迹，返回探索位移统计。"""
    # 平滑系数由 env cfg 在构造时读环境变量决定，必须在 parse_env_cfg 之前设置
    os.environ["NAV_COMMAND_SMOOTHING"] = str(alpha)

    env_cfg = parse_env_cfg(NAV_TASK, device="cuda:0",
                            num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(NAV_TASK, cfg=env_cfg)

    torch.manual_seed(args_cli.seed)
    obs, _ = env.reset()
    act_dim = env.unwrapped.action_manager.total_action_dim
    global _HL_DT
    _HL_DT = float(getattr(env.unwrapped, "step_dt", _HL_DT))

    start_xy = _root_xy(env.unwrapped)
    prev_xy = start_xy.clone()
    path_len = torch.zeros(args_cli.num_envs, device="cuda:0")
    max_disp = torch.zeros(args_cli.num_envs, device="cuda:0")
    # 只统计"从未被重置过"的 env。IsaacLab 在 episode 结束时会自动重生，
    # 机器人瞬移到新出生点；若在更新掩码前就统计位移，会把瞬移当成移动
    # （初版就踩了这个坑：200 步内量出 26 m，而 G1 最快也就 ~1 m/s）。
    alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device="cuda:0")
    alive_steps = 0  # 累计"有效 env·步"，用于算不受 episode 长短影响的平均速度
    n_done = 0

    for _ in range(args_cli.steps):
        # 随机初始化策略 = 零均值高斯（真实 PPO 起点的均值也接近 0）
        action = torch.randn(args_cli.num_envs, act_dim, device="cuda:0") * args_cli.std
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated

        cur_xy = _root_xy(env.unwrapped)
        # 本步就结束的 env 已经被重生，它的 cur_xy 无意义 —— 本步起即排除
        valid = alive & ~done
        zeros = torch.zeros_like(path_len)
        path_len += torch.where(valid, torch.norm(cur_xy - prev_xy, dim=1), zeros)
        max_disp = torch.maximum(
            max_disp, torch.where(valid, torch.norm(cur_xy - start_xy, dim=1), zeros))
        prev_xy = cur_xy

        n_done += int((alive & done).sum().item())
        alive = valid
        alive_steps += int(valid.sum().item())

    n_alive = int(alive.sum().item())
    total_path = float(path_len.sum().item())
    res = {
        "alpha": alpha,
        # 平均速度对"活多久"不敏感，是跨组比较的主指标；
        # 累计路程会被早摔的组系统性低估，只作参考。
        "speed": total_path / max(alive_steps, 1) / _HL_DT,
        "path_len_mean": float(path_len.mean().item()),
        "max_disp_mean": float(max_disp.mean().item()),
        "max_disp_max": float(max_disp.max().item()),
        "alive_steps": alive_steps,
        "n_done": n_done,
        "still_alive": n_alive,
    }
    env.close()
    return res


def main() -> int:
    print("=" * 68, flush=True)
    print("实践 5 探索探针：EMA 平滑对随机策略位移的影响", flush=True)
    print(f"  随机策略 std = {args_cli.std}（与 init_noise_std 对齐）", flush=True)
    print(f"  步数 = {args_cli.steps}，并行环境 = {args_cli.num_envs}", flush=True)
    print("=" * 68, flush=True)

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 累积到同一个 JSON：一个进程里建两次 IsaacLab 环境非常慢（实测 >40 min
    # 还没跑完第二组），所以改成一次一个 alpha、多次调用，用文件汇总结果。
    saved = json.loads(out.read_text()) if out.exists() else {}

    for alpha in args_cli.alphas:
        print(f"\n--- 平滑 alpha = {alpha} "
              f"({'关闭' if alpha >= 1.0 else '开启'}) ---", flush=True)
        r = run_one(alpha)
        saved[f"{alpha}"] = r
        out.write_text(json.dumps(saved, indent=2, ensure_ascii=False))
        print(f"  平均速度   {r['speed']:.4f} m/s   ← 主指标", flush=True)
        print(f"  累计路程   {r['path_len_mean']:.3f} m/env", flush=True)
        print(f"  最远位移   {r['max_disp_mean']:.3f} m (最大 {r['max_disp_max']:.3f})",
              flush=True)
        print(f"  有效步数   {r['alive_steps']}   存活/结束 "
              f"{r['still_alive']}/{r['n_done']}", flush=True)
        print(f"  已写入 {out}", flush=True)

    _summarize(saved)
    return 0


def _summarize(saved: dict) -> None:
    """有了两组结果就下结论；只有一组就提示还差哪一组。"""
    on, off = saved.get("0.1"), saved.get("1.0")
    print("\n" + "=" * 68, flush=True)
    print("结论", flush=True)
    print("=" * 68, flush=True)
    if not (on and off):
        have = sorted(saved.keys())
        print(f"  已有 alpha = {have}，还需另一组才能对照。", flush=True)
        print("  跑法： --alphas 1.0   （或 0.1）", flush=True)
        return

    a = on["alpha"]
    # EMA 对时间不相关噪声的稳态衰减：sqrt(a / (2 - a))
    pred = (a / (2 - a)) ** 0.5
    ratio = on["speed"] / off["speed"] if off["speed"] > 1e-9 else float("nan")
    print(f"  平均速度  alpha={a}: {on['speed']:.4f} m/s   "
          f"alpha=1.0: {off['speed']:.4f} m/s", flush=True)
    print(f"  实测比值  {ratio:.3f}", flush=True)
    print(f"  EMA 噪声衰减理论值 sqrt(a/(2-a)) = {pred:.3f}", flush=True)
    if ratio < 0.6:
        print("\n  ✅ 平滑显著压制了探索位移 —— 支持「EMA 扼杀探索」的假设。", flush=True)
        print("     对策：低层已换成鲁棒得多的自训策略，平滑的初衷"
              "（保护脆弱预训练低层）已不成立，可关闭。", flush=True)
    else:
        print("\n  ❌ 平滑未显著压制位移 —— 探索塌缩另有原因。", flush=True)
        print("     下一步查：entropy_coef 是否过小、动作缩放、低层指令响应。", flush=True)


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
