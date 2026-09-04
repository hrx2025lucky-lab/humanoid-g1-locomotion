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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=200, help="每组仿真步数")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--std", type=float, default=0.20,
                    help="随机策略的动作噪声 std（与 init_noise_std 对齐）")
parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0],
                    help="要对照的 EMA 平滑系数")
parser.add_argument("--seed", type=int, default=0)
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

    start_xy = _root_xy(env.unwrapped)
    # 累计路程要逐步累加：直线距离会把"走出去又走回来"误记为没动
    path_len = torch.zeros(args_cli.num_envs, device="cuda:0")
    prev_xy = start_xy.clone()
    max_disp = torch.zeros(args_cli.num_envs, device="cuda:0")

    # 只统计没被 reset 打断的 env，避免把"重生瞬移"算成位移
    alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device="cuda:0")
    n_fell = 0

    for _ in range(args_cli.steps):
        # 随机初始化策略 = 零均值高斯（真实 PPO 起点的均值也接近 0）
        action = torch.randn(args_cli.num_envs, act_dim, device="cuda:0") * args_cli.std
        obs, _, terminated, truncated, _ = env.step(action)

        cur_xy = _root_xy(env.unwrapped)
        step_d = torch.norm(cur_xy - prev_xy, dim=1)
        # 单步位移超过 1m 只可能是 reset 瞬移，不计入路程
        moved = torch.where(step_d < 1.0, step_d, torch.zeros_like(step_d))
        path_len += torch.where(alive, moved, torch.zeros_like(moved))
        disp = torch.norm(cur_xy - start_xy, dim=1)
        max_disp = torch.maximum(max_disp, torch.where(alive & (disp < 100), disp, max_disp))
        prev_xy = cur_xy

        done = terminated | truncated
        if done.any():
            n_fell += int(done.sum().item())
            alive &= ~done

    res = {
        "alpha": alpha,
        "path_len_mean": float(path_len.mean().item()),
        "max_disp_mean": float(max_disp.mean().item()),
        "max_disp_p90": float(torch.quantile(max_disp, 0.9).item()),
        "n_done": n_fell,
        "still_alive": int(alive.sum().item()),
    }
    env.close()
    return res


def main() -> int:
    print("=" * 68)
    print("实践 5 探索探针：EMA 平滑对随机策略位移的影响")
    print(f"  随机策略 std = {args_cli.std}（与 init_noise_std 对齐）")
    print(f"  步数 = {args_cli.steps}，并行环境 = {args_cli.num_envs}")
    print("=" * 68)

    results = []
    for alpha in args_cli.alphas:
        print(f"\n--- 平滑 alpha = {alpha} "
              f"({'关闭' if alpha >= 1.0 else '开启'}) ---")
        r = run_one(alpha)
        results.append(r)
        print(f"  累计路程   {r['path_len_mean']:.3f} m")
        print(f"  最远位移   {r['max_disp_mean']:.3f} m (p90 {r['max_disp_p90']:.3f})")
        print(f"  存活/结束  {r['still_alive']} / {r['n_done']}")

    print("\n" + "=" * 68)
    print("结论")
    print("=" * 68)
    if len(results) == 2:
        on, off = results[0], results[1]
        # 理论预测：噪声幅度衰减到 sqrt(a/(2-a))
        a = on["alpha"]
        pred = (a / (2 - a)) ** 0.5
        ratio = (on["path_len_mean"] / off["path_len_mean"]
                 if off["path_len_mean"] > 1e-6 else float("nan"))
        print(f"  路程比 (alpha={a}) / (alpha=1.0) = {ratio:.3f}")
        print(f"  EMA 噪声衰减理论值               = {pred:.3f}")
        if ratio < 0.6:
            print("\n  ✅ 平滑显著压制了探索位移 —— 支持「EMA 扼杀探索」的假设")
        else:
            print("\n  ❌ 平滑未显著压制位移 —— 探索塌缩另有原因，需继续查")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
