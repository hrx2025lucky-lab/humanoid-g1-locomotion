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
parser.add_argument("--oracle", action="store_true",
                    help="改用脚本化『朝目标走』的高层策略，测任务的性能上界")
parser.add_argument("--sweep", action="store_true",
                    help="扫描恒定前进速度，找低层策略不摔倒的安全速度上限")
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


def _oracle_action(env) -> torch.Tensor:
    """脚本化高层策略：转向目标 + 前进。用来测这个任务的性能上界。

    动机：训练侧 goals_reached 恒为 0，但"学不会"有两种完全不同的原因 ——
      A. RL 探索/优化不到位（策略问题，调超参有救）
      B. 任务本身不可达（成功半径太小、目标在障碍里、时限不够 …… 调什么都没用）
    用一个不需要学习的控制器就能把两者分开：oracle 到得了就是 A，到不了就是 B。

    pose_command 的前两维是目标在**基座系**下的 xy 偏移，所以航向误差直接
    就是 atan2(dy, dx)，不需要再做世界系到基座系的变换。
    """
    cmd = env.command_manager.get_command("pose_command")
    dx, dy = cmd[:, 0], cmd[:, 1]
    heading_err = torch.atan2(dy, dx)

    act = torch.zeros(env.num_envs, 3, device=cmd.device)
    # 先对准再走：航向偏差大时慢速转身，对准后全速前进。
    # 这两个阈值不必精调 —— 只要能证明"存在一条策略能到达"即可。
    aligned = heading_err.abs() < 0.6
    act[:, 0] = torch.where(aligned, torch.full_like(dx, 1.0), torch.full_like(dx, 0.2))
    act[:, 2] = torch.clamp(1.5 * heading_err, -0.5, 0.5)
    return act


def run_oracle() -> dict:
    """让 oracle 跑满一个 episode，统计到达率与末端距离。"""
    os.environ["NAV_COMMAND_SMOOTHING"] = "1.0"
    env_cfg = parse_env_cfg(NAV_TASK, device="cuda:0",
                            num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(NAV_TASK, cfg=env_cfg)
    torch.manual_seed(args_cli.seed)
    env.reset()
    u = env.unwrapped

    cmd0 = u.command_manager.get_command("pose_command")
    d0 = torch.norm(cmd0[:, :2], dim=1).clone()
    # success_radius 决定"多近算到"，是判定 B 类故障的关键参数之一
    radius = float(u.command_manager.get_term("pose_command").cfg.success_radius)

    reached = torch.zeros(args_cli.num_envs, dtype=torch.bool, device="cuda:0")
    fell = torch.zeros(args_cli.num_envs, dtype=torch.bool, device="cuda:0")
    d_min = d0.clone()

    for _ in range(args_cli.steps):
        obs, _, terminated, truncated, _ = env.step(_oracle_action(u))
        d = torch.norm(u.command_manager.get_command("pose_command")[:, :2], dim=1)
        # 只在未结束的 env 上更新，避免重生后的新目标污染统计
        live = ~(reached | fell)
        d_min = torch.where(live, torch.minimum(d_min, d), d_min)
        reached |= live & (d < radius)
        fell |= live & terminated

    res = {
        "mode": "oracle",
        "success_radius": radius,
        "start_dist_mean": float(d0.mean().item()),
        "reached_rate": float(reached.float().mean().item()),
        "fell_rate": float(fell.float().mean().item()),
        "min_dist_mean": float(d_min.mean().item()),
        "min_dist_best": float(d_min.min().item()),
    }
    env.close()
    return res


def run_sweep() -> dict:
    """扫描恒定前进指令，测出低层策略的安全工作区间。

    由来：oracle（转向目标 + vx=1.0 前进）摔倒率 84%、到达率 0。
    这说明失败不在高层 RL，而在低层跟不住大速度指令。
    但"跟不住"要量化成一条可用的边界才有指导意义 ——
    我们需要知道：**最快能跑多少而不摔**，以及那个速度够不够在时限内走到目标。

    判据不是"摔没摔"，而是"在时限内能推进多远"：
    时限 = steps × step_dt，目标在 5~10 m 外，
    所以有效射程 = 安全速度 × 时限，必须 > 目标距离才可能完成任务。
    """
    os.environ["NAV_COMMAND_SMOOTHING"] = "1.0"
    env_cfg = parse_env_cfg(NAV_TASK, device="cuda:0",
                            num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(NAV_TASK, cfg=env_cfg)
    u = env.unwrapped
    dt = float(getattr(u, "step_dt", 0.2))
    horizon = args_cli.steps * dt

    rows = []
    for vx in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        torch.manual_seed(args_cli.seed)
        env.reset()
        act = torch.zeros(args_cli.num_envs, 3, device="cuda:0")
        act[:, 0] = vx

        start_xy = u.scene["robot"].data.root_pos_w[:, :2].clone()
        prev_xy = start_xy.clone()
        path = torch.zeros(args_cli.num_envs, device="cuda:0")
        alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device="cuda:0")
        alive_steps = 0
        first_fall = torch.full((args_cli.num_envs,), float(args_cli.steps),
                                device="cuda:0")

        for t in range(args_cli.steps):
            _, _, terminated, truncated, _ = env.step(act)
            cur = u.scene["robot"].data.root_pos_w[:, :2]
            valid = alive & ~(terminated | truncated)
            path += torch.where(valid, torch.norm(cur - prev_xy, dim=1),
                                torch.zeros_like(path))
            just_fell = alive & terminated
            first_fall = torch.where(just_fell,
                                     torch.full_like(first_fall, float(t)), first_fall)
            prev_xy = cur
            alive = valid
            alive_steps += int(valid.sum().item())

        fell = (first_fall < args_cli.steps)
        speed = float(path.sum().item()) / max(alive_steps, 1) / dt
        rows.append({
            "cmd_vx": vx,
            "actual_speed": speed,
            "fall_rate": float(fell.float().mean().item()),
            "survive_steps": float(first_fall.mean().item()),
            # 有效射程：按实际速度和"平均能活多久"折算，比单看速度更贴近任务
            "reach_in_horizon": speed * min(horizon,
                                            float(first_fall.mean().item()) * dt),
        })
        print(f"  vx={vx:<4} 实际速度 {speed:.3f} m/s   摔倒率 "
              f"{rows[-1]['fall_rate']*100:5.1f}%   存活 "
              f"{rows[-1]['survive_steps']:5.1f} 步   有效射程 "
              f"{rows[-1]['reach_in_horizon']:.2f} m", flush=True)

    env.close()
    return {"horizon_s": horizon, "step_dt": dt, "rows": rows}


def main() -> int:
    if args_cli.sweep:
        print("=" * 68, flush=True)
        print("实践 5 低层安全速度扫描", flush=True)
        print(f"  每档 {args_cli.steps} 步，并行环境 = {args_cli.num_envs}", flush=True)
        print("=" * 68, flush=True)
        r = run_sweep()
        out = Path(args_cli.out).with_name("speed_sweep.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print("\n" + "=" * 68, flush=True)
        best = max(r["rows"], key=lambda x: x["reach_in_horizon"])
        print(f"  时限 {r['horizon_s']:.0f} s   目标距离 5~10 m", flush=True)
        print(f"  最优档位 vx={best['cmd_vx']}：有效射程 "
              f"{best['reach_in_horizon']:.2f} m（摔倒率 "
              f"{best['fall_rate']*100:.1f}%）", flush=True)
        if best["reach_in_horizon"] >= 5.0:
            print("\n  ✅ 存在可用速度档能覆盖最近的目标 —— "
                  "把高层指令限幅收到该档即可让任务可解。", flush=True)
        else:
            print("\n  ❌ 任何速度档的有效射程都够不到 5 m —— "
                  "必须先把低层练好，或缩短目标距离课程。", flush=True)
        print(f"  已写入 {out}", flush=True)
        return 0

    if args_cli.oracle:
        print("=" * 68, flush=True)
        print("实践 5 性能上界探针：脚本化『朝目标走』的高层策略", flush=True)
        print(f"  步数 = {args_cli.steps}，并行环境 = {args_cli.num_envs}", flush=True)
        print("=" * 68, flush=True)
        r = run_oracle()
        print(f"\n  成功半径     {r['success_radius']:.2f} m", flush=True)
        print(f"  初始距离     {r['start_dist_mean']:.2f} m", flush=True)
        print(f"  到达率       {r['reached_rate']*100:.1f} %", flush=True)
        print(f"  摔倒率       {r['fell_rate']*100:.1f} %", flush=True)
        print(f"  最近距离     均值 {r['min_dist_mean']:.2f} m / "
              f"最好 {r['min_dist_best']:.2f} m", flush=True)
        print("\n" + "=" * 68, flush=True)
        if r["reached_rate"] > 0.3:
            print("  ✅ oracle 能到达 —— 任务可解，失败原因在 RL 探索/优化，", flush=True)
            print("     继续调 entropy_coef / init_noise_std / 目标距离课程。", flush=True)
        elif r["min_dist_mean"] < r["start_dist_mean"] * 0.5:
            print("  ⚠️ oracle 能明显靠近但进不了成功半径 —— 先查 success_radius "
                  "是否过小、时限是否够。", flush=True)
        else:
            print("  ❌ oracle 也走不过去 —— 任务侧有硬伤（低层跟踪、地形阻挡、"
                  "指令朝向定义），调 RL 超参无用。", flush=True)
        out = Path(args_cli.out).with_name("oracle_probe.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(f"  已写入 {out}", flush=True)
        return 0

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
