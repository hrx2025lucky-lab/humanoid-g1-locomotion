"""实践 5 §13：逐项对比低层观测在两个环境里的分布。

已知（§12）
----------
同一个低层策略：在自己的环境里 8 秒不摔，在导航环境里 1.86 秒必摔。
直接触发点是低层输出触顶 ±10（零指令 0%、vx=0.5 时 0.2% 触顶 → 每 env 摔 2.75 次）。

但"输出异常"只是症状。低层是个确定性函数，输出异常必然来自**输入异常**。
低层观测 480 维 = 5 帧 × 96 维，单帧构成：

    base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
    + joint_pos_rel(29) + joint_vel_rel(29) + last_action(29) = 96

其中只有两项被 HRL 重接线过：
  velocity_commands → 高层的 processed_actions
  last_action       → 本 action term 维护的 low_level_actions

要验证的假设
----------
**正反馈回路**：
    输出触顶 ±10 → last_action 里出现 ±10（训练时是 ±1 量级）
    → 观测分布外 → 输出更极端 → 再次触顶 …

若成立，`last_action` 这一项在导航环境里的量级会显著大于低层环境，
而其余四项应当接近。这是个可证伪的预测：
如果偏差平均分布在所有项上，那就是别的原因（如资产/物理差异）。

方法
----
两个环境各跑同样的恒定速度指令，逐项统计观测的 |值| 均值与最大值。
**必须逐项拆开看** —— 只看 480 维整体的范数会把 29 维的异常稀释掉。

用法
    python diagnose_p5_obs_gap.py --cmd 0.5 0 0 --steps 150
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                    metavar=("VX", "VY", "WZ"))
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--smoothing", type=float, default=1.0,
                    help="导航侧的 EMA 平滑（1.0=关闭，与 §12.2 扫描一致）")
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

LOW_TASK = "Unitree-G1-29dof-LowLevel"
NAV_TASK = "Unitree-G1-29dof-Navigation-HRL-Baseline"

# 被 HRL 重接线的两项 —— 归因时要和"原生项"分开看
REWIRED = ("velocity_commands", "last_action")


def build_slices(obs_mgr, group: str) -> list[tuple[str, int, int]]:
    """从 ObservationManager 自报的维度推导每一项的切片。

    ★ 不能硬编码。实测布局是**按项拼接**而非按帧：
        base_ang_vel      (15,)  = 3  × 5 帧
        projected_gravity (15,)  = 3  × 5
        velocity_commands (15,)  = 3  × 5
        joint_pos_rel     (145,) = 29 × 5
        joint_vel_rel     (145,) = 29 × 5
        last_action       (145,) = 29 × 5   合计 480
    我最初按"5 帧 × 96 维"切，切出来的每一段都横跨多个物理量，
    统计出的"某项异常"完全是假的。用管理器自报的维度就不会错。
    """
    names = obs_mgr.active_terms[group]
    dims = obs_mgr.group_obs_term_dim[group]
    out, off = [], 0
    for name, d in zip(names, dims):
        n = int(d[0]) if hasattr(d, "__len__") else int(d)
        out.append((name, off, off + n))
        off += n
    return out


def per_term_stats(obs_flat: torch.Tensor,
                   slices: list[tuple[str, int, int]]) -> dict:
    """逐项统计 |值| 的均值与最大值。

    每一项内部是该项的 5 帧历史，一并统计即可 ——
    我们要判断的是"这一项的量级是否异常"，不需要区分是哪一帧异常。
    """
    out = {}
    for name, lo, hi in slices:
        seg = obs_flat[:, lo:hi]
        out[name] = (float(seg.abs().mean()), float(seg.abs().max()))
    return out


def run_low_level() -> dict:
    env_cfg = parse_env_cfg(LOW_TASK, device="cuda:0",
                            num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(LOW_TASK, cfg=env_cfg)
    u = env.unwrapped
    env.reset()   # 必须用包装后的 env，unwrapped.reset() 不会解除 gymnasium 的 ResetNeeded

    import os
    from isaaclab.utils.assets import read_file
    policy_path = os.environ.get(
        "UNITREE_G1_LOW_LEVEL_POLICY_PATH",
        "/home/limx/workspace/Roxan_warmup/repos/unitree_rl_lab/logs/rsl_rl"
        "/unitree_g1_29dof_velocity/2026-08-31_10-49-33/exported/policy.pt")
    policy = torch.jit.load(read_file(policy_path)).to(u.device).eval()

    cmd_term = u.command_manager.get_term("base_velocity")
    cmd_vec = torch.tensor(args_cli.cmd, device=u.device, dtype=torch.float32)

    slices = build_slices(u.observation_manager, "policy")
    acc, n_term = {k: [0.0, 0.0] for k, _, _ in slices}, 0
    out_abs, out_max = [], []
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            cmd_term.vel_command_b[:] = cmd_vec
            obs = u.observation_manager.compute_group("policy")
            action = policy(obs)
            out_abs.append(float(action.abs().mean()))
            out_max.append(float(action.abs().max()))
            st = per_term_stats(obs, slices)
            for k in acc:
                acc[k][0] += st[k][0]
                acc[k][1] = max(acc[k][1], st[k][1])
            _, _, terminated, _, _ = env.step(action)
            n_term += int(terminated.sum().item())

    res = {k: (v[0] / args_cli.steps, v[1]) for k, v in acc.items()}
    res["_output"] = (sum(out_abs) / len(out_abs), max(out_max))
    res["_terminations"] = (n_term, n_term / args_cli.num_envs)
    env.close()
    return res


def run_nav() -> dict:
    import os
    os.environ["NAV_COMMAND_SMOOTHING"] = str(args_cli.smoothing)
    env_cfg = parse_env_cfg(NAV_TASK, device="cuda:0",
                            num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.actions.pre_trained_policy_action.command_smoothing = args_cli.smoothing
    env = gym.make(NAV_TASK, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    term = u.action_manager.get_term("pre_trained_policy_action")

    action = torch.zeros(u.num_envs, 3, device=u.device)
    action[:, 0], action[:, 1], action[:, 2] = args_cli.cmd

    slices = build_slices(term._low_level_obs_manager, "ll_policy")
    acc, n_term = {k: [0.0, 0.0] for k, _, _ in slices}, 0
    out_abs, out_max = [], []
    with torch.no_grad():
        for _ in range(args_cli.steps):
            _, _, terminated, _, _ = env.step(action)
            n_term += int(terminated.sum().item())
            # 低层的观测由 action term 自己的 ObservationManager 维护
            obs = term._low_level_obs_manager.compute_group("ll_policy")
            st = per_term_stats(obs, slices)
            for k in acc:
                acc[k][0] += st[k][0]
                acc[k][1] = max(acc[k][1], st[k][1])
            ll = term.low_level_actions
            out_abs.append(float(ll.abs().mean()))
            out_max.append(float(ll.abs().max()))

    res = {k: (v[0] / args_cli.steps, v[1]) for k, v in acc.items()}
    res["_output"] = (sum(out_abs) / len(out_abs), max(out_max))
    res["_terminations"] = (n_term, n_term / args_cli.num_envs)
    env.close()
    return res


def main() -> int:
    print("=" * 74, flush=True)
    print(f"实践 5 §13：低层观测逐项对比   指令 {tuple(args_cli.cmd)}", flush=True)
    print("=" * 74, flush=True)

    print("\n[1/2] 低层自己的环境 …", flush=True)
    low = run_low_level()
    print("[2/2] 导航环境 …", flush=True)
    nav = run_nav()

    print("\n" + "=" * 74)
    print(f"  {'观测项':<20}{'低层环境':>18}{'导航环境':>18}{'倍数':>10}")
    print(f"  {'':20}{'均值 / 最大':>18}{'均值 / 最大':>18}")
    print("  " + "-" * 68)

    ratios = {}
    names = [k for k in low if not k.startswith("_")]
    for name in names:
        lm, lx = low[name]
        nm, nx = nav[name]
        r = nm / lm if lm > 1e-9 else float("inf")
        ratios[name] = r
        mark = "  ★" if name in ("velocity_commands", "last_action") else ""
        print(f"  {name:<20}{lm:>8.3f} /{lx:>7.2f}{nm:>9.3f} /{nx:>7.2f}"
              f"{r:>9.2f}x{mark}")

    print("  " + "-" * 68)
    lm, lx = low["_output"]
    nm, nx = nav["_output"]
    print(f"  {'低层输出 |a|':<20}{lm:>8.3f} /{lx:>7.2f}{nm:>9.3f} /{nx:>7.2f}"
          f"{(nm/lm if lm>1e-9 else 0):>9.2f}x")
    print(f"  {'非超时终止':<20}{low['_terminations'][0]:>8d} 次"
          f"{nav['_terminations'][0]:>13d} 次"
          f"   （每 env {low['_terminations'][1]:.2f} vs "
          f"{nav['_terminations'][1]:.2f}）")

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    rewired = {k: ratios[k] for k in ("velocity_commands", "last_action")}
    native = {k: v for k, v in ratios.items() if k not in rewired}
    worst_rw = max(rewired, key=lambda k: rewired[k])
    worst_nv = max(native, key=lambda k: native[k])
    print(f"  重接线项最大偏差：{worst_rw} {rewired[worst_rw]:.2f}x")
    print(f"  原生项最大偏差：  {worst_nv} {native[worst_nv]:.2f}x")

    if rewired[worst_rw] > 3 * native[worst_nv]:
        print(f"\n  ✅ 偏差集中在重接线项 `{worst_rw}` 上 —— 支持"
              "「HRL 接线导致观测分布外」的假设。")
        if worst_rw == "last_action":
            print("     且是 last_action，说明存在正反馈：")
            print("     输出触顶 → last_action 异常 → 观测分布外 → 输出更极端")
    elif native[worst_nv] > 3 * rewired[worst_rw]:
        print(f"\n  ❌ 偏差主要在原生项 `{worst_nv}` 上 —— 不是接线问题，"
              "应查资产/物理/初始状态差异。")
    else:
        print("\n  ⚠️ 各项偏差量级相当，无法归因到单一项，需要换角度。")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
