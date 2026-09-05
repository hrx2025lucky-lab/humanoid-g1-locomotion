"""实践 5 §14：验证低层观测历史是否被冻结。

假设
----
`ObservationManager.compute_group()` 的 `update_history` **默认为 False**。
IsaacLab 正常环境走的是 `compute(update_history=True)`（manager_based_rl_env.py:238），
但导航侧的低层 ObservationManager 调用的是：

    low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

没有传 `update_history=True`，所以低层观测里的 **5 帧历史从未被更新**。

若假设成立，低层策略看到的是"同一帧重复 5 次"，而它训练时看到的是真实时序。
这属于彻底的分布外输入，但**量级完全正常** ——
所以前面 11 项基于均值/最大值的对比全都发现不了它。

判据（可证伪）
------------
低层观测按项拼接，每项含 5 帧：

    base_ang_vel  (15,) = 3 × 5 帧
    joint_pos_rel (145,) = 29 × 5 帧
    ...

把某一项拆成 [5, dim]，若 5 帧**逐位相同**，历史就是冻结的。
反之若帧间有差异，假设不成立。

对照：低层自己的环境走正常路径，它的 5 帧必然有差异。
两边一起测才能排除"这个动作本身就没有帧间变化"的可能。

用法
    python verify_p5_history_frozen.py --cmd 0.5 0 0 --steps 30
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--num-envs", type=int, default=4)
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
LOW_TASK = "Unitree-G1-29dof-LowLevel"
HISTORY = 5


def frame_spread(obs: torch.Tensor, mgr, group: str) -> dict[str, float]:
    """逐项把 [B, dim*5] 拆成 [B, 5, dim]，返回帧间标准差。

    帧间标准差 ≈ 0 就是历史被冻结（5 帧完全相同）。
    用标准差而不是"是否逐位相等"，是因为浮点计算可能有极小抖动，
    而真实的行走历史会有量级明显的差异，两者不会混淆。
    """
    names = mgr.active_terms[group]
    dims = mgr.group_obs_term_dim[group]
    out, off = {}, 0
    for name, d in zip(names, dims):
        n = int(d[0]) if hasattr(d, "__len__") else int(d)
        seg = obs[:, off:off + n]
        off += n
        if n % HISTORY:
            continue
        per = n // HISTORY
        # 按项拼接时布局是 [帧0的dim维, 帧1的dim维, ...]
        frames = seg.reshape(seg.shape[0], HISTORY, per)
        out[name] = float(frames.std(dim=1).mean())
    return out


def run_nav() -> dict:
    os.environ["NAV_COMMAND_SMOOTHING"] = "1.0"
    cfg = parse_env_cfg(NAV_TASK, device="cuda:0",
                        num_envs=args_cli.num_envs, use_fabric=True)
    cfg.actions.pre_trained_policy_action.command_smoothing = 1.0
    env = gym.make(NAV_TASK, cfg=cfg)
    env.reset()
    u = env.unwrapped
    term = u.action_manager.get_term("pre_trained_policy_action")

    action = torch.zeros(u.num_envs, 3, device=u.device)
    action[:, 0], action[:, 1], action[:, 2] = args_cli.cmd

    acc = None
    with torch.no_grad():
        for _ in range(args_cli.steps):
            env.step(action)
            obs = term._low_level_obs_manager.compute_group("ll_policy")
            st = frame_spread(obs, term._low_level_obs_manager, "ll_policy")
            acc = st if acc is None else {k: acc[k] + v for k, v in st.items()}
    env.close()
    return {k: v / args_cli.steps for k, v in acc.items()}


def run_low() -> dict:
    cfg = parse_env_cfg(LOW_TASK, device="cuda:0",
                        num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(LOW_TASK, cfg=cfg)
    env.reset()
    u = env.unwrapped

    from isaaclab.utils.assets import read_file
    path = os.environ.get(
        "UNITREE_G1_LOW_LEVEL_POLICY_PATH",
        "/home/limx/workspace/Roxan_warmup/repos/unitree_rl_lab/logs/rsl_rl"
        "/unitree_g1_29dof_velocity/2026-08-31_10-49-33/exported/policy.pt")
    policy = torch.jit.load(read_file(path)).to(u.device).eval()

    cmd_term = u.command_manager.get_term("base_velocity")
    cmd_vec = torch.tensor(args_cli.cmd, device=u.device, dtype=torch.float32)

    acc = None
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            cmd_term.vel_command_b[:] = cmd_vec
            # 用 env 缓存的观测：它由环境按正常路径（update_history=True）维护
            obs = u.obs_buf["policy"] if isinstance(u.obs_buf, dict) else u.obs_buf
            st = frame_spread(obs, u.observation_manager, "policy")
            acc = st if acc is None else {k: acc[k] + v for k, v in st.items()}
            env.step(policy(obs))
    env.close()
    return {k: v / args_cli.steps for k, v in acc.items()}


def main() -> int:
    print("=" * 68, flush=True)
    print(f"实践 5 §14：低层观测历史是否被冻结   指令 {tuple(args_cli.cmd)}")
    print("=" * 68, flush=True)

    print("\n[1/2] 低层自己的环境（正常路径，对照组）…", flush=True)
    low = run_low()
    print("[2/2] 导航环境（HRL 路径）…", flush=True)
    nav = run_nav()

    print("\n" + "=" * 68)
    print(f"  {'观测项':<22}{'低层环境':>14}{'导航环境':>14}{'判定':>12}")
    print(f"  {'':22}{'帧间标准差':>14}{'帧间标准差':>14}")
    print("  " + "-" * 62)
    frozen = []
    for name in low:
        if name not in nav:
            continue
        lo, nv = low[name], nav[name]
        is_frozen = nv < 1e-6 and lo > 1e-6
        if is_frozen:
            frozen.append(name)
        print(f"  {name:<22}{lo:>14.6f}{nv:>14.6f}"
              f"{'  ❄️ 冻结' if is_frozen else '':>12}")

    print("\n" + "=" * 68)
    if frozen:
        print(f"  ✅ 假设成立：{len(frozen)} 项的历史在导航环境里完全没有变化")
        print(f"     {', '.join(frozen)}")
        print("\n  根因：pre_trained_policy_action.py 调用")
        print("        compute_group('ll_policy') 时没传 update_history=True，")
        print("        而该参数默认为 False。IsaacLab 正常环境走的是")
        print("        observation_manager.compute(update_history=True)。")
        print("\n  低层策略训练时看到真实时序，现在看到的是同一帧重复 5 次 ——")
        print("  量级完全正常，所以前面 11 项基于均值/最大值的对比全都发现不了。")
    else:
        print("  ❌ 假设不成立：导航环境的历史确实在更新，另找原因。")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
