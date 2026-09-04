"""实践 5 §13.6：摔倒瞬间，机器人离最近的障碍有多远？

为什么用这个方法
--------------
想回答"机器人是撞障碍摔的，还是自己走不稳"。
直觉做法是关掉障碍做对照，但我试了两次都翻车（见文档 §13.5）：
一次改错常量，一次读到崩溃后留下的陈旧文件。

换个思路：**不改任何配置，直接测摔倒那一刻到最近障碍的距离。**

判据很硬：
  近（< 半径 + 0.5 m）  → 撞上了，障碍是主因
  远（> 1.5 m）        → 摔的时候周围没东西，障碍无关

比开关对照更好的三点：
  1. 不动配置，没有"开关没生效"的风险
  2. 直接测因果现场，而不是比较两组统计量
  3. 顺便给出分布，能看出是"少数撞上"还是"普遍自己摔"

用法
    python diagnose_p5_fall_distance.py --cmd 0.5 0 0 --steps 200
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                    metavar=("VX", "VY", "WZ"))
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--smoothing", type=float, default=1.0)
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


def nearest_obstacle_dist(layout, robot_xy: torch.Tensor,
                          env_origins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """每个 env 里机器人到最近**已激活**障碍的表面距离。

    返回 (中心距 − 足迹半径, 该 env 的激活障碍数)。
    减去 footprint_radius 是因为我们关心的是"碰没碰到"，不是"离中心多远"。

    centers_xy 是**环境局部坐标**，而 root_pos_w 是世界坐标，
    必须先减掉 env_origins 才能比 —— 多环境并行时每个 env 有自己的原点偏移，
    忘了减会让所有距离都偏大好几米，直接得出"从没撞过"的假结论。
    """
    local_xy = robot_xy - env_origins[:, :2]
    centers = layout.centers_xy                      # [E, M, 2]
    active = layout.active_mask                      # [E, M]
    radius = layout.footprint_radius                 # [M] 或 [E, M]

    d = torch.norm(centers - local_xy.unsqueeze(1), dim=-1)   # [E, M]
    if radius.ndim == 1:
        radius = radius.unsqueeze(0).expand_as(d)
    surf = d - radius
    # 未激活的槽位填 +inf，不参与取最小
    surf = torch.where(active, surf, torch.full_like(surf, float("inf")))
    return surf.min(dim=1).values, active.sum(dim=1)


def main() -> int:
    os.environ["NAV_COMMAND_SMOOTHING"] = str(args_cli.smoothing)
    cfg = parse_env_cfg(NAV_TASK, device="cuda:0",
                        num_envs=args_cli.num_envs, use_fabric=True)
    cfg.actions.pre_trained_policy_action.command_smoothing = args_cli.smoothing
    env = gym.make(NAV_TASK, cfg=cfg)
    env.reset()
    u = env.unwrapped

    from unitree_rl_lab.tasks.navigation.mdp.obstacles import get_obstacle_layout
    layout = get_obstacle_layout(u)
    if layout is None:
        print("❌ 取不到障碍布局，无法测距")
        env.close()
        return 1

    action = torch.zeros(u.num_envs, 3, device=u.device)
    action[:, 0], action[:, 1], action[:, 2] = args_cli.cmd

    robot = u.scene["robot"]
    origins = u.scene.env_origins

    fall_dists: list[float] = []
    fall_steps: list[int] = []
    n_active_at_fall: list[int] = []
    dist_all: list[float] = []

    print(f"跑 {args_cli.steps} 步，指令 {tuple(args_cli.cmd)} …", flush=True)
    with torch.no_grad():
        for t in range(args_cli.steps):
            _, _, terminated, _, _ = env.step(action)
            xy = robot.data.root_pos_w[:, :2]
            d, n_act = nearest_obstacle_dist(layout, xy, origins)
            finite = d[torch.isfinite(d)]
            if finite.numel():
                dist_all.append(float(finite.mean()))
            # ★ 必须在 step 之后立刻取：IsaacLab 会在下一次 step 开头重置，
            # 晚一步取到的就是重生后的位置，距离完全无意义
            fell = terminated.nonzero(as_tuple=False).flatten()
            for i in fell.tolist():
                if torch.isfinite(d[i]):
                    fall_dists.append(float(d[i]))
                    fall_steps.append(t)
                    n_active_at_fall.append(int(n_act[i]))

    env.close()

    print("\n" + "=" * 64)
    print("摔倒瞬间到最近障碍的距离")
    print("=" * 64)
    if not fall_dists:
        print("  本次没有摔倒事件 —— 换更长步数或更大指令再试")
        return 0

    dt = torch.tensor(fall_dists)
    print(f"  摔倒次数        {len(fall_dists)}（{args_cli.num_envs} envs）")
    print(f"  激活障碍数      {sum(n_active_at_fall)/len(n_active_at_fall):.0f} 个/env")
    print(f"  平均首次摔倒步  {sum(fall_steps)/len(fall_steps):.1f}")
    if dist_all:
        print(f"  全程平均最近距离 {sum(dist_all)/len(dist_all):.2f} m")
    print()
    print(f"  距离  最小 {dt.min():.2f} m   中位 {dt.median():.2f} m   "
          f"最大 {dt.max():.2f} m")
    for th in (0.0, 0.3, 0.5, 1.0, 1.5):
        frac = float((dt < th).float().mean())
        print(f"    < {th:.1f} m : {frac*100:5.1f}%")

    print("\n" + "=" * 64)
    near = float((dt < 0.5).float().mean())
    far = float((dt > 1.5).float().mean())
    if near > 0.5:
        print(f"  ✅ {near*100:.0f}% 的摔倒发生在障碍 0.5 m 内 —— **撞障碍是主因**")
        print("     对策：给终止项加碰撞判据，或降低障碍密度做课程")
    elif far > 0.5:
        print(f"  ❌ {far*100:.0f}% 的摔倒发生在离障碍 1.5 m 以外 —— "
              "**障碍无关，机器人是自己走不稳**")
        print("     对策：回到低层与导航环境的物理/初始状态差异上查")
    else:
        print("  ⚠️ 距离分布分散，两种成因都有，需要按距离分组再看")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
