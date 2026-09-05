"""实践 5 底层策略诊断 —— 隔离"机器人/环境"与"HRL 接线"两类故障。

背景：实践 5 的两组训练都在 ~20 iter 内发散（std 变 NaN）。
根因指标是 `Episode_Termination/bad_orientation ≈ 1.0`、
`Mean episode length ≈ 13`（完整 episode 是 1000 步）—— 机器人一出生就摔。

摔倒可能来自两处，必须先分开：
  A. 机器人/环境本身有问题（例如换成 URDF 后与预训练策略不匹配）
  B. HRL 接线有问题（高层指令超界、观测重接线错误等）

本脚本只跑**底层任务**、只喂**零速度指令**，把 B 完全排除。
零指令下机器人应当原地站稳；若它照样摔，问题就在 A。

用法：
    python diagnose_p5_lowlevel.py                    # 用 hw5 自带预训练策略
    python diagnose_p5_lowlevel.py --policy /path/to/policy.pt
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", default=None, help="底层 TorchScript 策略路径")
parser.add_argument("--steps", type=int, default=200, help="仿真步数")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--nav", action="store_true",
                    help="改跑导航环境 + 固定高层指令的接线验证")
parser.add_argument("--nan-hunt", action="store_true",
                    help="随机高层指令下检测观测/奖励的 NaN 与 inf，并统计摔倒率")
parser.add_argument("--smoothing", type=float, default=1.0,
                    help="高层指令 EMA 平滑系数（1.0 = 关闭）")
parser.add_argument("--resample-every", type=int, default=0,
                    help="每 N 步重采样一次速度指令（0=恒定）。"
                         "低层训练时指令每 10 秒才换一次，而 HRL 里高层每 0.2 秒"
                         "就下发新指令 —— 用这个开关在低层自己的环境里复现那种"
                         "高变化率，判断它是不是导航环境摔倒的病因。")
parser.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("VX", "VY", "WZ"),
                    help="底层任务下发的恒定速度指令。默认全零（测站立）；"
                         "给非零值可测『这个策略到底会不会走』")
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import copy  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Unitree-G1-29dof-LowLevel"
NAV_TASK = "Unitree-G1-29dof-Navigation-HRL-Baseline"
DEFAULT_POLICY = (
    "/home/limx/workspace/Roxan_warmup/shenlan_hw/hw5_navigation/unitree_rl_lab"
    "/pretrained/g1_29dof_lowlevel/policy.pt"
)


def run_nav_zero_command() -> int:
    """在**导航环境**里下发指定的高层动作，检查两件事。

    ① 零指令能否站住 —— 与底层任务对照，排查导航环境侧（障碍物/出生点/地形）。
    ② 恒定非零指令能否走出去 —— 这一条才是关键。

    ★ 只测零指令是不够的：接线正常传了 0，与接线断了永远读到 0，
      表现完全相同（都站着不动）。而"闭包捕获旧张量导致低层永远读到零指令"
      正是本实践文档点名的陷阱。因此必须下发非零指令并测净位移：
      走得动 → 接线通；原地不动 → 接线断。
    """
    print("\n" + "═" * 66)
    print("对照实验：导航环境 + 固定高层指令")
    print("═" * 66)

    env_cfg = parse_env_cfg(NAV_TASK, num_envs=args_cli.num_envs)
    # 与 --nan-hunt 模式保持一致：不显式设置的话会静默沿用 cfg 默认的
    # command_smoothing=0.1（EMA 开启），于是"低层读到的指令"永远不等于
    # 下发值，看起来像接线 bug 实则是 EMA 未收敛。此前就险些据此得出错误结论。
    env_cfg.actions.pre_trained_policy_action.command_smoothing = args_cli.smoothing
    env = gym.make(NAV_TASK, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("pre_trained_policy_action")

    results = {}
    # 速度阶梯：零指令在导航环境里 0 次摔倒，vx=0.5 每 env 摔 2.88 次。
    # 静态差异（物理/材质/初始状态）已全部排除，剩下唯一变量就是"走起来"。
    # 逐档加速能给出摔倒的**速度阈值** —— 阈值本身就是线索：
    #   阈值很低（0.1 就摔）→ 一迈步就失衡，问题在步态启动
    #   阈值居中（0.3 附近） → 与低层的实际能力边界有关
    #   没有阈值（都不摔）  → 之前的 vx=0.5 结论有问题，需重测
    for label, cmd in (
        ("零指令", (0.0, 0.0, 0.0)),
        ("vx=0.1", (0.1, 0.0, 0.0)),
        ("vx=0.2", (0.2, 0.0, 0.0)),
        ("vx=0.3", (0.3, 0.0, 0.0)),
        ("前进 vx=0.5", (0.5, 0.0, 0.0)),
        # 三轴复合：策略实际输出的是三个分量都非零的指令。
        # 只测纯前进会高估低层的鲁棒性 —— 侧移与转向叠加时难得多。
        ("复合 (0.4,0.3,0.3)", (0.4, 0.3, 0.3)),
    ):
        env.reset()
        action = torch.zeros(env.num_envs, 3, device=env.device)
        action[:, 0], action[:, 1], action[:, 2] = cmd

        # 用机体系前向速度而不是净位移：episode 会在中途因终止而重置，
        # 机器人被传送回出生点，跨重置累计的位移毫无意义
        #（实测零指令也能"走出" 26 m，全是重置跳变）。
        # 机体系速度是瞬时量，不受重置影响，且与指令同一坐标系可直接比。
        gz_hist, vx_hist, proc_hist = [], [], []
        # 低层输出的量级：这是区分"输入没接对"与"输出被玩坏了"的关键。
        # 正常关节动作在 ±1 量级（×action_scale=0.25 → ±0.25 rad）。
        # 若接近 _LOW_LEVEL_ACTION_LIMIT=10，说明低层已经在分布外乱吐 ——
        # 限幅只是拦住了数值爆炸，机器人照样会被 2.5 rad 的目标角扭到摔倒。
        ll_abs_hist, ll_max_hist, clip_frac_hist = [], [], []
        # ★ 必须统计终止次数，不能只看最后一帧的姿态。
        # IsaacLab 在 episode 终止时会自动重置并把机器人放回直立初始姿态，
        # 所以"最后一帧投影重力 = -1.0"在自动重置的环境里恒成立，
        # 哪怕机器人一路上摔了几十次。本脚本早期版本就是被这一点骗过，
        # 给出过"接线验证通过"的错误结论（实际摔倒率 92%）。
        n_term = 0
        limit = 10.0
        try:
            # 限幅常量的唯一真源在 action term 所在模块，别在这里复制一份数值
            from unitree_rl_lab.tasks.navigation.mdp import pre_trained_policy_action as _pt
            limit = float(getattr(_pt, "_LOW_LEVEL_ACTION_LIMIT", limit))
        except Exception:
            pass
        with torch.no_grad():
            for _ in range(args_cli.steps):
                _, _, terminated, _, _ = env.step(action)
                n_term += int(terminated.sum().item())
                gz_hist.append(robot.data.projected_gravity_b[:, 2].mean().item())
                vx_hist.append(robot.data.root_lin_vel_b[:, 0].mean().item())
                proc_hist.append(action_term.processed_actions[:, 0].mean().item())
                ll = getattr(action_term, "low_level_actions", None)
                if ll is not None:
                    ll_abs_hist.append(ll.abs().mean().item())
                    ll_max_hist.append(ll.abs().max().item())
                    clip_frac_hist.append((ll.abs() >= limit * 0.99).float().mean().item())

        # 取后半段均值，跳过起步瞬态
        half = len(vx_hist) // 2
        vx_mean = sum(vx_hist[half:]) / max(len(vx_hist) - half, 1)
        results[label] = (gz_hist[-1], vx_mean, proc_hist[-1])
        print(f"\n  ── {label} ──")
        print(f"     投影重力 z 最终   {gz_hist[-1]:+.3f}   （直立 = -1.0）")
        print(f"     机体系实际 vx     {vx_mean:+.3f} m/s   （指令 {cmd[0]}）")
        print(f"     低层读到的 vx     {proc_hist[-1]:+.3f}   （应等于指令）")
        if ll_abs_hist:
            print(f"     低层输出 |a| 均值 {sum(ll_abs_hist)/len(ll_abs_hist):.3f}"
                  f"   （正常 ≈0.3~1.0）")
            print(f"     低层输出 |a| 峰值 {max(ll_max_hist):.3f}"
                  f"   （限幅 {limit}）")
            print(f"     触顶比例          "
                  f"{max(clip_frac_hist)*100:.1f}%   （>0 即已在分布外）")
        # 每 env 平均终止次数才是"摔没摔"的真判据
        per_env = n_term / max(env.num_envs, 1)
        print(f"     非超时终止次数    {n_term}   （每 env {per_env:.2f} 次）"
              f"{'  ← 在摔' if per_env > 0.1 else ''}")

    print("\n" + "═" * 66)
    gz0, vx0, _ = results["零指令"]
    gz1, vx1, proc1 = results["前进 vx=0.5"]

    if gz0 > -0.7:
        print("❌ 导航环境里零指令就摔 → 问题在导航环境侧（障碍物/出生点/地形）。")
        code = 1
    elif abs(proc1 - 0.5) > 1e-3:
        print(f"❌ 低层读到的 vx 是 {proc1:+.3f} 而非 0.5 → 高层→低层接线断了")
        print("   （典型原因：闭包捕获了旧张量，低层永远读到初始零指令）。")
        code = 1
    elif vx1 < 0.2:
        print(f"❌ 低层读到了 vx=0.5，但机器人实际只有 {vx1:+.3f} m/s → 指令没驱动运动。")
        code = 1
    else:
        print(f"✅ 零指令站得稳（{gz0:+.3f}），vx=0.5 实际走出 {vx1:+.3f} m/s")
        print("   → 环境与两层接线都正常，摔倒来自高层输出的**指令变化率**：")
        print("     高层每 0.2 s 重新采样一次随机指令，而低层是在")
        print("     指令数十秒才换一次的分布上训练的。")
        code = 0
    print("═" * 66 + "\n")
    env.close()
    return code


def main() -> int:
    policy_path = args_cli.policy or os.environ.get(
        "UNITREE_G1_LOW_LEVEL_POLICY_PATH", DEFAULT_POLICY
    )
    print("\n" + "═" * 66)
    print("实践 5 底层策略诊断（零指令站立测试）")
    print("═" * 66)
    print(f"  任务  : {TASK}")
    print(f"  策略  : {policy_path}")

    env_cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]

    print(f"\n── 机器人资产 ──")
    print(f"  关节数: {len(robot.joint_names)}")
    print(f"  asset 顺序前 8: {robot.joint_names[:8]}")
    print(f"  默认关节角前 6: {robot.data.default_joint_pos[0, :6].tolist()}")

    # 底层观测按训练配置 deepcopy 构造，与 HRL 里的做法一致
    from isaaclab.managers import ObservationManager

    obs_cfg = copy.deepcopy(env_cfg.observations.policy)
    obs_cfg.enable_corruption = False
    for term in ("base_ang_vel", "projected_gravity", "joint_pos_rel", "joint_vel_rel"):
        if hasattr(obs_cfg, term):
            getattr(obs_cfg, term).noise = None

    policy = torch.jit.load(policy_path, map_location=env.device)
    policy.eval()

    env.reset()

    # 恒定速度指令：每步重写，防止 CommandManager 到点重采样把它覆盖掉。
    # 默认全零测站立；给非零值就变成"这个低层到底会不会走"的直接测试 ——
    # 这正是隔离"策略不会走"与"导航环境把策略搞坏了"的关键对照。
    cmd_term = env.command_manager.get_term("base_velocity")
    cmd_vec = torch.tensor(args_cli.cmd, device=env.device, dtype=torch.float32)
    is_zero_cmd = bool(cmd_vec.abs().sum() < 1e-6)

    gz_hist, height_hist = [], []
    start_xy = robot.data.root_pos_w[:, :2].clone()
    n_term_low = 0
    with torch.inference_mode():
        for step in range(args_cli.steps):
            if args_cli.resample_every > 0 and step % args_cli.resample_every == 0:
                # 在 velocity_clip 同量级的范围内均匀重采样，模拟高层输出
                cmd_vec = torch.stack([
                    torch.empty(1, device=env.device).uniform_(-0.5, 1.0)[0],
                    torch.empty(1, device=env.device).uniform_(-0.5, 0.5)[0],
                    torch.empty(1, device=env.device).uniform_(-0.5, 0.5)[0],
                ])
            cmd_term.vel_command_b[:] = cmd_vec
            obs = env.observation_manager.compute_group("policy")
            action = policy(obs)
            _, _, terminated, _, _ = env.step(action)
            n_term_low += int(terminated.sum().item())

            gz = robot.data.projected_gravity_b[:, 2]
            gz_hist.append(gz.mean().item())
            height_hist.append(robot.data.root_pos_w[:, 2].mean().item())

    travelled = float(torch.norm(
        robot.data.root_pos_w[:, :2] - start_xy, dim=1).mean().item())

    label = ("零指令" if is_zero_cmd else f"指令 {tuple(args_cli.cmd)}")
    if args_cli.resample_every > 0:
        label = f"每 {args_cli.resample_every} 步重采样的随机指令"
    print(f"\n── {label}下的姿态演化（{args_cli.steps} 步）──")
    print(f"  {'步':>6}{'投影重力 z':>14}{'根节点高度 m':>14}")
    for i in (0, 10, 25, 50, 100, args_cli.steps - 1):
        if i < len(gz_hist):
            print(f"  {i:>6}{gz_hist[i]:>14.3f}{height_hist[i]:>14.3f}")

    final_gz = gz_hist[-1]
    print(f"\n  投影重力 z 最终 {final_gz:+.3f}   （直立 = -1.0，> -0.7 判为摔倒）")
    print(f"  根节点高度 最终 {height_hist[-1]:.3f} m   （站立约 0.78 m）")
    # ★ 判据用"过程中摔了几次"，不是"最后一帧姿态" —— 自动重置会把最终姿态复位
    print(f"  非超时终止 {n_term_low} 次   （每 env {n_term_low/env.num_envs:.2f} 次）"
          f"{'  ← 在摔' if n_term_low/env.num_envs > 0.1 else ''}")
    if not is_zero_cmd:
        dt = float(getattr(env, "step_dt", 0.02))
        want = (args_cli.cmd[0] ** 2 + args_cli.cmd[1] ** 2) ** 0.5
        got = travelled / max(args_cli.steps * dt, 1e-6)
        print(f"  实际位移   {travelled:.2f} m   平均速度 {got:.3f} m/s"
              f"（指令 {want:.3f} m/s）")

    print("\n" + "═" * 66)
    if final_gz >= -0.7:
        print(f"❌ {label}下机器人摔了 → 低层策略本身撑不住这个指令。")
        print("   若零指令能站住而带速指令会摔，说明策略只会站不会走。")
        verdict = 1
    elif is_zero_cmd:
        print("✅ 底层策略能让机器人站住 → 机器人/环境没问题，")
        print("   实践 5 的摔倒来自 HRL 接线（高层指令或观测重接线）。")
        verdict = 0
    else:
        print(f"✅ 低层能在{label}下稳定行走 → 策略没问题，")
        print("   导航环境里的摔倒来自 HRL 侧（指令分布、地形或观测重接线）。")
        verdict = 0
    print("═" * 66 + "\n")

    env.close()
    return verdict


def run_nan_hunt() -> int:
    """用随机高层指令跑，直接检测观测/奖励里的 NaN 与 inf。

    实践 5 两组训练都在 ~20 iter 内以 `normal expects all elements of std >= 0.0`
    发散 —— 策略分布的 std 变成了 NaN。机器人摔倒只会让回报变差，
    不会产生 NaN；NaN 一定来自数值污染。

    高层观测里的 height_scan_pooled 有 clip=(-1.5,1.5)，能挡住 inf，
    但 **torch.clip(nan) 仍然是 nan** —— 这正是实践 2 踩过的同一类坑
    （那次是 ray_hits_w 的 inf 污染了 base_height_l2 的均值）。
    """
    print("\n" + "═" * 66)
    print("NaN 溯源：随机高层指令下检测观测与奖励")
    print("═" * 66)

    env_cfg = parse_env_cfg(NAV_TASK, num_envs=args_cli.num_envs)
    if args_cli.smoothing < 1.0:
        env_cfg.actions.pre_trained_policy_action.command_smoothing = args_cli.smoothing
        print(f"  指令 EMA 平滑: α = {args_cli.smoothing}")
    else:
        print("  指令 EMA 平滑: 关闭（原行为）")
    env = gym.make(NAV_TASK, cfg=env_cfg).unwrapped
    env.reset()

    bad = {"obs_nan": 0, "obs_inf": 0, "rew_nan": 0, "rew_inf": 0}
    first_hit = None
    scan_extremes = []
    fall_events = 0
    total_terms = 0
    obs_absmax = (0.0, -1, -1)
    obs_p99 = []
    om = env.observation_manager
    term_names = list(om.active_terms['policy'])
    term_dims = [int(d[0]) if hasattr(d, '__len__') else int(d)
                 for d in om.group_obs_term_dim['policy']]
    term_max = {}

    with torch.no_grad():
        for step in range(args_cli.steps):
            # 模拟未训练的高层：init_noise_std=0.2 的高斯，每个 env step 重采样一次。
            # 这正是训练早期低层实际收到的指令流 —— 值不大，但每 0.2 s 就换一次。
            action = torch.randn(env.num_envs, 3, device=env.device) * 0.2
            obs, rew, *_ = env.step(action)

            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            n_nan = torch.isnan(policy_obs).sum().item()
            n_inf = torch.isinf(policy_obs).sum().item()
            r_nan = torch.isnan(rew).sum().item()
            r_inf = torch.isinf(rew).sum().item()
            bad["obs_nan"] += n_nan
            bad["obs_inf"] += n_inf
            bad["rew_nan"] += r_nan
            bad["rew_inf"] += r_inf
            if first_hit is None and (n_nan or n_inf or r_nan or r_inf):
                first_hit = (step, n_nan, n_inf, r_nan, r_inf)

            # 只查 NaN/inf 是不够的：学习率已被自适应调度压到下限 1e-5、
            # 且有 max_grad_norm=1.0 裁剪，权重不可能自己涨到 10^12。
            # 所以要盯观测的**量级** —— V(s)=W·obs，输入巨大时输出同样巨大。
            amax = policy_obs.abs().max().item()
            if amax > obs_absmax[0]:
                idx = policy_obs.abs().argmax().item()
                obs_absmax = (amax, step, idx % policy_obs.shape[1])
            # 逐项统计，把"第几维"翻译成"哪个观测项"
            off = 0
            for tname, tdim in zip(term_names, term_dims):
                seg = policy_obs[:, off:off + tdim]
                m = seg.abs().max().item()
                if m > term_max.get(tname, 0.0):
                    term_max[tname] = m
                off += tdim
            obs_p99.append(policy_obs.abs().flatten().kthvalue(
                max(int(policy_obs.numel() * 0.99), 1)).values.item())

            # 摔倒统计：terminated 是"非超时终止"，在本任务里几乎等同于摔倒
            term = env.termination_manager.terminated
            fall_events += term.sum().item()
            total_terms += env.num_envs

            scanner = env.scene.sensors["height_scanner"]
            z = scanner.data.ray_hits_w[..., 2]
            scan_extremes.append(
                (torch.isnan(z).sum().item(), torch.isinf(z).sum().item())
            )

    tot_scan_nan = sum(a for a, _ in scan_extremes)
    tot_scan_inf = sum(b for _, b in scan_extremes)

    print(f"\n  共 {args_cli.steps} 步 × {args_cli.num_envs} env")
    print(f"  高层观测   NaN {bad['obs_nan']:>8}   inf {bad['obs_inf']:>8}")
    print(f"  奖励       NaN {bad['rew_nan']:>8}   inf {bad['rew_inf']:>8}")
    print(f"  射线命中 z NaN {tot_scan_nan:>8}   inf {tot_scan_inf:>8}   （clip 之前的原始值）")
    print(f"\n  摔倒事件   {fall_events} 次 / {total_terms} env-步"
          f"   → 平均每 {total_terms / max(fall_events, 1):.1f} 步摔一次")
    amax, astep, adim = obs_absmax
    p99 = sum(obs_p99) / max(len(obs_p99), 1)
    print(f"\n  观测量级   |obs| 最大 {amax:.3e}  (第 {astep} 步, 第 {adim} 维)")
    print(f"             |obs| 99 分位均值 {p99:.3f}   ← 正常应为个位数")
    print("\n  逐项最大绝对值：")
    for tname, m in sorted(term_max.items(), key=lambda x: -x[1]):
        flag = "  ← 异常" if m > 100 else ""
        print(f"    {tname:<28}{m:>14.3e}{flag}")
    if first_hit:
        s, a, b, c, d = first_hit
        print(f"\n  首次异常于第 {s} 步：obs_nan={a} obs_inf={b} rew_nan={c} rew_inf={d}")

    print("\n" + "═" * 66)
    if bad["obs_nan"] or bad["rew_nan"]:
        print("❌ 观测或奖励里出现 NaN → 这就是 std 变 NaN 的来源。")
        print("   注意 clip 挡得住 inf 但挡不住 NaN，必须在源头做掩码。")
        code = 1
    elif bad["obs_inf"] or bad["rew_inf"]:
        print("❌ 观测或奖励里出现 inf → 会在网络里迅速退化成 NaN。")
        code = 1
    elif tot_scan_inf:
        print(f"⚠️  射线有 {tot_scan_inf} 个 inf，但被观测项的 clip 挡住了。")
        print("   奖励项若也用到该传感器且没有防护，仍是隐患。")
        code = 0
    else:
        print("✅ 本次采样未发现 NaN/inf。发散可能需要更长 rollout 才能复现，")
        print("   或来自优化器侧（学习率 / 梯度爆炸）。")
        code = 0
    print("═" * 66 + "\n")
    env.close()
    return code


if __name__ == "__main__":
    if args_cli.nan_hunt:
        code = run_nan_hunt()
    elif args_cli.nav:
        code = run_nav_zero_command()
    else:
        code = main()
    simulation_app.close()
    raise SystemExit(code)
