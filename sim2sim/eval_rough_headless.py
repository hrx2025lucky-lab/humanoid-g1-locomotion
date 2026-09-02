#!/usr/bin/env python3
"""粗糙地形 sim2sim 的无头量化评估。

课程自带的 `sim2sim_raycaster.py` 只能靠键盘下发速度指令、靠肉眼看视频判断好坏。
问题是"看着还行"和"真的在跟踪指令"是两回事 —— 实践 2 的训练阶段已经栽过一次：
episode_length 和 reward 全线上涨，录像里机器人却在原地踏步。

所以部署侧也需要同一套判读方式：给定指令，量出实际速度、净位移和姿态，
而不是看视频。本脚本复用课程 `RaycasterSim2Sim` 的观测拼装与控制链路
（第三方代码零改动），只替换掉"键盘输入 + 实时渲染"这一层。

输出三类证据：
  ① 跨仿真器观测契约   观测项顺序/维度/历史长度，与训练侧逐项核对
  ② 指令跟踪           机体系实际速度 vs 指令速度，分段统计
  ③ 姿态与存活         投影重力 z 分量、离地高度、是否摔倒

用法：
    export ROXAN_ROOT=/path/to/workspace          # 可选，默认见下
    python sim2sim/eval_rough_headless.py                       # 用 config.py 默认策略
    python sim2sim/eval_rough_headless.py --run-dir /path/to/run  # 指定自训 run
    python sim2sim/eval_rough_headless.py --vx 0.5 --seconds 12
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from dataclasses import dataclass, field

import numpy as np

ROXAN_ROOT = os.environ.get("ROXAN_ROOT", "/home/limx/workspace/Roxan_warmup")
SIM2SIM_PKG = os.environ.get(
    "HW2_SIM2SIM_DIR", os.path.join(ROXAN_ROOT, "shenlan_hw/hw2_sim2sim/sim2sim")
)

# 姿态失败判据：投影重力的 z 分量。直立时为 -1，越接近 0 越倒。
# -0.7 约对应躯干倾斜 45°，与训练侧 bad_orientation 终止项同源。
UPRIGHT_FAIL_GZ = -0.7


@dataclass
class Segment:
    """一段固定指令的评估区间。"""

    name: str
    seconds: float
    command: tuple[float, float, float]  # vx, vy, yaw
    # 采样量在这一段内累积
    vx_actual: list[float] = field(default_factory=list)
    vyaw_actual: list[float] = field(default_factory=list)
    gz: list[float] = field(default_factory=list)
    clearance: list[float] = field(default_factory=list)
    start_xy: np.ndarray | None = None
    end_xy: np.ndarray | None = None


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_app(run_dir: str | None, policy: str | None):
    """构造课程的 RaycasterSim2Sim，可选覆盖策略路径。

    config.py 是模块级常量，`RaycasterSim2Sim.__init__` 在调用时才读取，
    因此在构造前改模块属性即可切换策略，不必改课程代码。
    """
    sys.path.insert(0, SIM2SIM_PKG)
    import config  # noqa: E402
    from sim2sim_raycaster import RaycasterSim2Sim  # noqa: E402

    if run_dir:
        run_dir = run_dir.rstrip("/")
        config.TRAIN_RUN_DIR = run_dir
        config.DEPLOY_CONFIG = os.path.join(run_dir, "params", "deploy.yaml")
        config.POLICY_PATH = policy or os.path.join(run_dir, "exported", "policy.pt")
    elif policy:
        config.POLICY_PATH = policy

    for path in (config.POLICY_PATH, config.DEPLOY_CONFIG,
                 config.ROBOT_SCENE, config.RAYCASTER_PLUGIN_LIBRARY):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    app = RaycasterSim2Sim(types.SimpleNamespace(no_viewer=True))
    return app, config


def report_obs_contract(app) -> None:
    """① 观测契约：MuJoCo 侧拼装顺序必须与训练侧逐项一致。

    这六项里任何一项错了都不会报错，只会让策略读到错位的数值。
    """
    print("── ① 跨仿真器观测契约 ──")
    total = 0
    print(f"  {'观测项':<20}{'单帧维度':>8}{'历史帧':>7}{'小计':>8}")
    for name in app.observation_terms:
        dim = len(app.history[name][-1])
        hist = app._history_length(name)
        total += dim * hist
        print(f"  {name:<20}{dim:>8}{hist:>7}{dim * hist:>8}")
    print(f"  {'合计':<20}{'':>8}{'':>7}{total:>8}")
    obs = app.get_history_obs()
    flag = "✅" if obs.shape[0] == total else "❌"
    print(f"  {flag} 实际拼出的 obs 向量: {obs.shape[0]} 维")
    print(f"  控制频率: policy_dt={app.policy_dt:.3f}s "
          f"(decimation={app.decimation} × sim_dt={app.policy_dt / app.decimation:.3f}s)")
    print()


def run_segments(app, segments: list[Segment]) -> tuple[bool, int]:
    """跑完所有指令段，途中采样。返回 (是否摔倒, 总步数)。"""
    data = app.env.data
    fell = False
    total_steps = 0

    for seg in segments:
        n_steps = int(round(seg.seconds / app.policy_dt))
        app.command[:] = np.asarray(seg.command, dtype=np.float32)
        seg.start_xy = data.qpos[:3].copy()[:2]

        for _ in range(n_steps):
            app.update_observation()
            raw_action = app.policy(app.get_history_obs())
            target = app.action_to_joint_pos_sdk(raw_action)
            for _ in range(app.decimation):
                data.ctrl[:] = app.compute_torque(target)
                app.env.step()
            total_steps += 1

            # 机体系速度：把世界系线速度转回机体系，与指令同一坐标系
            rot = quat_to_rotmat(data.qpos[3:7])
            v_body = rot.T @ data.qvel[:3]
            seg.vx_actual.append(float(v_body[0]))
            seg.vyaw_actual.append(float(data.qvel[5]))

            # 投影重力：世界系 -z 转到机体系，直立时 z 分量为 -1
            gz = float((rot.T @ np.array([0.0, 0.0, -1.0]))[2])
            seg.gz.append(gz)

            # 离地高度用高度扫描估计，而不是世界系 z —— 粗糙地形上地面本身不为 0
            scan = app.env.height_scanner.update()
            finite = scan[np.isfinite(scan)]
            seg.clearance.append(float(np.mean(finite)) if finite.size else float("nan"))

            if gz > UPRIGHT_FAIL_GZ:
                fell = True
                break

        seg.end_xy = data.qpos[:3].copy()[:2]
        if fell:
            break

    return fell, total_steps


def report_segments(segments: list[Segment], fell: bool, total_steps: int,
                    policy_dt: float) -> bool:
    print("── ② 指令跟踪（机体系）──")
    print(f"  {'指令段':<12}{'cmd vx':>8}{'实际 vx':>9}{'误差':>8}"
          f"{'cmd yaw':>9}{'实际 yaw':>10}{'净位移 m':>10}")
    ok = True
    for seg in segments:
        if not seg.vx_actual:
            continue
        vx = float(np.mean(seg.vx_actual))
        vyaw = float(np.mean(seg.vyaw_actual))
        disp = float(np.linalg.norm(seg.end_xy - seg.start_xy))
        err = abs(vx - seg.command[0])
        print(f"  {seg.name:<12}{seg.command[0]:>8.2f}{vx:>9.3f}{err:>8.3f}"
              f"{seg.command[2]:>9.2f}{vyaw:>10.3f}{disp:>10.3f}")
        # 前进段要求：实际速度至少达到指令的一半，否则就是"原地踏步"复现
        if seg.command[0] > 0.1 and vx < 0.5 * seg.command[0]:
            ok = False
    print()

    print("── ③ 姿态与存活 ──")
    all_gz = [g for s in segments for g in s.gz]
    all_cl = [c for s in segments for c in s.clearance if np.isfinite(c)]
    if all_gz:
        print(f"  投影重力 z      均值 {np.mean(all_gz):+.3f}  最差 {max(all_gz):+.3f}"
              f"  （直立 = -1.0，> {UPRIGHT_FAIL_GZ} 判为摔倒）")
    if all_cl:
        print(f"  离地高度(扫描)  均值 {np.mean(all_cl):+.3f} m  最低 {min(all_cl):+.3f} m")
    print(f"  仿真时长        {total_steps * policy_dt:.2f} s / {total_steps} 个策略步")
    print(f"  结果            {'❌ 中途摔倒' if fell else '✅ 全程未摔倒'}")
    print()
    return ok and not fell


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None, help="训练 run 目录，覆盖 config.TRAIN_RUN_DIR")
    p.add_argument("--policy", default=None, help="策略文件路径，覆盖 config.POLICY_PATH")
    p.add_argument("--vx", type=float, default=0.5, help="前进速度指令 (m/s)")
    p.add_argument("--vyaw", type=float, default=0.5, help="转向角速度指令 (rad/s)")
    p.add_argument("--seconds", type=float, default=8.0, help="每个运动段时长 (s)")
    args = p.parse_args()

    app, cfg = load_app(args.run_dir, args.policy)
    print()
    print("═" * 68)
    print("粗糙地形 sim2sim 无头评估")
    print("═" * 68)
    print(f"  策略: {cfg.POLICY_PATH}")
    print(f"  场景: {cfg.ROBOT_SCENE}")
    print()

    report_obs_contract(app)

    segments = [
        Segment("站立", 1.5, (0.0, 0.0, 0.0)),
        Segment("前进", args.seconds, (args.vx, 0.0, 0.0)),
        Segment("前进+转向", args.seconds, (args.vx, 0.0, args.vyaw)),
        Segment("停止", 1.5, (0.0, 0.0, 0.0)),
    ]
    fell, total_steps = run_segments(app, segments)
    passed = report_segments(segments, fell, total_steps, app.policy_dt)

    print("═" * 68)
    print("判定：" + ("✅ 通过 —— 策略在 MuJoCo 中跟踪指令且保持站立"
                      if passed else
                      "❌ 未通过 —— 见上方分项"))
    print("═" * 68)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
