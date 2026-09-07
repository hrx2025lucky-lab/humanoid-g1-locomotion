#!/usr/bin/env python3
"""录制实践 11 sim2sim 的验收视频（自动驱动 + 合适视角）。

直接跑 `sim2sim.py --task stand` 会看到机器人**站着不动**——
这不是 bug：`_default_velocity_commands()` 返回 (0,0,0)，
而 stand 任务连键盘回调都直接 return，它本来就只演示站立平衡。

要拍到走动，得用 parkour 任务并**注入速度指令**。
原脚本靠键盘方向键改 `velocity_commands`，没有非交互入口，
所以这里在 viewer 循环外定时改写它。

另一个问题是默认视角：config.py 里
    VIEWER_LOOKAT   = (15.0, 0.0, 0.5)
    VIEWER_DISTANCE = 32.0
相机看向 15 米外、拉远 32 米——那是为跑酷全景设的，
机器人在原点附近时会小成一个点。录像时改成跟随机器人。

用法：
    envs/isaaclab/bin/python scripts/record_p11_sim2sim.py
    envs/isaaclab/bin/python scripts/record_p11_sim2sim.py --seconds 20 --vx 0.6
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/limx/workspace/Roxan_warmup")
SIM = ROOT / "motion control/humanoid_practice/course_code/sim2sim/sim2sim"
OUT_DIR = Path.home() / "humanoid_logs/p11_parkour/videos"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--vx", type=float, default=0.5, help="前进速度指令")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="先站稳几秒再给指令（冷启动就给速度容易摔）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SIM))
    os.chdir(SIM)

    import numpy as np
    import mujoco

    import config as cfg
    # ★ 必须在 import sim2sim 之前改 ★
    # sim2sim.py 在模块顶层 from config import VIEWER_* 取值，
    # 导入之后再改 cfg 就没用了。
    cfg.VIEWER_LOOKAT = (0.0, 0.0, 0.7)
    cfg.VIEWER_DISTANCE = 4.0
    cfg.VIEWER_ELEVATION = -15.0

    import sim2sim as s2s

    print(f"视角改为 lookat={cfg.VIEWER_LOOKAT} distance={cfg.VIEWER_DISTANCE}")
    runner = s2s.Sim2simInstance(sim2sim_type="parkour")

    # 相机跟随机器人 —— 固定 lookat 的话机器人一走就出画
    cam = runner.sim_env.viewer.cam
    cam.lookat[:] = cfg.VIEWER_LOOKAT
    cam.distance = cfg.VIEWER_DISTANCE
    cam.elevation = cfg.VIEWER_ELEVATION

    frames = []
    t0 = time.perf_counter()
    commanded = False
    n = 0

    while runner.sim_env.viewer.is_running():
        el = time.perf_counter() - t0
        if el > args.seconds:
            break

        # 先站稳再给前进指令
        if not commanded and el >= args.warmup:
            runner.velocity_commands[:] = np.array([args.vx, 0.0, 0.0], np.float32)
            commanded = True
            print(f"  t={el:.1f}s 注入速度指令 vx={args.vx}")

        runner.update_observation()
        hist = runner.get_history_obs()[None, ...]
        raw = runner.actor({"input": hist}).ravel()
        runner.last_action = raw.copy()
        act = raw[runner.policy_to_robot] * runner.action_scale + runner.default_joint_pos
        act = act * runner.joint_signs

        for _ in range(s2s.DECIMATION):
            tau = (runner.stiffness * (act - runner.sim_env.model_data.sensordata[:29])
                   - runner.damping * runner.sim_env.model_data.sensordata[29:58])
            tau = np.clip(tau, -runner.torque_limit, runner.torque_limit)
            runner.sim_env.model_data.ctrl[:] = tau
            runner.sim_env.physical_step()

        # 相机跟着机器人走
        cam.lookat[0] = runner.sim_env.model_data.qpos[0]
        cam.lookat[1] = runner.sim_env.model_data.qpos[1]
        runner.sim_env.viewer.sync()
        n += 1

    print(f"跑了 {n} 步 / {time.perf_counter()-t0:.1f}s")
    print(f"机器人最终位置 x={runner.sim_env.model_data.qpos[0]:.3f} "
          f"y={runner.sim_env.model_data.qpos[1]:.3f} "
          f"z={runner.sim_env.model_data.qpos[2]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
