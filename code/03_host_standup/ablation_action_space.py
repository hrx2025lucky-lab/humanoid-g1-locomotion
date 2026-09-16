#!/usr/bin/env python3
"""实践 3 消融实验：增量动作空间 vs 残差动作空间。

参考实现建议的进阶实验：
    "对比『固定标称姿态残差动作空间』和 HoST 增量动作空间的站起效果，
     从而直观看到增量动作空间对接触丰富任务的优势。"

两组唯一的差别是动作映射的基准：
    A 增量式(HoST)  q* = q_current + 0.25·a     基准 = 当前实际姿态
    B 残差式        q* = q_default + 0.25·a     基准 = 固定标称站姿

其余全部相同：同一个预训练策略、同一份观测拼装、同一组 PD 参数、同一个随机种子。
观测里的 previous_policy_action 也照常反馈，所以两组的策略输入分布只会因为
机器人自身状态不同而分化,这正是我们要观察的因果。

用法:
    python ablation_action_space.py            # 跑两组并对比
    SIM_SECONDS=20 python ablation_action_space.py

产物:
    outputs/ablation_incremental.mp4
    outputs/ablation_residual.mp4
    终端输出两组的高度曲线统计
"""
import os
import sys

import imageio
import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_mujoco_host_student import (  # noqa: E402
    build_single_observation,
    load_config,
    pd_control,
    read_robot_state,
    update_observation_history,
)


def run(mode: str, cfg: dict, duration: float, record: bool):
    """跑一组仿真。mode ∈ {"incremental", "residual"}。"""
    sim_dt = cfg["simulation_dt"]
    decimation = cfg["control_decimation"]
    total_steps = int(duration / sim_dt)

    kp = np.array(cfg["kps"], dtype=np.float32)
    kd = np.array(cfg["kds"], dtype=np.float32)
    torque_limits = np.array(cfg["torque_limits"], dtype=np.float32)
    default_q = np.array(cfg["default_angles"], dtype=np.float32)
    action_scale = cfg["action_scale"]

    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt
    policy = torch.jit.load(cfg["policy_path"])

    prev_action = np.zeros(cfg["num_actions"], dtype=np.float32)
    target_q = default_q.copy()
    hist = np.zeros(cfg["single_observation_dim"] * cfg["history_length"], dtype=np.float32)

    writer = None
    renderer = None
    if record:
        W, H, fps = cfg["video_width"], cfg["video_height"], cfg["video_fps"]
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H)
        renderer = mujoco.Renderer(m, height=H, width=W)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance, cam.azimuth, cam.elevation = 2.5, 135, -20
        cam.lookat[:] = [0.0, 0.0, 0.8]
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"ablation_{mode}.mp4")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        writer = imageio.get_writer(out, fps=fps, macro_block_size=1)
        frame_stride = max(1, round((1.0 / fps) / sim_dt))

    base_z, delta_norm = [], []

    for step in range(total_steps):
        tau = pd_control(target_q, d.qpos[7:], kp, np.zeros_like(kd), d.qvel[6:], kd)
        d.ctrl[:] = np.clip(tau, -torque_limits, torque_limits)
        mujoco.mj_step(m, d)

        if (step + 1) % decimation == 0:
            cur_q, q_obs, dq_obs, w_obs, g = read_robot_state(
                d, cfg["dof_pos_scale"], cfg["dof_vel_scale"], cfg["ang_vel_scale"]
            )
            obs = build_single_observation(w_obs, g, q_obs, dq_obs, prev_action, action_scale)
            hist = update_observation_history(hist, obs, cfg["single_observation_dim"], cfg["history_length"])
            action = policy(torch.from_numpy(hist).unsqueeze(0).float()).detach().numpy().squeeze()
            prev_action = action.copy()

            # 两组唯一的差别就在这里
            if mode == "incremental":
                target_q = (cur_q + action_scale * action).astype(np.float32)
            else:
                target_q = (default_q + action_scale * action).astype(np.float32)

            base_z.append(float(d.qpos[2]))
            # 目标角相对当前姿态的位移量：反映 PD 每一步被要求"拉"多远
            delta_norm.append(float(np.linalg.norm(target_q - cur_q)))

        if record and (step + 1) % frame_stride == 0:
            renderer.update_scene(d, camera=cam)
            writer.append_data(renderer.render())

    if record:
        renderer.close()
        writer.close()

    return np.array(base_z), np.array(delta_norm)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(root, "configs", "g1.yaml"))
    duration = float(os.environ.get("SIM_SECONDS", 20.0))

    results = {}
    for mode, label in [("incremental", "A 增量式 q*=q_cur+0.25a (HoST)"),
                        ("residual", "B 残差式 q*=q_def+0.25a")]:
        print(f"\n跑 {label} ...")
        z, dn = run(mode, cfg, duration, record=True)
        results[mode] = (z, dn)
        stood = z[len(z) // 2:].min() > 0.5
        print(f"  起始高度   {z[0]:.3f} m")
        print(f"  峰值高度   {z.max():.3f} m")
        print(f"  结束高度   {z[-1]:.3f} m")
        print(f"  后半程均值 {z[len(z) // 2:].mean():.3f} m   最低 {z[len(z) // 2:].min():.3f} m")
        print(f"  ‖q*-q_cur‖ 均值 {dn.mean():.3f} rad  最大 {dn.max():.3f} rad")
        print(f"  站起判定   {'✅ 成功且维持' if stood else '❌ 未能站起或已倒下'}")

    za, da = results["incremental"]
    zb, db = results["residual"]
    print("\n" + "=" * 62)
    print(f"{'指标':<24}{'A 增量式':>16}{'B 残差式':>16}")
    print("-" * 62)
    print(f"{'峰值高度 (m)':<24}{za.max():>16.3f}{zb.max():>16.3f}")
    print(f"{'后半程均值 (m)':<24}{za[len(za)//2:].mean():>16.3f}{zb[len(zb)//2:].mean():>16.3f}")
    print(f"{'后半程最低 (m)':<24}{za[len(za)//2:].min():>16.3f}{zb[len(zb)//2:].min():>16.3f}")
    print(f"{'‖q*-q_cur‖ 均值 (rad)':<24}{da.mean():>16.3f}{db.mean():>16.3f}")
    print(f"{'‖q*-q_cur‖ 最大 (rad)':<24}{da.max():>16.3f}{db.max():>16.3f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
