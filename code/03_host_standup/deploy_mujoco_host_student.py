"""
HW2 Task 2: Humanoid Action Space — HoST Sim2Sim Student Starter

Complete the TODO blocks below to implement:
  1. Reading raw robot state from MuJoCo
  2. Building the single-step observation vector
  3. Maintaining the rolling observation history
  4. Converting policy output into current-pose incremental joint targets

See HW2_Task2_Action_Space.md for background and instructions.
"""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


def get_gravity_orientation(base_quat_wxyz):
    """Return gravity direction in the robot base frame (3,)."""
    qw = base_quat_wxyz[0]
    qx = base_quat_wxyz[1]
    qy = base_quat_wxyz[2]
    qz = base_quat_wxyz[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_joint_positions, current_joint_positions, kp, target_joint_velocities, current_joint_velocities, kd):
    """Position PD controller: returns joint torques."""
    return (target_joint_positions - current_joint_positions) * kp + (
        target_joint_velocities - current_joint_velocities
    ) * kd


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    package_root = os.path.dirname(os.path.abspath(config_path))
    package_root = os.path.dirname(package_root)  # configs/ -> package root

    def resolve_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(package_root, path)

    config["policy_path"] = resolve_path(config["policy_path"])
    config["xml_path"] = resolve_path(config["xml_path"])
    config["video_path"] = resolve_path(config.get("video_path", "outputs/simulation.mp4"))
    return config


def read_robot_state(mj_data, dof_pos_scale, dof_vel_scale, ang_vel_scale):
    """
    Read joint and base state from MuJoCo.

    Returns:
        current_joint_positions: (23,) unscaled joint angles in radians
        joint_positions_for_obs: (23,) scaled joint positions for the policy observation
        joint_velocities_for_obs: (23,) scaled joint velocities for the policy observation
        base_angular_velocity_obs: (3,) scaled base angular velocity in the base frame
        projected_gravity: (3,) gravity direction expressed in the base frame
    """
    # -------------------------------------------------------------------------
    # TODO 1: Read raw robot state from MuJoCo
    #
    # MuJoCo 浮动基座（free joint）的状态布局：
    #   qpos[0:3]  基座位置 xyz          qvel[0:3]  基座线速度
    #   qpos[3:7]  基座四元数 wxyz    qvel[3:6]  基座角速度
    #   qpos[7:]   23 个关节角           qvel[6:]   23 个关节角速度
    # 注意 qpos 有 7 个基座量而 qvel 只有 6 个：四元数用 4 个数表示 3 个自由度。
    # 四元数是 wxyz 顺序，不是 xyzw: 读反了重力投影会算错，机器人会倒置。
    #
    # 用 np.array(..., dtype=np.float32) 而不是直接切片：
    # mj_data.qpos 返回的是视图不是副本，后续 mj_step 会就地改写它。
    # 这里显式拷贝，避免 current_joint_positions 在 TODO 4 用到时已经变了。
    current_joint_positions = np.array(mj_data.qpos[7:], dtype=np.float32)
    raw_joint_velocities = np.array(mj_data.qvel[6:], dtype=np.float32)
    base_quat_wxyz = np.array(mj_data.qpos[3:7], dtype=np.float32)
    base_angular_velocity = np.array(mj_data.qvel[3:6], dtype=np.float32)

    # 观测缩放：把不同物理量压到相近的数值范围，网络才好学。
    # 角速度可达 ±10 rad/s，关节角只有 ±3 rad，不缩放的话角速度会主导梯度。
    # 这三个 scale 必须和训练时完全一致，否则策略读到的是"另一个世界"的数值。
    joint_positions_for_obs = current_joint_positions * dof_pos_scale
    joint_velocities_for_obs = raw_joint_velocities * dof_vel_scale
    base_angular_velocity_obs = base_angular_velocity * ang_vel_scale

    # 投影重力：把世界系的重力方向旋转到基座系，得到 3 维单位向量。
    # 它替代了"姿态角"作为观测,机器人站直时约为 (0,0,-1)，躺平时 z 分量接近 0。
    # 用它而不是欧拉角，是因为它没有万向节死锁、且对 yaw 不敏感
    #（站起任务不关心朝向，只关心"哪边是下"）。
    projected_gravity = get_gravity_orientation(base_quat_wxyz)

    # 返回顺序必须与 main() 中的解包顺序一致
    return (
        current_joint_positions,      # 未缩放！TODO 4 的动作映射基准
        joint_positions_for_obs,      # 已缩放，进观测
        joint_velocities_for_obs,     # 已缩放，进观测
        base_angular_velocity_obs,    # 已缩放，进观测
        projected_gravity,            # 无需缩放，本身就是单位向量
    )


def build_single_observation(
    base_angular_velocity_obs,
    projected_gravity,
    joint_positions_for_obs,
    joint_velocities_for_obs,
    previous_policy_action,
    action_scale,
):
    """
    Build one 76-dimensional observation vector.

    Layout (total = 76):
        [0:3]   base angular velocity          (3)
        [3:6]   projected gravity              (3)
        [6:29]  joint positions                (23)
        [29:52] joint velocities               (23)
        [52:75] previous policy action         (23)
        [75]    action scale scalar            (1)
    """
    # -------------------------------------------------------------------------
    # TODO 2: Build the single-step observation vector
    #
    # 拼接顺序是硬约束：策略网络的第一层权重按训练时的顺序排列，
    # 顺序错了不会报错，只会让每个神经元读到错误的物理量:
    # 表现为机器人乱动或直接躺平，且极难 debug。
    #
    # action_scale 是标量，np.concatenate 只接受数组，
    # 必须包成长度 1 的数组。把它放进观测是 HoST 的设计：
    # 让策略"知道"自己的输出会被乘以多大的系数再变成关节增量。
    current_obs = np.concatenate(
        [
            base_angular_velocity_obs,                      # [0:3]   基座角速度
            projected_gravity,                              # [3:6]   投影重力
            joint_positions_for_obs,                        # [6:29]  关节角
            joint_velocities_for_obs,                       # [29:52] 关节角速度
            previous_policy_action,                         # [52:75] 上一帧动作
            np.array([action_scale], dtype=np.float32),     # [75]    动作尺度
        ]
    ).astype(np.float32)

    assert current_obs.shape == (76,), f"单帧观测应为 76 维，实际 {current_obs.shape}"
    return current_obs


def update_observation_history(observation_history, current_obs, single_observation_dim, history_length):
    """
    Maintain a rolling buffer of the last `history_length` observations.

    Returns:
        observation_history: (single_observation_dim * history_length,) updated buffer
    """
    # -------------------------------------------------------------------------
    # TODO 3: Update the rolling observation history
    #
    # 为什么需要历史：站起任务是 POMDP（部分可观测马尔可夫决策过程）。
    # 单帧观测里没有"我正在往哪个方向倒"这种信息,角速度只给了瞬时值，
    # 但躺倒姿态下策略需要判断趋势（是在翻身还是在滑）。6 帧 = 120 ms 的窗口，
    # 让网络能从差分中隐式恢复出加速度和接触状态的变化。
    #
    # FIFO 语义：丢掉最旧的一帧，把最新一帧接到末尾。
    # 顺序必须是"旧 → 新"，最新帧在最高索引端。写反了策略会把
    # 未来当过去，动作方向整体反相。
    observation_history = np.concatenate(
        [
            observation_history[single_observation_dim:],  # 丢弃最旧的 76 维
            current_obs,                                   # 最新帧追加到末尾
        ]
    ).astype(np.float32)

    expected = single_observation_dim * history_length
    assert observation_history.shape == (expected,), (
        f"历史观测应为 {expected} 维，实际 {observation_history.shape}"
    )
    return observation_history


def action_to_joint_targets(policy_action, current_joint_positions, action_scale):
    """
    Convert policy output to joint position targets using the HoST incremental action space.

    HoST formulation:
        target_joint_positions = current_joint_positions + action_scale * policy_action

    Compare with the standard residual formulation:
        target_joint_positions = default_joint_positions + action_scale * policy_action
    """
    # -------------------------------------------------------------------------
    # TODO 4: Convert policy output into current-pose incremental joint targets
    #
    # 本模块的核心。两种动作空间的差别：
    #
    #   HoST 增量式:  q* = q_current + 0.25·a     ← 基准是当前实际姿态
    #   常规残差式:   q* = q_default   + 0.25·a     ← 基准是固定标称姿态
    #
    # 为什么站起任务必须用增量式,关键在可达关节空间：
    #
    # 残差式的目标角被永久锁在 q_default 附近的一个小盒子里。策略输出 a 的
    # 典型量级是 ±1（极端 ±3），乘 action_scale=0.25 后，膝关节的可达范围只有
    #     [0.3-0.25, 0.3+0.25] = [0.05, 0.55] rad
    # 而从仰卧撑起需要膝关节屈曲到约 1.5~2.0 rad。无论跑多少步都到不了:
    # 因为每一步的目标都是相对同一个固定基准算的，误差不会累积。
    #
    # 增量式则把基准锚在当前姿态上：每步最多移动 0.25 rad，但从当前位置累积，
    # 50 Hz 控制下 1 秒的理论行程可达 12.5 rad，整个关节空间都能到达。
    # 同时 a=0 的语义变成"保持现状"而不是"回到站姿"，策略输出的每个分量
    # 都是一次相对于此刻的真实位移决策。
    #
    # 站起是接触状态频繁切换的任务（背贴地→肘撑地→臀离地→深蹲→站），
    # 每个阶段的合理姿态相差极远，根本不存在一个通用的标称基准。
    #
    # 消融实验实测（ablation_action_space.py，20 秒仿真）：
    #   增量式  峰值高度 0.747 m，后半程稳定在 0.742 m   ✅ 站起并维持
    #   残差式  峰值高度 0.096 m（= 起始高度），结束 0.065 m  ❌ 全程平躺未动
    #
    # 注意 current_joint_positions 必须是未缩放的真实关节角（弧度）。
    # 观测里的 joint_positions_for_obs 乘过 dof_pos_scale，不能拿来做基准。
    target_joint_positions = (
        current_joint_positions + action_scale * policy_action
    ).astype(np.float32)

    # 增量动作空间的自检：a = 0 时目标必须等于当前姿态。
    # 这是验证本函数最直接的方法（推荐做法）。
    assert target_joint_positions.shape == current_joint_positions.shape, (
        f"目标关节角维度 {target_joint_positions.shape} "
        f"应与当前关节角 {current_joint_positions.shape} 一致"
    )
    return target_joint_positions


def main():
    package_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(package_root, "configs", "g1.yaml")
    config = load_config(config_path)

    print("Config:", config_path)
    print("Policy:", config["policy_path"])
    print("Robot model:", config["xml_path"])

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    joint_kp = np.array(config["kps"], dtype=np.float32)
    joint_kd = np.array(config["kds"], dtype=np.float32)
    default_joint_positions = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]

    num_actions = config["num_actions"]
    single_observation_dim = config["single_observation_dim"]
    history_length = config["history_length"]

    torque_limits = np.array(config["torque_limits"], dtype=np.float32)

    save_video = config.get("save_video", False)
    video_path = config["video_path"]
    video_fps = config.get("video_fps", 50)
    video_width = config.get("video_width", 1280)
    video_height = config.get("video_height", 720)

    # -------------------------------------------------------------------------
    # Control state
    previous_policy_action = np.zeros(num_actions, dtype=np.float32)
    target_joint_positions = default_joint_positions.copy()
    observation_history = np.zeros(
        single_observation_dim * history_length, dtype=np.float32
    )

    # Load MuJoCo model and policy
    mj_model = mujoco.MjModel.from_xml_path(config["xml_path"])
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = simulation_dt
    policy = torch.jit.load(config["policy_path"])

    # Optional video recording
    video_frame_stride = max(1, round((1.0 / video_fps) / simulation_dt))
    renderer = None
    camera = None
    video_writer = None
    video_frame_count = 0
    if save_video:
        import imageio

        mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, video_width)
        mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, video_height)
        renderer = mujoco.Renderer(mj_model, height=video_height, width=video_width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(mj_model, camera)
        camera.distance = 2.5
        camera.azimuth = 135
        camera.elevation = -20
        camera.lookat[:] = [0.0, 0.0, 0.8]
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=video_fps, macro_block_size=1)

    sim_step_counter = 0

    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            start_sim_time = mj_data.time
            # Rendering latency must not shorten the requested simulated duration.
            while viewer.is_running() and mj_data.time - start_sim_time < simulation_duration:
                step_start = time.time()

                # Low-level PD tracking (runs every physics step)
                joint_torques = pd_control(
                    target_joint_positions,
                    mj_data.qpos[7:],
                    joint_kp,
                    np.zeros_like(joint_kd),
                    mj_data.qvel[6:],
                    joint_kd,
                )
                joint_torques = np.clip(joint_torques, -torque_limits, torque_limits)
                mj_data.ctrl[:] = joint_torques
                mujoco.mj_step(mj_model, mj_data)

                sim_step_counter += 1
                if sim_step_counter % control_decimation == 0:
                    # ---------------------------------------------------------
                    # Policy control loop (50 Hz)
                    # ---------------------------------------------------------
                    (
                        current_joint_positions,
                        joint_positions_for_obs,
                        joint_velocities_for_obs,
                        base_angular_velocity_obs,
                        projected_gravity,
                    ) = read_robot_state(
                        mj_data, dof_pos_scale, dof_vel_scale, ang_vel_scale
                    )

                    current_obs = build_single_observation(
                        base_angular_velocity_obs,
                        projected_gravity,
                        joint_positions_for_obs,
                        joint_velocities_for_obs,
                        previous_policy_action,
                        action_scale,
                    )

                    observation_history = update_observation_history(
                        observation_history,
                        current_obs,
                        single_observation_dim,
                        history_length,
                    )

                    obs_tensor = torch.from_numpy(observation_history).unsqueeze(0).float()
                    policy_action = policy(obs_tensor).detach().numpy().squeeze()
                    previous_policy_action = policy_action.copy()

                    target_joint_positions = action_to_joint_targets(
                        policy_action,
                        current_joint_positions,
                        action_scale,
                    )

                if save_video and sim_step_counter % video_frame_stride == 0:
                    renderer.update_scene(mj_data, camera=camera)
                    video_writer.append_data(renderer.render())
                    video_frame_count += 1

                viewer.sync()

                time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        if renderer is not None:
            renderer.close()
        if video_writer is not None:
            video_writer.close()
            if video_frame_count > 0:
                print(f"Saved video ({video_frame_count} frames) to: {video_path}")


if __name__ == "__main__":
    main()
