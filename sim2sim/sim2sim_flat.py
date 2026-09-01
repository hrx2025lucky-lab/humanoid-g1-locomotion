"""
实践 1 Sim2Sim：把 Isaac Lab 训练的 G1 平地行走策略搬到 MuJoCo 里跑。

改编自课程 `ch2_sim2sim_v1/sim2sim/sim2sim_raycaster.py`，
去掉了 raycaster 高度扫描（实践 2 才需要），只保留本体观测。

═══════════════════════════════════════════════════════════════
这个脚本的全部价值，在于演示 Sim2Sim 必须对齐的 6 件事：

  ① 关节顺序   MuJoCo(SDK) 顺序 ≠ Isaac Lab(asset) 顺序 → joint_ids_map
  ② 默认姿态   动作是相对默认姿态的偏移 → default_joint_pos
  ③ 动作尺度   目标角 = 默认 + scale × 网络输出 → action scale/offset
  ④ PD 参数    Isaac 内置 PD，MuJoCo 要自己算力矩 → stiffness/damping
  ⑤ 观测顺序   拼接顺序、缩放、历史帧数必须逐项一致
  ⑥ 控制频率   decimation × sim_dt = policy_dt

  错任何一项，机器人都会立刻摔倒。这就是 Sim2Sim 的全部难点。
═══════════════════════════════════════════════════════════════

用法：
    # 训练产物与 MuJoCo 场景都在仓库之外，位置因机器而异，用环境变量指定：
    export ROXAN_ROOT=/path/to/workspace       # 缺省 ~/workspace/Roxan_warmup
    export RL_LAB_RUN_DIR=/path/to/train/run   # 缺省 ROOT 下的训练输出目录

    # 无窗口快速自检（20 步）
    python sim2sim/sim2sim_flat.py --no-viewer --steps 20

    # 带窗口，键盘控制
    MUJOCO_GL=glfw python sim2sim/sim2sim_flat.py

键盘（在 MuJoCo 窗口里按）：
    ↑ / ↓     前进 / 后退
    ← / →     左转 / 右转
    W / S     左移 / 右移
    Space     停止
    R         重置
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

# ══════════════════════════════════════════════════════════════
# 配置：改这里就能换策略 / 换场景
# ══════════════════════════════════════════════════════════════

# 外部依赖都在本仓库之外（训练产物几百 MB、MuJoCo 场景来自 unitree_mujoco），
# 位置因机器而异，因此一律走环境变量，缺省值沿用开发机布局。
ROOT = os.environ.get("ROXAN_ROOT", os.path.expanduser("~/workspace/Roxan_warmup"))

# 训练产物目录（含 exported/policy.pt 和 params/deploy.yaml）
TRAIN_RUN_DIR = os.environ.get(
    "RL_LAB_RUN_DIR",
    os.path.join(
        ROOT, "repos/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-08-31_10-49-33"
    ),
)

# MuJoCo 场景（unitree_mujoco 自带的干净平地场景，不含 raycaster 插件）
ROBOT_SCENE = os.environ.get(
    "MUJOCO_SCENE",
    os.path.join(ROOT, "repos/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"),
)

POLICY_PATH = os.path.join(TRAIN_RUN_DIR, "exported", "policy.pt")
DEPLOY_CONFIG = os.path.join(TRAIN_RUN_DIR, "params", "deploy.yaml")

SIM_DT = 0.002          # MuJoCo 物理步长
DECIMATION = 10         # 每 10 个物理步跑一次策略 → policy_dt = 0.02 s（与训练一致）
DEVICE = "cpu"

INIT_BASE_POS = (0.0, 0.0, 0.80)
INIT_BASE_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

COMMAND_STEP = (0.2, 0.1, 0.2)      # 每次按键的增量 (vx, vy, wz)
COMMAND_RANGES = {
    "lin_vel_x": (-0.5, 1.0),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.2, 0.2),
}

VIEWER_DISTANCE = 4.0
VIEWER_AZIMUTH = 120.0
VIEWER_ELEVATION = -20.0


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def quat_apply_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """把世界系向量转到机身系。用于算 projected_gravity。

    四元数共轭旋转的展开式，等价于 R(q)^T @ vec。
    """
    w, x, y, z = quat_wxyz
    q_vec = np.array([x, y, z], dtype=np.float32)
    a = vec * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, vec) * (2.0 * w)
    c = q_vec * (2.0 * float(np.dot(q_vec, vec)))
    return a - b + c


def key_code(name: str, fallback: int) -> int:
    """取 GLFW 键码，取不到就用 fallback。"""
    try:
        import glfw
        return int(getattr(glfw, name))
    except Exception:
        return fallback


class PolicyInference:
    """加载 .pt 策略：numpy 观测进，numpy 动作出。

    支持两种文件：
      - TorchScript（play.py 导出的 exported/policy.pt）
      - RSL-RL 原始 checkpoint（model_xxxxx.pt）
    """

    def __init__(self, policy_file: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.policy_file = policy_file
        try:
            policy = torch.jit.load(policy_file, map_location=self.device)
            policy.eval()
            self.policy, self.kind = policy.to(self.device), "torchscript"
        except Exception:
            ckpt = torch.load(policy_file, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.policy = _ActorFromCheckpoint(state).to(self.device).eval()
            self.kind = "rsl_rl_checkpoint"

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.policy(t)
        return a.detach().cpu().numpy().reshape(-1).astype(np.float32)


class _ActorFromCheckpoint(torch.nn.Module):
    """从 RSL-RL checkpoint 里把 actor 网络重建出来。"""

    def __init__(self, state_dict: dict):
        super().__init__()
        actor_state = {
            k.removeprefix("actor."): v
            for k, v in state_dict.items()
            if k.startswith("actor.")
        }
        if not actor_state:
            raise ValueError("checkpoint 里找不到 actor.* 权重")
        layer_ids = sorted({int(k.split(".")[0]) for k in actor_state if k.endswith(".weight")})
        layers = []
        for lid in layer_ids:
            out_f, in_f = actor_state[f"{lid}.weight"].shape
            layers.append(torch.nn.Linear(in_f, out_f))
            if lid != layer_ids[-1]:
                layers.append(torch.nn.ELU())
        self.actor = torch.nn.Sequential(*layers)
        self.actor.load_state_dict(actor_state)

    def forward(self, obs):
        return self.actor(obs)


# ══════════════════════════════════════════════════════════════
# 主体
# ══════════════════════════════════════════════════════════════

class FlatSim2Sim:
    # 观测拼接顺序，必须与训练时 PolicyCfg 里的声明顺序完全一致
    OBS_ORDER = (
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    )

    def __init__(self, args):
        # ── 读部署合同（train.py 训练时自动导出的） ──
        with open(DEPLOY_CONFIG, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.step_dt = float(self.cfg["step_dt"])
        self.decimation = DECIMATION
        self.sim_dt = SIM_DT

        # ── 对齐 ①：关节顺序映射 ──
        # asset_to_sdk[i] = 「asset 顺序的第 i 个关节」在 SDK 顺序里的下标
        self.asset_to_sdk = np.asarray(self.cfg["joint_ids_map"], dtype=np.int64)
        self.num_joints = len(self.asset_to_sdk)

        # ── 对齐 ②：默认姿态 ──
        self.default_joint_pos_asset = np.asarray(
            self.cfg["default_joint_pos"], dtype=np.float32
        )
        self.default_joint_pos_sdk = np.zeros(self.num_joints, dtype=np.float32)
        self.default_joint_pos_sdk[self.asset_to_sdk] = self.default_joint_pos_asset

        # ── 对齐 ③：动作尺度与偏移 ──
        action_cfg = self.cfg["actions"]["JointPositionAction"]
        self.action_scale_asset = np.asarray(action_cfg["scale"], dtype=np.float32)
        self.action_offset_asset = np.asarray(action_cfg["offset"], dtype=np.float32)

        # ── 对齐 ④：PD 增益 ──
        self.stiffness = np.asarray(self.cfg["stiffness"], dtype=np.float32)
        self.damping = np.asarray(self.cfg["damping"], dtype=np.float32)

        # ── 对齐 ⑤：观测项配置（scale / clip / history_length） ──
        self.obs_cfg = self.cfg["observations"]
        missing = [n for n in self.OBS_ORDER if n not in self.obs_cfg]
        if missing:
            raise ValueError(
                f"deploy.yaml 缺少观测项 {missing}。\n"
                f"实际有的观测项：{list(self.obs_cfg.keys())}"
            )
        extra = [n for n in self.obs_cfg if n not in self.OBS_ORDER]
        if extra:
            raise ValueError(
                f"deploy.yaml 里有本脚本未处理的观测项 {extra}。\n"
                f"若含 height_scan，说明这是粗糙地形策略，请用实践 2 的脚本。"
            )

        # ── MuJoCo 模型 ──
        self.model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        self.torque_max = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.torque_min = self.model.actuator_ctrlrange[:, 0].astype(np.float32)

        self.imu_gyro_adr = self._sensor_adr("imu-body-gyro", fallback_name="imu_gyro")
        self.imu_quat_adr = self._sensor_adr("imu-body-quat", fallback_name="imu_quat")

        # ── 策略 ──
        self.policy = PolicyInference(POLICY_PATH, DEVICE)

        # ── 运行时状态 ──
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.history: dict[str, deque] = {}
        self.viewer = None
        self.no_viewer = args.no_viewer

        self.reset()

    # ---------- 辅助 ----------

    def _sensor_adr(self, name: str, fallback_name: str | None = None) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0 and fallback_name:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, fallback_name)
        if sid < 0:
            names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                for i in range(self.model.nsensor)
            ]
            raise ValueError(f"找不到传感器 '{name}'。模型里有：{names}")
        return int(self.model.sensor_adr[sid])

    def _hist_len(self, term: str) -> int:
        return int(self.obs_cfg[term].get("history_length", 1) or 1)

    def _scale_and_clip(self, term: str, obs: np.ndarray) -> np.ndarray:
        """对齐 ⑤ 的一部分：观测缩放与截断，必须与训练时一致。"""
        c = self.obs_cfg[term]
        scale = c.get("scale")
        if scale is not None:
            obs = obs * np.asarray(scale, dtype=np.float32)
        clip = c.get("clip")
        if clip is not None:
            obs = np.clip(obs, clip[0], clip[1])
        return obs.astype(np.float32)

    # ---------- 生命周期 ----------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = INIT_BASE_POS
        self.data.qpos[3:7] = INIT_BASE_QUAT_WXYZ
        self.data.qpos[7:7 + self.num_joints] = self.default_joint_pos_sdk
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.command[:] = 0.0
        self.last_action[:] = 0.0
        # 历史缓冲用零填满（与训练时首帧行为一致）
        self.history = {
            name: deque(
                [np.zeros(self._term_dim(name), dtype=np.float32)] * self._hist_len(name),
                maxlen=self._hist_len(name),
            )
            for name in self.OBS_ORDER
        }

    def _term_dim(self, name: str) -> int:
        return {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "velocity_commands": 3,
            "joint_pos_rel": self.num_joints,
            "joint_vel_rel": self.num_joints,
            "last_action": self.num_joints,
        }[name]

    # ---------- 观测 / 动作 ----------

    def update_observation(self):
        sd = self.data.sensordata
        joint_pos_sdk = sd[: self.num_joints].astype(np.float32)
        joint_vel_sdk = sd[self.num_joints: 2 * self.num_joints].astype(np.float32)

        imu_gyro = sd[self.imu_gyro_adr: self.imu_gyro_adr + 3].astype(np.float32)
        imu_quat = sd[self.imu_quat_adr: self.imu_quat_adr + 4].astype(np.float32)
        projected_gravity = quat_apply_inverse(
            imu_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )

        # ★ 对齐 ①：SDK 顺序 → asset 顺序
        joint_pos_asset = joint_pos_sdk[self.asset_to_sdk]
        joint_vel_asset = joint_vel_sdk[self.asset_to_sdk]

        raw = {
            "base_ang_vel": imu_gyro,
            "projected_gravity": projected_gravity,
            "velocity_commands": self.command.copy(),
            # ★ 对齐 ②：关节角是相对默认姿态的偏移
            "joint_pos_rel": joint_pos_asset - self.default_joint_pos_asset,
            "joint_vel_rel": joint_vel_asset,
            "last_action": self.last_action.copy(),
        }
        for name in self.OBS_ORDER:
            self.history[name].append(self._scale_and_clip(name, raw[name]))

    def get_obs(self) -> np.ndarray:
        """★ 对齐 ⑤：按 OBS_ORDER + 历史帧顺序拼成一维向量。"""
        parts = []
        for name in self.OBS_ORDER:
            for frame in self.history[name]:
                parts.append(frame)
        return np.concatenate(parts, dtype=np.float32)

    def action_to_target_sdk(self, raw_action_asset: np.ndarray) -> np.ndarray:
        """★ 对齐 ③：目标角 = offset + scale × 网络输出，再转回 SDK 顺序。"""
        self.last_action[:] = raw_action_asset
        target_asset = raw_action_asset * self.action_scale_asset + self.action_offset_asset
        target_sdk = np.zeros(self.num_joints, dtype=np.float32)
        target_sdk[self.asset_to_sdk] = target_asset
        return target_sdk

    def compute_torque(self, target_sdk: np.ndarray) -> np.ndarray:
        """★ 对齐 ④：Isaac Sim 内置 PD，MuJoCo 要自己算。"""
        sd = self.data.sensordata
        q = sd[: self.num_joints].astype(np.float32)
        dq = sd[self.num_joints: 2 * self.num_joints].astype(np.float32)
        tau = self.stiffness * (target_sdk - q) - self.damping * dq
        return np.clip(tau, self.torque_min, self.torque_max)

    # ---------- 键盘 ----------

    def key_callback(self, keycode: int):
        k = {
            "up": key_code("KEY_UP", 265),
            "down": key_code("KEY_DOWN", 264),
            "left": key_code("KEY_LEFT", 263),
            "right": key_code("KEY_RIGHT", 262),
            "space": key_code("KEY_SPACE", 32),
            "w": key_code("KEY_W", 87),
            "s": key_code("KEY_S", 83),
            "r": key_code("KEY_R", 82),
        }
        dx, dy, dz = COMMAND_STEP
        if keycode == k["up"]:
            self.command[0] += dx
        elif keycode == k["down"]:
            self.command[0] -= dx
        elif keycode == k["w"]:
            self.command[1] += dy
        elif keycode == k["s"]:
            self.command[1] -= dy
        elif keycode == k["left"]:
            self.command[2] += dz
        elif keycode == k["right"]:
            self.command[2] -= dz
        elif keycode == k["space"]:
            self.command[:] = 0.0
        elif keycode == k["r"]:
            self.reset()
            print("[reset]")
            return

        lo = np.array([COMMAND_RANGES["lin_vel_x"][0],
                       COMMAND_RANGES["lin_vel_y"][0],
                       COMMAND_RANGES["ang_vel_z"][0]], dtype=np.float32)
        hi = np.array([COMMAND_RANGES["lin_vel_x"][1],
                       COMMAND_RANGES["lin_vel_y"][1],
                       COMMAND_RANGES["ang_vel_z"][1]], dtype=np.float32)
        self.command[:] = np.clip(self.command, lo, hi)
        print(f"[cmd] vx={self.command[0]:+.2f}  vy={self.command[1]:+.2f}  wz={self.command[2]:+.2f}")

    # ---------- 运行 ----------

    def print_info(self):
        obs_dim = sum(self._term_dim(n) * self._hist_len(n) for n in self.OBS_ORDER)
        print("=" * 62)
        print("实践 1 Sim2Sim（平地，无 height_scan）")
        print("=" * 62)
        print(f"  场景          {ROBOT_SCENE}")
        print(f"  策略          {POLICY_PATH}")
        print(f"                ({self.policy.kind})")
        print(f"  部署合同      {DEPLOY_CONFIG}")
        print(f"  关节数        {self.num_joints}")
        print(f"  观测维度      {obs_dim}")
        for n in self.OBS_ORDER:
            print(f"      {n:<20} dim={self._term_dim(n):<3} × hist={self._hist_len(n)}")
        print(f"  policy_dt     {self.decimation * self.sim_dt:.4f} s "
              f"(deploy.yaml 记录 {self.step_dt:.4f} s)")
        if abs(self.decimation * self.sim_dt - self.step_dt) > 1e-6:
            print("  ⚠️  控制频率与训练时不一致！对齐 ⑥ 失败")
        print("=" * 62)
        if not self.no_viewer:
            print("键盘（在 MuJoCo 窗口里按）：")
            print("  ↑/↓ 前进后退   ←/→ 左右转   W/S 左右平移   Space 停   R 重置")
            print("=" * 62)

    def run(self, steps: int | None = None):
        self.print_info()
        policy_dt = self.decimation * self.sim_dt

        if not self.no_viewer:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.key_callback
            )
            self.viewer.cam.distance = VIEWER_DISTANCE
            self.viewer.cam.azimuth = VIEWER_AZIMUTH
            self.viewer.cam.elevation = VIEWER_ELEVATION
            self.viewer.cam.lookat[:] = self.data.qpos[0:3]

        n = 0
        try:
            while True:
                if self.viewer is not None and not self.viewer.is_running():
                    break
                t0 = time.perf_counter()

                self.update_observation()
                obs = self.get_obs()
                raw_action = self.policy(obs)
                if raw_action.shape[0] != self.num_joints:
                    raise ValueError(
                        f"策略输出维度 {raw_action.shape[0]} != 关节数 {self.num_joints}"
                    )

                target_sdk = self.action_to_target_sdk(raw_action)
                for _ in range(self.decimation):
                    self.data.ctrl[:] = self.compute_torque(target_sdk)
                    mujoco.mj_step(self.model, self.data)

                n += 1
                if self.viewer is not None:
                    self.viewer.cam.lookat[:] = self.data.qpos[0:3]
                    self.viewer.sync()

                if self.no_viewer and n % 10 == 0:
                    z = self.data.qpos[2]
                    print(f"step {n:5d}  base_z={z:.3f}  "
                          f"vx={self.data.qvel[0]:+.3f}  "
                          f"|action|={np.abs(raw_action).max():.3f}")

                if steps is not None and n >= steps:
                    break

                if self.viewer is not None:
                    sleep = policy_dt - (time.perf_counter() - t0)
                    if sleep > 0:
                        time.sleep(sleep)
        finally:
            if self.viewer is not None:
                self.viewer.close()

        print(f"\n跑完 {n} 步。最终 base_z = {self.data.qpos[2]:.3f} m")
        if self.data.qpos[2] < 0.3:
            print("⚠️  机器人趴下了（base_z < 0.3）")
        elif not np.isfinite(self.data.qpos).all():
            print("⚠️  出现 NaN")
        else:
            print("✅ 机器人保持站立")


def main():
    p = argparse.ArgumentParser(description="实践 1 平地 Sim2Sim")
    p.add_argument("--no-viewer", action="store_true", help="不开窗口，用于快速自检")
    p.add_argument("--steps", type=int, default=None, help="跑多少个策略步后停")
    args = p.parse_args()
    FlatSim2Sim(args).run(args.steps)


if __name__ == "__main__":
    main()
