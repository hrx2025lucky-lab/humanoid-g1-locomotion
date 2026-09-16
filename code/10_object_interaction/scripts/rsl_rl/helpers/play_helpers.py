import re
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from unitree_rl_lab.tasks.mimic.mdp.hoi_height_scan_vis import get_hoi_height_scan_result


def _runner_expects_actor_critic_cfg() -> bool:
    """安装的 rsl_rl 到底要 ``actor``/``critic`` 还是单个 ``policy`` 键？

    两代 API 不兼容：
      - 旧版（含本环境的 rsl-rl-lib 3.1.2）：``train_cfg["policy"]``
      - 新版：拆成 ``train_cfg["actor"]`` 和 ``train_cfg["critic"]``

    下面的转换对旧版是有害的：弹掉 policy 后 runner 立刻
    ``KeyError: 'policy'``（on_policy_runner.py:29）。

    不按版本号猜（版本边界本身就是猜测），直接问这一版的 __init__
    读的是哪个键,那才是真正决定行为的事实。
    """
    try:
        import inspect

        from rsl_rl.runners import OnPolicyRunner

        src = inspect.getsource(OnPolicyRunner.__init__)
        return '"actor"' in src or "'actor'" in src
    except Exception:
        # 探测失败就不转换：保持配置原样，让 runner 报它自己的真实错误，
        # 而不是被一次错误的转换掩盖成误导性的 KeyError。
        return False


def convert_legacy_policy_cfg(train_cfg: dict) -> dict:
    """Convert legacy IsaacLab `policy` config to rsl_rl `actor`/`critic` format."""
    if not _runner_expects_actor_critic_cfg():
        return train_cfg
    if "actor" in train_cfg and "critic" in train_cfg:
        return train_cfg

    policy_cfg = train_cfg.get("policy")
    if policy_cfg is None:
        return train_cfg

    if policy_cfg.get("class_name", "ActorCritic") != "ActorCritic":
        raise ValueError(
            f"Unsupported legacy policy class '{policy_cfg.get('class_name')}'. "
            "Expected 'ActorCritic' for automatic conversion."
        )

    activation = policy_cfg.get("activation", "elu")
    actor_hidden_dims = policy_cfg.get("actor_hidden_dims", [256, 256, 256])
    critic_hidden_dims = policy_cfg.get("critic_hidden_dims", actor_hidden_dims)
    init_std = policy_cfg.get("init_noise_std", 1.0)
    std_type = policy_cfg.get("noise_std_type", "scalar")

    train_cfg["actor"] = {
        "class_name": "MLPModel",
        "hidden_dims": actor_hidden_dims,
        "activation": activation,
        "obs_normalization": bool(policy_cfg.get("actor_obs_normalization")),
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": init_std,
            "std_type": std_type,
        },
    }
    train_cfg["critic"] = {
        "class_name": "MLPModel",
        "hidden_dims": critic_hidden_dims,
        "activation": activation,
        "obs_normalization": bool(policy_cfg.get("critic_obs_normalization")),
    }
    train_cfg.pop("policy", None)
    return train_cfg


def apply_fixed_velocity_command(env_cfg, args) -> None:
    """Optionally pin play-time velocity/yaw commands to user-provided values."""
    cmd_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if cmd_cfg is None or not hasattr(cmd_cfg, "ranges"):
        return

    yaw_cmd = args.cmd_yaw_rate if args.cmd_yaw_rate is not None else args.cmd_raw_rate
    if yaw_cmd is not None and args.cmd_yaw_rate is None:
        print("[INFO] Using --cmd-raw-rate as alias of --cmd-yaw-rate.")

    if args.cmd_lin_x is not None:
        cmd_cfg.ranges.lin_vel_x = (args.cmd_lin_x, args.cmd_lin_x)
    if args.cmd_lin_y is not None:
        cmd_cfg.ranges.lin_vel_y = (args.cmd_lin_y, args.cmd_lin_y)
    if yaw_cmd is not None:
        cmd_cfg.ranges.ang_vel_z = (yaw_cmd, yaw_cmd)

    if args.cmd_lin_x is not None or args.cmd_lin_y is not None or yaw_cmd is not None:
        print(
            "[INFO] Fixed command override:"
            f" lin_x={cmd_cfg.ranges.lin_vel_x},"
            f" lin_y={cmd_cfg.ranges.lin_vel_y},"
            f" yaw_rate={cmd_cfg.ranges.ang_vel_z}"
        )


class HeightScanLivePrint:
    """Terminal printout of height-scan ray heights (matches ``mdp.height_scan`` definition)."""

    def __init__(
        self,
        env,
        print_hz: float = 5.0,
        sensor_name: str = "height_scanner",
        height_offset: float = 0.5,
    ):
        self.env = env
        self.print_hz = max(float(print_hz), 0.5)
        self.sensor_name = sensor_name
        self.height_offset = height_offset
        self._last_print_t = 0.0

        if sensor_name not in env.scene.sensors:
            raise KeyError(f"Scene has no sensor '{sensor_name}' for height scan printout.")

        print(f"[INFO] Height scan terminal print enabled at {self.print_hz:.1f} Hz (sensor={sensor_name}, offset={height_offset})")

    def update(self):
        now_t = time.time()
        if now_t - self._last_print_t < 1.0 / self.print_hz:
            return
        self._last_print_t = now_t

        sensor = self.env.scene.sensors[self.sensor_name]
        # Same as isaaclab.envs.mdp.observations.height_scan
        heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - self.height_offset
        h0 = heights[0].detach().cpu().reshape(-1)
        n = int(h0.numel())
        sample_n = min(16, n)
        print(
            f"[HEIGHT_SCAN] env0 n={n} min={float(h0.min()):.3f} max={float(h0.max()):.3f} "
            f"mean={float(h0.mean()):.3f} first_{sample_n}={h0[:sample_n].tolist()}"
        )


class HoiHeightScanLivePrint:
    """Terminal printout for analytical HOI height scan (``mdp.hoi_height_scan``)."""

    def __init__(self, env, print_hz: float = 5.0):
        self.env = env
        self.print_hz = max(float(print_hz), 0.5)
        self._last_print_t = 0.0
        print(f"[INFO] HOI height scan terminal print enabled at {self.print_hz:.1f} Hz")

    def update(self):
        now_t = time.time()
        if now_t - self._last_print_t < 1.0 / self.print_hz:
            return
        self._last_print_t = now_t

        result = get_hoi_height_scan_result(self.env)
        h0 = result.heights[0].detach().cpu().reshape(-1)
        n = int(h0.numel())
        sample_n = min(16, n)
        print(
            f"[HOI_HEIGHT_SCAN] env0 n={n} min={float(h0.min()):.3f} max={float(h0.max()):.3f} "
            f"mean={float(h0.mean()):.3f} first_{sample_n}={h0[:sample_n].tolist()}"
        )


class FootstepMarkerTracker:
    """Persistent touchdown marker tracker for evaluation-time visualization."""

    _FOOT_PATTERNS = [r"ankle_roll", r"ankle", r"_foot"]

    def __init__(
        self,
        env,
        history: int = 120,
        marker_scale: float = 0.05,
        contact_force_threshold: float = 5.0,
    ):
        self.env = env
        self.device = env.device
        self.history = max(int(history), 1)

        self.robot = self.env.scene["robot"]
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
        self.force_body_names = self._resolve_force_body_names()
        self.foot_force_body_ids = self._resolve_foot_body_ids()
        self.foot_body_names = [self.force_body_names[idx] for idx in self.foot_force_body_ids]
        self.foot_robot_body_ids = [self.robot.body_names.index(name) for name in self.foot_body_names]
        self.num_feet = len(self.foot_force_body_ids)

        self.num_envs = self.env.num_envs
        self.prev_contact = torch.zeros((self.num_envs, self.num_feet), dtype=torch.bool, device=self.device)
        self.write_idx = torch.zeros((self.num_envs, self.num_feet), dtype=torch.long, device=self.device)
        self.contact_force_threshold = float(contact_force_threshold)

        self.history_pos = torch.full(
            (self.num_envs, self.history, self.num_feet, 3), torch.nan, dtype=torch.float, device=self.device
        )
        self.history_quat = torch.zeros((self.num_envs, self.history, self.num_feet, 4), dtype=torch.float, device=self.device)
        self.history_quat[..., 0] = 1.0

        self.left_foot_ids = [i for i, name in enumerate(self.foot_body_names) if "left" in name.lower()]
        self.right_foot_ids = [i for i, name in enumerate(self.foot_body_names) if "right" in name.lower()]

        if not self.left_foot_ids and self.num_feet > 0:
            self.left_foot_ids = [0]
        if not self.right_foot_ids and self.num_feet > 1:
            self.right_foot_ids = [i for i in range(self.num_feet) if i not in self.left_foot_ids]

        self.marker_radius = marker_scale * 1.15
        self.marker_height = max(marker_scale * 0.2, 0.006)
        # Keep marker just above terrain while avoiding z-fighting.
        self.marker_z_offset = 0.5 * self.marker_height + 0.001

        marker_cfg_left = VisualizationMarkersCfg(
            prim_path="/Visuals/Footsteps/touchdown_left",
            markers={
                "step": sim_utils.CylinderCfg(
                    radius=self.marker_radius,
                    height=self.marker_height,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
            },
        )
        marker_cfg_right = VisualizationMarkersCfg(
            prim_path="/Visuals/Footsteps/touchdown_right",
            markers={
                "step": sim_utils.CylinderCfg(
                    radius=self.marker_radius,
                    height=self.marker_height,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                ),
            },
        )
        self.left_marker_visualizer = VisualizationMarkers(marker_cfg_left)
        self.right_marker_visualizer = VisualizationMarkers(marker_cfg_right)
        self.left_marker_visualizer.set_visibility(False)
        self.right_marker_visualizer.set_visibility(False)

        foot_names = list(self.foot_body_names)
        print(
            "[INFO] Footstep markers enabled:"
            f" feet={foot_names}, history={self.history}, scale={marker_scale},"
            f" force_threshold={self.contact_force_threshold}N, z_offset={self.marker_z_offset:.4f}m"
        )

    def _resolve_force_body_names(self) -> list[str]:
        num_force_bodies = int(self.contact_sensor.data.net_forces_w.shape[1])
        sensor_body_names = getattr(self.contact_sensor, "body_names", None)
        if sensor_body_names is not None and len(sensor_body_names) == num_force_bodies:
            return list(sensor_body_names)
        if len(self.robot.body_names) == num_force_bodies:
            return list(self.robot.body_names)
        # Fallback generic names if no reliable mapping is available.
        return [f"body_{i}" for i in range(num_force_bodies)]

    def _resolve_foot_body_ids(self) -> list[int]:
        for pattern in self._FOOT_PATTERNS:
            body_ids = [i for i, name in enumerate(self.force_body_names) if re.search(pattern, name)]
            if body_ids:
                return body_ids
        raise RuntimeError(
            "Could not resolve foot body ids for marker visualization. "
            f"Tried patterns: {self._FOOT_PATTERNS}, available bodies(sample)={self.force_body_names[:20]}"
        )

    def update(self):
        foot_forces = torch.linalg.norm(self.contact_sensor.data.net_forces_w[:, self.foot_force_body_ids, :], dim=-1)
        current_contact = foot_forces > self.contact_force_threshold
        touchdown = torch.logical_and(current_contact, torch.logical_not(self.prev_contact))

        if torch.any(touchdown):
            foot_pos_w = self.robot.data.body_pos_w[:, self.foot_robot_body_ids]
            env_ids, foot_ids = torch.nonzero(touchdown, as_tuple=True)
            history_ids = self.write_idx[env_ids, foot_ids]

            touch_pos = foot_pos_w[env_ids, foot_ids].clone()
            touch_pos[:, 2] += self.marker_z_offset
            self.history_pos[env_ids, history_ids, foot_ids] = touch_pos
            self.write_idx[env_ids, foot_ids] = (history_ids + 1) % self.history

        self.prev_contact.copy_(current_contact)

        if self.left_foot_ids:
            left_pos = self.history_pos[:, :, self.left_foot_ids, :].reshape(-1, 3)
            left_quat = self.history_quat[:, :, self.left_foot_ids, :].reshape(-1, 4)
            left_mask = torch.isfinite(left_pos).all(dim=1)
            if torch.any(left_mask):
                self.left_marker_visualizer.set_visibility(True)
                self.left_marker_visualizer.visualize(left_pos[left_mask], left_quat[left_mask])
            else:
                self.left_marker_visualizer.set_visibility(False)

        if self.right_foot_ids:
            right_pos = self.history_pos[:, :, self.right_foot_ids, :].reshape(-1, 3)
            right_quat = self.history_quat[:, :, self.right_foot_ids, :].reshape(-1, 4)
            right_mask = torch.isfinite(right_pos).all(dim=1)
            if torch.any(right_mask):
                self.right_marker_visualizer.set_visibility(True)
                self.right_marker_visualizer.visualize(right_pos[right_mask], right_quat[right_mask])
            else:
                self.right_marker_visualizer.set_visibility(False)


class ComPredictionMarkerTracker:
    """Live-overwrite COM XY future prediction markers from unicycle integration."""

    def __init__(
        self,
        env,
        horizon_steps: int = 15,
        marker_interval: int = 5,
        marker_scale: float = 0.06,
    ):
        self.env = env
        self.device = env.device
        self.robot = self.env.scene["robot"]
        self.step_dt = float(self.env.step_dt)
        self.horizon_steps = max(1, int(horizon_steps))
        self.marker_interval = max(1, int(marker_interval))

        self.marker_steps = [k for k in range(1, self.horizon_steps + 1) if k % self.marker_interval == 0]
        if not self.marker_steps:
            self.marker_steps = [self.horizon_steps]

        self.num_envs = int(self.env.num_envs)
        self.torso_body_id = self._resolve_torso_body_id()
        self.identity_quat = torch.zeros((self.num_envs, len(self.marker_steps), 4), dtype=torch.float, device=self.device)
        self.identity_quat[..., 0] = 1.0
        # Lift COM prediction markers for clearer visibility in the viewport.
        self.marker_z = 0.78

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/COM/future_prediction",
            markers={
                "step": sim_utils.SphereCfg(
                    radius=max(float(marker_scale), 0.005),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                ),
            },
        )
        self.marker_visualizer = VisualizationMarkers(marker_cfg)
        self.marker_visualizer.set_visibility(False)

        print(
            "[INFO] COM prediction markers enabled:"
            f" torso='{self.robot.body_names[self.torso_body_id]}',"
            f" horizon_steps={self.horizon_steps},"
            f" marker_steps={self.marker_steps}, dt={self.step_dt:.4f}s,"
            f" marker_z={self.marker_z:.3f}m"
        )

    def _resolve_torso_body_id(self) -> int:
        torso_name_candidates = ["torso_link", "torso", "base_link", "base", "pelvis"]
        lower_names = [name.lower() for name in self.robot.body_names]
        for candidate in torso_name_candidates:
            if candidate in lower_names:
                return lower_names.index(candidate)

        for idx, name in enumerate(lower_names):
            if "torso" in name:
                return idx
        for idx, name in enumerate(lower_names):
            if "base" in name:
                return idx
        return 0

    @staticmethod
    def _yaw_from_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w = quat_wxyz[..., 0]
        x = quat_wxyz[..., 1]
        y = quat_wxyz[..., 2]
        z = quat_wxyz[..., 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def update(self):
        base_cmd = self.env.command_manager.get_command("base_velocity")
        if base_cmd is None or base_cmd.shape[-1] < 3:
            self.marker_visualizer.set_visibility(False)
            return

        torso_pos_w = self.robot.data.body_pos_w[:, self.torso_body_id]
        torso_quat_w = self.robot.data.body_quat_w[:, self.torso_body_id]
        yaw = self._yaw_from_wxyz(torso_quat_w)

        pred_xy = torso_pos_w[:, :2].clone()
        cmd_lin_x = base_cmd[:, 0]
        cmd_lin_y = base_cmd[:, 1]
        cmd_yaw = base_cmd[:, 2]

        marker_points = []
        dt = self.step_dt
        for step_k in range(1, self.horizon_steps + 1):
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            vel_x_w = cmd_lin_x * cos_yaw - cmd_lin_y * sin_yaw
            vel_y_w = cmd_lin_x * sin_yaw + cmd_lin_y * cos_yaw
            pred_xy[:, 0] += vel_x_w * dt
            pred_xy[:, 1] += vel_y_w * dt
            yaw += cmd_yaw * dt

            if step_k in self.marker_steps:
                marker_points.append(pred_xy.clone())

        if not marker_points:
            self.marker_visualizer.set_visibility(False)
            return

        marker_xy = torch.stack(marker_points, dim=1)
        marker_pos = torch.zeros((self.num_envs, len(self.marker_steps), 3), dtype=torch.float, device=self.device)
        marker_pos[..., :2] = marker_xy
        marker_pos[..., 2] = self.marker_z
        self.marker_visualizer.set_visibility(True)
        self.marker_visualizer.visualize(marker_pos.reshape(-1, 3), self.identity_quat.reshape(-1, 4))
