from __future__ import annotations

import copy
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

from bymic_commands_mjlab_test import (
    assert_close,
    assert_command_matches_expected,
    load_target_module,
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

DEFAULT_TARGET_MODULE = os.environ.get("BYMIC_TEST_MODULE", "bymic_commands_isaaclab_todo")

EXPECTED_BODY_INDEXES = torch.tensor([0, 1], dtype=torch.long)
EXPECTED_BODY_POS_SHAPE = (3, 2, 3)

EXPECTED_ADAPTIVE_COMMAND = {
    "time_steps": torch.tensor([1, 3, 3], dtype=torch.long),
    "_current_bin_failed": torch.tensor([1.0, 1.0]),
    "bin_failed_count": torch.tensor([0.0, 2.0]),
    "body_pos_relative_w": torch.zeros(3, 2, 3),
    "body_quat_relative_w": torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ]
    ),
    "sampling_entropy": torch.tensor([0.266764939, 0.266764939, 0.266764939]),
    "sampling_top1_prob": torch.tensor([0.954545498, 0.954545498, 0.954545498]),
    "sampling_top1_bin": torch.tensor([0.5, 0.5, 0.5]),
}

EXPECTED_UPDATED_COMMAND = {
    "time_steps": torch.tensor([1, 3, 3], dtype=torch.long),
    "_current_bin_failed": torch.tensor([0.0, 0.0]),
    "bin_failed_count": torch.tensor([0.0, 0.5]),
    "body_pos_relative_w": torch.tensor(
        [
            [
                [10.0, 0.0, 1.049999952],
                [10.248750687, -0.024958352, 1.399999976],
            ],
            [
                [2.299999952, 0.0, 1.049999952],
                [2.549999952, 0.0, 1.399999976],
            ],
            [
                [30.0, 0.0, 1.049999952],
                [30.219394684, -0.119856402, 1.399999976],
            ],
        ]
    ),
    "body_quat_relative_w": torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.000000004],
                [0.998750269, 0.0, 0.0, 0.049979176],
            ],
            [
                [0.988771081, 0.0, 0.0, 0.149438143],
                [0.980066597, 0.0, 0.0, 0.198669329],
            ],
            [
                [0.995004177, 0.0, 0.0, -0.099833444],
                [0.998750329, 0.0, 0.0, -0.049979210],
            ],
        ]
    ),
    "sampling_entropy": torch.tensor([1.0, 1.0, 1.0]),
    "sampling_top1_prob": torch.tensor([0.5, 0.5, 0.5]),
    "sampling_top1_bin": torch.tensor([0.0, 0.0, 0.0]),
}

EXPECTED_WRITE_ENV_IDS = torch.tensor([1], dtype=torch.long)


class Marker:
    def __init__(self) -> None:
        self.scale = (1.0, 1.0, 1.0)


class VisualizationMarkersCfg:
    def __init__(self, prim_path: str = "/Visuals/Command/pose") -> None:
        self.prim_path = prim_path
        self.markers = {"frame": Marker()}

    def replace(self, prim_path: str):
        cfg = copy.deepcopy(self)
        cfg.prim_path = prim_path
        return cfg


class VisualizationMarkers:
    def __init__(self, cfg: VisualizationMarkersCfg) -> None:
        self.cfg = cfg
        self.visible = False

    def set_visibility(self, visible: bool) -> None:
        self.visible = visible

    def visualize(self, *args, **kwargs) -> None:
        return None


def configclass(cls):
    annotations = getattr(cls, "__annotations__", {})

    def __init__(self, **kwargs):
        for name in annotations:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
            elif hasattr(cls, name):
                value = getattr(cls, name)
                if value is _MISSING:
                    raise TypeError(f"missing required config value: {name}")
                setattr(self, name, copy.deepcopy(value))
            else:
                raise TypeError(f"missing required config value: {name}")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected config values: {unexpected}")

    cls.__init__ = __init__
    return cls


class _Missing:
    pass


_MISSING = _Missing()


def install_isaaclab_stubs() -> None:
    class Articulation:
        pass

    class CommandTermCfg:
        pass

    class CommandTerm:
        def __init__(self, cfg, env) -> None:
            self.cfg = cfg
            self._env = env
            self.device = env.device
            self.num_envs = env.num_envs
            self.metrics: dict[str, torch.Tensor] = {}

    assets = types.ModuleType("isaaclab.assets")
    assets.Articulation = Articulation

    managers = types.ModuleType("isaaclab.managers")
    managers.CommandTerm = CommandTerm
    managers.CommandTermCfg = CommandTermCfg

    markers = types.ModuleType("isaaclab.markers")
    markers.VisualizationMarkers = VisualizationMarkers
    markers.VisualizationMarkersCfg = VisualizationMarkersCfg

    markers_config = types.ModuleType("isaaclab.markers.config")
    markers_config.FRAME_MARKER_CFG = VisualizationMarkersCfg()

    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = configclass

    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.matrix_from_quat = matrix_from_quat
    math_mod.quat_apply = quat_apply
    math_mod.quat_error_magnitude = quat_error_magnitude
    math_mod.quat_from_euler_xyz = quat_from_euler_xyz
    math_mod.quat_inv = quat_inv
    math_mod.quat_mul = quat_mul
    math_mod.sample_uniform = sample_uniform
    math_mod.yaw_quat = yaw_quat

    sys.modules["isaaclab"] = types.ModuleType("isaaclab")
    sys.modules["isaaclab.assets"] = assets
    sys.modules["isaaclab.managers"] = managers
    sys.modules["isaaclab.markers"] = markers
    sys.modules["isaaclab.markers.config"] = markers_config
    sys.modules["isaaclab.utils"] = utils
    sys.modules["isaaclab.utils.math"] = math_mod

    import dataclasses

    dataclasses.MISSING = _MISSING


class MockIsaacRobotData:
    def __init__(
        self, num_envs: int, joint_count: int, body_count: int, device: str
    ) -> None:
        self.joint_pos = torch.zeros(
            num_envs, joint_count, dtype=torch.float32, device=device
        )
        self.joint_vel = torch.zeros(
            num_envs, joint_count, dtype=torch.float32, device=device
        )
        self.body_pos_w = torch.zeros(
            num_envs, body_count, 3, dtype=torch.float32, device=device
        )
        self.body_quat_w = torch.zeros(
            num_envs, body_count, 4, dtype=torch.float32, device=device
        )
        self.body_quat_w[..., 0] = 1.0
        self.body_lin_vel_w = torch.zeros(
            num_envs, body_count, 3, dtype=torch.float32, device=device
        )
        self.body_ang_vel_w = torch.zeros(
            num_envs, body_count, 3, dtype=torch.float32, device=device
        )
        lower = -torch.ones(num_envs, joint_count, dtype=torch.float32, device=device)
        upper = torch.ones(num_envs, joint_count, dtype=torch.float32, device=device)
        self.soft_joint_pos_limits = torch.stack([lower, upper], dim=-1)


class MockIsaacRobot:
    def __init__(
        self, num_envs: int, joint_count: int, body_names: tuple[str, ...], device: str
    ) -> None:
        self.body_names = body_names
        self.data = MockIsaacRobotData(num_envs, joint_count, len(body_names), device)
        self.is_initialized = True
        self.joint_write_env_ids: torch.Tensor | None = None
        self.root_write_env_ids: torch.Tensor | None = None

    def find_bodies(
        self, body_names: list[str] | tuple[str, ...], preserve_order: bool = True
    ):
        indexes = [self.body_names.index(name) for name in body_names]
        return indexes, list(body_names)

    def write_joint_state_to_sim(
        self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.joint_write_env_ids = env_ids.clone()
        self.data.joint_pos[env_ids] = joint_pos
        self.data.joint_vel[env_ids] = joint_vel

    def write_root_state_to_sim(
        self, root_state: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.root_write_env_ids = env_ids.clone()
        self.data.body_pos_w[env_ids, 0] = root_state[:, 0:3]
        self.data.body_quat_w[env_ids, 0] = root_state[:, 3:7]
        self.data.body_lin_vel_w[env_ids, 0] = root_state[:, 7:10]
        self.data.body_ang_vel_w[env_ids, 0] = root_state[:, 10:13]


class MockIsaacScene(dict):
    def __init__(self, robot: MockIsaacRobot, num_envs: int, device: str) -> None:
        super().__init__({"robot": robot})
        self.env_origins = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.env_origins[:, 0] = 2.0 * torch.arange(
            num_envs, dtype=torch.float32, device=device
        )


class MockTerminationManager:
    def __init__(self, num_envs: int, device: str) -> None:
        self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)


class MockIsaacEnv:
    def __init__(self, num_envs: int = 3, device: str = "cpu") -> None:
        self.device = device
        self.num_envs = num_envs
        self.scene = MockIsaacScene(
            MockIsaacRobot(num_envs, 2, ("root", "hand"), device), num_envs, device
        )
        self.termination_manager = MockTerminationManager(num_envs, device)
        self.cfg = types.SimpleNamespace(
            decimation=1, sim=types.SimpleNamespace(dt=0.2)
        )


def write_motion_file(path: str) -> None:
    root_pos = torch.tensor(
        [
            [0.0, 0.0, 1.00],
            [0.1, 0.0, 1.05],
            [0.2, 0.0, 1.10],
            [0.3, 0.0, 1.05],
            [0.4, 0.0, 1.00],
        ],
        dtype=torch.float32,
    )
    hand_offset = torch.tensor(
        [
            [0.20, 0.00, 0.30],
            [0.25, 0.00, 0.35],
            [0.30, 0.00, 0.40],
            [0.25, 0.00, 0.35],
            [0.20, 0.00, 0.30],
        ],
        dtype=torch.float32,
    )
    body_pos_w = torch.stack([root_pos, root_pos + hand_offset], dim=1)
    yaws = torch.tensor([0.0, 0.10, 0.20, 0.30, 0.40], dtype=torch.float32)
    zeros = torch.zeros_like(yaws)
    body_quat_w = torch.stack(
        [
            quat_from_euler_xyz(zeros, zeros, yaws),
            quat_from_euler_xyz(zeros, zeros, yaws + 0.10),
        ],
        dim=1,
    )
    body_lin_vel_w = torch.zeros_like(body_pos_w)
    body_lin_vel_w[1:] = body_pos_w[1:] - body_pos_w[:-1]
    body_ang_vel_w = torch.zeros_like(body_pos_w)
    body_ang_vel_w[:, :, 2] = 0.1
    np.savez(
        path,
        fps=np.array(5, dtype=np.int64),
        joint_pos=np.array(
            [[0.0, 0.0], [0.2, -0.1], [0.4, -0.2], [0.3, -0.1], [0.1, 0.0]],
            dtype=np.float32,
        ),
        joint_vel=np.array(
            [[0.0, 0.0], [0.2, -0.1], [0.2, -0.1], [-0.1, 0.1], [-0.2, 0.1]],
            dtype=np.float32,
        ),
        body_pos_w=body_pos_w.numpy(),
        body_quat_w=body_quat_w.numpy(),
        body_lin_vel_w=body_lin_vel_w.numpy(),
        body_ang_vel_w=body_ang_vel_w.numpy(),
    )


def get_target_arg() -> str:
    if len(sys.argv) > 2:
        print(f"Usage: python {Path(sys.argv[0]).name} [path/to/todo.py]")
        raise SystemExit(2)
    if len(sys.argv) == 2:
        return sys.argv[1]
    return DEFAULT_TARGET_MODULE


def make_command(module, motion_file: str):
    env = MockIsaacEnv(num_envs=3, device="cpu")
    cfg = module.MotionCommandCfg(
        asset_name="robot",
        motion_file=motion_file,
        body_names=["root", "hand"],
        anchor_body_name="root",
        adaptive_alpha=0.5,
        adaptive_uniform_ratio=0.2,
        adaptive_kernel_size=1,
    )
    return module.MotionCommand(cfg, env)


def setup_isaac_robot_anchor(command) -> None:
    command.robot.data.body_pos_w[:, 0] = torch.tensor(
        [[10.0, 0.0, 0.9], [20.0, 0.0, 0.8], [30.0, 0.0, 0.7]], dtype=torch.float32
    )
    yaws = torch.tensor([0.0, 0.3, -0.2], dtype=torch.float32)
    zeros = torch.zeros_like(yaws)
    command.robot.data.body_quat_w[:, 0] = quat_from_euler_xyz(zeros, zeros, yaws)


def run_tests(target: str | None = None) -> None:
    install_isaaclab_stubs()
    target_module, target_label = load_target_module(target or DEFAULT_TARGET_MODULE)

    with tempfile.TemporaryDirectory() as temp_dir:
        motion_file = os.path.join(temp_dir, "toy_motion.npz")
        write_motion_file(motion_file)

        target = make_command(target_module, motion_file)
        assert_close("body_indexes", target.body_indexes, EXPECTED_BODY_INDEXES)
        assert hasattr(target.robot.data, "body_pos_w")
        assert target.body_pos_w.shape == EXPECTED_BODY_POS_SHAPE
        print("Test 0 passed: IsaacLab interfaces are intact.")

        env_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        target.time_steps[:] = torch.tensor([0, 2, 4])
        target._env.termination_manager.terminated[:] = torch.tensor(
            [False, True, True]
        )
        target.bin_failed_count[:] = torch.tensor([0.0, 2.0])

        torch.manual_seed(3)
        target._adaptive_sampling(env_ids)
        assert_command_matches_expected(target, EXPECTED_ADAPTIVE_COMMAND)
        print("Test 1 passed: _adaptive_sampling matches the expected values.")

        target = make_command(target_module, motion_file)
        target.time_steps[:] = torch.tensor([0, 4, 2])
        target._env.termination_manager.terminated[:] = torch.tensor(
            [False, True, False]
        )
        setup_isaac_robot_anchor(target)

        torch.manual_seed(5)
        target._update_command()
        assert_command_matches_expected(target, EXPECTED_UPDATED_COMMAND)
        assert_close(
            "joint_write_env_ids",
            target.robot.joint_write_env_ids,
            EXPECTED_WRITE_ENV_IDS,
        )
        assert_close(
            "root_write_env_ids",
            target.robot.root_write_env_ids,
            EXPECTED_WRITE_ENV_IDS,
        )
        print("Test 2 passed: _update_command matches the expected values.")

    print(f"\nAll tests passed for {target_label}.")


if __name__ == "__main__":
    target_arg = get_target_arg()
    try:
        run_tests(target_arg)
    except NotImplementedError as exc:
        print(f"\n{target_arg} still has unfinished TODO code.")
        print(exc)
        raise SystemExit(1)
