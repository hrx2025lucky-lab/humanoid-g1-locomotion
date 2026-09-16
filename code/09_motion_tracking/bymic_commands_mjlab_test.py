from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

DEFAULT_TARGET_MODULE = os.environ.get("BYMIC_TEST_MODULE", "bymic_commands_mjlab_todo")

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


def sample_uniform(lower, upper, shape=None, device=None, size=None):
    shape = shape if shape is not None else size
    lower = torch.as_tensor(lower, dtype=torch.float32, device=device)
    upper = torch.as_tensor(upper, dtype=torch.float32, device=device)
    return lower + (upper - lower) * torch.rand(shape, device=device)


def quat_from_euler_xyz(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=-1,
    )


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    q_conj = q.clone()
    q_conj[..., 1:] = -q_conj[..., 1:]
    return q_conj / torch.clamp((q * q).sum(dim=-1, keepdim=True), min=1e-12)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 1:]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zeros = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zeros, zeros, yaw)


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dq = quat_mul(quat_inv(q1), q2)
    return 2.0 * torch.atan2(
        torch.linalg.norm(dq[..., 1:], dim=-1), torch.abs(dq[..., 0])
    )


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    two = torch.tensor(2.0, dtype=q.dtype, device=q.device)
    return torch.stack(
        [
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def install_mjlab_stubs() -> None:
    class CommandTermCfg:
        pass

    class CommandTerm:
        def __init__(self, cfg, env) -> None:
            self.cfg = cfg
            self._env = env
            self.device = env.device
            self.num_envs = env.num_envs
            self.metrics: dict[str, torch.Tensor] = {}

    class DebugVisualizer:
        pass

    managers = types.ModuleType("mjlab.managers")
    managers.CommandTerm = CommandTerm
    managers.CommandTermCfg = CommandTermCfg

    math_mod = types.ModuleType("mjlab.utils.lab_api.math")
    math_mod.matrix_from_quat = matrix_from_quat
    math_mod.quat_apply = quat_apply
    math_mod.quat_error_magnitude = quat_error_magnitude
    math_mod.quat_from_euler_xyz = quat_from_euler_xyz
    math_mod.quat_inv = quat_inv
    math_mod.quat_mul = quat_mul
    math_mod.sample_uniform = sample_uniform
    math_mod.yaw_quat = yaw_quat

    debug_mod = types.ModuleType("mjlab.viewer.debug_visualizer")
    debug_mod.DebugVisualizer = DebugVisualizer

    sys.modules["mujoco"] = types.ModuleType("mujoco")
    sys.modules["mjlab"] = types.ModuleType("mjlab")
    sys.modules["mjlab.managers"] = managers
    sys.modules["mjlab.utils"] = types.ModuleType("mjlab.utils")
    sys.modules["mjlab.utils.lab_api"] = types.ModuleType("mjlab.utils.lab_api")
    sys.modules["mjlab.utils.lab_api.math"] = math_mod
    sys.modules["mjlab.viewer"] = types.ModuleType("mjlab.viewer")
    sys.modules["mjlab.viewer.debug_visualizer"] = debug_mod


class MockMjlabRobotData:
    def __init__(
        self, num_envs: int, joint_count: int, body_count: int, device: str
    ) -> None:
        self.joint_pos = torch.zeros(
            num_envs, joint_count, dtype=torch.float32, device=device
        )
        self.joint_vel = torch.zeros(
            num_envs, joint_count, dtype=torch.float32, device=device
        )
        self.body_link_pos_w = torch.zeros(
            num_envs, body_count, 3, dtype=torch.float32, device=device
        )
        self.body_link_quat_w = torch.zeros(
            num_envs, body_count, 4, dtype=torch.float32, device=device
        )
        self.body_link_quat_w[..., 0] = 1.0
        self.body_link_lin_vel_w = torch.zeros(
            num_envs, body_count, 3, dtype=torch.float32, device=device
        )
        self.body_link_ang_vel_w = torch.zeros(
            num_envs, body_count, 3, dtype=torch.float32, device=device
        )
        lower = -torch.ones(num_envs, joint_count, dtype=torch.float32, device=device)
        upper = torch.ones(num_envs, joint_count, dtype=torch.float32, device=device)
        self.soft_joint_pos_limits = torch.stack([lower, upper], dim=-1)


class MockMjlabRobot:
    def __init__(
        self, num_envs: int, joint_count: int, body_names: tuple[str, ...], device: str
    ) -> None:
        self.body_names = body_names
        self.data = MockMjlabRobotData(num_envs, joint_count, len(body_names), device)
        self.joint_write_env_ids: torch.Tensor | None = None
        self.root_write_env_ids: torch.Tensor | None = None
        self.clear_state_env_ids: torch.Tensor | None = None

    def find_bodies(self, body_names: tuple[str, ...], preserve_order: bool = True):
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
        self.data.body_link_pos_w[env_ids, 0] = root_state[:, 0:3]
        self.data.body_link_quat_w[env_ids, 0] = root_state[:, 3:7]
        self.data.body_link_lin_vel_w[env_ids, 0] = root_state[:, 7:10]
        self.data.body_link_ang_vel_w[env_ids, 0] = root_state[:, 10:13]

    def clear_state(self, env_ids: torch.Tensor) -> None:
        self.clear_state_env_ids = env_ids.clone()


class MockMjlabScene(dict):
    def __init__(self, robot: MockMjlabRobot, num_envs: int, device: str) -> None:
        super().__init__({"robot": robot})
        self.env_origins = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.env_origins[:, 0] = 2.0 * torch.arange(
            num_envs, dtype=torch.float32, device=device
        )


class MockTerminationManager:
    def __init__(self, num_envs: int, device: str) -> None:
        self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)


class MockMjlabEnv:
    def __init__(self, num_envs: int = 3, device: str = "cpu") -> None:
        self.device = device
        self.num_envs = num_envs
        self.step_dt = 0.2
        robot = MockMjlabRobot(
            num_envs, joint_count=2, body_names=("root", "hand"), device=device
        )
        self.scene = MockMjlabScene(robot, num_envs, device)
        self.termination_manager = MockTerminationManager(num_envs, device)


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
    root_quat = quat_from_euler_xyz(zeros, zeros, yaws)
    hand_quat = quat_from_euler_xyz(zeros, zeros, yaws + 0.10)
    body_quat_w = torch.stack([root_quat, hand_quat], dim=1)
    body_lin_vel_w = torch.zeros_like(body_pos_w)
    body_lin_vel_w[1:] = body_pos_w[1:] - body_pos_w[:-1]
    body_ang_vel_w = torch.zeros_like(body_pos_w)
    body_ang_vel_w[:, :, 2] = 0.1
    np.savez(
        path,
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


def import_fresh(module_name: str):
    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def load_target_module(target: str):
    path_like = target.endswith(".py") or os.path.sep in target or os.path.exists(target)
    if not path_like:
        return import_fresh(target), target

    path = Path(target).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TODO file not found: {path}")

    module_name = f"_bymic_target_{path.stem}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import TODO file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, str(path)


def get_target_arg() -> str:
    if len(sys.argv) > 2:
        print(f"Usage: python {Path(sys.argv[0]).name} [path/to/todo.py]")
        raise SystemExit(2)
    if len(sys.argv) == 2:
        return sys.argv[1]
    return DEFAULT_TARGET_MODULE


def make_command(module, motion_file: str):
    env = MockMjlabEnv(num_envs=3, device="cpu")
    cfg = module.MotionCommandCfg(
        motion_file=motion_file,
        entity_name="robot",
        body_names=("root", "hand"),
        anchor_body_name="root",
        adaptive_alpha=0.5,
        adaptive_uniform_ratio=0.2,
        adaptive_kernel_size=1,
    )
    return module.MotionCommand(cfg, env)


def assert_close(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if lhs.dtype == torch.bool or rhs.dtype == torch.bool:
        assert torch.equal(lhs, rhs), name
    elif lhs.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        assert torch.equal(lhs, rhs), name
    else:
        assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-6), name


def assert_command_matches_expected(
    command, expected: dict[str, torch.Tensor]
) -> None:
    for name in (
        "time_steps",
        "_current_bin_failed",
        "bin_failed_count",
        "body_pos_relative_w",
        "body_quat_relative_w",
    ):
        assert_close(name, getattr(command, name), expected[name])
    for key in ("sampling_entropy", "sampling_top1_prob", "sampling_top1_bin"):
        assert_close(key, command.metrics[key], expected[key])


def setup_robot_anchor(command) -> None:
    command.robot.data.body_link_pos_w[:, 0] = torch.tensor(
        [[10.0, 0.0, 0.9], [20.0, 0.0, 0.8], [30.0, 0.0, 0.7]], dtype=torch.float32
    )
    yaws = torch.tensor([0.0, 0.3, -0.2], dtype=torch.float32)
    zeros = torch.zeros_like(yaws)
    command.robot.data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zeros, zeros, yaws)


def run_tests(target: str | None = None) -> None:
    install_mjlab_stubs()
    target_module, target_label = load_target_module(target or DEFAULT_TARGET_MODULE)

    with tempfile.TemporaryDirectory() as temp_dir:
        motion_file = os.path.join(temp_dir, "toy_motion.npz")
        write_motion_file(motion_file)

        target = make_command(target_module, motion_file)
        assert_close("body_indexes", target.body_indexes, EXPECTED_BODY_INDEXES)
        assert hasattr(target.robot.data, "body_link_pos_w")
        assert target.body_pos_w.shape == EXPECTED_BODY_POS_SHAPE
        print("Test 0 passed: MJLab interfaces are intact.")

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
        setup_robot_anchor(target)

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
        assert_close(
            "clear_state_env_ids",
            target.robot.clear_state_env_ids,
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
