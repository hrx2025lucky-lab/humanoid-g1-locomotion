from humanoid_hw6.teacher_env_cfg import DEFAULT_TEACHER_FUTURE_STEPS
from humanoid_hw6.tracking_base_env_cfg import HW6_HISTORY_LENGTH


def test_teacher_future_offsets_include_current_and_future() -> None:
  command_step_offsets = (0, *DEFAULT_TEACHER_FUTURE_STEPS)
  assert len(command_step_offsets) == 21
  assert command_step_offsets[0] == 0


def test_teacher_history_length() -> None:
  assert HW6_HISTORY_LENGTH == 10


def test_teacher_policy_observation_layout_constants() -> None:
  dof = 29
  per_step = dof + dof + 2 + 1 + 1 + 1 + 1  # joint pos/vel + anchor features
  assert per_step == 64
  command_steps = len((0, *DEFAULT_TEACHER_FUTURE_STEPS))
  command_dim = command_steps * per_step
  proprio_dim = dof + dof + 3 + 3 + dof  # noisy proprio block
  history_dim = HW6_HISTORY_LENGTH * (per_step + proprio_dim)
  assert command_dim == 1344
  assert proprio_dim == 93
  assert history_dim == 1570
  assert command_dim + proprio_dim + history_dim == 3007
