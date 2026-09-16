"""Tests specific to velocity tasks."""

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.mdp import (
  UniformBaseHeightCommandCfg,
  UniformVelocityCommandCfg,
)


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


@pytest.fixture(scope="module")
def g1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all G1 velocity task IDs."""
  return [t for t in velocity_task_ids if "G1" in t]


@pytest.fixture(scope="module")
def flat_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all flat terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Flat" in t]


def test_velocity_tasks_have_velocity_command(velocity_task_ids: list[str]) -> None:
  """All velocity tasks should have a velocity command."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "velocity" in cfg.commands, f"Task {task_id} missing 'velocity' command"

    velocity_cmd = cfg.commands["velocity"]
    assert isinstance(velocity_cmd, UniformVelocityCommandCfg), (
      f"Task {task_id} velocity command is not UniformVelocityCommandCfg"
    )


def test_g1_velocity_has_required_sensors(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should have feet/ground and self collision sensors."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_flat_velocity_tasks_have_plane_terrain(
  flat_velocity_task_ids: list[str],
) -> None:
  """Flat velocity tasks should have terrain_type='plane' and no terrain_generator."""
  for task_id in flat_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "plane", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, expected 'plane'"
    )
    assert cfg.scene.terrain.terrain_generator is None, (
      f"Task {task_id} has terrain_generator, expected None for flat terrain"
    )


def test_g1_velocity_has_correct_action_scale(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should use G1_ACTION_SCALE."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


G1_VELOCITY_HEIGHT_TASK_ID = "Mjlab-VelocityHeight-Flat-Unitree-G1"
G1_VELOCITY_FLAT_TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"


def test_g1_velocity_height_task_is_registered() -> None:
  """The G1 velocity + height task should be registered."""
  assert G1_VELOCITY_HEIGHT_TASK_ID in list_tasks()


def test_g1_velocity_tasks_and_ablation_contracts() -> None:
  """Keep the two base tasks and verify the two explicit homework ablations."""
  g1_tasks = [t for t in list_tasks() if "Unitree-G1" in t]
  assert sorted(g1_tasks) == sorted(
    [
      G1_VELOCITY_FLAT_TASK_ID,
      G1_VELOCITY_HEIGHT_TASK_ID,
      G1_VELOCITY_HEIGHT_TASK_ID + "-BlindActor",
      G1_VELOCITY_HEIGHT_TASK_ID + "-NoHeightRew",
    ]
  )
  baseline = load_env_cfg(G1_VELOCITY_HEIGHT_TASK_ID)
  no_reward = load_env_cfg(G1_VELOCITY_HEIGHT_TASK_ID + "-NoHeightRew")
  blind = load_env_cfg(G1_VELOCITY_HEIGHT_TASK_ID + "-BlindActor")
  assert no_reward.rewards["track_base_height"].weight == 0.0
  assert no_reward.observations == baseline.observations
  assert blind.rewards == baseline.rewards
  assert blind.observations["critic"] == baseline.observations["critic"]
  expected_actor = baseline.observations["actor"]
  expected_actor.terms.pop("height_command")
  assert blind.observations["actor"] == expected_actor


def test_g1_velocity_height_task_has_commands() -> None:
  """The G1 velocity + height task should have velocity and base_height commands."""
  cfg = load_env_cfg(G1_VELOCITY_HEIGHT_TASK_ID)

  assert "velocity" in cfg.commands
  assert isinstance(cfg.commands["velocity"], UniformVelocityCommandCfg)

  assert "base_height" in cfg.commands
  assert isinstance(cfg.commands["base_height"], UniformBaseHeightCommandCfg)
  assert cfg.commands["base_height"].ranges.height == (0.45, 0.80)


def test_g1_velocity_height_task_has_observations_and_rewards() -> None:
  """The G1 velocity + height task should expose height obs and reward terms."""
  cfg = load_env_cfg(G1_VELOCITY_HEIGHT_TASK_ID)

  assert "height_command" in cfg.observations["actor"].terms
  assert "height_command" in cfg.observations["critic"].terms
  assert "track_base_height" in cfg.rewards


def test_flat_velocity_task_does_not_have_base_height() -> None:
  """The flat velocity-only task should not include the base_height command."""
  cfg = load_env_cfg(G1_VELOCITY_FLAT_TASK_ID)
  assert "base_height" not in cfg.commands
