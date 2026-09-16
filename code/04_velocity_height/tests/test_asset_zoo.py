import mujoco

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity


def test_g1_robot_compiles() -> None:
  """Tests that the G1 robot in the asset zoo compiles without errors."""
  robot_cfg = get_g1_robot_cfg()
  assert isinstance(Entity(robot_cfg).compile(), mujoco.MjModel)
