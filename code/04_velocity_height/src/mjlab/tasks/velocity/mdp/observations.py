"""Velocity task MDP observations.

Homework TODOs in this file: 10  (of 10 total)
Index: 本文件 · grep: 【实现要点
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  # >>> IMPL_10_START
  # ==============================================================================
  # 【实现要点 10/10】足部接触特权观测（Critic）
  # ==============================================================================
  # 特权观测：只给 Critic，不给 Actor。
  #
  # 为什么这样分：Critic 只在训练时用来估计状态价值 V(s)，训练结束就丢弃；
  # Actor 才是最终要部署到真机上的网络。真机上"脚是否接触地面"需要足底力传感器，
  # 未必可靠或存在，所以 Actor 不能依赖它。但仿真里这个信息是免费且精确的，
  # 给 Critic 用能让价值估计更准 → 优势函数 A = R - V 的方差更小 → 梯度更稳。
  # 这就是 asymmetric actor-critic：让"打分的人"比"做事的人"看得更多。
  #
  # 转 float 而非保留 bool：观测张量要和其他项拼接成一个连续的浮点向量，
  # bool 会在 concatenate 时报 dtype 错误。
  return (sensor_data.found > 0).float()
  # <<< IMPL_10_END


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
