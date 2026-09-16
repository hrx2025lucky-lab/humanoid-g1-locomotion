from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation


class MotionEncoder(nn.Module):
  """Temporal encoder for stacked observations."""

  @staticmethod
  def infer_conv_out_dim(
    num_steps: int,
    conv_channels: tuple[int, ...] | list[int],
    conv_kernel_sizes: tuple[int, ...] | list[int],
    conv_strides: tuple[int, ...] | list[int],
  ) -> int:
    conv_channels = tuple(int(v) for v in conv_channels)
    conv_kernel_sizes = tuple(int(v) for v in conv_kernel_sizes)
    conv_strides = tuple(int(v) for v in conv_strides)

    conv_out_length = int(num_steps)
    for kernel, stride in zip(conv_kernel_sizes, conv_strides, strict=True):
      conv_out_length = (conv_out_length - kernel) // stride + 1
      if conv_out_length <= 0:
        raise ValueError(
          "Invalid temporal conv config for given `num_steps`: "
          f"{num_steps}, kernels={conv_kernel_sizes}, strides={conv_strides}."
        )
    return int(conv_channels[-1] * conv_out_length)

  def __init__(
    self,
    input_dim_per_step: int,
    num_steps: int,
    activation: str = "elu",
    conv_channels: tuple[int, ...] | list[int] = (48, 24),
    conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    conv_strides: tuple[int, ...] | list[int] = (2, 2),
    projection_dim: int = 64,
  ) -> None:
    super().__init__()

    self.num_steps = int(num_steps)
    conv_channels = tuple(int(v) for v in conv_channels)
    conv_kernel_sizes = tuple(int(v) for v in conv_kernel_sizes)
    conv_strides = tuple(int(v) for v in conv_strides)
    self.out_dim = int(projection_dim)

    if self.num_steps <= 0:
      raise ValueError(f"`num_steps` must be positive, got {self.num_steps}.")
    if input_dim_per_step <= 0:
      raise ValueError(
        f"`input_dim_per_step` must be positive, got {input_dim_per_step}."
      )
    if self.out_dim <= 0:
      raise ValueError(f"`projection_dim` must be positive, got {self.out_dim}.")
    if len(conv_channels) == 0:
      raise ValueError("`conv_channels` must contain at least one value.")
    if not (len(conv_channels) == len(conv_kernel_sizes) == len(conv_strides)):
      raise ValueError("Conv config lengths must match.")

    conv_layers: list[nn.Module] = []
    in_channels = int(input_dim_per_step)
    for out_channels, kernel, stride in zip(
      conv_channels, conv_kernel_sizes, conv_strides, strict=True
    ):
      conv_layers.append(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride)
      )
      conv_layers.append(resolve_nn_activation(activation))
      in_channels = out_channels
    self.conv = nn.Sequential(*conv_layers)
    self.flatten = nn.Flatten()

    self.conv_out_dim = self.infer_conv_out_dim(
      num_steps=self.num_steps,
      conv_channels=conv_channels,
      conv_kernel_sizes=conv_kernel_sizes,
      conv_strides=conv_strides,
    )
    self.proj = nn.Sequential(
      nn.Linear(self.conv_out_dim, self.out_dim),
      resolve_nn_activation(activation),
    )

  def forward(self, motion_obs: torch.Tensor) -> torch.Tensor:
    batch_size = motion_obs.shape[0]
    step_obs = motion_obs.reshape(batch_size, self.num_steps, -1).permute(0, 2, 1)
    return self.proj(self.flatten(self.conv(step_obs)))


def build_mlp(
  input_dim: int,
  output_dim: int | tuple[int, ...] | list[int],
  hidden_dims: tuple[int, ...] | list[int],
  activation: str,
  layer_norm: bool,
) -> nn.Sequential:
  if len(hidden_dims) == 0:
    raise ValueError("`hidden_dims` must contain at least one element.")

  def new_activation() -> nn.Module:
    return resolve_nn_activation(activation)

  layers: list[nn.Module] = []
  in_dim = input_dim
  for idx, hidden_dim in enumerate(hidden_dims):
    layers.append(nn.Linear(in_dim, hidden_dim))
    if layer_norm and len(hidden_dims) > 1 and idx == len(hidden_dims) - 1:
      layers.append(nn.LayerNorm(hidden_dim))
    layers.append(new_activation())
    in_dim = hidden_dim

  if isinstance(output_dim, int):
    layers.append(nn.Linear(in_dim, output_dim))
  else:
    total_out = reduce(lambda x, y: x * y, output_dim)
    layers.append(nn.Linear(in_dim, total_out))
    layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

  return nn.Sequential(*layers)
