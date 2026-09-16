from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from humanoid_hw6.rl.models.temporal_encoder import MotionEncoder, build_mlp


class StudentActor(MLPModel):
  """Actor with a history encoder."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    current_motion_obs_dim: int = 0,
    proprio_obs_dim: int = 0,
    history_steps: int = 10,
    history_latent_dim: int = 64,
    history_conv_channels: tuple[int, ...] | list[int] = (48, 24),
    history_conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    history_conv_strides: tuple[int, ...] | list[int] = (2, 2),
    layer_norm: bool = True,
  ) -> None:
    self.current_motion_obs_dim = int(current_motion_obs_dim)
    self.proprio_obs_dim = int(proprio_obs_dim)
    self.history_steps = int(history_steps)
    self.history_latent_dim = int(history_latent_dim)
    self.layer_norm = bool(layer_norm)

    if self.current_motion_obs_dim <= 0:
      raise ValueError(
        f"`current_motion_obs_dim` must be positive, got {self.current_motion_obs_dim}."
      )
    if self.proprio_obs_dim <= 0:
      raise ValueError(
        f"`proprio_obs_dim` must be positive, got {self.proprio_obs_dim}."
      )
    if self.history_steps <= 0:
      raise ValueError(f"`history_steps` must be positive, got {self.history_steps}.")

    self.current_obs_dim = self.current_motion_obs_dim + self.proprio_obs_dim
    self.history_obs_dim = self.current_obs_dim * self.history_steps

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    expected_obs_dim = self.current_obs_dim + self.history_obs_dim
    if self.obs_dim != expected_obs_dim:
      raise ValueError(
        "StudentActor observation dimension mismatch: "
        f"got {self.obs_dim}, expected {expected_obs_dim} "
        f"({self.current_obs_dim} current + {self.history_obs_dim} history)."
      )

    self.history_encoder = MotionEncoder(
      input_dim_per_step=self.current_obs_dim,
      num_steps=self.history_steps,
      activation=activation,
      conv_channels=history_conv_channels,
      conv_kernel_sizes=history_conv_kernel_sizes,
      conv_strides=history_conv_strides,
      projection_dim=self.history_latent_dim,
    )

    mlp_output_dim = (
      self.distribution.input_dim if self.distribution is not None else output_dim
    )
    self.mlp = build_mlp(
      input_dim=self._get_latent_dim(),
      output_dim=mlp_output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      layer_norm=self.layer_norm,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def _get_latent_dim(self) -> int:
    return self.current_obs_dim + self.history_latent_dim

  def _split_obs(
    self, obs_flat: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    motion_end = self.current_motion_obs_dim
    current_end = self.current_obs_dim
    history_end = current_end + self.history_obs_dim
    return (
      obs_flat[:, :motion_end],
      obs_flat[:, motion_end:current_end],
      obs_flat[:, current_end:history_end],
    )

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
  ) -> torch.Tensor:
    obs_flat = super().get_latent(obs, masks, hidden_state)
    motion_obs, proprio_obs, history_obs = self._split_obs(obs_flat)
    history_latent = self.history_encoder(history_obs)
    return torch.cat((motion_obs, proprio_obs, history_latent), dim=-1)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return OnnxStudentActor(self, verbose=verbose)




class OnnxStudentActor(nn.Module):
  """ONNX wrapper for StudentActor."""

  is_recurrent: bool = False

  def __init__(self, model: StudentActor, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_encoder = copy.deepcopy(model.history_encoder)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.obs_dim
    self.current_motion_obs_dim = model.current_motion_obs_dim
    self.current_obs_dim = model.current_obs_dim
    self.proprio_obs_dim = model.proprio_obs_dim
    self.history_obs_dim = model.history_obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    motion_end = self.current_motion_obs_dim
    current_end = self.current_obs_dim
    history_end = current_end + self.history_obs_dim

    motion_obs = x[:, :motion_end]
    proprio_obs = x[:, motion_end:current_end]
    history_obs = x[:, current_end:history_end]
    history_latent = self.history_encoder(history_obs)
    out = self.mlp(torch.cat((motion_obs, proprio_obs, history_latent), dim=-1))
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


