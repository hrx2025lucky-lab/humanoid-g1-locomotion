import os
import shutil
from pathlib import Path
from typing import Any

import torch
import wandb
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from humanoid_hw6.rl.exporter import attach_onnx_metadata
from humanoid_hw6.rl.model_cfg import configure_model_cfg
from humanoid_hw6.rl.models.student_teacher import StudentTeacherActor
from humanoid_hw6.rl.motion_stats import dump_motion_stats
from humanoid_hw6.rl.reward_logging import (
  install_actual_episode_length_reward_logging,
  install_merged_timeout_termination_logging,
  install_reward_logger,
)
from humanoid_hw6.utils import get_wandb_checkpoint_path


class OnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    configure_model_cfg(env, train_cfg)
    if isinstance(env, RslRlVecEnvWrapper):
      install_actual_episode_length_reward_logging(env)
      install_merged_timeout_termination_logging(env)
    super().__init__(env, train_cfg, log_dir, device)
    install_reward_logger(self)
    self.registry_name = registry_name

  def _upload_model_mode(self) -> str:
    mode = str(self.cfg.get("upload_model_mode", "rolling_latest"))
    if mode not in {"all", "rolling_latest"}:
      raise ValueError(
        f"Unsupported upload_model_mode `{mode}`. Expected one of: all, rolling_latest."
      )
    return mode

  def _maybe_upload_checkpoint(self, checkpoint_path: Path) -> None:
    if not self.cfg.get("upload_model", True):
      return

    mode = self._upload_model_mode()
    if mode == "all":
      self.logger.save_model(str(checkpoint_path), self.current_learning_iteration)
      return

    latest_path = checkpoint_path.with_name("model_latest.pt")
    shutil.copy2(checkpoint_path, latest_path)
    self.logger.save_model(str(latest_path), self.current_learning_iteration)

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    self._maybe_upload_checkpoint(Path(path))
    dump_motion_stats(
      self.env,
      self.cfg,
      Path(path),
      self.current_learning_iteration,
      logger_type=getattr(self.logger, "logger_type", None),
    )

    policy_path = Path(path).parent
    filename = f"{policy_path.parent.name}.onnx"
    try:
      self.export_policy_to_onnx(str(policy_path), filename)
      run_name = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )
      attach_onnx_metadata(self.env.unwrapped, run_name, str(policy_path / filename))
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(
          str(policy_path / filename), base_path=os.path.dirname(str(policy_path))
        )
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore[union-attr]
          self.registry_name = None
    except Exception as exc:
      print(f"[WARN] ONNX export failed (training continues): {exc}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    actor_sd = loaded_dict.get("actor_state_dict", {})
    _normalize_gaussian_distribution_state_dict(actor_sd)

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if load_iteration and infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos


def _normalize_gaussian_distribution_state_dict(state_dict: dict[str, Any]) -> None:
  if "std" in state_dict:
    state_dict["distribution.std_param"] = state_dict.pop("std")
  if "log_std" in state_dict:
    state_dict["distribution.log_std_param"] = state_dict.pop("log_std")


class StudentOnPolicyRunner(OnPolicyRunner):
  """On-policy runner with explicit teacher checkpoint loading for student tasks."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    self._tracking_log_dir = Path(log_dir).resolve() if log_dir is not None else None
    super().__init__(env, train_cfg, log_dir, device, registry_name=registry_name)
    self._maybe_load_teacher_checkpoint()

  def _teacher_checkpoint_path(self) -> Path | None:
    checkpoint_file = self.cfg.get("teacher_checkpoint_file")
    if checkpoint_file:
      checkpoint_path = Path(str(checkpoint_file)).expanduser().resolve()
      if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint file not found: {checkpoint_path}")
      print(f"[INFO]: Using local teacher checkpoint: {checkpoint_path}")
      return checkpoint_path

    wandb_run_path = self.cfg.get("teacher_wandb_run_path")
    if not wandb_run_path:
      return None

    if self._tracking_log_dir is None:
      raise ValueError("Cannot resolve teacher W&B checkpoint without a log directory.")
    log_root_path = self._tracking_log_dir.parent
    checkpoint_path, was_cached = get_wandb_checkpoint_path(
      log_root_path,
      Path(str(wandb_run_path)),
      self.cfg.get("teacher_wandb_checkpoint_name"),
    )
    run_id = checkpoint_path.parent.name
    checkpoint_name = checkpoint_path.name
    cached_str = "cached" if was_cached else "downloaded"
    print(
      "[INFO]: Resolved teacher checkpoint: "
      f"{checkpoint_name} (run: {run_id}, {cached_str})"
    )
    return checkpoint_path

  def _maybe_load_teacher_checkpoint(self) -> None:
    actor = getattr(self.alg, "actor", None)
    if not isinstance(actor, StudentTeacherActor):
      if any(
        self.cfg.get(key)
        for key in (
          "teacher_wandb_run_path",
          "teacher_wandb_checkpoint_name",
          "teacher_checkpoint_file",
        )
      ):
        raise ValueError(
          "Teacher checkpoint options are only supported for "
          "StudentTeacherActor student runs."
        )
      return

    checkpoint_path = self._teacher_checkpoint_path()
    if checkpoint_path is None:
      return

    loaded_dict = torch.load(
      checkpoint_path, map_location=self.device, weights_only=False
    )
    teacher_state_dict = loaded_dict.get("teacher_state_dict")
    if teacher_state_dict is None:
      teacher_state_dict = loaded_dict.get("actor_state_dict")
    if teacher_state_dict is None:
      raise ValueError(
        "Teacher checkpoint does not contain `teacher_state_dict` or "
        "`actor_state_dict`."
      )
    _normalize_gaussian_distribution_state_dict(teacher_state_dict)
    actor.teacher.load_state_dict(
      teacher_state_dict,
      strict=bool(self.cfg.get("teacher_strict_load", True)),
    )
    actor.loaded_teacher = True
    actor.teacher.eval()

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    actor_sd = loaded_dict.get("actor_state_dict", {})
    _normalize_gaussian_distribution_state_dict(actor_sd)

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if load_iteration and infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]

    actor = getattr(self.alg, "actor", None)
    if isinstance(actor, StudentTeacherActor):
      if any(key.startswith("teacher.") for key in actor_sd):
        actor.loaded_teacher = True
        actor.teacher.eval()
    # Allow an explicit teacher checkpoint to override embedded teacher weights
    # from a resumed student checkpoint.
    self._maybe_load_teacher_checkpoint()
    return infos
