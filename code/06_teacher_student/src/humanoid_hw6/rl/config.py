from dataclasses import dataclass
from typing import Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

UploadModelMode = Literal["all", "rolling_latest"]


@dataclass
class OnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  upload_model_mode: UploadModelMode = "rolling_latest"


@dataclass
class StudentOnPolicyRunnerCfg(OnPolicyRunnerCfg):
  teacher_wandb_run_path: str | None = None
  teacher_wandb_checkpoint_name: str | None = None
  teacher_checkpoint_file: str | None = None
  teacher_strict_load: bool = True


@dataclass
class ActionMatchingPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "humanoid_hw6.rl.algorithms.action_matching:ActionMatchingPPO"
  bc_coef_start: float = 1.0
  bc_coef_end: float = 0.0
  bc_anneal_iters: int = 10_000
  bc_loss_type: Literal["mse", "huber"] = "mse"


@dataclass
class KlMatchingPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "humanoid_hw6.rl.algorithms.kl_matching:KlMatchingPPO"
  kl_coef: float = 0.1
  kl_coef_min: float = 0.0
  kl_coef_anneal_iters: int = 10_000
