from mjlab.utils.lab_api.tasks.importer import import_packages

from humanoid_hw6.utils import get_wandb_checkpoint_path as _get_wandb_checkpoint_path

_BLACKLIST_PKGS = ["scripts", ".mdp"]


def _patch_mjlab_wandb_checkpoint_loading() -> None:
  """Use HW6's W&B checkpoint resolver for mjlab train resume paths."""
  try:
    import mjlab.utils.os as mjlab_os

    mjlab_os.get_wandb_checkpoint_path = _get_wandb_checkpoint_path
  except Exception:
    pass

  try:
    import mjlab.scripts.train as mjlab_train

    mjlab_train.get_wandb_checkpoint_path = _get_wandb_checkpoint_path
  except Exception:
    pass


_patch_mjlab_wandb_checkpoint_loading()

import_packages(__name__, _BLACKLIST_PKGS)
