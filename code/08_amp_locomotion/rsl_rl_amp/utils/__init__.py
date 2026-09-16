# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions."""

from importlib import import_module

from .log_writer import LogWriter
from .config import normalize_runner_config
from .utils import (
    check_nan,
    compile_model,
    get_param,
    resolve_callable,
    resolve_nn_activation,
    resolve_obs_groups,
    resolve_optimizer,
    split_and_pad_trajectories,
    unpad_trajectories,
)


def __getattr__(name: str):
    """Load optional experiment backends only when explicitly requested."""
    optional_backends = {
        "NeptuneLogWriter": ("rsl_rl_amp.utils.neptune_log_writer", "NeptuneLogWriter"),
        "WandbLogWriter": ("rsl_rl_amp.utils.wandb_log_writer", "WandbLogWriter"),
    }
    if name not in optional_backends:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = optional_backends[name]
    resolved = getattr(import_module(module_name), attribute)
    globals()[name] = resolved
    return resolved

__all__ = [
    "LogWriter",
    "NeptuneLogWriter",
    "WandbLogWriter",
    "check_nan",
    "normalize_runner_config",
    "compile_model",
    "get_param",
    "resolve_callable",
    "resolve_nn_activation",
    "resolve_obs_groups",
    "resolve_optimizer",
    "split_and_pad_trajectories",
    "unpad_trajectories",
]
