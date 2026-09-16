"""Replay HOI ``.npz`` trajectories using the Unitree G1 29DoF USD from ``unitree_model``.

Robot asset (default ``UNITREE_MODEL_DIR`` = ``unitree_rl_lab/unitree_model``)::

    $UNITREE_MODEL_DIR/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd

Set ``UNITREE_MODEL_DIR`` to override where USD assets are resolved from.

Equivalent CLI::

    python scripts/mimic/view_hoi_npz.py --robot-source usd ...

Examples::

    python scripts/mimic/view_hoi_npz_unitree_usd.py \\
        -f motion_dataset/HOI/robot-terrain/climb_00_z_scale_1.0.npz

    python scripts/mimic/view_hoi_npz_unitree_usd.py \\
        --task object-terrain --filter scene_00 --hoi-root motion_dataset/HOI
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_VIEWER = Path(__file__).resolve().with_name("view_hoi_npz.py")


def main() -> None:
    argv = sys.argv[1:]
    if not any(a == "--robot-source" or a.startswith("--robot-source=") for a in argv):
        argv = ["--robot-source", "usd", *argv]
    sys.argv = [str(_VIEWER), *argv]
    runpy.run_path(str(_VIEWER), run_name="__main__")


if __name__ == "__main__":
    main()
