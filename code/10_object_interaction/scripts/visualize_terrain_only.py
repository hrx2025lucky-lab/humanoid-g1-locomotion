#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize generated terrain only (no robot, no policy).

Example:
    python scripts/visualize_terrain_only.py \
        --task Unitree-G1-29dof-Velocity-Heightfield-Warmup \
        --terrain-rows 1 --terrain-cols 1 --num-envs 1
"""

import argparse

from isaaclab.app import AppLauncher

# local package imports must happen after AppLauncher setup in Isaac Lab scripts
parser = argparse.ArgumentParser(description="Visualize terrain generation without loading robots or policies.")
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-G1-29dof-Velocity-Heightfield-Warmup",
    help="Registered task name used only to fetch scene terrain configuration.",
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=1,
    help="Number of env origins to compute (terrain-only visualization typically uses 1).",
)
parser.add_argument("--terrain-rows", type=int, default=None, help="Override terrain generator rows.")
parser.add_argument("--terrain-cols", type=int, default=None, help="Override terrain generator cols.")
parser.add_argument(
    "--max-init-level",
    type=int,
    default=None,
    help="Override max initial terrain level for curriculum-based terrain origins.",
)
parser.add_argument(
    "--debug-vis",
    action="store_true",
    default=False,
    help="Show terrain origin frame markers.",
)
parser.add_argument(
    "--disable-fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _build_terrain_from_task():
    """Load task config and create only the terrain importer."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="env_cfg_entry_point",
    )

    terrain_cfg = env_cfg.scene.terrain
    terrain_cfg.num_envs = max(1, int(args_cli.num_envs))
    terrain_cfg.debug_vis = bool(args_cli.debug_vis)

    if getattr(terrain_cfg, "terrain_generator", None) is not None:
        if args_cli.terrain_rows is not None:
            terrain_cfg.terrain_generator.num_rows = max(1, int(args_cli.terrain_rows))
        if args_cli.terrain_cols is not None:
            terrain_cfg.terrain_generator.num_cols = max(1, int(args_cli.terrain_cols))

    if args_cli.max_init_level is not None and hasattr(terrain_cfg, "max_init_terrain_level"):
        terrain_cfg.max_init_terrain_level = max(0, int(args_cli.max_init_level))

    return TerrainImporter(terrain_cfg)


def main():
    """Launch simulator and keep stepping for visual inspection."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Add a bright dome light so terrain remains visible without robot assets.
    light_cfg = sim_utils.DomeLightCfg(intensity=4000.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/Light", light_cfg)

    terrain = _build_terrain_from_task()

    # Center camera around the first origin if available, otherwise around world origin.
    if getattr(terrain, "env_origins", None) is not None and len(terrain.env_origins) > 0:
        target = terrain.env_origins[0].tolist()
    else:
        target = [0.0, 0.0, 0.0]
    sim.set_camera_view(
        eye=[target[0] + 5.5, target[1] + 5.5, target[2] + 4.5],
        target=target,
    )

    sim.reset()

    print("[INFO] Terrain-only scene ready.")
    print(f"[INFO] task={args_cli.task}")
    if getattr(terrain, "terrain_origins", None) is not None:
        rows, cols = terrain.terrain_origins.shape[:2]
        print(f"[INFO] generated terrain grid rows={rows}, cols={cols}")
    print("[INFO] Close the simulator window to exit.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
