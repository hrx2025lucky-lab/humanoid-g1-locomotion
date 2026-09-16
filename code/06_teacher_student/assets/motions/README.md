# Motion Data

HW6 uses only the 25-clip ACCAD walk subset under `assets/motions/g1_accad_walk/`.

Default motion config:

- [`src/humanoid_hw6/config/g1/motion_data_cfg.yaml`](../../src/humanoid_hw6/config/g1/motion_data_cfg.yaml)
- [`src/humanoid_hw6/config/g1/motion_data_cfg_g1_accad_walk.yaml`](../../src/humanoid_hw6/config/g1/motion_data_cfg_g1_accad_walk.yaml)

Each NPZ clip contains joint and body reference trajectories consumed by the motion command module.

## Preview clips in Viser

```bash
cd shenlan_humanoid_hw6
uv run python -m humanoid_hw6.scripts.data.visualize_motion_curate_viser \
  --motion assets/motions/g1_accad_walk
```

Details: [`ASSIGNMENT.md`](../../ASSIGNMENT.md) §3.7.

## Archived local datasets

Larger legacy datasets (full ACCAD library, curation exports, preview videos, smoke-test subsets) were moved to `assets/motions_unused/` and are not referenced by HW6 configs.
