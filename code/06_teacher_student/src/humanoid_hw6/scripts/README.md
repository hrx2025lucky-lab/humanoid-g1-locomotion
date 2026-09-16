# Humanoid HW6 Scripts

Utility scripts kept for the teaching workflow.

## Preview reference motions (Viser)

Browse the 25 ACCAD walk NPZ clips in a web viewer (no policy, reference pose only):

```bash
uv run python -m humanoid_hw6.scripts.data.visualize_motion_curate_viser \
  --motion assets/motions/g1_accad_walk
```

See [`ASSIGNMENT.md`](../../../ASSIGNMENT.md) §3.7 for full usage.

## Export checkpoint to ONNX

```bash
uv run python -m humanoid_hw6.scripts.deploy.export_checkpoint_to_onnx --help
```

## Play checkpoint helper

```bash
uv run play Mjlab-Humanoid-HW6-Teacher-G1 --help
```

## Instructor solution verification

```bash
uv run python scripts/verify_instructor_solutions.py
```
