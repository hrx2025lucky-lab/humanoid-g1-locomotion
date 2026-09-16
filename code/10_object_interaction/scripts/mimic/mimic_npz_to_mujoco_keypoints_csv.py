#!/usr/bin/env python3
"""Export mimic motion body keypoints to a MuJoCo debug-visualization CSV.

CSV format:
  time,p0_x,p0_y,p0_z,p1_x,p1_y,p1_z,...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input mimic npz file path.")
    parser.add_argument("--output", required=True, help="Output CSV file path.")
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback FPS if not present in NPZ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {in_path}")

    data = np.load(in_path)
    if "body_pos_w" not in data:
        raise ValueError("NPZ does not contain 'body_pos_w'.")
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
    if body_pos.ndim != 3 or body_pos.shape[2] != 3:
        raise ValueError(f"Expected body_pos_w shape [T,B,3], got {body_pos.shape}")

    fps = float(np.asarray(data["fps"]).item()) if "fps" in data else float(args.fps)
    if fps <= 0.0:
        raise ValueError(f"Invalid fps: {fps}")

    frame_count, body_count, _ = body_pos.shape
    times = np.arange(frame_count, dtype=np.float32) / np.float32(fps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        header = ["time"]
        for i in range(body_count):
            header.extend((f"p{i}_x", f"p{i}_y", f"p{i}_z"))
        f.write(",".join(header) + "\n")
        for t in range(frame_count):
            vals = [f"{float(times[t]):.9g}"]
            pts = body_pos[t].reshape(-1)
            vals.extend(f"{float(v):.9g}" for v in pts)
            f.write(",".join(vals) + "\n")

    print(f"[OK] Wrote keypoint CSV: {out_path}")
    print(f"[Info] frames={frame_count}, keypoints_per_frame={body_count}, fps={fps:.6g}")


if __name__ == "__main__":
    main()
