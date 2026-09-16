"""Convert mimic-format NPZ motion to g1_ctrl Mimic CSV format.

Expected input NPZ schema (from hoi_to_mimic_npz.py):
  - joint_pos: [T, 29]
  - body_pos_w: [T, B, 3]
  - body_quat_w: [T, B, 4]

Output CSV schema (per row) expected by State_Mimic::MotionLoader_:
  [root_pos_xyz, root_quat_xyzw, joint_pos_29]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert mimic NPZ to g1_ctrl mimic CSV.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to mimic npz file (contains joint_pos/body_pos_w/body_quat_w).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV path. Defaults to <input_stem>_g1_ctrl.csv next to input file.",
    )
    parser.add_argument(
        "--root-body-index",
        type=int,
        default=0,
        help="Body index used as root pose source. Default 0 (pelvis for G1 mimic exports).",
    )
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion order in body_quat_w stored in NPZ. Output is always xyzw.",
    )
    return parser.parse_args()


def _resolve_output(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).resolve()
    return input_path.with_name(f"{input_path.stem}_g1_ctrl.csv")


def _to_xyzw(quat: np.ndarray, order: str) -> np.ndarray:
    if order == "xyzw":
        return quat
    # Input is wxyz -> output xyzw
    return quat[:, [1, 2, 3, 0]]


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    data = np.load(input_path)
    required = ("joint_pos", "body_pos_w", "body_quat_w")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {input_path}: {missing}")

    joint_pos = data["joint_pos"].astype(np.float32)
    body_pos = data["body_pos_w"].astype(np.float32)
    body_quat = data["body_quat_w"].astype(np.float32)

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
        raise ValueError(f"Expected joint_pos shape [T,29], got {joint_pos.shape}")
    if body_pos.ndim != 3 or body_pos.shape[2] != 3:
        raise ValueError(f"Expected body_pos_w shape [T,B,3], got {body_pos.shape}")
    if body_quat.ndim != 3 or body_quat.shape[2] != 4:
        raise ValueError(f"Expected body_quat_w shape [T,B,4], got {body_quat.shape}")
    if body_pos.shape[0] != joint_pos.shape[0] or body_quat.shape[0] != joint_pos.shape[0]:
        raise ValueError("Frame count mismatch among joint_pos/body_pos_w/body_quat_w")
    if args.root_body_index < 0 or args.root_body_index >= body_pos.shape[1]:
        raise IndexError(
            f"root-body-index {args.root_body_index} out of bounds for body count {body_pos.shape[1]}"
        )

    root_pos = body_pos[:, args.root_body_index, :]
    root_quat = _to_xyzw(body_quat[:, args.root_body_index, :], args.quat_order)
    out = np.concatenate([root_pos, root_quat, joint_pos], axis=1)

    output_path = _resolve_output(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, out, delimiter=",", fmt="%.6f")

    print(f"Wrote {out.shape[0]} frames to: {output_path}")
    print(f"CSV columns: {out.shape[1]} (expected 36)")


if __name__ == "__main__":
    main()
