"""CLI: compute ``mjcf_boxes`` from a HOI terrain URDF (mesh collision + OBJ AABB) for ``*.terrain.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyHOIsim2sim.scene_builder import compute_mjcf_boxes_from_urdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export mjcf_boxes (MuJoCo half-extents, wxyz, positions in terrain body frame) from URDF."
    )
    parser.add_argument("--urdf", type=Path, required=True, help="Terrain URDF path.")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Added to each half-axis (same as PYHOI_TERRAIN_BOX_MARGIN).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output terrain JSON path; use --merge to patch an existing file.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge mjcf_boxes into existing --out-json (keeps terrain_init_pos/quat and other keys).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print mjcf_boxes to stdout when writing --out-json.",
    )
    args = parser.parse_args()

    boxes = compute_mjcf_boxes_from_urdf(args.urdf.resolve(), args.margin)
    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        if args.merge:
            if not out_path.is_file():
                raise FileNotFoundError(f"--merge requires an existing terrain JSON: {out_path}")
            meta = json.loads(out_path.read_text(encoding="utf-8"))
            meta["mjcf_boxes"] = boxes
            out_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"[export_terrain_boxes] Merged mjcf_boxes into {out_path}", file=sys.stderr)
        else:
            payload = {"mjcf_boxes": boxes}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"[export_terrain_boxes] Wrote {out_path}", file=sys.stderr)
    if args.stdout or args.out_json is None:
        print(json.dumps(boxes, indent=2))


if __name__ == "__main__":
    main()
