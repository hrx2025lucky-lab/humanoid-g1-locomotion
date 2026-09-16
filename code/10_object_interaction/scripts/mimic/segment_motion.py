"""Lightweight segmentation tool for HOI qpos and mimic motion files.

Examples:
    python unitree_rl_lab/scripts/mimic/segment_motion.py view \
      -f motion_dataset/HOI/robot-terrain/climb_15_z_scale_1.0.npz

    python unitree_rl_lab/scripts/mimic/segment_motion.py label \
      -f motion_dataset/HOI/robot-terrain/climb_15_z_scale_1.0.npz \
      --output-dir /home/sustech/unitree_project/logs/hoi_mimic_data/climb_15_segments_raw

    python unitree_rl_lab/scripts/mimic/segment_motion.py export \
      -f /home/sustech/unitree_project/logs/hoi_mimic_data/climb_15_z_scale_1.0_mimic.npz \
      --manifest /home/sustech/unitree_project/logs/hoi_mimic_data/climb_15_z_scale_1.0.segments.json \
      --output-dir /home/sustech/unitree_project/logs/hoi_mimic_data/climb_15_segments_mimic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPO_ROOT / "source" / "unitree_rl_lab"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from unitree_rl_lab.utils.motion_segmentation.export import export_segment_clips, read_manifest, write_manifest
from unitree_rl_lab.utils.motion_segmentation.io import load_motion_npz
from unitree_rl_lab.utils.motion_segmentation.types import SegmentLabel
from unitree_rl_lab.utils.motion_segmentation.viser_ui import run_labeling_session, run_motion_viewer


def _default_manifest_path(motion_path: Path, output_dir: Path | None) -> Path:
    base_dir = output_dir if output_dir is not None else motion_path.parent
    return base_dir / f"{motion_path.stem}.segments.json"


def _load_segments_from_manifest(manifest_path: Path) -> list[SegmentLabel]:
    payload = read_manifest(manifest_path)
    items = payload.get("segments", [])
    out: list[SegmentLabel] = []
    for i, item in enumerate(items):
        out.append(
            SegmentLabel(
                segment_id=str(item.get("segment_id", f"seg_{i:03d}")),
                label=str(item.get("label", "segment")),
                start_frame=int(item["start_frame"]),
                end_frame=int(item["end_frame"]),
                notes=str(item.get("notes", "")),
            )
        )
    return out


def cmd_view(args: argparse.Namespace) -> None:
    motion = load_motion_npz(args.file)
    run_motion_viewer(
        motion,
        host=args.host,
        port=args.port,
        title="HOI Motion Viewer",
    )


def cmd_label(args: argparse.Namespace) -> None:
    motion = load_motion_npz(args.file)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else _default_manifest_path(Path(args.file).expanduser().resolve(), output_dir)
    )
    initial_segments = []
    if manifest_path.is_file():
        initial_segments = _load_segments_from_manifest(manifest_path)

    segments = run_labeling_session(
        motion,
        initial_segments=initial_segments,
        host=args.host,
        port=args.port,
        title="HOI Motion Segmentation",
    )

    written_manifest = write_manifest(manifest_path, motion, segments)
    print(f"[INFO] Wrote manifest: {written_manifest}")

    if args.export_clips:
        if output_dir is None:
            output_dir = written_manifest.parent / f"{Path(args.file).stem}_segments"
        written = export_segment_clips(motion, segments, output_dir, keep_schema=True)
        print(f"[INFO] Exported {len(written)} segment clip(s) into: {output_dir}")
        for p in written:
            print(f"  - {p}")


def cmd_export(args: argparse.Namespace) -> None:
    motion = load_motion_npz(args.file)
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    segments = _load_segments_from_manifest(manifest_path)
    written = export_segment_clips(motion, segments, output_dir, keep_schema=True)
    print(f"[INFO] Exported {len(written)} segment clip(s) into: {output_dir}")
    for p in written:
        print(f"  - {p}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment HOI/mimic motion files with a lightweight Viser UI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_view = subparsers.add_parser("view", help="Open viewer for timeline inspection.")
    parser_view.add_argument("--file", "-f", required=True, type=str, help="Path to HOI qpos or mimic npz.")
    parser_view.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host.")
    parser_view.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser_view.set_defaults(func=cmd_view)

    parser_label = subparsers.add_parser("label", help="Interactive segment labeling UI.")
    parser_label.add_argument("--file", "-f", required=True, type=str, help="Path to HOI qpos or mimic npz.")
    parser_label.add_argument("--manifest", type=str, default=None, help="Optional manifest output path.")
    parser_label.add_argument("--output-dir", type=str, default=None, help="Optional clip output directory.")
    parser_label.add_argument(
        "--export-clips",
        action="store_true",
        help="Export segmented clips immediately after finishing labeling.",
    )
    parser_label.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host.")
    parser_label.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser_label.set_defaults(func=cmd_label)

    parser_export = subparsers.add_parser("export", help="Export clip files from an existing manifest.")
    parser_export.add_argument("--file", "-f", required=True, type=str, help="Path to source HOI qpos or mimic npz.")
    parser_export.add_argument("--manifest", required=True, type=str, help="Path to segments JSON manifest.")
    parser_export.add_argument("--output-dir", required=True, type=str, help="Directory to write clip npz files.")
    parser_export.set_defaults(func=cmd_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

