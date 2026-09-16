from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from unitree_rl_lab.utils.motion_segmentation.types import NormalizedMotion, SegmentLabel


def _segment_payload(motion: NormalizedMotion, segment: SegmentLabel) -> dict:
    start_s = float(segment.start_frame / max(motion.fps, 1e-6))
    end_s = float(segment.end_frame / max(motion.fps, 1e-6))
    return {
        **segment.to_dict(),
        "start_time_s": start_s,
        "end_time_s": end_s,
        "duration_s": float((segment.end_frame - segment.start_frame + 1) / max(motion.fps, 1e-6)),
    }


def write_manifest(manifest_path: str | Path, motion: NormalizedMotion, segments: list[SegmentLabel]) -> Path:
    p = Path(manifest_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": str(motion.source_path),
        "schema": motion.schema,
        "fps": float(motion.fps),
        "num_frames": int(motion.num_frames),
        "segments": [_segment_payload(motion, seg) for seg in segments],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def read_manifest(manifest_path: str | Path) -> dict:
    p = Path(manifest_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_name(text: str) -> str:
    clean = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in text.strip())
    return clean or "segment"


def export_segment_clips(
    motion: NormalizedMotion,
    segments: list[SegmentLabel],
    output_dir: str | Path,
    keep_schema: bool = True,
) -> list[Path]:
    """Export clips as npz files.

    Notes:
    - `keep_schema=True` preserves source schema.
    - For HOI qpos source, preserve mode slices `qpos` and `fps`.
    - For mimic source, preserve mode slices all arrays whose first dimension equals T.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if not keep_schema:
        raise ValueError("Only keep_schema=True is supported in v1 export.")

    for i, seg in enumerate(segments):
        sliced = motion.with_slice(seg.start_frame, seg.end_frame)
        label_name = _sanitize_name(seg.label)
        stem = f"{motion.source_path.stem}__{i:02d}_{label_name}"
        out_path = out_dir / f"{stem}.npz"

        if motion.schema == "hoi_qpos":
            qpos = sliced.raw_arrays.get("qpos")
            if qpos is None:
                # fallback reconstruction for HOI layout
                if sliced.object_pos is not None and sliced.object_quat_wxyz is not None:
                    qpos = np.concatenate(
                        [sliced.root_quat_wxyz, sliced.root_pos, sliced.joint_pos, sliced.object_quat_wxyz, sliced.object_pos],
                        axis=1,
                    )
                else:
                    qpos = np.concatenate([sliced.root_quat_wxyz, sliced.root_pos, sliced.joint_pos], axis=1)
            np.savez(out_path, qpos=qpos.astype(np.float32), fps=np.asarray([sliced.fps], dtype=np.float32))
        else:
            arrays: dict[str, np.ndarray] = {}
            for k, v in sliced.raw_arrays.items():
                arr = np.asarray(v)
                if arr.ndim > 0 and arr.shape[0] == sliced.num_frames:
                    arrays[k] = arr
                elif k == "fps":
                    arrays[k] = np.asarray([sliced.fps], dtype=np.float32)
                else:
                    arrays[k] = arr
            # Ensure primary arrays exist after slicing.
            arrays.setdefault("joint_pos", sliced.joint_pos.astype(np.float32))
            arrays.setdefault("joint_vel", sliced.joint_vel.astype(np.float32))
            arrays.setdefault("body_pos_w", sliced.body_pos_w.astype(np.float32) if sliced.body_pos_w is not None else None)
            arrays.setdefault(
                "body_quat_w", sliced.body_quat_w.astype(np.float32) if sliced.body_quat_w is not None else None
            )
            arrays.setdefault(
                "body_lin_vel_w",
                sliced.body_lin_vel_w.astype(np.float32) if sliced.body_lin_vel_w is not None else None,
            )
            arrays.setdefault(
                "body_ang_vel_w",
                sliced.body_ang_vel_w.astype(np.float32) if sliced.body_ang_vel_w is not None else None,
            )
            arrays = {k: v for k, v in arrays.items() if v is not None}
            np.savez(out_path, **arrays)

        written.append(out_path)
    return written

