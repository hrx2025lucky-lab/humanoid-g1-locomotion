"""Browse G1 motion NPZs in Viser with skeleton playback and keep/reject export.

Loads clips via :func:`humanoid_hw6.mdp.motion.library.load_motion_file` (FK when needed).
Skeleton edges follow the Unitree G1 MJCF parent chain from mjlab.
Optional Unitree G1 visual meshes (STL assets from the MJCF) are drawn per
geom and posed each frame via MuJoCo ``mj_forward``.
An XZ ground grid (horizontal reference plane at ``z=0``) matches MuJoCo Z-up.

Paths written to export files are absolute (see ``--help``).

Usage:
  uv run python -m humanoid_hw6.scripts.data.visualize_motion_curate_viser \\
    --motion assets/motions/g1_accad_walk
"""



# uv run python -m humanoid_hw6.scripts.data.visualize_motion_curate_viser \
#   --motion assets/motions/g1_accad_walk/accad_B3___walk1.npz


from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import viser

from humanoid_hw6.mdp.motion.library import (
  LoadedMotionData,
  _joint_qpos_indices,
  load_motion_file,
)

STATE_VERSION = 1


def _collect_npz_paths(motion: Path) -> list[Path]:
  motion = motion.expanduser().resolve()
  if motion.is_file():
    if motion.suffix.lower() != ".npz":
      raise ValueError(f"Expected .npz file, got: {motion}")
    return [motion]
  if motion.is_dir():
    paths = sorted(motion.rglob("*.npz"))
    if not paths:
      raise ValueError(f"No .npz files under directory: {motion}")
    return paths
  raise FileNotFoundError(f"Not a file or directory: {motion}")


def _build_skeleton_edges(
  body_names: tuple[str, ...],
  *,
  model: object,
) -> tuple[np.ndarray, np.ndarray]:
  """Return (parent_col, child_col) indices into ``body_names`` for bone segments."""
  import mujoco

  name_to_idx = {n: i for i, n in enumerate(body_names)}
  pairs: list[tuple[int, int]] = []
  seen: set[tuple[int, int]] = set()

  for child_idx, child_name in enumerate(body_names):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_name)
    if bid < 0:
      print(f"[WARN] Body `{child_name}` not found in G1 MJCF; skipping edge.")
      continue
    parent_bid = int(model.body_parentid[bid])
    found = False
    while parent_bid > 0:
      pname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_bid)
      if pname is None:
        break
      if pname in name_to_idx:
        pi = name_to_idx[pname]
        key = (pi, child_idx)
        if key not in seen:
          seen.add(key)
          pairs.append((pi, child_idx))
        found = True
        break
      parent_bid = int(model.body_parentid[parent_bid])
    if not found:
      print(f"[WARN] No motion ancestor in chain for `{child_name}`; skipping edge.")

  if not pairs:
    raise RuntimeError("No skeleton edges resolved; check body_names vs G1 MJCF.")

  parent_col = np.array([p[0] for p in pairs], dtype=np.int64)
  child_col = np.array([p[1] for p in pairs], dtype=np.int64)
  return parent_col, child_col


def _segment_points_world(
  body_pos_w: np.ndarray,
  frame_idx: int,
  parent_idx: np.ndarray,
  child_idx: np.ndarray,
) -> np.ndarray:
  """Shape (N, 2, 3) line segment endpoints in world frame."""
  pos = body_pos_w[frame_idx]
  starts = pos[parent_idx]
  ends = pos[child_idx]
  return np.stack([starts, ends], axis=1).astype(np.float32)


def _visual_mesh_geom_ids(model: object) -> list[int]:
  """Geom ids for group-2 mesh visuals on the G1 MJCF."""
  import mujoco

  out: list[int] = []
  for gid in range(model.ngeom):
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    if int(model.geom_group[gid]) != 2:
      continue
    out.append(gid)
  return out


def _trimesh_for_mesh_geom(model: object, gid: int) -> object:
  """Build a local-frame mesh for geom ``gid`` (MuJoCo mesh asset vertices)."""
  import trimesh

  mid = int(model.geom_dataid[gid])
  vadr = int(model.mesh_vertadr[mid])
  vn = int(model.mesh_vertnum[mid])
  verts = np.asarray(model.mesh_vert[vadr : vadr + 3 * vn], dtype=np.float64).reshape(
    -1, 3
  )
  fadr = int(model.mesh_faceadr[mid])
  fn = int(model.mesh_facenum[mid])
  faces = np.asarray(model.mesh_face[fadr : fadr + fn], dtype=np.int64)
  return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _mat9_to_wxyz(xmat: np.ndarray) -> tuple[float, float, float, float]:
  from scipy.spatial.transform import Rotation as SciRotation

  m = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
  q_xyzw = SciRotation.from_matrix(m).as_quat()
  return (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def _pelvis_motion_index(payload: LoadedMotionData) -> int:
  try:
    return payload.body_names.index("pelvis")
  except ValueError as e:
    raise ValueError(
      "Robot mesh visualization requires `pelvis` in motion `body_names`."
    ) from e


def _apply_motion_to_mujoco(
  model: object,
  data: object,
  payload: LoadedMotionData,
  frame_idx: int,
  *,
  free_qadr: int,
  joint_q_indices: np.ndarray,
  pelvis_idx: int,
) -> None:
  """Set ``data.qpos`` from motion frame and run ``mj_forward`` (poses mesh geoms)."""
  import mujoco

  n_j = int(joint_q_indices.shape[0])
  if payload.joint_pos.shape[1] != n_j:
    raise ValueError(
      "Motion DoF count does not match G1 MJCF: "
      f"motion={payload.joint_pos.shape[1]} model={n_j}"
    )
  n_frames = int(payload.joint_pos.shape[0])
  fi = int(np.clip(frame_idx, 0, n_frames - 1))
  root_pos = payload.body_pos_w[fi, pelvis_idx].astype(np.float64)
  root_quat = payload.body_quat_w[fi, pelvis_idx].astype(np.float64)
  rn = float(np.linalg.norm(root_quat))
  if rn > 1e-9:
    root_quat = root_quat / rn
  qpos = np.zeros(model.nq, dtype=np.float64)
  qpos[free_qadr : free_qadr + 3] = root_pos
  qpos[free_qadr + 3 : free_qadr + 7] = root_quat
  qpos[joint_q_indices] = payload.joint_pos[fi].astype(np.float64)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)


@dataclass
class ViewerState:
  clip_paths: list[Path]
  clip_idx: int = 0
  frame_idx: int = 0
  playing: bool = False
  speed: float = 1.0
  labels: dict[str, str | None] = field(default_factory=dict)
  payload: LoadedMotionData | None = None
  parent_idx: np.ndarray | None = None
  child_idx: np.ndarray | None = None
  shutdown: threading.Event = field(default_factory=threading.Event)
  lock: threading.Lock = field(default_factory=threading.Lock)


def _load_labels_json(path: Path | None) -> dict[str, str | None]:
  if path is None or not path.is_file():
    return {}
  data = json.loads(path.read_text(encoding="utf-8"))
  raw = data.get("labels", {})
  out: dict[str, str | None] = {}
  for k, v in raw.items():
    key = str(Path(k).expanduser().resolve())
    if v in ("keep", "reject"):
      out[key] = v
    elif v is None:
      out[key] = None
  return out


def _save_labels_json(path: Path, labels: dict[str, str | None]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  serializable = {
    "version": STATE_VERSION,
    "labels": {k: labels[k] for k in sorted(labels.keys())},
  }
  path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _export_lists(export_prefix: Path, labels: dict[str, str | None]) -> None:
  export_prefix.parent.mkdir(parents=True, exist_ok=True)
  stem = export_prefix
  keep_paths = sorted(p for p, lab in labels.items() if lab == "keep")
  reject_paths = sorted(p for p, lab in labels.items() if lab == "reject")
  (stem.parent / f"{stem.name}_keep_paths.txt").write_text(
    "\n".join(keep_paths) + ("\n" if keep_paths else ""),
    encoding="utf-8",
  )
  (stem.parent / f"{stem.name}_reject_paths.txt").write_text(
    "\n".join(reject_paths) + ("\n" if reject_paths else ""),
    encoding="utf-8",
  )
  snapshot = {
    "version": STATE_VERSION,
    "labels": {k: labels[k] for k in sorted(labels.keys())},
  }
  (stem.parent / f"{stem.name}_state.json").write_text(
    json.dumps(snapshot, indent=2),
    encoding="utf-8",
  )
  print(
    f"[INFO] Exported keep/reject lists and state under prefix `{stem}` "
    f"(keep={len(keep_paths)}, reject={len(reject_paths)})."
  )


def _load_clip_into_state(state: ViewerState, mj_model: object) -> None:
  path = state.clip_paths[state.clip_idx]
  payload = load_motion_file(path)
  parent_idx, child_idx = _build_skeleton_edges(payload.body_names, model=mj_model)
  with state.lock:
    state.payload = payload
    state.parent_idx = parent_idx
    state.child_idx = child_idx
    state.frame_idx = 0


def run_viewer(
  *,
  motion: Path,
  state_json: Path | None,
  export_prefix: Path | None,
  export_on_exit: bool,
  host: str,
  port: int,
  bone_color_rgb: tuple[int, int, int],
  line_width: float,
  robot_meshes: bool,
  ground_grid: bool,
  autoplay: bool = True,
) -> None:
  import mujoco
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

  clip_paths = _collect_npz_paths(motion)
  mj_model = mujoco.MjModel.from_xml_path(str(G1_XML))
  mj_data = mujoco.MjData(mj_model)
  free_qadr, joint_q_indices_list = _joint_qpos_indices(mj_model)
  joint_q_indices_arr = np.asarray(joint_q_indices_list, dtype=np.int64)

  labels: dict[str, str | None] = {}
  if state_json is not None:
    labels.update(_load_labels_json(state_json))
  for p in clip_paths:
    key = str(p.resolve())
    labels.setdefault(key, None)

  state = ViewerState(clip_paths=clip_paths, labels=labels)
  _load_clip_into_state(state, mj_model)

  server = viser.ViserServer(host=host, port=port, label="motion curator")
  server.scene.set_up_direction("+z")

  grid_handle = None
  if ground_grid:
    grid_handle = server.scene.add_grid(
      "/world/ground_grid",
      plane="xy",
      infinite_grid=True,
      cell_size=0.25,
      section_size=1.0,
      cell_color=(190, 190, 200),
      section_color=(125, 125, 140),
      plane_color=(248, 248, 252),
      plane_opacity=0.14,
      shadow_opacity=0.2,
      position=(0.0, 0.0, 0.0),
    )

  def bone_colors(n_seg: int) -> np.ndarray:
    base = np.array(bone_color_rgb, dtype=np.uint8).reshape(1, 1, 3)
    return np.broadcast_to(base, (n_seg, 2, 3)).copy()

  assert state.payload is not None and state.parent_idx is not None
  n_seg0 = int(state.parent_idx.shape[0])
  pts0 = _segment_points_world(
    state.payload.body_pos_w,
    0,
    state.parent_idx,
    state.child_idx,
  )
  bone_handle = server.scene.add_line_segments(
    "/skeleton/bones",
    points=pts0,
    colors=bone_colors(n_seg0),
    line_width=line_width,
    visible=True,
  )

  mesh_handles: list[tuple[int, object]] = []
  mesh_draw = {"enabled": robot_meshes}
  if robot_meshes:
    _pelvis_motion_index(state.payload)
    visual_gids = _visual_mesh_geom_ids(mj_model)
    for gid in visual_gids:
      mesh_tm = _trimesh_for_mesh_geom(mj_model, gid)
      gname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "geom"
      safe = "".join(ch if ch.isalnum() else "_" for ch in gname)
      mh = server.scene.add_mesh_trimesh(
        f"/robot/{safe}_{gid}",
        mesh_tm,
        cast_shadow=True,
        receive_shadow=True,
      )
      mesh_handles.append((gid, mh))
    print(f"[INFO] Loaded {len(mesh_handles)} G1 visual mesh geoms (embodiment).")

  status_md = server.gui.add_markdown("Loading…")
  frame_slider = server.gui.add_slider(
    "Frame",
    min=0,
    max=max(0, state.payload.body_pos_w.shape[0] - 1),
    step=1,
    initial_value=0,
  )
  playing_cb = server.gui.add_checkbox("Play", initial_value=autoplay)
  state.playing = autoplay
  speed_slider = server.gui.add_slider(
    "Playback speed",
    min=0.1,
    max=4.0,
    step=0.05,
    initial_value=1.0,
  )

  if ground_grid:
    assert grid_handle is not None

    def on_ground_grid_vis(v: bool) -> None:
      grid_handle.visible = bool(v)

    ground_grid_cb = server.gui.add_checkbox("Ground grid", initial_value=True)
    ground_grid_cb.on_update(on_ground_grid_vis)

  if robot_meshes:

    def on_mesh_vis(v: bool) -> None:
      mesh_draw["enabled"] = bool(v)
      for _, mh in mesh_handles:
        mh.visible = bool(v)

    mesh_vis_cb = server.gui.add_checkbox("Robot meshes", initial_value=True)
    mesh_vis_cb.on_update(on_mesh_vis)

  with server.gui.add_folder("Clip"):
    prev_btn = server.gui.add_button("Previous clip")
    next_btn = server.gui.add_button("Next clip")
    export_btn = server.gui.add_button("Export lists now")

  with server.gui.add_folder("Label"):
    keep_btn = server.gui.add_button("Keep")
    reject_btn = server.gui.add_button("Reject")
    clear_btn = server.gui.add_button("Clear label")

  programmatic_slider = {"active": False}

  def current_label_str() -> str:
    key = str(state.clip_paths[state.clip_idx].resolve())
    lab = state.labels.get(key)
    if lab == "keep":
      return "**keep**"
    if lab == "reject":
      return "**reject**"
    return "(none)"

  def refresh_status() -> None:
    path = state.clip_paths[state.clip_idx]
    with state.lock:
      payload = state.payload
      n_frames = int(payload.body_pos_w.shape[0]) if payload is not None else 0
      fps = float(payload.fps) if payload is not None else 0.0
    idx = state.clip_idx + 1
    n_clips = len(state.clip_paths)
    status_md.content = (
      f"**Clip** {idx}/{n_clips}\n\n`{path.name}`\n\n"
      f"Frames: {n_frames} @ {fps:.2f} Hz\n\n"
      f"Label: {current_label_str()}\n\n"
      f"Paths in export/state files are absolute."
    )

  def apply_frame_to_scene(frame_i: int) -> None:
    with state.lock:
      payload = state.payload
      pi = state.parent_idx
      ci = state.child_idx
    if payload is None or pi is None or ci is None:
      return
    fi = int(np.clip(frame_i, 0, payload.body_pos_w.shape[0] - 1))
    pts = _segment_points_world(payload.body_pos_w, fi, pi, ci)
    with server.atomic():
      bone_handle.points = pts
      if mesh_handles and mesh_draw.get("enabled"):
        pidx = _pelvis_motion_index(payload)
        _apply_motion_to_mujoco(
          mj_model,
          mj_data,
          payload,
          fi,
          free_qadr=free_qadr,
          joint_q_indices=joint_q_indices_arr,
          pelvis_idx=pidx,
        )
        for gid, mh in mesh_handles:
          mh.position = np.asarray(mj_data.geom_xpos[gid], dtype=np.float32)
          mh.wxyz = np.asarray(_mat9_to_wxyz(mj_data.geom_xmat[gid]), dtype=np.float64)

  def refresh_bone_colors_for_clip() -> None:
    with state.lock:
      pi = state.parent_idx
    if pi is None:
      return
    with server.atomic():
      bone_handle.colors = bone_colors(int(pi.shape[0]))

  def sync_slider_bounds() -> None:
    with state.lock:
      payload = state.payload
    if payload is None:
      return
    n_frames = int(payload.body_pos_w.shape[0])
    programmatic_slider["active"] = True
    try:
      frame_slider.min = 0
      frame_slider.max = max(0, n_frames - 1)
      fi = int(np.clip(state.frame_idx, 0, n_frames - 1))
      frame_slider.value = fi
      state.frame_idx = fi
    finally:
      programmatic_slider["active"] = False

  def on_frame_slider(val: float) -> None:
    if programmatic_slider["active"]:
      return
    state.frame_idx = int(val)
    apply_frame_to_scene(state.frame_idx)

  def on_play_toggle(v: bool) -> None:
    state.playing = bool(v)

  def on_speed(val: float) -> None:
    state.speed = float(val)

  frame_slider.on_update(on_frame_slider)
  playing_cb.on_update(on_play_toggle)
  speed_slider.on_update(on_speed)

  def persist_labels() -> None:
    if state_json is not None:
      _save_labels_json(state_json, state.labels)

  def set_label(value: str | None) -> None:
    key = str(state.clip_paths[state.clip_idx].resolve())
    state.labels[key] = value
    persist_labels()
    refresh_status()

  keep_btn.on_click(lambda *_: set_label("keep"))
  reject_btn.on_click(lambda *_: set_label("reject"))
  clear_btn.on_click(lambda *_: set_label(None))

  def go_clip(delta: int) -> None:
    n = len(state.clip_paths)
    state.clip_idx = (state.clip_idx + delta) % n
    _load_clip_into_state(state, mj_model)
    if robot_meshes:
      pl = state.payload
      assert pl is not None
      _pelvis_motion_index(pl)
    sync_slider_bounds()
    refresh_bone_colors_for_clip()
    apply_frame_to_scene(state.frame_idx)
    refresh_status()

  prev_btn.on_click(lambda *_: go_clip(-1))
  next_btn.on_click(lambda *_: go_clip(1))

  def do_export() -> None:
    if export_prefix is None:
      print("[WARN] Export skipped: pass --export-prefix to enable.")
      return
    _export_lists(export_prefix, state.labels)

  export_btn.on_click(lambda *_: do_export())

  refresh_status()
  sync_slider_bounds()
  refresh_bone_colors_for_clip()
  apply_frame_to_scene(state.frame_idx)

  def playback_loop() -> None:
    while not state.shutdown.is_set():
      with state.lock:
        playing = state.playing
        payload = state.payload
        speed = max(state.speed, 0.05)
      if not playing or payload is None:
        time.sleep(0.04)
        continue
      fps = max(float(payload.fps), 1e-3)
      sleep_dt = (1.0 / fps) / speed
      time.sleep(max(1e-4, sleep_dt))
      with state.lock:
        if not state.playing or state.payload is None:
          continue
        n_frames = int(state.payload.body_pos_w.shape[0])
        state.frame_idx = (state.frame_idx + 1) % n_frames
        fi = state.frame_idx
      programmatic_slider["active"] = True
      try:
        frame_slider.value = fi
      finally:
        programmatic_slider["active"] = False
      apply_frame_to_scene(fi)

  thread = threading.Thread(target=playback_loop, daemon=True)
  thread.start()

  try:
    server.sleep_forever()
  finally:
    state.shutdown.set()
    if export_on_exit and export_prefix is not None:
      _export_lists(export_prefix, state.labels)
    elif export_on_exit and export_prefix is None:
      print("[WARN] --export-on-exit ignored: --export-prefix not set.")


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description=(
      "Viser viewer for G1 motion NPZs with skeleton playback and keep/reject curation. "
      "Export files use absolute paths."
    )
  )
  p.add_argument(
    "--motion",
    type=Path,
    default=Path("assets/motions/g1_accad_walk"),
    help="Path to a .npz clip or a directory (recursive *.npz).",
  )
  p.add_argument(
    "--state-json",
    type=Path,
    default=None,
    help="Optional JSON file to load/save labels {abs_path: keep|reject|null}.",
  )
  p.add_argument(
    "--export-prefix",
    type=Path,
    default=None,
    help=(
      "Stem for export files: writes "
      "{prefix}_keep_paths.txt, {prefix}_reject_paths.txt, {prefix}_state.json "
      "(under prefix.parent)."
    ),
  )
  p.add_argument(
    "--export-on-exit",
    action="store_true",
    help="Write export files when the server stops (requires --export-prefix).",
  )
  p.add_argument("--host", type=str, default="0.0.0.0", help="Viser bind host.")
  p.add_argument("--port", type=int, default=8080, help="Viser port.")
  p.add_argument(
    "--bone-color",
    type=int,
    nargs=3,
    default=(80, 160, 255),
    metavar=("R", "G", "B"),
    help="RGB bone color 0-255.",
  )
  p.add_argument("--line-width", type=float, default=3.0, help="Bone line width.")
  p.add_argument(
    "--no-robot-meshes",
    action="store_true",
    help="Disable G1 STL mesh embodiment (skeleton lines only; faster load).",
  )
  p.add_argument(
    "--no-ground-grid",
    action="store_true",
    help="Disable the infinite XZ ground plane / grid (Z-up reference at z=0).",
  )
  p.add_argument(
    "--no-autoplay",
    action="store_true",
    help="Start paused on frame 0 instead of auto-playing.",
  )
  return p


def main() -> None:
  args = _build_argparser().parse_args()
  rgb = (int(args.bone_color[0]), int(args.bone_color[1]), int(args.bone_color[2]))
  run_viewer(
    motion=args.motion,
    state_json=args.state_json,
    export_prefix=args.export_prefix,
    export_on_exit=args.export_on_exit,
    host=args.host,
    port=args.port,
    bone_color_rgb=rgb,
    line_width=float(args.line_width),
    robot_meshes=not args.no_robot_meshes,
    ground_grid=not args.no_ground_grid,
    autoplay=not args.no_autoplay,
  )


if __name__ == "__main__":
  main()
