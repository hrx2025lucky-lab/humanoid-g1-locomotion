from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from unitree_rl_lab.utils.motion_segmentation.kinematics import compute_motion_features
from unitree_rl_lab.utils.motion_segmentation.types import NormalizedMotion, SegmentLabel


def _require_viser():
    try:
        import viser  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Viser is required for motion segmentation UI. Install with `pip install viser`."
        ) from exc
    return viser


def _attach_click(handler, callback):
    if hasattr(handler, "on_click"):
        handler.on_click(callback)
        return
    if hasattr(handler, "on_update"):
        handler.on_update(callback)
        return
    raise AttributeError("GUI handler does not support click/update callbacks.")


def _attach_update(handler, callback):
    if hasattr(handler, "on_update"):
        handler.on_update(callback)
        return
    raise AttributeError("GUI handler does not support update callbacks.")


def _add_markdown(gui_api, name: str, content: str):
    """Compat wrapper across viser versions.

    Newer versions: add_markdown(content)
    Older versions: add_markdown(name, initial_value=content)
    """
    try:
        return gui_api.add_markdown(content)
    except TypeError:
        return gui_api.add_markdown(name, initial_value=content)


def _set_markdown(handle, content: str) -> None:
    if hasattr(handle, "content"):
        handle.content = content
    else:
        handle.value = content


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    rot = Rotation.from_quat(quat_xyzw)
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot.as_matrix()
    return tf


def _matrix_to_pose(tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = Rotation.from_matrix(tf[:3, :3])
    q_xyzw = rot.as_quat()
    q_wxyz = np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    pos = np.asarray(tf[:3, 3], dtype=np.float64)
    return pos, q_wxyz


class _UrdfMeshActor:
    """Render a URDF scene as Viser meshes and update by cfg/base pose."""

    def __init__(self, server, urdf_path: str, node_prefix: str):
        from yourdfpy import URDF

        self.urdf = URDF.load(urdf_path)
        self.node_prefix = node_prefix
        self.handles: dict[str, object] = {}
        self.node_geom_names: list[tuple[str, str]] = []

        scene = self.urdf.scene
        for node_name in scene.graph.nodes:
            if node_name not in scene.graph.nodes_geometry:
                continue
            tf_local, geom_name = scene.graph[node_name]
            mesh = scene.geometry[geom_name]
            handle_name = f"{self.node_prefix}/{node_name}"
            handle = server.scene.add_mesh_trimesh(
                handle_name,
                mesh=mesh,
                position=tf_local[:3, 3],
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            self.handles[node_name] = handle
            self.node_geom_names.append((node_name, geom_name))

        self.actuated_joint_names = list(getattr(self.urdf, "actuated_joint_names", []))

    def update(self, *, joint_values: np.ndarray | None = None, base_pos: np.ndarray | None = None, base_quat_wxyz: np.ndarray | None = None):
        if joint_values is not None and len(self.actuated_joint_names) > 0:
            n = min(len(joint_values), len(self.actuated_joint_names))
            cfg = {self.actuated_joint_names[i]: float(joint_values[i]) for i in range(n)}
            self.urdf.update_cfg(cfg)

        base_tf = np.eye(4, dtype=np.float64)
        if base_quat_wxyz is not None:
            base_tf = _quat_wxyz_to_matrix(np.asarray(base_quat_wxyz, dtype=np.float64))
        if base_pos is not None:
            base_tf[:3, 3] = np.asarray(base_pos, dtype=np.float64)

        scene = self.urdf.scene
        for node_name, _geom_name in self.node_geom_names:
            tf_local, _ = scene.graph[node_name]
            tf_world = base_tf @ tf_local
            pos, quat_wxyz = _matrix_to_pose(tf_world)
            handle = self.handles[node_name]
            handle.position = pos
            handle.wxyz = quat_wxyz


@dataclass
class _SessionState:
    current_frame: int = 0
    marked_start: int | None = None
    marked_end: int | None = None
    playback_accum_s: float = 0.0


def _format_segments(segments: list[SegmentLabel], fps: float) -> str:
    if not segments:
        return "No segments saved yet."
    lines: list[str] = []
    for seg in segments:
        duration = (seg.end_frame - seg.start_frame + 1) / max(fps, 1e-6)
        lines.append(
            f"- `{seg.segment_id}` `{seg.label}`: frames [{seg.start_frame}, {seg.end_frame}] ({duration:.2f}s)"
        )
    return "\n".join(lines)


def run_motion_viewer(
    motion: NormalizedMotion,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    title: str = "HOI Motion Viewer",
) -> None:
    """Launch a lightweight Viser viewer for timeline inspection."""
    _run_session(motion, host=host, port=port, title=title, editable=False)


def run_labeling_session(
    motion: NormalizedMotion,
    *,
    initial_segments: list[SegmentLabel] | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    title: str = "HOI Motion Segmentation",
) -> list[SegmentLabel]:
    """Launch manual timeline labeling UI and return saved segments."""
    return _run_session(
        motion,
        host=host,
        port=port,
        title=title,
        editable=True,
        initial_segments=initial_segments or [],
    )


def _run_session(
    motion: NormalizedMotion,
    *,
    host: str,
    port: int,
    title: str,
    editable: bool,
    initial_segments: list[SegmentLabel] | None = None,
) -> list[SegmentLabel]:
    viser = _require_viser()
    server = viser.ViserServer(host=host, port=port, verbose=False)
    state = _SessionState()
    segments: list[SegmentLabel] = list(initial_segments or [])
    features = compute_motion_features(motion)

    frame_slider = server.gui.add_slider(
        "frame",
        min=0,
        max=max(motion.num_frames - 1, 0),
        step=1,
        initial_value=0,
    )
    play_toggle = server.gui.add_checkbox("play", initial_value=False)
    fps_slider = server.gui.add_slider("playback_fps", min=1, max=240, step=1, initial_value=max(1, int(motion.fps)))
    summary_md = _add_markdown(
        server.gui,
        "summary",
        (
            f"### {title}\n"
            f"- schema: `{motion.schema}`\n"
            f"- frames: `{motion.num_frames}`\n"
            f"- fps: `{motion.fps:.3f}`\n"
            f"- source: `{motion.source_path}`"
        ),
    )
    _ = summary_md  # keep handle alive

    if editable:
        label_text = server.gui.add_text("label", initial_value="segment")
        notes_text = server.gui.add_text("notes", initial_value="")
        mark_start_btn = server.gui.add_button("mark_start")
        mark_end_btn = server.gui.add_button("mark_end")
        save_btn = server.gui.add_button("save_segment")
        delete_btn = server.gui.add_button("delete_last_segment")
        done_btn = server.gui.add_button("finish_and_return")
    else:
        label_text = None
        notes_text = None
        mark_start_btn = None
        mark_end_btn = None
        save_btn = None
        delete_btn = None
        done_btn = server.gui.add_button("close_session")

    markers_md = _add_markdown(server.gui, "markers", "start: `-`  end: `-`")
    segments_md = _add_markdown(server.gui, "segments", _format_segments(segments, motion.fps))
    features_md = _add_markdown(server.gui, "frame_features", "frame feature values appear here")

    robot_actor = None
    terrain_actor = None
    object_actor = None
    assets = motion.extra_metadata.get("assets", {}) if isinstance(motion.extra_metadata, dict) else {}
    robot_urdf = assets.get("robot_urdf")
    terrain_urdf = assets.get("terrain_urdf")
    object_urdf = assets.get("object_urdf")
    try:
        if robot_urdf:
            robot_actor = _UrdfMeshActor(server, robot_urdf, "/urdf/robot")
        if terrain_urdf:
            terrain_actor = _UrdfMeshActor(server, terrain_urdf, "/urdf/terrain")
        if object_urdf and motion.object_pos is not None and motion.object_quat_wxyz is not None:
            object_actor = _UrdfMeshActor(server, object_urdf, "/urdf/object")
    except Exception as exc:
        print(f"[WARN] URDF mesh rendering disabled: {exc}")

    root_handle = server.scene.add_frame(
        "/motion/root",
        position=motion.root_pos[0],
        wxyz=motion.root_quat_wxyz[0],
    )
    object_handle = None
    if motion.object_pos is not None and motion.object_quat_wxyz is not None:
        object_handle = server.scene.add_frame(
            "/motion/object",
            position=motion.object_pos[0],
            wxyz=motion.object_quat_wxyz[0],
        )
    body_pc = None
    if motion.body_pos_w is not None:
        colors = np.tile(np.asarray([[0.2, 0.7, 1.0]], dtype=np.float32), (motion.body_pos_w.shape[1], 1))
        body_pc = server.scene.add_point_cloud(
            "/motion/body_points",
            points=motion.body_pos_w[0],
            colors=colors,
            point_size=0.02,
        )
    if terrain_actor is not None:
        terrain_actor.update()

    done_state = {"done": False}

    def refresh_markers():
        start_txt = "-" if state.marked_start is None else str(state.marked_start)
        end_txt = "-" if state.marked_end is None else str(state.marked_end)
        _set_markdown(markers_md, f"start: `{start_txt}`  end: `{end_txt}`")

    def refresh_features(frame: int):
        frame = int(np.clip(frame, 0, motion.num_frames - 1))
        _set_markdown(
            features_md,
            (
            f"frame `{frame}` / `{motion.num_frames - 1}`\n\n"
            f"- t: `{features['time_s'][frame]:.3f}s`\n"
            f"- base_z: `{features['base_height'][frame]:.3f}`\n"
            f"- root_speed: `{features['root_speed'][frame]:.3f}`\n"
            f"- yaw_rate: `{features['yaw_rate'][frame]:.3f}`\n"
            f"- joint_speed: `{features['joint_speed'][frame]:.3f}`"
            ),
        )

    def render_frame(frame: int):
        frame = int(np.clip(frame, 0, motion.num_frames - 1))
        state.current_frame = frame
        root_handle.position = motion.root_pos[frame]
        root_handle.wxyz = motion.root_quat_wxyz[frame]
        if object_handle is not None and motion.object_pos is not None and motion.object_quat_wxyz is not None:
            object_handle.position = motion.object_pos[frame]
            object_handle.wxyz = motion.object_quat_wxyz[frame]
        if body_pc is not None and motion.body_pos_w is not None:
            body_pc.points = motion.body_pos_w[frame]
        if robot_actor is not None:
            robot_actor.update(
                joint_values=motion.joint_pos[frame],
                base_pos=motion.root_pos[frame],
                base_quat_wxyz=motion.root_quat_wxyz[frame],
            )
        if object_actor is not None and motion.object_pos is not None and motion.object_quat_wxyz is not None:
            object_actor.update(
                base_pos=motion.object_pos[frame],
                base_quat_wxyz=motion.object_quat_wxyz[frame],
            )
        refresh_features(frame)

    def on_frame_update(_event=None):
        frame = int(frame_slider.value)
        render_frame(frame)

    _attach_update(frame_slider, on_frame_update)

    if editable and mark_start_btn is not None:
        def on_mark_start(_event=None):
            state.marked_start = int(frame_slider.value)
            refresh_markers()

        def on_mark_end(_event=None):
            state.marked_end = int(frame_slider.value)
            refresh_markers()

        def on_save(_event=None):
            if state.marked_start is None or state.marked_end is None:
                return
            lo = min(state.marked_start, state.marked_end)
            hi = max(state.marked_start, state.marked_end)
            seg = SegmentLabel(
                segment_id=f"seg_{len(segments):03d}",
                label=str(label_text.value).strip() if label_text is not None else "segment",
                start_frame=lo,
                end_frame=hi,
                notes=str(notes_text.value).strip() if notes_text is not None else "",
            )
            segments.append(seg)
            _set_markdown(segments_md, _format_segments(segments, motion.fps))

        def on_delete_last(_event=None):
            if segments:
                segments.pop()
                _set_markdown(segments_md, _format_segments(segments, motion.fps))

        _attach_click(mark_start_btn, on_mark_start)
        _attach_click(mark_end_btn, on_mark_end)
        _attach_click(save_btn, on_save)
        _attach_click(delete_btn, on_delete_last)

    def on_done(_event=None):
        done_state["done"] = True

    _attach_click(done_btn, on_done)

    print(f"[INFO] Viser running at http://{host}:{port}")
    print("[INFO] Use GUI controls to inspect and label segments. Ctrl+C also exits.")

    refresh_markers()
    render_frame(0)

    last_t = time.time()
    try:
        while not done_state["done"]:
            now = time.time()
            dt = now - last_t
            last_t = now
            if play_toggle.value:
                state.playback_accum_s += dt
                step_dt = 1.0 / max(float(fps_slider.value), 1e-6)
                advanced = 0
                while state.playback_accum_s >= step_dt:
                    state.playback_accum_s -= step_dt
                    advanced += 1
                if advanced > 0 and motion.num_frames > 0:
                    next_frame = (int(frame_slider.value) + advanced) % motion.num_frames
                    frame_slider.value = int(next_frame)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    return segments

