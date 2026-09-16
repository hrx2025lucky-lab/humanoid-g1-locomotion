from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .config import PyHOIConfig
from .mjcf_primitives import add_box_geom, fmt_vec as _fmt, quat_mul as _quat_mul, quat_rotate as _quat_rotate, rpy_to_quat_wxyz as _rpy_to_quat_wxyz


G1_29DOF_MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

MOTION_FULL_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

IK_ALIGNMENT_BODIES: tuple[str, ...] = (
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
)

FRAME0_KEY_NAME = "pyhoi_frame0"


def _parse_vec(text: str | None, n: int, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.array(default, dtype=np.float64)
    values = [float(x) for x in text.split()]
    if len(values) != n:
        raise ValueError(f"Expected {n} values, got {len(values)} from {text!r}")
    return np.array(values, dtype=np.float64)


def _load_motion_frame0(
    motion_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    motion = np.load(motion_file)
    for key in ("body_pos_w", "body_quat_w", "joint_pos"):
        if key not in motion:
            raise KeyError(f"Motion file {motion_file} missing required key {key!r}")
    if motion["body_pos_w"].shape[1] != len(MOTION_FULL_BODY_NAMES):
        raise ValueError(
            f"body_pos_w has {motion['body_pos_w'].shape[1]} bodies, expected {len(MOTION_FULL_BODY_NAMES)}"
        )
    root_pos = np.asarray(motion["body_pos_w"][0, 0], dtype=np.float64)
    root_quat = np.asarray(motion["body_quat_w"][0, 0], dtype=np.float64)
    joint_pos = np.asarray(motion["joint_pos"][0], dtype=np.float64)
    if root_pos.shape != (3,):
        raise ValueError(f"body_pos_w[0,0] must be shape (3,), got {root_pos.shape}")
    if root_quat.shape != (4,):
        raise ValueError(f"body_quat_w[0,0] must be shape (4,), got {root_quat.shape}")
    if joint_pos.shape != (len(G1_29DOF_MUJOCO_JOINT_NAMES),):
        raise ValueError(
            f"joint_pos[0] must have {len(G1_29DOF_MUJOCO_JOINT_NAMES)} values, got {joint_pos.shape}"
        )
    body_pos_targets = {
        name: np.asarray(motion["body_pos_w"][0, i], dtype=np.float64)
        for i, name in enumerate(MOTION_FULL_BODY_NAMES)
    }
    body_quat_targets = {
        name: np.asarray(motion["body_quat_w"][0, i], dtype=np.float64)
        for i, name in enumerate(MOTION_FULL_BODY_NAMES)
    }
    return root_pos, root_quat, joint_pos, body_pos_targets, body_quat_targets


def _load_flat_scene_init(base_scene: Path, xy_offset: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(base_scene))
    qpos0 = np.asarray(model.qpos0, dtype=np.float64)
    expected_nq = 7 + len(G1_29DOF_MUJOCO_JOINT_NAMES)
    if qpos0.shape[0] < expected_nq:
        raise RuntimeError(f"Flat scene qpos0 has {qpos0.shape[0]} values, expected at least {expected_nq}")
    root_pos = qpos0[:3].copy()
    root_pos[0] += float(xy_offset[0])
    root_pos[1] += float(xy_offset[1])
    root_quat = qpos0[3:7].copy()
    joint_pos = qpos0[7:expected_nq].copy()
    return root_pos, root_quat, joint_pos


def _write_exact_robot_xml(
    base_robot_xml: Path,
    output_robot_xml: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> None:
    tree = ET.parse(base_robot_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"Robot XML missing worldbody: {base_robot_xml}")
    pelvis = worldbody.find("body[@name='pelvis']")
    if pelvis is None:
        raise RuntimeError(f"Robot XML missing pelvis body: {base_robot_xml}")
    pelvis.attrib["pos"] = _fmt(root_pos)
    pelvis.attrib["quat"] = _fmt(root_quat)

    joint_by_name = {joint.attrib.get("name"): joint for joint in root.findall(".//joint")}
    for name, value in zip(G1_29DOF_MUJOCO_JOINT_NAMES, joint_pos, strict=True):
        joint = joint_by_name.get(name)
        if joint is None:
            raise RuntimeError(f"Robot XML missing joint {name!r}: {base_robot_xml}")
        joint.attrib["ref"] = f"{float(value):.9g}"

    output_robot_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_robot_xml, encoding="utf-8", xml_declaration=False)


def _load_terrain_meta_full(meta_file: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    pos = np.asarray(meta.get("terrain_init_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    quat = np.asarray(meta.get("terrain_init_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    if pos.shape != (3,) or quat.shape != (4,):
        raise ValueError(f"Invalid terrain metadata pose in {meta_file}")
    raw_boxes = meta.get("mjcf_boxes")
    boxes: list[dict[str, np.ndarray]] = []
    if raw_boxes is None:
        return pos, quat, boxes
    if not isinstance(raw_boxes, list):
        raise ValueError(f"{meta_file}: mjcf_boxes must be a list, got {type(raw_boxes)}")
    for i, entry in enumerate(raw_boxes):
        if not isinstance(entry, dict):
            raise ValueError(f"{meta_file}: mjcf_boxes[{i}] must be an object")
        p = np.asarray(entry.get("pos"), dtype=np.float64).reshape(3)
        q = np.asarray(entry.get("quat"), dtype=np.float64).reshape(4)
        name = str(entry.get("name", f"mjcf_box_{i}"))
        if "half_size" in entry:
            hs = np.asarray(entry["half_size"], dtype=np.float64).reshape(3)
        elif "full_size" in entry:
            hs = 0.5 * np.asarray(entry["full_size"], dtype=np.float64).reshape(3)
        else:
            raise ValueError(f"{meta_file}: mjcf_boxes[{i}] needs half_size or full_size")
        boxes.append({"name": name, "pos": p, "quat": q, "half_size": hs})
    return pos, quat, boxes


def _load_terrain_meta(meta_file: Path) -> tuple[np.ndarray, np.ndarray]:
    pos, quat, _boxes = _load_terrain_meta_full(meta_file)
    return pos, quat


def compute_mjcf_boxes_from_urdf(terrain_urdf: Path, box_margin: float) -> list[dict[str, list[float]]]:
    """Compute MuJoCo box geoms (in URDF body frame, before terrain_init pose) from mesh collision entries."""
    urdf_root = ET.parse(terrain_urdf).getroot()
    child_poses = _fixed_joint_child_poses(urdf_root)
    out: list[dict[str, list[float]]] = []
    mesh_count = 0
    for link in urdf_root.findall("link"):
        link_name = link.attrib.get("name", f"link_{mesh_count}")
        link_pos, link_quat = child_poses.get(
            link_name, (np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        )
        for geom_source in link.findall("collision"):
            origin = geom_source.find("origin")
            geom_pos = _parse_vec(origin.attrib.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0))
            geom_rpy = _parse_vec(origin.attrib.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0))
            mesh = geom_source.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename")
            if not filename:
                continue
            mesh_path = (terrain_urdf.parent / filename).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Terrain mesh not found: {mesh_path}")
            scale = _parse_vec(mesh.attrib.get("scale"), 3, (1.0, 1.0, 1.0))
            geom_name = f"pyhoi_terrain_geom_{mesh_count}"
            geom_quat = _quat_mul(link_quat, _rpy_to_quat_wxyz(geom_rpy))
            geom_base_pos = link_pos + geom_pos
            aabb_min, aabb_max = _obj_mesh_aabb(mesh_path, scale)
            box_center = 0.5 * (aabb_min + aabb_max)
            half = 0.5 * (aabb_max - aabb_min) + float(box_margin)
            half = np.maximum(half, 1e-6)
            pos = geom_base_pos + _quat_rotate(geom_quat, box_center)
            out.append(
                {
                    "name": f"{geom_name}_collision",
                    "pos": pos.astype(float).tolist(),
                    "quat": geom_quat.astype(float).tolist(),
                    "half_size": half.astype(float).tolist(),
                }
            )
            mesh_count += 1
    if mesh_count == 0:
        raise RuntimeError(f"No mesh collision geometry found in terrain URDF: {terrain_urdf}")
    return out


def _insert_mjcf_boxes_from_meta(
    terrain_body: ET.Element,
    mjcf_boxes: list[dict[str, np.ndarray]],
    contact_attrs: dict[str, str],
) -> int:
    for box in mjcf_boxes:
        add_box_geom(
            terrain_body,
            name=str(box["name"]),
            pos=box["pos"],
            quat_wxyz=box["quat"],
            half_size=box["half_size"],
            rgba="0.55 0.55 0.55 0.35",
            contact=True,
            extra=contact_attrs or None,
        )
    return len(mjcf_boxes)


def _fixed_joint_child_poses(urdf_root: ET.Element) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for joint in urdf_root.findall("joint"):
        if joint.attrib.get("type") != "fixed":
            continue
        child = joint.find("child")
        if child is None:
            continue
        child_name = child.attrib.get("link")
        if not child_name:
            continue
        origin = joint.find("origin")
        xyz = _parse_vec(origin.attrib.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0))
        rpy = _parse_vec(origin.attrib.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0))
        poses[child_name] = (xyz, _rpy_to_quat_wxyz(rpy))
    return poses


def _terrain_contact_attrs(cfg: PyHOIConfig) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if cfg.terrain_friction is not None:
        attrs["friction"] = _fmt(cfg.terrain_friction)
    if cfg.terrain_condim is not None:
        attrs["condim"] = str(cfg.terrain_condim)
    if cfg.terrain_solref is not None:
        attrs["solref"] = _fmt(cfg.terrain_solref)
    if cfg.terrain_solimp is not None:
        attrs["solimp"] = _fmt(cfg.terrain_solimp)
    if cfg.terrain_priority is not None:
        attrs["priority"] = str(cfg.terrain_priority)
    return attrs


def _obj_mesh_aabb(mesh_path: Path, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    for line in mesh_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise RuntimeError(f"OBJ mesh has no vertices for AABB terrain collision: {mesh_path}")
    verts = np.asarray(vertices, dtype=np.float64) * scale
    return np.min(verts, axis=0), np.max(verts, axis=0)


def _insert_static_terrain(scene_root: ET.Element, cfg: PyHOIConfig) -> int:
    terrain_urdf = cfg.terrain_urdf
    terrain_meta_file = cfg.terrain_meta_file
    asset = scene_root.find("asset")
    worldbody = scene_root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("Base scene missing asset/worldbody.")

    for child in list(worldbody):
        if child.tag == "body" and child.attrib.get("name") == "hoi_terrain_exact":
            worldbody.remove(child)

    terrain_pos, terrain_quat, mjcf_boxes = _load_terrain_meta_full(terrain_meta_file)
    terrain_body = ET.SubElement(worldbody, "body", name="hoi_terrain_exact")
    terrain_body.attrib["pos"] = _fmt(terrain_pos)
    terrain_body.attrib["quat"] = _fmt(terrain_quat)
    contact_attrs = _terrain_contact_attrs(cfg)
    if cfg.terrain_collision_mode == "box":
        print("[pyHOI] Terrain collision mode: box (mesh visual + primitive box collision).")
    else:
        print("[pyHOI] Terrain collision mode: mesh.")
    if contact_attrs:
        print(f"[pyHOI] Terrain contact attrs: {contact_attrs}")

    if mjcf_boxes and cfg.terrain_collision_mode == "mesh":
        print(
            "[pyHOI][Warn] terrain.json defines mjcf_boxes but PYHOI_TERRAIN_COLLISION_MODE=mesh; "
            "ignoring mjcf_boxes and using URDF mesh collision."
        )
    if mjcf_boxes and cfg.terrain_collision_mode == "box":
        print(f"[pyHOI] Using {len(mjcf_boxes)} box(es) from terrain.json mjcf_boxes (skipping URDF mesh assets).")
        return _insert_mjcf_boxes_from_meta(terrain_body, mjcf_boxes, contact_attrs)

    urdf_root = ET.parse(terrain_urdf).getroot()
    child_poses = _fixed_joint_child_poses(urdf_root)
    mesh_count = 0
    for link in urdf_root.findall("link"):
        link_name = link.attrib.get("name", f"link_{mesh_count}")
        link_pos, link_quat = child_poses.get(
            link_name, (np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        )
        for geom_source in link.findall("collision"):
            origin = geom_source.find("origin")
            geom_pos = _parse_vec(origin.attrib.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0))
            geom_rpy = _parse_vec(origin.attrib.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0))
            mesh = geom_source.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename")
            if not filename:
                continue
            mesh_path = (terrain_urdf.parent / filename).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Terrain mesh not found: {mesh_path}")
            scale = _parse_vec(mesh.attrib.get("scale"), 3, (1.0, 1.0, 1.0))
            mesh_name = f"pyhoi_terrain_mesh_{mesh_count}"
            geom_name = f"pyhoi_terrain_geom_{mesh_count}"
            ET.SubElement(asset, "mesh", name=mesh_name, file=str(mesh_path), scale=_fmt(scale))
            geom_quat = _quat_mul(link_quat, _rpy_to_quat_wxyz(geom_rpy))
            geom_base_pos = link_pos + geom_pos
            if cfg.terrain_collision_mode == "mesh":
                ET.SubElement(
                    terrain_body,
                    "geom",
                    name=geom_name,
                    type="mesh",
                    mesh=mesh_name,
                    pos=_fmt(geom_base_pos),
                    quat=_fmt(geom_quat),
                    rgba="0.55 0.55 0.55 1",
                    **contact_attrs,
                )
            else:
                ET.SubElement(
                    terrain_body,
                    "geom",
                    name=f"{geom_name}_visual",
                    type="mesh",
                    mesh=mesh_name,
                    pos=_fmt(geom_base_pos),
                    quat=_fmt(geom_quat),
                    rgba="0.55 0.55 0.55 1",
                    contype="0",
                    conaffinity="0",
                    group="1",
                )
                aabb_min, aabb_max = _obj_mesh_aabb(mesh_path, scale)
                box_center = 0.5 * (aabb_min + aabb_max)
                half_size = 0.5 * (aabb_max - aabb_min) + float(cfg.terrain_box_margin)
                half_size = np.maximum(half_size, 1e-6)
                pos = geom_base_pos + _quat_rotate(geom_quat, box_center)
                add_box_geom(
                    terrain_body,
                    name=f"{geom_name}_collision",
                    pos=pos,
                    quat_wxyz=geom_quat,
                    half_size=half_size,
                    rgba="0.55 0.55 0.55 0.15",
                    contact=True,
                    extra=contact_attrs if contact_attrs else None,
                )
            mesh_count += 1
    if mesh_count == 0:
        raise RuntimeError(f"No mesh collision geometry found in terrain URDF: {terrain_urdf}")
    return mesh_count


def _apply_floor_collision_mode(scene_root: ET.Element, cfg: PyHOIConfig) -> None:
    if cfg.floor_collision:
        return
    worldbody = scene_root.find("worldbody")
    if worldbody is None:
        return
    for geom in worldbody.findall("geom"):
        if geom.attrib.get("name") == "floor":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            print("[pyHOI] Floor collision disabled (checker visual only; PYHOI_FLOOR_COLLISION=0).")
            return
    print("[pyHOI][Warn] No worldbody geom name='floor'; floor collision toggle skipped.")


def _write_scene(cfg: PyHOIConfig) -> int:
    tree = ET.parse(cfg.base_scene)
    root = tree.getroot()
    include = root.find("include")
    if include is None:
        include = ET.Element("include")
        root.insert(0, include)
    include.attrib["file"] = cfg.generated_robot_xml.name
    mesh_count = _insert_static_terrain(root, cfg)
    _apply_floor_collision_mode(root, cfg)
    cfg.generated_scene.parent.mkdir(parents=True, exist_ok=True)
    tree.write(cfg.generated_scene, encoding="utf-8", xml_declaration=False)
    return mesh_count


def _write_frame0_keyframe(scene_xml: Path, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> None:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for key in list(keyframe):
        if key.tag == "key" and key.attrib.get("name") == FRAME0_KEY_NAME:
            keyframe.remove(key)

    qpos = np.concatenate((root_pos, root_quat, joint_pos))
    ET.SubElement(keyframe, "key", name=FRAME0_KEY_NAME, qpos=_fmt(qpos))
    tree.write(scene_xml, encoding="utf-8", xml_declaration=False)


def _apply_frame0_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, FRAME0_KEY_NAME)
    if key_id < 0:
        return False
    data.qpos[:] = model.key_qpos[key_id]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return True


def _init_solver_mode() -> str:
    mode = os.getenv("PYHOI_INIT_SOLVER", "ik").strip().lower()
    if mode not in {"direct", "ik"}:
        raise ValueError(f"Unsupported PYHOI_INIT_SOLVER={mode!r}; expected 'direct' or 'ik'.")
    return mode


def _init_root_offset() -> np.ndarray:
    raw = os.getenv("PYHOI_INIT_ROOT_OFFSET_XYZ", "0 0 0").strip()
    if not raw:
        return np.zeros(3, dtype=np.float64)
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 3:
        raise ValueError(
            f"PYHOI_INIT_ROOT_OFFSET_XYZ must be 3 floats (x y z), got {raw!r}"
        )
    try:
        offset = np.asarray([float(t) for t in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"PYHOI_INIT_ROOT_OFFSET_XYZ has non-float token: {raw!r}") from exc
    return offset


def _joint_qpos_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_indices: list[int] = []
    dof_indices: list[int] = []
    ranges: list[np.ndarray] = []
    for name in G1_29DOF_MUJOCO_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo model missing joint {name!r}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        dof_indices.append(int(model.jnt_dofadr[joint_id]))
        ranges.append(np.asarray(model.jnt_range[joint_id], dtype=np.float64))
    return np.asarray(qpos_indices), np.asarray(dof_indices), np.asarray(ranges)


def _set_model_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    qpos_indices: np.ndarray | None = None,
) -> None:
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    if qpos_indices is None:
        data.qpos[7:] = joint_pos
    else:
        data.qpos[qpos_indices] = joint_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _body_position_errors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_pos_targets: dict[str, np.ndarray],
    body_names: tuple[str, ...] = IK_ALIGNMENT_BODIES,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"MuJoCo model missing body {name!r}")
        errors[name] = float(np.linalg.norm(body_pos_targets[name] - data.xpos[body_id]))
    return errors


def _print_body_errors(prefix: str, errors: dict[str, float]) -> None:
    joined = ", ".join(f"{name}={err:.4f}" for name, err in errors.items())
    print(f"[pyHOI] {prefix}: {joined}")


def _solve_frame0_ik(
    scene_xml: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    initial_joint_pos: np.ndarray,
    body_pos_targets: dict[str, np.ndarray],
    *,
    max_iters: int = 120,
    damping: float = 1e-3,
    step_scale: float = 0.6,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    qpos_indices, dof_indices, joint_ranges = _joint_qpos_indices(model)
    q = initial_joint_pos.astype(np.float64).copy()
    _set_model_state(model, data, root_pos, root_quat, q, qpos_indices)
    before_errors = _body_position_errors(model, data, body_pos_targets)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    weights = {
        "torso_link": 1.5,
        "left_ankle_roll_link": 2.0,
        "right_ankle_roll_link": 2.0,
        "left_wrist_yaw_link": 2.5,
        "right_wrist_yaw_link": 2.5,
        "left_elbow_link": 1.0,
        "right_elbow_link": 1.0,
    }

    for _ in range(max_iters):
        residual_blocks: list[np.ndarray] = []
        jac_blocks: list[np.ndarray] = []
        for name in IK_ALIGNMENT_BODIES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
            weight = weights.get(name, 1.0)
            residual_blocks.append(weight * (body_pos_targets[name] - data.xpos[body_id]))
            jac_blocks.append(weight * jacp[:, dof_indices])

        residual = np.concatenate(residual_blocks)
        jac = np.vstack(jac_blocks)
        if float(np.linalg.norm(residual)) < 1e-5:
            break
        lhs = jac.T @ jac + damping * np.eye(jac.shape[1])
        rhs = jac.T @ residual
        delta = np.linalg.solve(lhs, rhs)
        delta = np.clip(delta, -0.08, 0.08)
        q = q + step_scale * delta
        q = np.clip(q, joint_ranges[:, 0], joint_ranges[:, 1])
        _set_model_state(model, data, root_pos, root_quat, q, qpos_indices)

    after_errors = _body_position_errors(model, data, body_pos_targets)
    return q, before_errors, after_errors


def _validate_generated_robot(
    robot_xml: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray | None,
) -> None:
    root = ET.parse(robot_xml).getroot()
    pelvis = root.find("worldbody/body[@name='pelvis']")
    if pelvis is None:
        raise RuntimeError(f"Generated robot XML missing pelvis: {robot_xml}")
    out_pos = _parse_vec(pelvis.attrib.get("pos"), 3, (0.0, 0.0, 0.0))
    out_quat = _parse_vec(pelvis.attrib.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
    pos_err = float(np.max(np.abs(out_pos - root_pos)))
    quat_err = float(np.max(np.abs(out_quat - root_quat)))
    joint_err: float | None = None
    if joint_pos is not None:
        joint_by_name = {joint.attrib.get("name"): joint for joint in root.findall(".//joint")}
        joint_err = 0.0
        for name, ref in zip(G1_29DOF_MUJOCO_JOINT_NAMES, joint_pos, strict=True):
            joint = joint_by_name[name]
            joint_err = max(joint_err, abs(float(joint.attrib["ref"]) - float(ref)))
    print(f"[pyHOI] expected_init_root_pos: {_fmt(root_pos)}")
    print(f"[pyHOI] expected_init_root_quat_wxyz: {_fmt(root_quat)}")
    print(f"[pyHOI] generated_pelvis_pos: {_fmt(out_pos)}")
    print(f"[pyHOI] generated_pelvis_quat_wxyz: {_fmt(out_quat)}")
    joint_err_text = "skipped" if joint_err is None else f"{joint_err:.3e}"
    print(f"[pyHOI] init_pose_max_abs_err: pos={pos_err:.3e}, quat={quat_err:.3e}, joints={joint_err_text}")
    if pos_err > 1e-8 or quat_err > 1e-8 or (joint_err is not None and joint_err > 1e-8):
        raise RuntimeError("Generated pyHOI robot init does not match motion first frame exactly.")


def prepare_scene(cfg: PyHOIConfig, *, force: bool = False) -> Path:
    """Return a MuJoCo scene path, generating the HOI scene when needed."""
    if cfg.scene_override is not None:
        print(f"[pyHOI] Using scene override: {cfg.scene_override}")
        return cfg.scene_override

    if cfg.generated_scene.is_file() and cfg.generated_robot_xml.is_file() and not force:
        if cfg.validate_cached_scene:
            try:
                if cfg.init_pose_mode == "flat_offset":
                    root_pos, root_quat, raw_joint_pos = _load_flat_scene_init(
                        cfg.base_scene, cfg.flat_init_xy_offset
                    )
                    solver_mode = "direct"
                    body_pos_targets = {}
                    expected_cached_joints = raw_joint_pos
                else:
                    root_pos, root_quat, raw_joint_pos, body_pos_targets, _ = _load_motion_frame0(cfg.motion_file)
                    solver_mode = _init_solver_mode()
                    root_offset = _init_root_offset()
                    if float(np.linalg.norm(root_offset)) > 0.0:
                        root_pos = root_pos + root_offset
                        body_pos_targets = {name: pos + root_offset for name, pos in body_pos_targets.items()}
                    expected_cached_joints = raw_joint_pos if solver_mode == "direct" else None
                _validate_generated_robot(cfg.generated_robot_xml, root_pos, root_quat, expected_cached_joints)
                if solver_mode == "ik" and body_pos_targets:
                    model = mujoco.MjModel.from_xml_path(str(cfg.generated_scene))
                    data = mujoco.MjData(model)
                    if not _apply_frame0_keyframe(model, data):
                        raise RuntimeError(f"existing IK scene missing {FRAME0_KEY_NAME!r} keyframe")
                    errors = _body_position_errors(model, data, body_pos_targets)
                    if max(errors.values()) > 0.03:
                        raise RuntimeError(f"existing IK body error too high: max={max(errors.values()):.4f}")
                    _print_body_errors("cached_frame0_ik_body_error_m", errors)
            except Exception as exc:
                print(f"[pyHOI][Warn] Existing generated scene is stale: {exc}")
                print("[pyHOI] Regenerating exact frame-0 scene.")
            else:
                print(f"[pyHOI] Using existing generated HOI scene: {cfg.generated_scene}")
                return cfg.generated_scene
        else:
            print(f"[pyHOI] Reusing existing generated HOI scene (no validation): {cfg.generated_scene}")
            print("[pyHOI] Pass --force-scene or set PYHOI_VALIDATE_SCENE=1 to rebuild/validate.")
            return cfg.generated_scene

    if cfg.init_pose_mode == "flat_offset":
        root_pos, root_quat, raw_joint_pos = _load_flat_scene_init(cfg.base_scene, cfg.flat_init_xy_offset)
        solver_mode = "direct"
        body_pos_targets: dict[str, np.ndarray] = {}
        print(
            "[pyHOI] Init pose mode: flat_offset; using flat-scene qpos0 "
            f"with xy offset {_fmt(cfg.flat_init_xy_offset)} m."
        )
    else:
        root_pos, root_quat, raw_joint_pos, body_pos_targets, _body_quat_targets = _load_motion_frame0(cfg.motion_file)
        solver_mode = _init_solver_mode()
        root_offset = _init_root_offset()
        if float(np.linalg.norm(root_offset)) > 0.0:
            print(f"[pyHOI] Applying root offset to frame 0: {_fmt(root_offset)} (xyz, meters)")
            root_pos = root_pos + root_offset
            body_pos_targets = {name: pos + root_offset for name, pos in body_pos_targets.items()}

    cfg.generated_scene.parent.mkdir(parents=True, exist_ok=True)
    cfg.generated_robot_xml.parent.mkdir(parents=True, exist_ok=True)

    print("[pyHOI] Generating HOI MuJoCo scene with exact motion frame-0 init.")
    _write_exact_robot_xml(cfg.base_robot_xml, cfg.generated_robot_xml, root_pos, root_quat, raw_joint_pos)
    mesh_count = _write_scene(cfg)
    ref_joint_pos = raw_joint_pos
    runtime_joint_pos = raw_joint_pos
    if solver_mode == "ik":
        ref_joint_pos, before_errors, ref_after_errors = _solve_frame0_ik(
            cfg.generated_scene, root_pos, root_quat, raw_joint_pos, body_pos_targets
        )
        _print_body_errors("frame0_ik_body_error_before_m", before_errors)
        _print_body_errors("frame0_ik_body_error_after_ref_solve_m", ref_after_errors)
        print(f"[pyHOI] frame0_ik_max_ref_delta_rad: {float(np.max(np.abs(ref_joint_pos - raw_joint_pos))):.6f}")
        _write_exact_robot_xml(cfg.base_robot_xml, cfg.generated_robot_xml, root_pos, root_quat, ref_joint_pos)
        runtime_joint_pos, _, after_errors = _solve_frame0_ik(
            cfg.generated_scene, root_pos, root_quat, ref_joint_pos, body_pos_targets
        )
        _write_frame0_keyframe(cfg.generated_scene, root_pos, root_quat, runtime_joint_pos)
        _print_body_errors("frame0_ik_body_error_after_keyframe_m", after_errors)
        print(f"[pyHOI] frame0_ik_max_key_qpos_delta_rad: {float(np.max(np.abs(runtime_joint_pos - raw_joint_pos))):.6f}")
    else:
        _write_frame0_keyframe(cfg.generated_scene, root_pos, root_quat, runtime_joint_pos)
        model = mujoco.MjModel.from_xml_path(str(cfg.generated_scene))
        data = mujoco.MjData(model)
        _apply_frame0_keyframe(model, data)
        if body_pos_targets:
            direct_errors = _body_position_errors(model, data, body_pos_targets)
            _print_body_errors("frame0_direct_body_error_m", direct_errors)

    _validate_generated_robot(cfg.generated_robot_xml, root_pos, root_quat, ref_joint_pos)
    if not cfg.generated_scene.is_file():
        raise FileNotFoundError(f"Scene generation did not create: {cfg.generated_scene}")
    print(f"[pyHOI] init_solver: {solver_mode}")
    print(f"[pyHOI] Inserted {mesh_count} static terrain geoms.")
    print(f"[pyHOI] Generated HOI scene: {cfg.generated_scene}")
    return cfg.generated_scene

