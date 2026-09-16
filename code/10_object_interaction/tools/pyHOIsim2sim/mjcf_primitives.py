"""Shared MuJoCo MJCF helpers for primitive geoms (used by scene_builder and terrain_tool)."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import xml.etree.ElementTree as ET


def fmt_vec(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    """URDF-style fixed-axis roll-pitch-yaw (rad) to MuJoCo wxyz quaternion."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    qx = np.array([cr, sr, 0.0, 0.0], dtype=np.float64)
    qy = np.array([cp, 0.0, sp, 0.0], dtype=np.float64)
    qz = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
    return quat_mul(quat_mul(qz, qy), qx)


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Same convention as legacy terrain_tool euler_to_quat (Z * Y * X)."""
    return rpy_to_quat_wxyz(np.array([roll, pitch, yaw], dtype=np.float64))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    q_vec = np.array([x, y, z], dtype=np.float64)
    return v + 2.0 * np.cross(q_vec, np.cross(q_vec, v) + w * v)


def add_box_geom(
    parent: ET.Element,
    *,
    name: str,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    half_size: np.ndarray,
    rgba: str | None = None,
    contact: bool = True,
    extra: Mapping[str, str] | None = None,
) -> ET.Element:
    """Append a MuJoCo box geom under ``parent`` (half extents in local frame)."""
    geo = ET.SubElement(parent, "geom")
    geo.attrib["name"] = name
    geo.attrib["type"] = "box"
    geo.attrib["pos"] = fmt_vec(pos)
    geo.attrib["quat"] = fmt_vec(quat_wxyz)
    geo.attrib["size"] = fmt_vec(half_size)
    if rgba is not None:
        geo.attrib["rgba"] = rgba
    if not contact:
        geo.attrib["contype"] = "0"
        geo.attrib["conaffinity"] = "0"
        geo.attrib["group"] = "1"
    if extra:
        for k, v in extra.items():
            geo.attrib[str(k)] = str(v)
    return geo


def add_box_geom_from_euler_xyz(
    parent: ET.Element,
    *,
    name: str,
    position: np.ndarray | list[float],
    euler_xyz: np.ndarray | list[float],
    full_size: np.ndarray | list[float],
    rgba: str | None = None,
    contact: bool = True,
    extra: Mapping[str, str] | None = None,
) -> ET.Element:
    """Box with orientation given as roll-pitch-yaw (same as legacy TerrainGenerator.AddBox)."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    euler = np.asarray(euler_xyz, dtype=np.float64).reshape(3)
    half = 0.5 * np.asarray(full_size, dtype=np.float64).reshape(3)
    quat = euler_xyz_to_quat_wxyz(float(euler[0]), float(euler[1]), float(euler[2]))
    return add_box_geom(
        parent,
        name=name,
        pos=pos,
        quat_wxyz=quat,
        half_size=half,
        rgba=rgba,
        contact=contact,
        extra=extra,
    )

