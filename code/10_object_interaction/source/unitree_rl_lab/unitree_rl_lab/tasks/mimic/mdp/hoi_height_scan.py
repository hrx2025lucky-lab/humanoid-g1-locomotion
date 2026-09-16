from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@dataclass
class HoiHeightScanResult:
    """Container returned by the analytical HOI terrain height scanner."""

    heights: torch.Tensor
    hits_w: torch.Tensor
    world_rays_xy: torch.Tensor


def _build_grid_xy(size_xy: tuple[float, float], resolution: float, device: torch.device) -> torch.Tensor:
    """Build a local XY scan grid centered at the robot torso.

    Legacy analytical-scanner outline (not part of the RayCaster assignment):
    1. Read size_x and size_y from size_xy.
    2. Validate that resolution is positive.
    3. Compute the number of samples along x/y as round(size / resolution) + 1.
       The default parameters should produce 17 * 11 = 187 points.
    4. Create x values from -size_x / 2 to +size_x / 2.
    5. Create y values from -size_y / 2 to +size_y / 2.
    6. Use torch.meshgrid and return a tensor with shape [num_points, 2].
    """
    raise NotImplementedError("The legacy analytical HOI height scanner is not implemented.")


def _normalize_box_record(entry: dict, idx: int) -> tuple[list[float], list[float], list[float]]:
    """Validate one metadata box and return pos, quat, half_size.

    Perceptive RayCaster notes:
    1. Read `pos`, `quat`, and `half_size` from entry.
    2. If `half_size` is missing but `full_size` exists, convert full_size to half_size.
    3. Check pos length is 3, quat length is 4, half_size length is 3.
    4. Raise ValueError with the box index when data is malformed.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"box[{idx}]: expected a dict record, got {type(entry).__name__}")

    pos = entry.get("pos")
    quat = entry.get("quat")
    half_size = entry.get("half_size")

    # 只给了 full_size 时按整边长的一半换算。方向是 full -> half，
    # 反过来写会让所有障碍物膨胀一倍，而且不会报错: 射线照常有命中，
    # 只是高度场整体偏大。
    if half_size is None:
        full_size = entry.get("full_size")
        if full_size is not None:
            if len(full_size) != 3:
                raise ValueError(
                    f"box[{idx}]: full_size must have length 3, got {len(full_size)}"
                )
            half_size = [float(v) / 2.0 for v in full_size]

    # 错误信息里必须带 idx：metadata 里的 box 没有稳定顺序之外的标识，
    # 少了下标就只能整份文件重新翻。
    if pos is None:
        raise ValueError(f"box[{idx}]: missing required field 'pos'")
    if len(pos) != 3:
        raise ValueError(f"box[{idx}]: pos must have length 3, got {len(pos)}")

    if quat is None:
        raise ValueError(f"box[{idx}]: missing required field 'quat'")
    if len(quat) != 4:
        raise ValueError(f"box[{idx}]: quat must have length 4 (wxyz), got {len(quat)}")

    if half_size is None:
        raise ValueError(f"box[{idx}]: missing both 'half_size' and 'full_size'")
    if len(half_size) != 3:
        raise ValueError(
            f"box[{idx}]: half_size must have length 3, got {len(half_size)}"
        )

    # quat 保持 wxyz 原样传出: 上游 RayCaster 与 IsaacLab 的 math_utils
    # 都按 wxyz 约定，这里转成 xyzw 会让障碍物旋转错乱。
    return (
        [float(v) for v in pos],
        [float(v) for v in quat],
        [float(v) for v in half_size],
    )


# 按 (metadata_file, device) 缓存。RayCaster 只在 mesh bake 阶段调 loader，
# 但 bake 可能被重复触发；缓存保证 JSON I/O 不会混进逐帧的 observation 计算。
_BOX_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _load_boxes(metadata_file: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load terrain boxes from a `*.terrain.json` file.

    Returns:
        box_pos_local: [num_boxes, 3], terrain-body local box centers.
        box_quat_local: [num_boxes, 4], terrain-body local box orientation in wxyz.
        box_half: [num_boxes, 3], box half extents.

    Perceptive RayCaster notes:
    1. Open metadata_file as JSON.
    2. Read `mjcf_boxes` and check it is a non-empty list.
    3. Normalize every box record with `_normalize_box_record`.
    4. Convert lists to torch.float32 tensors on `device`.
    5. Optionally cache by `(metadata_file, device)` to avoid repeated JSON loading.
    """
    cache_key = (str(metadata_file), str(device))
    cached = _BOX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    path = Path(metadata_file)
    if not path.is_file():
        raise FileNotFoundError(f"terrain metadata not found: {metadata_file}")

    # 显式 UTF-8：不指定就跟随 locale，在非 UTF-8 环境下会解码失败
    with open(path, encoding="utf-8") as f:
        metadata = json.load(f)

    boxes = metadata.get("mjcf_boxes")
    if not isinstance(boxes, list):
        raise ValueError(
            f"{metadata_file}: 'mjcf_boxes' must be a list, got {type(boxes).__name__}"
        )
    if len(boxes) == 0:
        raise ValueError(f"{metadata_file}: 'mjcf_boxes' is empty")

    pos_list: list[list[float]] = []
    quat_list: list[list[float]] = []
    half_list: list[list[float]] = []
    for idx, entry in enumerate(boxes):
        pos, quat, half = _normalize_box_record(entry, idx)
        pos_list.append(pos)
        quat_list.append(quat)
        half_list.append(half)

    # dtype 与 device 都由调用方决定：RayCaster 在 bake 时把它们和场景张量
    # 拼在一起，float64 或留在 CPU 都会在后面炸出难读的类型/设备错误。
    box_pos_local = torch.tensor(pos_list, dtype=torch.float32, device=device)
    box_quat_local = torch.tensor(quat_list, dtype=torch.float32, device=device)
    box_half = torch.tensor(half_list, dtype=torch.float32, device=device)

    result = (box_pos_local, box_quat_local, box_half)
    _BOX_CACHE[cache_key] = result
    return result


def _store_scan_cache(
    env: ManagerBasedEnv,
    *,
    result: HoiHeightScanResult,
    params_key: tuple,
) -> None:
    """Store latest scan result on env for visualization/debug tools."""
    env._hoi_height_scan_debug = {
        "heights": result.heights,
        "hits_w": result.hits_w,
        "world_rays_xy": result.world_rays_xy,
        "params_key": params_key,
    }


def compute_hoi_height_scan(
    env: ManagerBasedEnv,
    *,
    command_name: str | None = None,
    metadata_file: str,
    terrain_asset_name: str = "hoi_terrain",
    robot_asset_name: str = "robot",
    body_name: str = "torso_link",
    size_xy: tuple[float, float] = (1.6, 1.0),
    resolution: float = 0.1,
    offset: float = 0.5,
    cache_on_env: bool = True,
) -> HoiHeightScanResult:
    """Compute analytical HOI height-scan hits and height observations.

    The output observation should be:

        torso_z - hit_z - offset

    Legacy analytical-scanner outline (not part of the RayCaster assignment):
    1. Get robot and terrain assets from `env.scene`.
    2. Resolve `body_name` to a body id on the robot.
    3. Read torso position and quaternion from `robot.data.body_pos_w/body_quat_w`.
    4. Build a local XY scan grid and rotate it with torso yaw only.
    5. Translate the grid to the torso world position to get world sample XY points.
    6. Initialize hit_z with the ground plane height z=0.
    7. Read terrain pose from `terrain.data`.
    8. Load metadata boxes with `_load_boxes`.
    9. Transform each box from terrain local frame to world frame.
    10. For each box, test whether each sample point is inside the box top projection.
    11. For covered points, update hit_z with the highest box top z.
    12. Build and return HoiHeightScanResult.
    13. If cache_on_env is true, call `_store_scan_cache`.

    Legacy scope:
    - Validation boxes have horizontal top faces (world translation/yaw are allowed).
    - Keep env and scan-point operations batched; a loop over the small box list is OK.
    """
    del command_name
    raise NotImplementedError("The legacy analytical HOI height scanner is not implemented.")


def hoi_height_scan(
    env: ManagerBasedEnv,
    *,
    command_name: str | None = None,
    metadata_file: str,
    terrain_asset_name: str = "hoi_terrain",
    robot_asset_name: str = "robot",
    body_name: str = "torso_link",
    size_xy: tuple[float, float] = (1.6, 1.0),
    resolution: float = 0.1,
    offset: float = 0.5,
) -> torch.Tensor:
    """Observation term used by the perceptive task.

    Legacy analytical-scanner outline (not part of the RayCaster assignment):
    1. Call `compute_hoi_height_scan`.
    2. Return only `result.heights`.
    """
    raise NotImplementedError("The legacy analytical HOI height scanner is not implemented.")
