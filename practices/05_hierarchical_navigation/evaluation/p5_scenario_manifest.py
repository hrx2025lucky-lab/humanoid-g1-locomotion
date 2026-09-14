"""Create, validate, and replay paired Practice 5 navigation scenarios.

The manifest stores environment-local obstacle, robot, and goal state.  A normal
environment reset must run before ``apply_live_scenario`` so framework-owned
history and action buffers are cleared; the function then overwrites the random
physical state with the frozen scenario.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable
SCHEMA_VERSION = 1

def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()

def _as_list(value: Any) -> list[Any]:
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'tolist'):
        value = value.tolist()
    return list(value)

def _require_vector(name: str, value: Any, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{name} must contain exactly {length} values')
    result = [float(item) for item in value]
    if not all((math.isfinite(item) for item in result)):
        raise ValueError(f'{name} contains a non-finite value')
    return result

def validate_scenario(record: dict[str, Any], *, verify_hash: bool=True) -> None:
    required = {'scenario_id', 'generator_seed', 'env_index', 'obstacles', 'robot', 'target'}
    missing = required - record.keys()
    if missing:
        raise ValueError(f'scenario is missing fields: {sorted(missing)}')
    if not isinstance(record['scenario_id'], str) or not record['scenario_id']:
        raise ValueError('scenario_id must be a non-empty string')
    if not isinstance(record['generator_seed'], int) or not isinstance(record['env_index'], int):
        raise ValueError('generator_seed and env_index must be integers')
    obstacle_group = record['obstacles']
    if not isinstance(obstacle_group, dict) or not isinstance(obstacle_group.get('items'), list):
        raise ValueError('obstacles.items must be a list')
    max_slots = int(obstacle_group.get('max_slots', -1))
    if max_slots <= 0:
        raise ValueError('obstacles.max_slots must be positive')
    slot_ids: set[int] = set()
    for (index, obstacle) in enumerate(obstacle_group['items']):
        if not isinstance(obstacle, dict):
            raise ValueError(f'obstacles.items[{index}] must be an object')
        slot_id = int(obstacle.get('slot_id', -1))
        if slot_id < 0 or slot_id >= max_slots or slot_id in slot_ids:
            raise ValueError(f'invalid or duplicate obstacle slot_id {slot_id}')
        slot_ids.add(slot_id)
        _require_vector(f'obstacles.items[{index}].center_xy_local', obstacle.get('center_xy_local'), 2)
        _require_vector(f'obstacles.items[{index}].half_extents_xy', obstacle.get('half_extents_xy'), 2)
        for key in ('footprint_radius', 'height'):
            number = float(obstacle.get(key, math.nan))
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f'obstacles.items[{index}].{key} must be finite and non-negative')
    robot = record['robot']
    root_state = _require_vector('robot.root_state_local_wxyz', robot.get('root_state_local_wxyz'), 13)
    quaternion_norm = math.sqrt(sum((value * value for value in root_state[3:7])))
    if abs(quaternion_norm - 1.0) > 0.001:
        raise ValueError(f'robot root quaternion norm is {quaternion_norm}, expected 1')
    joint_pos = robot.get('joint_pos')
    joint_vel = robot.get('joint_vel')
    if not isinstance(joint_pos, list) or not joint_pos or (not isinstance(joint_vel, list)):
        raise ValueError('robot joint_pos and joint_vel must be non-empty lists')
    if len(joint_pos) != len(joint_vel):
        raise ValueError('robot joint_pos and joint_vel lengths differ')
    _require_vector('robot.joint_pos', joint_pos, len(joint_pos))
    _require_vector('robot.joint_vel', joint_vel, len(joint_vel))
    target = record['target']
    target_pos = _require_vector('target.pos_local', target.get('pos_local'), 3)
    heading = float(target.get('heading_w', math.nan))
    start_distance = float(target.get('start_distance_m', math.nan))
    if not math.isfinite(heading) or not math.isfinite(start_distance) or start_distance < 0.0:
        raise ValueError('target heading and start distance must be finite')
    recomputed_distance = math.hypot(target_pos[0] - root_state[0], target_pos[1] - root_state[1])
    if abs(recomputed_distance - start_distance) > 1e-05:
        raise ValueError('target.start_distance_m does not match robot and target positions')
    if verify_hash:
        expected = record.get('scenario_sha256')
        unhashed = {key: value for (key, value) in record.items() if key != 'scenario_sha256'}
        actual = content_sha256(unhashed)
        if expected != actual:
            raise ValueError(f"scenario hash mismatch for {record['scenario_id']}: {expected} != {actual}")

def seal_scenario(record: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(record)
    sealed.pop('scenario_sha256', None)
    validate_scenario(sealed, verify_hash=False)
    sealed['scenario_sha256'] = content_sha256(sealed)
    return sealed

def build_manifest(records: Iterable[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    sealed_records = [seal_scenario(record) for record in records]
    sealed_records.sort(key=lambda record: record['scenario_id'])
    identifiers = [record['scenario_id'] for record in sealed_records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('scenario_id values must be unique')
    manifest = {'schema_version': SCHEMA_VERSION, 'coordinate_convention': {'position_frame': 'environment-local; add env_origin for simulator world position', 'quaternion_order': 'wxyz', 'angles': 'radians', 'distance': 'metres'}, 'metadata': copy.deepcopy(metadata), 'scenarios': sealed_records}
    manifest['manifest_sha256'] = content_sha256(manifest)
    return manifest

def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {manifest.get('schema_version')}")
    if not isinstance(manifest.get('metadata'), dict) or not isinstance(manifest.get('scenarios'), list):
        raise ValueError('metadata must be an object and scenarios must be a list')
    identifiers = []
    for scenario in manifest['scenarios']:
        validate_scenario(scenario)
        identifiers.append(scenario['scenario_id'])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('scenario_id values must be unique')
    expected = manifest.get('manifest_sha256')
    unhashed = {key: value for (key, value) in manifest.items() if key != 'manifest_sha256'}
    actual = content_sha256(unhashed)
    if expected != actual:
        raise ValueError(f'manifest hash mismatch: {expected} != {actual}')

def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')

def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    return manifest

def export_live_scenario(env: Any, env_index: int, generator_seed: int, scenario_id: str, *, asset_name: str='robot', command_name: str='pose_command') -> dict[str, Any]:
    """Capture one already-reset Isaac Lab environment into a plain dictionary."""
    layout = env.obstacle_layout
    robot = env.scene[asset_name]
    command = env.command_manager.get_term(command_name)
    origin = _as_list(env.scene.env_origins[env_index])
    root_state = [float(value) for value in _as_list(robot.data.root_state_w[env_index])]
    root_state[:3] = [root_state[i] - float(origin[i]) for i in range(3)]
    target_pos = [float(value) for value in _as_list(command.pos_command_w[env_index])]
    target_pos = [target_pos[i] - float(origin[i]) for i in range(3)]
    obstacles = []
    count = int(layout.num_active[env_index].item())
    for row in range(count):
        if not bool(layout.active_mask[env_index, row].item()):
            raise RuntimeError(f'inactive obstacle inside active prefix at row {row}')
        slot_id = int(layout.active_slot_ids[env_index, row].item())
        obstacles.append({'slot_id': slot_id, 'slot_type': int(layout.slot_type[slot_id].item()), 'center_xy_local': [float(value) for value in _as_list(layout.centers_xy[env_index, row])], 'footprint_radius': float(layout.footprint_radius[slot_id].item()), 'half_extents_xy': [float(value) for value in _as_list(layout.half_extents_xy[slot_id])], 'height': float(layout.heights[slot_id].item())})
    record = {'scenario_id': scenario_id, 'generator_seed': int(generator_seed), 'env_index': int(env_index), 'obstacles': {'max_slots': int(layout.max_obstacles), 'items': obstacles}, 'robot': {'root_state_local_wxyz': root_state, 'joint_pos': [float(value) for value in _as_list(robot.data.joint_pos[env_index])], 'joint_vel': [float(value) for value in _as_list(robot.data.joint_vel[env_index])]}, 'target': {'pos_local': target_pos, 'heading_w': float(command.heading_command_w[env_index].item()), 'start_distance_m': math.hypot(target_pos[0] - root_state[0], target_pos[1] - root_state[1])}}
    return seal_scenario(record)

def _index_tensor(reference: Any, env_index: int) -> Any:
    if hasattr(reference, 'new_tensor'):
        return reference.new_tensor([env_index])
    return [env_index]

def validate_layout_compatibility(layout: Any, scenario: dict[str, Any]) -> None:
    """Reject a manifest whose obstacle slot mapping differs from the live task."""
    expected_slots = int(scenario['obstacles']['max_slots'])
    actual_slots = int(layout.max_obstacles)
    if actual_slots != expected_slots:
        raise ValueError(f'obstacle slot count differs: manifest={expected_slots}, live={actual_slots}')
    for obstacle in scenario['obstacles']['items']:
        slot_id = int(obstacle['slot_id'])
        expected = {'slot_type': int(obstacle['slot_type']), 'footprint_radius': float(obstacle['footprint_radius']), 'half_extents_xy': [float(value) for value in obstacle['half_extents_xy']], 'height': float(obstacle['height'])}
        actual = {'slot_type': int(layout.slot_type[slot_id].item()), 'footprint_radius': float(layout.footprint_radius[slot_id].item()), 'half_extents_xy': [float(value) for value in _as_list(layout.half_extents_xy[slot_id])], 'height': float(layout.heights[slot_id].item())}
        if expected['slot_type'] != actual['slot_type']:
            raise ValueError(f'obstacle slot {slot_id} type differs from the manifest')
        for key in ('footprint_radius', 'height'):
            if not math.isclose(expected[key], actual[key], rel_tol=0.0, abs_tol=1e-06):
                raise ValueError(f'obstacle slot {slot_id} {key} differs from the manifest')
        if any((not math.isclose(wanted, observed, rel_tol=0.0, abs_tol=1e-06) for (wanted, observed) in zip(expected['half_extents_xy'], actual['half_extents_xy']))):
            raise ValueError(f'obstacle slot {slot_id} half_extents_xy differs from the manifest')

def apply_live_scenario(env: Any, scenario: dict[str, Any], *, destination_env_index: int, asset_name: str='robot', command_name: str='pose_command') -> None:
    """Overwrite a normally reset environment with one verified scenario."""
    validate_scenario(scenario)
    layout = env.obstacle_layout
    robot = env.scene[asset_name]
    command = env.command_manager.get_term(command_name)
    env_ids = _index_tensor(layout.num_active, destination_env_index)
    items = scenario['obstacles']['items']
    validate_layout_compatibility(layout, scenario)
    layout.centers_xy[destination_env_index] = 0.0
    layout.active_mask[destination_env_index] = False
    layout.active_slot_ids[destination_env_index] = -1
    for (row, obstacle) in enumerate(items):
        slot_id = int(obstacle['slot_id'])
        layout.centers_xy[destination_env_index, row] = layout.centers_xy.new_tensor(obstacle['center_xy_local'])
        layout.active_mask[destination_env_index, row] = True
        layout.active_slot_ids[destination_env_index, row] = slot_id
    layout.num_active[destination_env_index] = len(items)
    layout.write_to_sim(env_ids)
    origin = env.scene.env_origins[destination_env_index]
    root_state = robot.data.root_state_w.new_tensor(scenario['robot']['root_state_local_wxyz'])
    root_state[:3] += origin
    robot.write_root_pose_to_sim(root_state[:7].unsqueeze(0), env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[7:13].unsqueeze(0), env_ids=env_ids)
    joint_pos = robot.data.joint_pos.new_tensor(scenario['robot']['joint_pos']).unsqueeze(0)
    joint_vel = robot.data.joint_vel.new_tensor(scenario['robot']['joint_vel']).unsqueeze(0)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    target_pos = command.pos_command_w.new_tensor(scenario['target']['pos_local'])
    target_pos += origin
    command.pos_command_w[destination_env_index] = target_pos
    command.heading_command_w[destination_env_index] = float(scenario['target']['heading_w'])
    command.goal_reached_this_step[destination_env_index] = False
    command._update_command_for_envs(env_ids)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    print(json.dumps({'status': 'valid', 'schema_version': manifest['schema_version'], 'scenarios': len(manifest['scenarios']), 'manifest_sha256': manifest['manifest_sha256']}, ensure_ascii=False))
if __name__ == '__main__':
    main()
