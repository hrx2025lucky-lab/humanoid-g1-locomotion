"""P5 paired smoke: frozen scenarios, deterministic policies, pre-reset telemetry.

No training. ``--preflight-only`` uses CPU and never imports Isaac Sim. Runtime
holds the shared GPU lock and refuses to run beside an existing compute process.
One runtime invocation evaluates both model_950 policies on one scene family.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
ROOT = Path('/path/to/workspace/motion control/humanoid_practice')
NAV = Path('/path/to/workspace/shenlan_hw/hw5_navigation/unitree_rl_lab')
AUDIT = ROOT / '验收证据/P5公平对照协议_2026-09-11/config_audit.json'
MODELS = ('baseline', 'random_arena')
TERMS = ('goal_reached', 'base_height', 'bad_orientation', 'time_out')
AXES = ('vx', 'vy', 'wz')
TASKS = {'FixedLayout': 'Unitree-G1-29dof-Navigation-HRL-Baseline', 'RandomArena': 'Unitree-G1-29dof-Navigation-HRL-RandomArena'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)

def pinned_low_level_path():
    import yaml
    audit = json.loads(AUDIT.read_text())
    paths = []
    for name in MODELS:
        path = Path(audit[name]['run']) / 'params/env.yaml'
        if sha(path) != audit[name]['env_yaml_sha256']:
            raise ValueError('Saved environment YAML changed')
        cfg = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        paths.append(Path(cfg['actions']['pre_trained_policy_action']['policy_path']))
    if paths[0] != paths[1]:
        raise ValueError('Paired policies used different low-level policy paths')
    if sha(paths[0]) != audit['common_training_contract']['low_level_policy_sha256']:
        raise ValueError('Low-level policy differs from paired audit')
    return paths[0]

def load_policy(model_name, device='cpu'):
    """Use the actual RSL-RL network and restricted tensor checkpoint loader."""
    import torch
    import yaml
    from rsl_rl.modules import ActorCritic
    audit = json.loads(AUDIT.read_text())
    pin = audit[model_name]
    run = Path(pin['run'])
    path = run / pin['checkpoint']
    agent = run / 'params/agent.yaml'
    if sha(path) != pin['checkpoint_sha256'] or sha(agent) != pin['agent_yaml_sha256']:
        raise ValueError('Policy checkpoint or saved agent config differs from audited pair')
    cfg = yaml.safe_load(agent.read_text())
    if cfg['empirical_normalization'] or cfg['policy']['state_dependent_std']:
        raise ValueError('This evaluator pins the audited non-normalized scalar-std policy')
    if cfg['policy']['actor_obs_normalization'] or cfg['policy']['critic_obs_normalization']:
        raise ValueError('Unexpected observation normalization')
    if cfg['clip_actions'] != 1.0 or cfg['policy']['activation'] != 'elu':
        raise ValueError('Unexpected action clipping or activation')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['iter'] != 950:
        raise ValueError('Expected model index 950')
    state = checkpoint['model_state_dict']
    if tuple(state['actor.0.weight'].shape) != (256, 376):
        raise ValueError('Actor input contract changed')
    if tuple(state['critic.0.weight'].shape) != (256, 378):
        raise ValueError('Critic input contract changed')
    if not all((torch.isfinite(v).all() for v in state.values())):
        raise ValueError('Non-finite checkpoint tensor')
    policy_cfg = dict(cfg['policy'])
    if policy_cfg.pop('class_name') != 'ActorCritic':
        raise ValueError('Unsupported policy class')
    dummy = {'policy': torch.zeros(1, 376), 'critic': torch.zeros(1, 378)}
    with contextlib.redirect_stdout(io.StringIO()):
        model = ActorCritic(dummy, {'policy': ['policy'], 'critic': ['critic']}, 3, **policy_cfg)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return (model, {'checkpoint': str(path), 'sha256': sha(path), 'index': checkpoint['iter'], 'actor_dim': 376, 'critic_dim': 378, 'output_dim': 3, 'outer_action_clip': 1.0, 'weights_only': True})

def cpu_preflight():
    import inspect
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    results = {}
    for name in MODELS:
        (model, metadata) = load_policy(name)
        inputs = torch.randn(64, 376, generator=torch.Generator().manual_seed(4201))
        with torch.no_grad():
            observed = model.act_inference({'policy': inputs})
            state = model.state_dict()
            expected = inputs.clone()
            for layer in (0, 2, 4, 6):
                expected = F.linear(expected, state[f'actor.{layer}.weight'], state[f'actor.{layer}.bias'])
                if layer != 6:
                    expected = F.elu(expected)
            torch.testing.assert_close(observed, expected, rtol=1e-06, atol=1e-06)
        if not torch.isfinite(observed).all():
            raise ValueError('Non-finite CPU inference')
        results[name] = {**metadata, 'samples': 64, 'manual_forward_max_error': float((observed - expected).abs().max()), 'rsl_actor_source': inspect.getfile(type(model))}
    low = pinned_low_level_path()
    expected = json.loads(AUDIT.read_text())['common_training_contract']['low_level_policy_sha256']
    if sha(low) != expected:
        raise ValueError('Low-level policy differs from paired audit')
    return {'status': 'cpu_policy_preflight_passed', 'policies': results, 'low_level_path': str(low), 'low_level_sha256': expected, 'cuda_initialized': torch.cuda.is_initialized(), 'limitation': 'Network loading and inference only; no physical rollout.'}

class EpisodeAccumulator:
    """Keep only each environment's first episode, including its terminal step."""

    def __init__(self, scenarios, model, family, step_dt):
        import numpy as np
        self.np = np
        (self.scenarios, self.model, self.family, self.dt) = (scenarios, model, family, step_dt)
        n = len(scenarios)
        self.active = np.ones(n, dtype=bool)
        self.steps = np.zeros(n, dtype=int)
        self.minimum = np.array([s['target']['start_distance_m'] for s in scenarios])
        self.sums = {key: np.zeros((n, 3)) for key in ('sat', 'clip', 'delta', 'tracking')}
        self.near_sum = np.zeros(n)
        self.previous = np.zeros((n, 3))
        self.records = []

    def add(self, snapshot, actor_raw):
        np = self.np
        raw = np.asarray(actor_raw, dtype=float)
        processed = snapshot['processed']
        if raw.shape != self.previous.shape or processed.shape != raw.shape:
            raise ValueError('Unexpected action shape')
        active = self.active.copy()
        for name in ('distance', 'cached_distance', 'processed', 'velocity', 'near', 'root_state'):
            if not np.isfinite(snapshot[name][active]).all():
                raise ValueError(f'Non-finite active telemetry: {name}')
        if not np.isfinite(raw[active]).all():
            raise ValueError('Non-finite active actor output')
        self.steps[active] += 1
        self.minimum[active] = np.minimum(self.minimum[active], snapshot['distance'][active])
        (lo, hi) = (snapshot['lower'], snapshot['upper'])
        self.sums['sat'][active] += (np.isclose(processed, lo, atol=1e-06, rtol=0) | np.isclose(processed, hi, atol=1e-06, rtol=0))[active]
        self.sums['clip'][active] += np.abs(raw - processed)[active]
        self.sums['delta'][active] += np.abs(processed - self.previous)[active]
        self.sums['tracking'][active] += np.abs(snapshot['velocity'] - processed)[active]
        self.near_sum[active] += snapshot['near'][active]
        self.previous[active] = processed[active]
        finished = active & snapshot['flags'].any(axis=1)
        for i in np.flatnonzero(finished):
            scenario = self.scenarios[i]
            record = {'model': self.model, 'scene_family': self.family, 'seed': scenario['generator_seed'], 'scenario_id': scenario['scenario_id'], 'scenario_sha256': scenario['scenario_sha256'], 'duration_s': float(self.steps[i] * self.dt), 'final_distance_m': float(snapshot['distance'][i]), 'min_distance_m': float(self.minimum[i]), 'termination': dict(zip(TERMS, map(bool, snapshot['flags'][i]))), 'near_obstacle_proxy_mean': float(self.near_sum[i] / self.steps[i]), 'cached_goal_distance_at_termination_m': float(snapshot['cached_distance'][i]), 'terminal_root_state_wxyz': snapshot['root_state'][i].tolist(), 'control_steps': int(self.steps[i]), 'measurement': '5 Hz post-physics, before auto-reset and command resampling'}
            fields = {'sat': 'high_level_saturation_fraction', 'clip': 'mean_abs_clip_delta', 'delta': 'mean_abs_command_delta', 'tracking': 'low_level_tracking_mae'}
            for (key, field) in fields.items():
                record[field] = dict(zip(AXES, (self.sums[key][i] / self.steps[i]).tolist()))
            self.records.append(record)
        self.active[finished] = False
        return [self.records[-k] for k in range(int(finished.sum()), 0, -1)]

def install_terminal_snapshot(env):
    """Capture AFTER raw term computation, BEFORE the environment resets anything."""
    import torch
    original = env.termination_manager.compute
    holder = {}

    def compute():
        result = original()
        robot = env.scene['robot']
        command = env.command_manager.get_term('pose_command')
        action = env.action_manager.get_term('pre_trained_policy_action')

        def copy(value):
            return value.detach().cpu().numpy().copy()
        holder['snapshot'] = {'flags': copy(torch.stack([env.termination_manager.get_term(k) for k in TERMS], dim=1)), 'distance': copy(torch.linalg.vector_norm(command.pos_command_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)), 'cached_distance': copy(torch.linalg.vector_norm(command.command[:, :2], dim=1)), 'processed': copy(action.processed_actions), 'lower': copy(action._clip_lower), 'upper': copy(action._clip_upper), 'velocity': copy(torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=1)), 'near': copy(env.obstacle_layout.soft_proximity_penalty(robot.data.root_pos_w[:, :2])), 'root_state': copy(robot.data.root_state_w)}
        return result
    env.termination_manager.compute = compute
    return (holder, original)

def assert_roundtrip(expected, observed, tolerance=1e-05):
    """Compare physical content, allowing normal float32 world/local roundoff."""
    import numpy as np
    for key in ('obstacles', 'robot', 'target'):

        def compare(a, b, name):
            if isinstance(a, dict):
                if a.keys() != b.keys():
                    raise ValueError(f'{name}: keys differ')
                for k in a:
                    compare(a[k], b[k], name + '.' + k)
            elif isinstance(a, list) and a and isinstance(a[0], dict):
                if len(a) != len(b):
                    raise ValueError(f'{name}: obstacle count differs')
                for (i, (x, y)) in enumerate(zip(a, b)):
                    compare(x, y, f'{name}[{i}]')
            else:
                np.testing.assert_allclose(a, b, rtol=0, atol=tolerance, err_msg=name)
        compare(expected[key], observed[key], key)

def restore_batch(env, scenarios):
    import torch
    from p5_scenario_manifest import apply_live_scenario, export_live_scenario
    env.reset(seed=scenarios[0]['generator_seed'])
    for (i, scenario) in enumerate(scenarios):
        apply_live_scenario(env, scenario, destination_env_index=i)
    env.scene.reset()
    env.action_manager.reset()
    env.observation_manager.reset()
    env.scene.write_data_to_sim()
    env.sim.forward()
    ids = torch.arange(env.num_envs, device=env.device)
    command = env.command_manager.get_term('pose_command')
    command._update_command_for_envs(ids)
    restored = []
    layout = env.obstacle_layout
    obstacles = env.scene[layout.cfg.obstacle_asset_name]
    data = obstacles.data
    poses = data._reshape_view_to_data(data._root_physx_view.get_transforms().clone())
    for (i, scenario) in enumerate(scenarios):
        actual = export_live_scenario(env, i, scenario['generator_seed'], scenario['scenario_id'])
        assert_roundtrip(scenario, actual)
        for item in scenario['obstacles']['items']:
            slot = item['slot_id']
            object_id = int(layout._object_ids[slot])
            xyz = poses[i, object_id, :3]
            target = xyz.new_tensor([*item['center_xy_local'], item['height'] / 2])
            target[:2] += env.scene.env_origins[i, :2]
            torch.testing.assert_close(xyz, target, rtol=0, atol=1e-05)
        restored.append(actual)
    obs = env.observation_manager.compute(update_history=True)
    return (obs, restored)

def run_smoke(args, out):
    import fcntl
    import os
    import subprocess
    lock = open('/tmp/humanoid_gpu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy; do not interrupt existing training')
    sys.path.insert(0, str(NAV / 'source/unitree_rl_lab'))
    from isaaclab.app import AppLauncher
    launcher = AppLauncher({'headless': True, 'device': 'cuda:0'})
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import unitree_rl_lab.tasks
        from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
        from p5_scenario_manifest import build_manifest, export_live_scenario, write_manifest
        from p5_evaluation_metrics import validate_record, aggregate
        from isaaclab.utils.io import dump_yaml
        torch.set_num_threads(2)
        task = TASKS[args.scene_family]
        cfg = parse_env_cfg(task, device='cuda:0', num_envs=16, entry_point_key='env_cfg_entry_point')
        cfg.seed = args.seed
        cfg.curriculum = None
        cfg.observations.policy.enable_corruption = False
        cfg.observations.critic.enable_corruption = False
        cfg.actions.pre_trained_policy_action.low_level_observations.enable_corruption = False
        cfg.actions.pre_trained_policy_action.debug_vis = False
        cfg.commands.pose_command.debug_vis = False
        action_cfg = cfg.actions.pre_trained_policy_action
        if sha(action_cfg.policy_path) != json.loads(AUDIT.read_text())['common_training_contract']['low_level_policy_sha256']:
            raise ValueError('Runtime low-level policy mismatch')
        if action_cfg.command_smoothing != 1.0 or tuple(map(tuple, action_cfg.velocity_clip)) != ((-0.5, 1.0), (-0.5, 0.5), (-0.5, 0.5)):
            raise ValueError('Runtime command bounds/smoothing mismatch')
        if cfg.sim.dt != 0.005 or cfg.decimation != 40 or action_cfg.low_level_decimation != 4:
            raise ValueError('Runtime clock mismatch')
        if cfg.episode_length_s != 30.0 or cfg.commands.pose_command.update_goal_on_success:
            raise ValueError('Unexpected episode/goal contract')
        if tuple(cfg.commands.pose_command.ranges.distance) != (5.0, 10.0):
            raise ValueError('Runtime goal range mismatch')
        dump_yaml(str(out / 'effective_env.yaml'), cfg)
        env = gym.make(task, cfg=cfg).unwrapped
        if set(env.termination_manager.active_terms) != set(TERMS):
            raise ValueError('Runtime termination set differs')
        if 'interval' in env.event_manager.available_modes:
            raise ValueError('Unfrozen interval randomization is not part of this protocol')
        env.reset(seed=args.seed)
        ids = torch.arange(env.num_envs, device=env.device)
        command = env.command_manager.get_term('pose_command')
        if not env.obstacle_layout.is_pose_free(env.scene['robot'].data.root_pos_w[:, :2], 0.5, ids).all():
            raise RuntimeError('Generated starting pose intersects an obstacle soft zone')
        if not env.obstacle_layout.is_pose_free(command.pos_command_w[:, :2], 0.5, ids).all():
            raise RuntimeError('Generated goal intersects an obstacle soft zone')
        scenarios = [export_live_scenario(env, i, args.seed, f'{args.scene_family}-s{args.seed}-e{i:04d}') for i in range(16)]
        manifest = build_manifest(scenarios, {'scene_family': args.scene_family, 'scope': '16-episode integration smoke', 'generator_seed': args.seed, 'num_envs': 16, 'physics': 'Isaac Lab'})
        write_manifest(out / 'scenarios.json', manifest)
        (snapshots, original_compute) = install_terminal_snapshot(env)
        (all_records, restores) = ([], {})
        initial_observation = None
        with (out / 'episodes.jsonl').open('x') as stream, torch.no_grad():
            for name in MODELS:
                (model, metadata) = load_policy(name, env.device)
                (obs, restored) = restore_batch(env, scenarios)
                if obs['policy'].shape != (16, 376) or obs['critic'].shape != (16, 378):
                    raise ValueError('Runtime observation dimensions differ')
                if not torch.isfinite(obs['policy']).all() or not torch.isfinite(obs['critic']).all():
                    raise ValueError('Non-finite initial observations')
                if initial_observation is None:
                    initial_observation = obs['policy'].clone()
                else:
                    torch.testing.assert_close(obs['policy'], initial_observation, rtol=0, atol=1e-05)
                restores[name] = {'checkpoint': metadata, 'scenarios': restored, 'policy_observations': obs['policy'].cpu().tolist()}
                write_json(out / f'{name}_restored_states.json', restores[name])
                stats = EpisodeAccumulator(scenarios, name, args.scene_family, env.step_dt)
                trace = []
                for step in range(env.max_episode_length + 1):
                    if not stats.active.any():
                        break
                    if not torch.isfinite(obs['policy']).all():
                        raise ValueError('Non-finite policy observation')
                    raw = model.act_inference(obs)
                    before = stats.active.copy()
                    (obs, reward, terminated, truncated, _) = env.step(raw.clamp(-1.0, 1.0))
                    snapshot = snapshots.pop('snapshot')
                    records = stats.add(snapshot, raw.cpu().numpy())
                    for record in records:
                        validate_record(record)
                        stream.write(json.dumps(record, allow_nan=False) + '\n')
                        stream.flush()
                    trace.append({**snapshot, 'actor_raw': raw.cpu().numpy().copy(), 'active': before})
                if stats.active.any():
                    raise RuntimeError('Missing terminal flag by the original episode horizon')
                all_records.extend(stats.records)
                np.savez_compressed(out / f'{name}_trace.npz', **{k: np.stack([s[k] for s in trace]) for k in trace[0]})
                print(json.dumps({'completed_model': name, 'episodes': len(stats.records)}, ensure_ascii=False), flush=True)
                del model
        env.termination_manager.compute = original_compute
        write_json(out / 'restored_states.json', restores)
        summary = aggregate(all_records, expected_per_cell=None)
        summary['protocol_conclusion'] = None
        summary['scope'] = 'Integration smoke; full matrix and three-seed conclusion gates are not evaluated.'
        write_json(out / 'smoke_statistics.json', summary)
        result = {'status': 'paired_smoke_completed', 'scene_family': args.scene_family, 'seed': args.seed, 'episodes': len(all_records), 'manifest_sha256': manifest['manifest_sha256'], 'initial_policy_observation_pair_passed': True, 'scope': '32 real episodes; integration smoke only, full 1536-episode evaluation not performed', 'successes': {name: sum((r['termination']['goal_reached'] for r in all_records if r['model'] == name)) for name in MODELS}}
        write_json(out / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if env is not None:
            env.close()
        launcher.app.close()
        lock.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--scene-family', choices=tuple(TASKS), default='FixedLayout')
    parser.add_argument('--seed', type=int, choices=(4201, 4202, 4203), default=4201)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        preflight = cpu_preflight()
        write_json(out / 'cpu_preflight.json', preflight)
        if args.preflight_only:
            print(json.dumps(preflight, indent=2))
        else:
            run_smoke(args, out)
    except Exception as exc:
        write_json(out / 'failure.json', {'error': str(exc), 'type': type(exc).__name__, 'time_unix': time.time()})
        raise
if __name__ == '__main__':
    main()
