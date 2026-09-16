"""Disclosed temporal crops of completed GMR outputs; preserve source and pose data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(os.environ.get('P7_EVIDENCE_ROOT', Path.home() / 'humanoid_practice'))
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT / '验收证据'
AMASS = Path(os.environ.get('AMASS_ACCAD_DIR', Path.home() / 'datasets/AMASS/ACCAD/Female1Running_c3d'))
PLAN = {
    'walk_C4_f050_105': (EVIDENCE / 'P7补充动作覆盖_2026-09-11/C4_run_to_walk/C4_run_to_walk.npz', AMASS / 'C4_-_Run_to_walk1_stageii.npz', 50, 106, 'walking phase after run-to-walk transition'),
    'run_C5_f120_161': (EVIDENCE / '周末实践推进_2026-09-10/p7_self_generated_required_set/raw/C5_walk_to_run.npz', AMASS / 'C5_-_walk_to_run_stageii.npz', 120, 162, 'accelerating running phase after walk-to-run transition'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    segments = OUT / 'segments'
    segments.mkdir(exist_ok=False)
    report = {'classification': 'pending_human_robot_review', 'operation': 'Temporal slice plus first-frame XY translation only; no smoothing, time warping, repetition or pose changes.', 'clips': {}}
    for name, (source, human, start, end, reason) in PLAN.items():
        with np.load(source, allow_pickle=False) as z:
            original = {k: z[k] for k in z.files}
        with np.load(human, allow_pickle=False) as h:
            source_frames = len(h['trans'])
            source_fps = float(h['mocap_frame_rate'])
        n = len(original['dof_pos'])
        assert 0 <= start < end <= n and end - start >= 3
        data = {}
        for key, value in original.items():
            data[key] = value[start:end].copy() if key in ('root_pos', 'root_rot', 'dof_pos', 'local_body_pos') else value.copy()
        xy_offset = data['root_pos'][0, :2].copy()
        data['root_pos'][:, :2] -= xy_offset
        dest = segments / f'{name}.npz'
        np.savez(dest, **data)
        with np.load(dest, allow_pickle=False) as check:
            for key in ('root_rot', 'dof_pos', 'local_body_pos'):
                assert np.array_equal(check[key], original[key][start:end])
            for key in ('fps', 'link_body_list'):
                assert np.array_equal(check[key], original[key])
            np.testing.assert_allclose(check['root_pos'][:, :2] + xy_offset, original['root_pos'][start:end, :2], atol=1e-14, rtol=0)
            assert np.array_equal(check['root_pos'][:, 2], original['root_pos'][start:end, 2])
        # The existing converter uses linspace across the full human source.
        source_positions = np.linspace(0, source_frames - 1, n)[start:end]
        report['clips'][name] = {
            'retargeted_source': str(source), 'retargeted_source_sha256': sha(source),
            'human_source': str(human), 'human_source_sha256': sha(human),
            'original_target_frames': n, 'target_frame_range_start_inclusive_end_exclusive': [start, end],
            'source_frame_coordinates': source_positions.tolist(), 'human_fps': source_fps,
            'source_time_endpoints_s': (source_positions[[0, -1]] / source_fps).tolist(),
            'fps': float(data['fps']), 'frames': end-start,
            'sample_span_s': (end-start-1)/float(data['fps']), 'reason': reason,
            'xy_translation_m': xy_offset.tolist(), 'segment_sha256': sha(dest),
            'original_source_preserved': True,
        }
        (OUT / 'segment_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        cmd = [sys.executable, str(ROOT / 'g1_locomotion/scripts/assess_p7_npz.py'), '--npz', str(dest), '--output-dir', str(OUT / name / 'numeric'), '--no-video']
        result = subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'), capture_output=True, text=True, timeout=60)
        (OUT / f'{name}_numeric.log').write_text(result.stdout+result.stderr)
        assert result.returncode == 0, result.stderr
        print(name, result.stdout, flush=True)
    report['script_sha256'] = sha(Path(__file__))
    (OUT / 'segment_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
