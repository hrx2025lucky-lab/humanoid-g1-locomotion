"""Sweep the constant velocity command and measure how far the policy walks.

Why: our exported policy was trained with PoseVelocityCommand, where the
observed command is not a setpoint but a P-controller output,

    vel_command_b[:2] = (target position in base frame) * velocity_control_stiffness
    vel_command_b[2]  = heading error * heading_control_stiffness

with stiffness 2.0. A constant 0.6 therefore reads as "the target is 0.3 m
ahead, permanently", which is a near-arrival command. The training ranges on
rough terrain were lin_vel_x in (0.45, 1.0), so the command the policy expects
while travelling is the clipped upper end, not 0.6.

This probe runs physics and ONNX inference only, with no rendering, so it does
not contend with GPU training. It reports forward distance per command value.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')


import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'sim2sim'))
import mujoco
import config as C
import mujoco_env


class _StubViewer:
    def __init__(self):
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def is_running(self):
        return True

    def sync(self):
        pass


def _init_viewer_stub(self):
    self.viewer = _StubViewer()
    self._setup_initial_free_camera()


def run_one(S, C, task, vx, wz, seconds, seed):
    np.random.seed(seed)
    inst = S.Sim2simInstance(task)
    env = inst.sim_env
    inst.velocity_commands = np.array([vx, 0.0, wz], dtype=np.float32)
    dt = C.SIM_DT * C.DECIMATION
    steps = int(round(seconds / dt))
    start = env.model_data.qpos[:3].copy()
    zmin = float(start[2])
    for _ in range(steps):
        inst.update_observation()
        obs = inst.get_history_obs()[None, ...]
        raw = inst.actor({'input': obs}).ravel()
        inst.last_action = raw.copy()
        act = (raw[inst.policy_to_robot] * inst.action_scale + inst.default_joint_pos) * inst.joint_signs
        for _ in range(C.DECIMATION):
            tau = (inst.stiffness * (act - env.model_data.sensordata[:29])
                   - inst.damping * env.model_data.sensordata[29:58])
            tau = np.clip(tau, a_min=-inst.torque_limit, a_max=inst.torque_limit)
            env.model_data.ctrl[:] = tau
            env.physical_step()
        zmin = min(zmin, float(env.model_data.qpos[2]))
    end = env.model_data.qpos[:3].copy()
    return dict(vx_command=vx, wz_command=wz,
                forward_m=float(end[0] - start[0]),
                lateral_m=float(end[1] - start[1]),
                mean_speed_m_s=float((end[0] - start[0]) / seconds),
                min_base_height_m=zmin, final_height_m=float(end[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actor', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--mjcf', default=None)
    ap.add_argument('--seconds', type=float, default=12.0)
    ap.add_argument('--commands', default='0.6,1.0,1.5,2.0')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    mujoco_env.MujocoEnv._init_viewer = _init_viewer_stub
    if args.mjcf:
        C.MJCF_FILE = args.mjcf
    C.PARKOUR_POLICY_FILE = args.actor
    C.PARKOUR_DEPTH_ENCODER_FILE = args.encoder
    import sim2sim as S
    if args.mjcf:
        S.MJCF_FILE = args.mjcf
    S.PARKOUR_POLICY_FILE = args.actor
    S.PARKOUR_DEPTH_ENCODER_FILE = args.encoder

    rows = []
    for vx in [float(x) for x in args.commands.split(',')]:
        r = run_one(S, C, 'parkour', vx, 0.0, args.seconds, seed=0)
        rows.append(r)
        print(json.dumps({args.label: r}, ensure_ascii=False), flush=True)
    args.out.write_text(json.dumps(dict(label=args.label, seconds=args.seconds,
                                        scene=C.MJCF_FILE, rows=rows),
                                   ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
