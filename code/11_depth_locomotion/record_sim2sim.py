"""Headless recorder for the MuJoCo Sim2Sim run.

The shipped sim2sim.py drives an interactive passive viewer and loops until the
window is closed, which cannot produce a file. This entry point runs the exact
same instance for a fixed number of control steps and renders frames offscreen.

Only the display layer is replaced:
  - MujocoEnv._init_viewer is swapped for a stub whose sync() does nothing,
  - frames come from a separate offscreen mujoco.Renderer.
Observation construction, ONNX inference, action decoding, the PD law and
mj_step are all executed by the original unmodified code, so the recording
reflects the same dynamics as the interactive run.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'sim2sim'))

import mujoco
import imageio.v2 as imageio
import config as C
import mujoco_env


class _StubViewer:
    """Stands in for the passive viewer so the shipped loop can run headless."""

    def __init__(self):
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._running = True

    def is_running(self):
        return self._running

    def sync(self):
        pass

    def close(self):
        self._running = False


def _init_viewer_stub(self):
    self.viewer = _StubViewer()
    self._setup_initial_free_camera()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['parkour', 'stand'], default='parkour')
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--vx', type=float, default=0.6)
    ap.add_argument('--wz', type=float, default=0.0)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mjcf', type=str, default=None,
                    help='override the scene, e.g. the flat scene, for diagnosis')
    ap.add_argument('--actor', type=str, default=None,
                    help='override the actor ONNX, e.g. our own exported policy')
    ap.add_argument('--encoder', type=str, default=None,
                    help='override the depth encoder ONNX')
    ap.add_argument('--trace', action='store_true',
                    help='record per-step height, speed and depth statistics')
    args = ap.parse_args()
    assert not args.output.exists(), args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    mujoco_env.MujocoEnv._init_viewer = _init_viewer_stub

    if args.mjcf:
        C.MJCF_FILE = args.mjcf
    if args.actor:
        C.PARKOUR_POLICY_FILE = args.actor
        C.STAND_POLICY_FILE = args.actor
    if args.encoder:
        C.PARKOUR_DEPTH_ENCODER_FILE = args.encoder
        C.STAND_DEPTH_ENCODER_FILE = args.encoder
    import sim2sim as S
    if args.mjcf:
        S.MJCF_FILE = args.mjcf
    if args.actor:
        S.PARKOUR_POLICY_FILE = args.actor
        S.STAND_POLICY_FILE = args.actor
    if args.encoder:
        S.PARKOUR_DEPTH_ENCODER_FILE = args.encoder
        S.STAND_DEPTH_ENCODER_FILE = args.encoder
    inst = S.Sim2simInstance(args.task)
    env = inst.sim_env
    inst.velocity_commands = np.array([args.vx, 0.0, args.wz], dtype=np.float32)

    control_dt = C.SIM_DT * C.DECIMATION
    steps = int(round(args.seconds / control_dt))
    fps = int(round(1.0 / control_dt))

    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, args.width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 120.0, -12.0, 3.4

    root = []
    trace = []
    terminated_at = None
    t0 = time.perf_counter()
    with imageio.get_writer(args.output, fps=fps, codec='libx264', quality=8,
                            ffmpeg_log_level='error',
                            ffmpeg_params=['-pix_fmt', 'yuv420p']) as writer:
        for k in range(steps):
            inst.update_observation()
            obs = inst.get_history_obs()[None, ...]
            raw_action = inst.actor({'input': obs}).ravel()
            inst.last_action = raw_action.copy()
            action = (raw_action[inst.policy_to_robot] * inst.action_scale
                      + inst.default_joint_pos) * inst.joint_signs
            for _ in range(C.DECIMATION):
                tau = (inst.stiffness * (action - env.model_data.sensordata[:29])
                       - inst.damping * env.model_data.sensordata[29:58])
                tau = np.clip(tau, a_min=-inst.torque_limit, a_max=inst.torque_limit)
                env.model_data.ctrl[:] = tau
                env.physical_step()

            pos = env.model_data.qpos[:3].copy()
            root.append(pos.tolist())
            if args.trace:
                d = inst.depth_pipeline.que[-1]
                trace.append(dict(step=k, t=round(k * control_dt, 3),
                                  z=float(pos[2]), x=float(pos[0]),
                                  vx=float(env.model_data.qvel[0]),
                                  depth_mean=float(d.mean()), depth_min=float(d.min()),
                                  depth_max=float(d.max())))
            cam.lookat[:] = [pos[0], pos[1], pos[2] + 0.1]
            renderer.update_scene(env.model_data, camera=cam)
            writer.append_data(renderer.render())

            if pos[2] < 0.3 and terminated_at is None:
                terminated_at = k * control_dt
            if k % 100 == 0:
                print(json.dumps(dict(step=k, x=round(float(pos[0]), 3),
                                      z=round(float(pos[2]), 3))), flush=True)

    root = np.asarray(root)
    dx = float(root[-1, 0] - root[0, 0])
    report = dict(
        status='sim2sim_recording_complete', task=args.task,
        command=dict(vx=args.vx, vy=0.0, wz=args.wz),
        seconds=args.seconds, control_steps=steps, control_dt=control_dt, fps=fps,
        mjcf=C.MJCF_FILE,
        policy=dict(actor=args.actor or 'course_example',
                    encoder=args.encoder or 'course_example'),
        depth_pipeline=dict(resize_shape=list(C.RESIZED_DEPTH_IMAGE_SHAPE),
                            crop_region=list(C.OBS_DEPTH_IMAGE_CROP_REGION),
                            obs_shape=list(inst.depth_pipeline.get_depth_obs().shape)),
        forward_distance_m=dx,
        mean_forward_speed_m_s=dx / args.seconds,
        final_base_height_m=float(root[-1, 2]),
        min_base_height_m=float(root[:, 2].min()),
        fell_below_0p3m_at_s=terminated_at,
        wall_clock_s=time.perf_counter() - t0,
        video=str(args.output),
        video_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        scene=C.MJCF_FILE,
        trace=trace if args.trace else None,
        note='Headless recording of the shipped Sim2simInstance. Only the viewer '
             'was stubbed; observation, inference, action decoding, PD control and '
             'mj_step come from the unmodified sim2sim.py.')
    # release the GL context explicitly; relying on __del__ at interpreter exit
    # raises EGL_NOT_INITIALIZED after EGL has already been torn down.
    renderer.close()
    # the depth camera holds a second GL context created inside MujocoEnv
    env.depth_render._render.close()
    args.output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in
                      ('status', 'forward_distance_m', 'mean_forward_speed_m_s',
                       'min_base_height_m', 'fell_below_0p3m_at_s')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
