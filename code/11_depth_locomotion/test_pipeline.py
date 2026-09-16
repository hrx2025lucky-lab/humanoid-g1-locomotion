"""Check the depth pipeline against the training-side definition.

The training side crops with end = size - margin (noise_model.crop_and_resize),
normalises over depth_range (0.0, 2.5) to (0, 1), and blurs with kernel 3 /
sigma 1. This test pins the behaviour that matters for matching that.
"""
import sys
import numpy as np

sys.path.insert(0, 'sim2sim')
import config as C
from sim2sim import DepthImagePipeline

H, W = C.RESIZED_DEPTH_IMAGE_SHAPE
CROP = C.OBS_DEPTH_IMAGE_CROP_REGION
y1, y2, x1, x2 = CROP
EXP_H, EXP_W = H - y1 - y2, W - x1 - x2

p = DepthImagePipeline(shape=(H, W), length=60, crop_region=CROP,
                       near_far_clip=(0.0, 2.5), nb_frames=C.HISTORY_LENGTH,
                       delay_ranges=(0, 0))

print(f'resize target (h,w) = ({H},{W}); crop margins = {CROP}; expected out = ({EXP_H},{EXP_W})')
assert (y2 == 0 or x2 == 0), 'this config is expected to exercise a zero margin'

raw = np.random.uniform(0.0, 3.0, size=C.DEPTH_IMAGE_SHAPE).astype(np.float32)
p.append(raw)
out = p.que[-1]
assert out.shape == (EXP_H, EXP_W), f'shape {out.shape} != {(EXP_H, EXP_W)}'
assert out.size > 0, 'empty crop: the zero bottom margin was mishandled'
print('shape ok:', out.shape)

assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6, (out.min(), out.max())
print(f'range ok: [{out.min():.4f}, {out.max():.4f}]')

near = np.full(C.DEPTH_IMAGE_SHAPE, 0.0, dtype=np.float32)
far = np.full(C.DEPTH_IMAGE_SHAPE, 2.5, dtype=np.float32)
p.append(far); hi = p.que[-1]
p.append(near); lo = p.que[-1]
assert abs(float(hi.mean()) - 1.0) < 1e-5, hi.mean()
print(f'far_clip -> {hi.mean():.6f} (expect 1.0)')
print(f'near_clip -> {lo.mean():.6f} (inpaint fills the all-zero mask)')

mid = np.full(C.DEPTH_IMAGE_SHAPE, 1.25, dtype=np.float32)
p.append(mid)
assert abs(float(p.que[-1].mean()) - 0.5) < 1e-5, p.que[-1].mean()
print(f'midpoint 1.25 m -> {p.que[-1].mean():.6f} (expect 0.5)')

obs = p.get_depth_obs()
assert obs.shape == (1, C.HISTORY_LENGTH, EXP_H, EXP_W), obs.shape
print('depth obs shape ok:', obs.shape)

grad = np.tile(np.linspace(0, 2.5, C.DEPTH_IMAGE_SHAPE[1], dtype=np.float32),
               (C.DEPTH_IMAGE_SHAPE[0], 1))
p.append(grad); g = p.que[-1]
assert np.all(np.diff(g.mean(0)) > 0), 'horizontal ordering lost'
col_lo = grad[0, int(x1 / W * C.DEPTH_IMAGE_SHAPE[1])] / 2.5
print(f'left-to-right ordering preserved; first col {g.mean(0)[0]:.3f} vs raw-left {col_lo:.3f}')
print('\nALL CHECKS PASSED')
