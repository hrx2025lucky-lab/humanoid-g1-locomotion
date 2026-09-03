#!/usr/bin/env python3
"""实践 11 —— 深度图处理流水线的可复现验证。

跑酷策略靠一张前视深度图判断台阶在哪。这张图从仿真相机出来后要经过
五步处理才能喂进网络，**训练侧与部署侧必须逐步一致**：

    resize → crop → inpaint → gaussian blur → clip & normalize

和 sim2sim 的其它环节一样，这五步错了都不会报错：
resize 的 dsize 传反只会把图转置、归一化的零点差一点只会让距离整体平移、
inpaint 漏掉会让无效点（值 0）被当成"紧贴镜头的障碍"。
只能靠断言逐项钉死。

被验证的实现直接从提交的 sim2sim.py import，不复制代码。

用法：
    python verify_practice11_depth.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROXAN_ROOT = os.environ.get("ROXAN_ROOT", "/home/limx/workspace/Roxan_warmup")
PKG = os.environ.get(
    "P11_SIM2SIM_DIR",
    os.path.join(ROXAN_ROOT, "motion control/humanoid_practice/course_code",
                 "sim2sim", "sim2sim"),
)
sys.path.insert(0, PKG)

_results: list[tuple[bool, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


def make_pipeline(**kw):
    from sim2sim import DepthImagePipeline
    defaults = dict(shape=(36, 64), length=60, crop_region=(2, 2, 4, 4),
                    near_far_clip=(0.0, 2.5), nb_frames=8, delay_ranges=(0, 0))
    defaults.update(kw)
    return DepthImagePipeline(**defaults)


def test_shape_and_crop() -> None:
    """① resize + ② crop：输出尺寸必须是 目标尺寸 − 裁剪量。"""
    h, w = 36, 64
    y1, y2, x1, x2 = 2, 3, 4, 5
    pipe = make_pipeline(shape=(h, w), crop_region=(y1, y2, x1, x2))

    # 故意用一个与目标尺寸不同、且长宽不等的输入，能暴露 dsize 传反
    raw = np.random.rand(120, 200).astype(np.float32)
    pipe.append(raw)
    out = pipe.que[-1]

    expect = (h - y1 - y2, w - x1 - x2)
    check("① resize+crop·输出尺寸正确", out.shape == expect,
          f"实测 {out.shape}，期望 {expect}")
    check("① resize·dsize 未传反（非方形也正确）", out.shape[0] != out.shape[1],
          f"{out.shape} —— 若把 (h,w) 当 dsize 传入会得到转置的结果")


def test_nearest_interpolation() -> None:
    """① resize 必须用最近邻：深度值是物理距离，不能插值混合。

    构造一张只有两种深度值的图（前景 0.5 / 背景 2.0）。
    最近邻缩放后仍然只有这两个值；双线性会在边界产生中间值 ——
    那个位置其实什么都没有，策略会把它当成真实的障碍面。
    """
    pipe = make_pipeline(shape=(16, 16), crop_region=(0, 0, 0, 0),
                         near_far_clip=(0.0, 4.0))
    raw = np.full((64, 64), 2.0, dtype=np.float32)
    raw[:32, :] = 0.5
    pipe.append(raw)
    out = pipe.que[-1] * 4.0  # 反归一化回物理距离

    uniq = np.unique(np.round(out, 3))
    # 经过 inpaint 与高斯模糊后会有少量新值，所以只检查"没有大面积中间值"
    mid = ((out > 0.7) & (out < 1.8)).mean()
    check("① resize·用最近邻（无大面积插值中间值）", mid < 0.25,
          f"介于前景/背景之间的像素占比 {mid:.1%}，唯一值数 {len(uniq)}")


def test_inpaint() -> None:
    """③ inpaint：深度相机的无效点是 0，必须补掉。

    0 在归一化后是**最近距离**，语义与"没测到"完全相反 ——
    不补的话策略会以为镜头前紧贴着一堵墙。
    """
    pipe = make_pipeline(shape=(32, 32), crop_region=(0, 0, 0, 0),
                         near_far_clip=(0.0, 2.5))
    raw = np.full((32, 32), 1.5, dtype=np.float32)
    raw[12:20, 12:20] = 0.0          # 中间挖一个无效洞
    pipe.append(raw)
    out = pipe.que[-1]

    hole = out[12:20, 12:20]
    check("③ inpaint·无效洞被填补（不再是 0）", float(hole.min()) > 0.05,
          f"洞内最小值 {hole.min():.4f}")
    check("③ inpaint·填补值接近邻域深度", abs(float(hole.mean()) - 1.5 / 2.5) < 0.25,
          f"洞内均值 {hole.mean():.4f}，邻域归一化值 {1.5 / 2.5:.4f}")


def test_blur_is_mild() -> None:
    """④ 高斯模糊必须是轻度的：台阶边缘正是跑酷策略最需要的信息。"""
    pipe = make_pipeline(shape=(32, 32), crop_region=(0, 0, 0, 0),
                         near_far_clip=(0.0, 2.0))
    raw = np.full((32, 32), 2.0, dtype=np.float32)
    raw[:, :16] = 0.5                 # 一条竖直阶跃边
    pipe.append(raw)
    out = pipe.que[-1]

    row = out[16]
    step = float(row[-1] - row[0])
    # 边缘过渡宽度：落在两端值之间 10%~90% 的像素数
    lo, hi = row.min(), row.max()
    trans = int((((row > lo + 0.1 * (hi - lo)) & (row < lo + 0.9 * (hi - lo)))).sum())
    check("④ 模糊·阶跃仍然保留", step > 0.5, f"两端落差 {step:.3f}")
    check("④ 模糊·过渡带足够窄（≤5 像素，边缘未被抹平）", trans <= 5,
          f"过渡带 {trans} 像素")


def test_normalization() -> None:
    """⑤ clip + 归一化：零点与量程的定义必须与训练侧一致。"""
    near, far = 0.3, 2.5
    pipe = make_pipeline(shape=(16, 16), crop_region=(0, 0, 0, 0),
                         near_far_clip=(near, far))

    for depth, label in ((near, "近裁剪面"), (far, "远裁剪面"),
                         ((near + far) / 2, "中点")):
        pipe.reset()
        pipe.append(np.full((16, 16), depth, dtype=np.float32))
        out = pipe.que[-1]
        expect = (depth - near) / (far - near)
        check(f"⑤ 归一化·{label} → {expect:.2f}",
              abs(float(out.mean()) - expect) < 0.02,
              f"实测 {out.mean():.4f}")

    # 超出量程的值必须被 clip 住，而不是外推
    pipe.reset()
    pipe.append(np.full((16, 16), far + 5.0, dtype=np.float32))
    check("⑤ 归一化·超远值被 clip 到 1.0",
          abs(float(pipe.que[-1].mean()) - 1.0) < 1e-3,
          f"{pipe.que[-1].mean():.4f}")
    pipe.reset()
    pipe.append(np.full((16, 16), 0.0, dtype=np.float32))
    check("⑤ 归一化·超近值被 clip 到 0.0",
          float(pipe.que[-1].mean()) < 0.05, f"{pipe.que[-1].mean():.4f}")

    check("⑤ 归一化·输出恒在 [0,1]",
          bool(0.0 <= pipe.que[-1].min() and pipe.que[-1].max() <= 1.0))


def test_history_stack() -> None:
    """历史帧堆叠：跑酷是 POMDP，单帧深度图看不出自己在接近还是远离。"""
    pipe = make_pipeline(shape=(16, 16), crop_region=(0, 0, 0, 0),
                         near_far_clip=(0.0, 2.0), nb_frames=8)
    for i in range(60):
        pipe.append(np.full((16, 16), 0.5 + i * 0.01, dtype=np.float32))
    obs = pipe.get_depth_obs()
    # 形状是 (batch, nb_frames, h, w) —— 批次维在前，帧维在轴 1。
    # 第一次写断言时误以为帧维在轴 0，测出 shape[0]==1 判失败；
    # 错的是测试不是实现。
    check("历史·堆叠 nb_frames=8 帧（形状 (1,8,h,w)）",
          obs.ndim == 4 and obs.shape[1] == 8, f"{obs.shape}")
    check("历史·缓冲长度为 length=60", len(pipe.que) == 60, f"{len(pipe.que)}")


def main() -> int:
    print()
    print("═" * 68)
    print("实践 11 验证 —— 深度图处理流水线")
    print("═" * 68)
    print(f"  被验证的实现: {PKG}/sim2sim.py")
    print("  流水线: resize → crop → inpaint → blur → clip&normalize")
    print()

    groups = [
        ("① resize 与 ② crop", test_shape_and_crop),
        ("① 最近邻插值", test_nearest_interpolation),
        ("③ inpaint 补洞", test_inpaint),
        ("④ 高斯模糊", test_blur_is_mild),
        ("⑤ clip 与归一化", test_normalization),
        ("历史帧堆叠", test_history_stack),
    ]
    for title, fn in groups:
        start = len(_results)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            check(f"{title} 执行异常", False, repr(exc))
        print(f"── {title} ──")
        for ok, name, detail in _results[start:]:
            mark = "✅" if ok else "❌"
            print(f"  {mark} {name}" + (f"   [{detail}]" if detail else ""))
        print()

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print("═" * 68)
    print(f"结果：{passed}/{total} 项通过" + ("" if passed == total else "   ❌ 见上方"))
    print("═" * 68)
    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
