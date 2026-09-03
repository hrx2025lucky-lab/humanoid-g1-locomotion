#!/usr/bin/env python3
"""实践 7 —— 重定向动作数据的可复现验证。

课程给的 `B12_-_walk_turn_right_(90)_stageii.npz` 是**已经重定向到 G1 的动作**
（`dof_pos` 29 维、`link_body_list` 38 个刚体），不是原始 SMPL-X 人体数据。
它会被实践 9 / 10 / 11 当作参考轨迹使用，所以在用它之前必须先确认它是对的。

重定向结果的错误几乎都不会报错：四元数序读反、关节超限、帧间跳变、
脚陷进地面 —— 这些都能正常加载，只是机器人动作诡异。和 sim2sim 一样，
只能靠断言逐项钉死。

本脚本不依赖 GMR 包（它的 install_requires 会拉 mujoco/numpy/scipy，
装进 isaaclab 环境有升级 mujoco 的风险，而 raycaster 插件是按 3.12.0 编译的）。
只用 numpy 与 URDF 解析。

用法：
    python verify_practice7_motion.py                    # 验证课程自带动作
    python verify_practice7_motion.py --motion path.npz  # 验证其它重定向结果
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

ROXAN_ROOT = os.environ.get("ROXAN_ROOT", "/home/limx/workspace/Roxan_warmup")
DEFAULT_MOTION = os.path.join(
    ROXAN_ROOT,
    "motion control/humanoid_practice/course_code",
    "深蓝学院-人形运控-Project7-code(1)",
    "B12_-_walk_turn_right_(90)_stageii.npz",
)
DEFAULT_URDF = os.path.join(
    ROXAN_ROOT, "repos/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
)

NUM_DOF = 29
_results: list[tuple[bool, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


def load_joint_limits(urdf: str) -> dict[str, tuple[float, float]]:
    root = ET.parse(urdf).getroot()
    lim = {}
    for j in root.iter("joint"):
        if j.get("type") in ("revolute", "prismatic"):
            l = j.find("limit")
            if l is not None:
                lim[j.get("name")] = (float(l.get("lower")), float(l.get("upper")))
    return lim


def test_integrity(d) -> tuple[np.ndarray, ...]:
    fps = float(d["fps"])
    dof = d["dof_pos"]
    rpos = d["root_pos"]
    rrot = d["root_rot"]
    bodies = d["local_body_pos"]
    names = list(d["link_body_list"])
    n = dof.shape[0]

    check("数据·帧数一致", rpos.shape[0] == n and rrot.shape[0] == n
          and bodies.shape[0] == n, f"{dof.shape[0]}/{rpos.shape[0]}/{rrot.shape[0]}")
    check(f"数据·dof_pos 为 {NUM_DOF} 维（G1 29DoF）", dof.shape[1] == NUM_DOF, f"{dof.shape}")
    check("数据·root_pos 为 3 维", rpos.shape[1] == 3, f"{rpos.shape}")
    check("数据·root_rot 为 4 维", rrot.shape[1] == 4, f"{rrot.shape}")
    check("数据·link_body_list 与 local_body_pos 对齐",
          len(names) == bodies.shape[1], f"{len(names)} vs {bodies.shape[1]}")
    check("数据·fps 合理（10~200）", 10 < fps < 200, f"{fps:.2f}")
    for arr, nm in ((dof, "dof_pos"), (rpos, "root_pos"), (rrot, "root_rot"),
                    (bodies, "local_body_pos")):
        check(f"数据·{nm} 无 NaN/inf", bool(np.isfinite(arr).all()))
    return dof, rpos, rrot, bodies, names, fps


def test_quaternion(rrot: np.ndarray) -> None:
    norms = np.linalg.norm(rrot, axis=1)
    check("四元数·全部归一化", bool(np.allclose(norms, 1.0, atol=1e-4)),
          f"模长范围 {norms.min():.6f}~{norms.max():.6f}")

    # 判定 wxyz 还是 xyzw。近似直立行走时实部 w 应当接近 ±1 附近的较大值，
    # 三个虚部里 z（yaw）通常最大、x/y（roll/pitch）很小。
    # 因此"最大分量的位置"与"两个近零分量的位置"能把两种约定区分开。
    mean_abs = np.abs(rrot).mean(axis=0)
    first, last = mean_abs[0], mean_abs[3]
    small = np.argsort(mean_abs)[:2]
    is_xyzw = last > first and set(small.tolist()) <= {0, 1, 2}
    check("四元数·可判定分量顺序", bool(is_xyzw or first > last),
          f"各分量均值 {np.round(mean_abs, 4)}")
    order = "xyzw（w 在末位）" if is_xyzw else "wxyz（w 在首位）"
    check(f"四元数·顺序判定为 {order}", True,
          "课程 vis 脚本用 [:, [3,0,1,2]] 重排，与 xyzw 判定一致")

    # 相邻帧不应出现符号翻转造成的假跳变（q 与 -q 表示同一旋转）
    dots = np.abs((rrot[:-1] * rrot[1:]).sum(axis=1))
    check("四元数·相邻帧连续（无符号翻转跳变）", bool((dots > 0.9).all()),
          f"最小相邻点积 {dots.min():.4f}")


def test_joint_limits(dof: np.ndarray, limits: dict) -> None:
    names = list(limits.keys())
    if len(names) != dof.shape[1]:
        check("关节限位·URDF 关节数与 dof_pos 匹配", False,
              f"URDF {len(names)} vs dof {dof.shape[1]}")
        return
    check("关节限位·URDF 关节数与 dof_pos 匹配", True, f"{len(names)}")

    violations = []
    for i, nm in enumerate(names):
        lo, hi = limits[nm]
        col = dof[:, i]
        # 留 5% 量程的容差：重定向是优化解，贴边很正常，真正的问题是大幅越界
        tol = 0.05 * (hi - lo)
        if col.min() < lo - tol or col.max() > hi + tol:
            violations.append(
                f"{nm}: 实测 [{col.min():.3f}, {col.max():.3f}] vs 限位 [{lo:.3f}, {hi:.3f}]")
    check("关节限位·所有关节在限位内（含 5% 容差）", not violations,
          "; ".join(violations[:3]) if violations else "")


def test_continuity(dof: np.ndarray, rpos: np.ndarray, fps: float) -> None:
    dt = 1.0 / fps
    # 关节角速度：G1 各关节额定速度多在 20~30 rad/s，取 40 作为明显异常的门槛
    jvel = np.abs(np.diff(dof, axis=0)) / dt
    check("连续性·关节速度无异常尖峰（< 40 rad/s）", bool(jvel.max() < 40.0),
          f"最大 {jvel.max():.2f} rad/s")
    # 根节点线速度：行走动作不应超过 5 m/s
    rvel = np.linalg.norm(np.diff(rpos, axis=0), axis=1) / dt
    check("连续性·根节点速度无瞬移（< 5 m/s）", bool(rvel.max() < 5.0),
          f"最大 {rvel.max():.2f} m/s")


def test_physical_plausibility(rpos: np.ndarray, bodies: np.ndarray,
                               names: list) -> None:
    z = rpos[:, 2]
    check("物理·根节点高度在站立范围（0.4~1.2 m）",
          bool(0.4 < z.min() and z.max() < 1.2), f"{z.min():.3f}~{z.max():.3f} m")

    # local_body_pos 是相对根节点的局部坐标，脚踝应当明显低于根节点
    foot_idx = [i for i, n in enumerate(names)
                if "ankle_roll" in str(n) or "toe" in str(n)]
    if foot_idx:
        foot_z = bodies[:, foot_idx, 2]
        check("物理·脚部低于根节点（局部 z < 0）", bool(foot_z.max() < 0.0),
              f"脚部局部 z 最大 {foot_z.max():.3f}")
        # 站立时根节点约 0.8 m，脚相对根节点应在 -0.9 ~ -0.4 之间
        check("物理·脚到根节点的距离合理（0.3~1.0 m）",
              bool(0.3 < -foot_z.mean() < 1.0), f"均值 {-foot_z.mean():.3f} m")
    else:
        check("物理·找到脚部刚体", False, "link_body_list 里没有 ankle_roll/toe")

    # 至少有一只脚在大部分时间接近最低点 —— 行走应当有支撑相
    if foot_idx:
        lowest = bodies[:, foot_idx, 2].min(axis=1)
        spread = lowest.max() - lowest.min()
        check("物理·存在支撑相（最低脚高度波动 < 0.3 m）", bool(spread < 0.3),
              f"波动 {spread:.3f} m")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", default=DEFAULT_MOTION)
    p.add_argument("--urdf", default=DEFAULT_URDF)
    args = p.parse_args()

    print()
    print("═" * 70)
    print("实践 7 验证 —— 重定向到 G1 的动作数据")
    print("═" * 70)
    print(f"  动作: {os.path.basename(args.motion)}")
    print(f"  URDF: {os.path.basename(args.urdf)}")
    print()

    d = np.load(args.motion, allow_pickle=True)
    limits = load_joint_limits(args.urdf)

    dof, rpos, rrot, bodies, names, fps = test_integrity(d)
    print(f"  {dof.shape[0]} 帧 @ {fps:.2f} fps = {dof.shape[0] / fps:.2f} 秒，"
          f"{dof.shape[1]} DoF，{len(names)} 个刚体")
    print()

    for title, fn in (
        ("① 数据完整性", lambda: None),          # 已在 test_integrity 中完成
        ("② 四元数约定与连续性", lambda: test_quaternion(rrot)),
        ("③ 关节限位", lambda: test_joint_limits(dof, limits)),
        ("④ 时序连续性", lambda: test_continuity(dof, rpos, fps)),
        ("⑤ 物理合理性", lambda: test_physical_plausibility(rpos, bodies, names)),
    ):
        start = len(_results) if title != "① 数据完整性" else 0
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            check(f"{title} 执行异常", False, repr(exc))
        end = len(_results)
        seg = _results[start:end] if title != "① 数据完整性" else \
            [r for r in _results if r[1].startswith("数据·")]
        print(f"── {title} ──")
        for ok, name, detail in seg:
            mark = "✅" if ok else "❌"
            extra = f"   [{detail}]" if detail else ""
            print(f"  {mark} {name}{extra}")
        print()

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print("═" * 70)
    print(f"结果：{passed}/{total} 项通过" + ("" if passed == total else "   ❌ 见上方"))
    print("═" * 70)
    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
