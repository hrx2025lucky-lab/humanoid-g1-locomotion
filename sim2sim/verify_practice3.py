#!/usr/bin/env python3
"""实践 3 —— HoST 增量动作空间的可复现验证。

文档里写着「16 项断言全部通过」，但那次验证是临时跑的，没有沉淀成脚本。
本文件把这些检查固化下来，任何时候都能重跑。

为什么这些检查值得单独写：sim2sim 的部署错误**几乎都不会报错**。
四元数读反、观测顺序拼错、历史帧方向写反、动作基准用了缩放值 ——
这些全都能正常跑完，只是机器人行为不对。所以只能靠断言逐项钉死。

四个被验证的核心函数直接从提交的实现 import，不复制代码，
保证验的就是跑的那份。

用法：
    python verify_practice3.py           # 只跑不依赖 MuJoCo 的纯函数检查
    python verify_practice3.py --full    # 额外跑 MuJoCo 状态读取检查
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# sim2sim/ -> g1_locomotion/ -> humanoid_practice/course_code/...
DEFAULT_PKG = os.path.join(
    HERE, "..", "..", "course_code", "shenlan_hw2_action_space", "shenlan_hw2_action_space"
)
PKG = os.path.abspath(os.environ.get("HW2_ACTION_SPACE_DIR", DEFAULT_PKG))
sys.path.insert(0, PKG)

from deploy_mujoco_host_student import (  # noqa: E402
    action_to_joint_targets,
    build_single_observation,
    get_gravity_orientation,
    read_robot_state,
    update_observation_history,
)

NUM_JOINTS = 23
OBS_DIM = 76
HISTORY_LEN = 6
ACTION_SCALE = 0.25

_results: list[tuple[bool, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


# ══════════════════════════════════════════════════════════════════════
# ① 投影重力
# ══════════════════════════════════════════════════════════════════════
def test_gravity() -> None:
    # 站直（单位四元数）：重力在机体系应指向 -z
    g_up = get_gravity_orientation(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    check("重力·站立姿态 ≈ (0,0,-1)",
          np.allclose(g_up, [0, 0, -1], atol=1e-6), f"{g_up}")

    # 模长恒为 1：它是方向向量，不是加速度。
    # 四元数必须精确归一化后再传入 —— 写 0.7071 的话输入模长就已经是
    # 0.99999，误差会原样传到重力向量上，测出来的是构造误差不是实现误差。
    r2 = float(np.sqrt(0.5))
    for label, q in (
        ("单位", [1.0, 0.0, 0.0, 0.0]),
        ("绕x90°侧躺", [r2, r2, 0.0, 0.0]),
        ("绕y90°俯仰", [r2, 0.0, r2, 0.0]),
        ("等权", [0.5, 0.5, 0.5, 0.5]),
    ):
        q = np.array(q, dtype=np.float32)
        q /= np.linalg.norm(q)
        g = get_gravity_orientation(q)
        check(f"重力·模长=1（{label}）",
              abs(np.linalg.norm(g) - 1.0) < 1e-5, f"‖g‖={np.linalg.norm(g):.6f}")

    # 仰面躺倒（绕 x 转 180°）：重力应指向 +z，这是站起任务的初始姿态
    g_supine = get_gravity_orientation(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    check("重力·仰躺姿态 z 分量为正",
          g_supine[2] > 0.99, f"{g_supine}")

    # wxyz vs xyzw：读反了会得到完全不同的结果，这里锁死顺序
    q_wxyz = np.array([r2, r2, 0.0, 0.0], dtype=np.float32)
    q_xyzw_misread = np.array([r2, 0.0, 0.0, r2], dtype=np.float32)
    check("重力·wxyz 与 xyzw 结果可区分（顺序被锁死）",
          not np.allclose(get_gravity_orientation(q_wxyz),
                          get_gravity_orientation(q_xyzw_misread), atol=1e-3))


# ══════════════════════════════════════════════════════════════════════
# ② 单帧观测布局
# ══════════════════════════════════════════════════════════════════════
def test_single_obs() -> None:
    # 每段填不同常数，这样能从数值直接反推出每段落在哪个区间
    ang_vel = np.full(3, 1.0, dtype=np.float32)
    grav = np.full(3, 2.0, dtype=np.float32)
    jpos = np.full(NUM_JOINTS, 3.0, dtype=np.float32)
    jvel = np.full(NUM_JOINTS, 4.0, dtype=np.float32)
    prev_a = np.full(NUM_JOINTS, 5.0, dtype=np.float32)

    obs = build_single_observation(ang_vel, grav, jpos, jvel, prev_a, ACTION_SCALE)

    check("单帧观测·维度 = 76", obs.shape == (OBS_DIM,), f"{obs.shape}")
    check("单帧观测·dtype = float32", obs.dtype == np.float32, f"{obs.dtype}")

    layout = [
        ("[0:3] 角速度", obs[0:3], 1.0),
        ("[3:6] 投影重力", obs[3:6], 2.0),
        ("[6:29] 关节角", obs[6:29], 3.0),
        ("[29:52] 关节角速度", obs[29:52], 4.0),
        ("[52:75] 上一帧动作", obs[52:75], 5.0),
    ]
    for label, seg, expect in layout:
        check(f"单帧观测·{label}",
              np.allclose(seg, expect), f"实际 {seg[0]}, 期望 {expect}")
    check("单帧观测·[75] action_scale 标量",
          np.isclose(obs[75], ACTION_SCALE), f"{obs[75]}")


# ══════════════════════════════════════════════════════════════════════
# ③ 历史缓冲的 FIFO 方向
# ══════════════════════════════════════════════════════════════════════
def test_history() -> None:
    total = OBS_DIM * HISTORY_LEN
    hist = np.zeros(total, dtype=np.float32)

    # 逐帧推入可辨识的常数帧，检查顺序是"旧 → 新"
    for i in range(1, HISTORY_LEN + 1):
        frame = np.full(OBS_DIM, float(i), dtype=np.float32)
        hist = update_observation_history(hist, frame, OBS_DIM, HISTORY_LEN)

    check("历史缓冲·总维度 = 456", hist.shape == (total,), f"{hist.shape}")
    check("历史缓冲·最新帧在末尾 [-76:]",
          np.allclose(hist[-OBS_DIM:], float(HISTORY_LEN)),
          f"{hist[-1]}")
    check("历史缓冲·次新帧在 [-152:-76]",
          np.allclose(hist[-2 * OBS_DIM:-OBS_DIM], float(HISTORY_LEN - 1)),
          f"{hist[-OBS_DIM - 1]}")
    check("历史缓冲·最旧帧在开头（方向未写反）",
          np.allclose(hist[:OBS_DIM], 1.0), f"{hist[0]}")

    # 再推一帧，最旧的应被挤掉
    hist = update_observation_history(
        hist, np.full(OBS_DIM, 99.0, dtype=np.float32), OBS_DIM, HISTORY_LEN)
    check("历史缓冲·推入新帧后最旧帧被挤出",
          np.allclose(hist[:OBS_DIM], 2.0) and np.allclose(hist[-OBS_DIM:], 99.0),
          f"头 {hist[0]}, 尾 {hist[-1]}")


# ══════════════════════════════════════════════════════════════════════
# ④ 增量动作空间（本实践核心）
# ══════════════════════════════════════════════════════════════════════
def test_action_space() -> None:
    rng = np.random.default_rng(0)
    current_q = rng.uniform(-1.0, 1.0, NUM_JOINTS).astype(np.float32)
    default_q = np.zeros(NUM_JOINTS, dtype=np.float32)

    # 零动作检查：最有效的单元测试，一次证明两件事
    zero_target = action_to_joint_targets(
        np.zeros(NUM_JOINTS, dtype=np.float32), current_q, ACTION_SCALE)
    check("增量式·a=0 时目标 = 当前姿态（是增量式）",
          np.allclose(zero_target, current_q), f"最大偏差 {np.abs(zero_target - current_q).max():.2e}")
    check("增量式·a=0 时目标 ≠ 默认姿态（不是残差式）",
          not np.allclose(zero_target, default_q))

    # 单位动作：位移量恰好等于 action_scale
    unit_target = action_to_joint_targets(
        np.ones(NUM_JOINTS, dtype=np.float32), current_q, ACTION_SCALE)
    check("增量式·a=1 时位移 = action_scale",
          np.allclose(unit_target - current_q, ACTION_SCALE),
          f"{(unit_target - current_q)[0]:.4f}")

    # 可达空间累积性：残差式做不到的关键性质。
    # 反复施加同方向动作，目标角应持续累积，而不是被锁在初始值附近。
    q = current_q.copy()
    for _ in range(50):
        q = action_to_joint_targets(np.ones(NUM_JOINTS, dtype=np.float32), q, ACTION_SCALE)
    drift = float(np.mean(q - current_q))
    check("增量式·50 步同向动作可累积到 12.5 rad（残差式不能）",
          abs(drift - 50 * ACTION_SCALE) < 1e-3, f"实际累积 {drift:.3f} rad")

    check("增量式·输出 dtype = float32", zero_target.dtype == np.float32)
    check("增量式·维度与输入一致", zero_target.shape == current_q.shape)


# ══════════════════════════════════════════════════════════════════════
# ⑤ MuJoCo 状态读取（需要真实模型，--full 才跑）
# ══════════════════════════════════════════════════════════════════════
def test_mujoco_state() -> None:
    import mujoco  # 延迟导入，纯函数检查不需要它
    import yaml

    with open(os.path.join(PKG, "configs", "g1.yaml")) as f:
        cfg = yaml.safe_load(f)
    xml = os.path.join(PKG, cfg["xml_path"])
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    q, q_obs, dq_obs, w_obs, g = read_robot_state(data, 1.0, 1.0, 1.0)

    check("MuJoCo·关节角维度 = 23", q.shape == (NUM_JOINTS,), f"{q.shape}")
    check("MuJoCo·投影重力维度 = 3", g.shape == (3,), f"{g.shape}")
    check("MuJoCo·qpos[7:] 与关节角一致",
          np.allclose(q, data.qpos[7:]), "")

    # 这条最容易踩：mj_data.qpos 是视图不是副本。
    # 若未显式拷贝，mj_step 之后 current_joint_positions 会被就地改写，
    # 导致 TODO 4 的动作基准用到的是"下一时刻"的姿态。
    before = q.copy()
    data.qpos[7:] += 0.5
    check("MuJoCo·关节角是副本而非 qpos 视图（不共享内存）",
          np.allclose(q, before), "读到的数组被 qpos 就地改写了")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true",
                   help="额外跑需要加载 MuJoCo 模型的检查")
    args = p.parse_args()

    print()
    print("═" * 64)
    print("实践 3 验证 —— HoST 增量动作空间")
    print("═" * 64)
    print(f"  被验证的实现: {PKG}/deploy_mujoco_host_student.py")
    print()

    groups = [
        ("① 投影重力", test_gravity),
        ("② 单帧观测布局", test_single_obs),
        ("③ 历史缓冲 FIFO", test_history),
        ("④ 增量动作空间", test_action_space),
    ]
    if args.full:
        groups.append(("⑤ MuJoCo 状态读取", test_mujoco_state))

    for title, fn in groups:
        start = len(_results)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            check(f"{title} 执行异常", False, repr(exc))
        print(f"── {title} ──")
        for ok, name, detail in _results[start:]:
            mark = "✅" if ok else "❌"
            print(f"  {mark} {name}" + (f"   [{detail}]" if detail and not ok else ""))
        print()

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print("═" * 64)
    print(f"结果：{passed}/{total} 项通过"
          + ("" if passed == total else "  ❌ 见上方失败项"))
    if not args.full:
        print("提示：加 --full 可另外验证 MuJoCo 状态读取（含 qpos 视图/副本陷阱）")
    print("═" * 64)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
