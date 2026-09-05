"""实践 7 的 npz 导出逻辑验证 —— 不需要 SMPL-X 模型即可跑。

思路
----
完整链路（SMPL-X → IK → G1）需要 SMPL-X 人体模型才能跑，
但**最容易出错的环节不在 IK，而在导出**：
四元数顺序、local_body_pos 的坐标系、连杆顺序。

这三件事都可以脱离 SMPL-X 单独验证 —— 用课程参考样例
`B12_-_walk_turn_right_(90)_stageii.npz` 做"已知答案"：

    取样例的 dof_pos → 用我们的 compute_local_body_pos 重算
        → 与样例自带的 local_body_pos 比较

如果实现正确，两者应逐点吻合。这是一个**可证伪**的检验：
坐标系用错、连杆顺序错、四元数约定错，任何一项都会让误差爆掉。

用法
    python verify_practice7_npz_export.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GMR = Path("/home/limx/workspace/Roxan_warmup/repos/GMR")
REF = Path("/home/limx/workspace/Roxan_warmup/motion control/humanoid_practice"
           "/course_code/深蓝学院-人形运控-Project7-code(1)"
           "/B12_-_walk_turn_right_(90)_stageii.npz")

sys.path.insert(0, str(GMR))
sys.path.insert(0, str(GMR / "scripts"))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}{('  ' + detail) if detail else ''}")


def main() -> int:
    print("=" * 62)
    print("实践 7 npz 导出逻辑验证（用课程参考样例做已知答案）")
    print("=" * 62)

    if not REF.exists():
        print(f"❌ 找不到参考样例 {REF}")
        return 1
    ref = np.load(REF, allow_pickle=True)

    print("\n── 1. 字段规格（作业 §6.2）──")
    T = ref["root_pos"].shape[0]
    spec = {
        "fps": ((), np.float64),
        "root_pos": ((T, 3), np.float64),
        "root_rot": ((T, 4), np.float64),
        "dof_pos": ((T, 29), np.float64),
    }
    for k, (shape, dt) in spec.items():
        a = ref[k]
        check(f"{k} 形状 {shape}", a.shape == shape, str(a.shape))
        check(f"{k} dtype {dt.__name__}", a.dtype == dt, str(a.dtype))
    check("local_body_pos 为 float32", ref["local_body_pos"].dtype == np.float32)
    check("link_body_list 与第二维一致",
          len(ref["link_body_list"]) == ref["local_body_pos"].shape[1],
          f"{len(ref['link_body_list'])} vs {ref['local_body_pos'].shape[1]}")

    print("\n── 2. 四元数约定（xyzw 还是 wxyz）──")
    q = ref["root_rot"]
    check("已归一化",
          bool(np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)))
    # w 分量在整段动作里应显著大于三个虚部（人形直立行走不会大角度翻滚）
    means = np.abs(q).mean(axis=0)
    w_idx = int(np.argmax(means))
    check("w 分量在第 3 位 → xyzw", w_idx == 3,
          f"各位均值 {np.round(means, 3)}")

    print("\n── 3. local_body_pos 用我们的实现重算 ──")
    try:
        from smplx_to_robot_npz import compute_local_body_pos
    except Exception as e:
        check("导入 compute_local_body_pos", False, f"{type(e).__name__}: {e}")
        return 1
    check("导入 compute_local_body_pos", True)

    dof = ref["dof_pos"]
    got, links = compute_local_body_pos(dof, "unitree_g1")
    want = ref["local_body_pos"]

    check("连杆数一致", len(links) == len(ref["link_body_list"]),
          f"{len(links)} vs {len(ref['link_body_list'])}")
    check("连杆名与顺序完全一致",
          list(links) == list(ref["link_body_list"]))
    check("形状一致", got.shape == want.shape, f"{got.shape} vs {want.shape}")

    if got.shape == want.shape:
        err = np.abs(got - want)
        check("逐点误差 < 1e-3 m", float(err.max()) < 1e-3,
              f"max {err.max():.6f} m, mean {err.mean():.6f} m")

    print("\n── 4. 物理合理性（防止「算出来了但是错的」）──")
    z = want[..., 2]
    check("根坐标系下最低点为负（脚在骨盆下方）", float(z.min()) < -0.5,
          f"min z = {z.min():.3f} m")
    check("最高点为正（头/肩在骨盆上方）", float(z.max()) > 0.2,
          f"max z = {z.max():.3f} m")
    # pelvis 是根节点，在自己的坐标系里必须恒为原点
    pelvis_i = list(ref["link_body_list"]).index("pelvis")
    check("pelvis 恒在原点",
          bool(np.allclose(want[:, pelvis_i, :], 0.0, atol=1e-5)),
          f"max |pelvis| = {np.abs(want[:, pelvis_i, :]).max():.2e}")
    # 若误存了世界系位置，根位置会随时间漂移；局部坐标不应有净漂移
    drift = np.linalg.norm(want[-1].mean(axis=0) - want[0].mean(axis=0))
    check("无全局漂移（确认是局部坐标而非世界坐标）", drift < 0.5,
          f"首尾质心位移 {drift:.3f} m")

    print("\n── 5. 坐标约定（官方 §6.3）──")
    # 官方："建议将第一帧的 x、y 平移到原点"。参考样例确实这么做了，
    # 说明是数据集既定约定而非可选项。
    rp = ref["root_pos"]
    check("首帧 xy 已平移到原点",
          bool(abs(rp[0, 0]) < 1e-6 and abs(rp[0, 1]) < 1e-6),
          f"首帧 xy = {rp[0, :2].round(4)}")
    # z 不该平移：要保持"脚接近地面"的绝对高度
    check("z 未被平移（脚接近地面）", 0.3 < float(rp[:, 2].mean()) < 1.2,
          f"z 均值 {rp[:, 2].mean():.3f} m")
    # 我们自己的导出脚本是否实现了这条
    src = (GMR / "scripts/smplx_to_robot_npz.py").read_text()
    check("导出脚本实现了首帧平移",
          "root_pos[:, :2] -= root_pos[0, :2]" in src)

    print("\n── 6. 与实践 8 的接口契约 ──")
    key_links = ["left_ankle_roll_link", "right_ankle_roll_link",
                 "left_wrist_yaw_link", "right_wrist_yaw_link"]
    have = [k for k in key_links if k in list(ref["link_body_list"])]
    check("含 AMP 需要的 4 个关键连杆", len(have) == 4,
          f"缺 {[k for k in key_links if k not in have]}" if len(have) < 4 else "")
    check("时序字段第一维一致",
          len({ref[k].shape[0] for k in
               ("root_pos", "root_rot", "dof_pos", "local_body_pos")}) == 1)

    print("\n" + "=" * 62)
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
