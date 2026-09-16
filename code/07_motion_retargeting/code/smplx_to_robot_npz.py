"""实践 7 TODO：单条 SMPL-X 动作 → G1，并导出 AMP 可用的 .npz。

为什么要新建这个文件
------------------
GMR 开源版的 `smplx_to_robot.py` 只导出 4 个字段
（`fps` / `root_pos` / `root_rot` / `dof_pos`），
缺少实践 8 的 AMP 判别器必需的 `local_body_pos` 与 `link_body_list`，
产物无法被 `motion_dataset.py` 加载。项目要求补齐这两项。

三个容易静默出错的地方
--------------------
1. 四元数顺序：MuJoCo 的 `qpos[3:7]` 是 wxyz，
   而 npz 规格要求 xyzw。实测参考样例
   `B12_-_walk_turn_right_(90)_stageii.npz` 的 `root_rot`
   各分量绝对值均值为 `[0.024, 0.025, 0.378, 0.862]`:
   最大值在第 3 位，确认是 xyzw。写错不会报错，
   只会让判别器看到一个"扭转的"机器人。

2. `local_body_pos` 不是世界系连杆位置。项目明确要求用
   `root_pos=0 / root_rot=单位四元数 / joint=dof_pos` 跑一次正运动学，
   得到的才是"根坐标系下的连杆位置"。
   直接存世界系位置或人体关键点都是错的:
   前者会把全局位移混进风格特征，后者根本不是机器人的量。

3. 连杆顺序：`link_body_list` 必须与 `local_body_pos` 第二维逐项对应。
   本脚本用 `KinematicsModel.body_names`，
   已实测与课程参考样例的 38 个连杆完全一致（含顺序）。

用法
----
    python smplx_to_robot_npz.py \
        --smplx_file motion_data/ACCAD/Male1Walking_c3d/B3_-_walk1_stageii.npz \
        --robot unitree_g1 \
        --save_path output/B3_-_walk1_stageii.npz

    # 只看可视化不存盘
    python smplx_to_robot_npz.py --smplx_file ... --no_save
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import torch

from general_motion_retargeting.kinematics_model import KinematicsModel

# SMPL-X 相关的导入放在 main() 里做惰性加载。
# 本模块的核心函数（qpos_to_fields / compute_local_body_pos / validate）
# 只依赖 numpy + torch + KinematicsModel，与 SMPL-X 无关。
# 顶层导入 smplx 会让"没装 SMPL-X 就无法校验导出逻辑"，
# 而导出逻辑恰恰是最容易出错、最该被单独测试的部分
# （见 sim2sim/verify_practice7_npz_export.py）。

HERE = pathlib.Path(__file__).parent
# 机器人 MJCF：正运动学与可视化必须用同一个模型，
# 否则算出的 local_body_pos 与看到的姿态对不上
ROBOT_XML = {
    "unitree_g1": HERE / ".." / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
    "unitree_g1_with_hands":
        HERE / ".." / "assets" / "unitree_g1" / "g1_mocap_29dof_with_hands.xml",
}


def qpos_to_fields(qpos_list: list[np.ndarray]) -> dict:
    """把逐帧 qpos 拆成 npz 的三个时序字段，并修正四元数顺序。

    qpos 布局（MuJoCo free joint）：
        [0:3]  root position
        [3:7]  root quaternion wxyz
        [7:]   29 个关节角
    """
    qpos = np.asarray(qpos_list, dtype=np.float64)
    root_pos = qpos[:, 0:3].copy()
    # 官方 §6.3：建议把第一帧的 x、y 平移到原点  p_xy(t) ← p_xy(t) − p_xy(0)
    # 实测课程参考样例 B12_-_walk_turn_right_(90)_stageii.npz 的首帧 xy 正是 [0, 0]，
    # 说明这是数据集的既定约定而非可选项。
    # 只平移 xy 不动 z：z 要保持"脚接近地面"的绝对高度，平移了会让机器人悬空或陷地。
    root_pos[:, :2] -= root_pos[0, :2]
    root_rot_wxyz = qpos[:, 3:7]
    # wxyz → xyzw：只是重排，不做归一化以外的任何改动
    root_rot = root_rot_wxyz[:, [1, 2, 3, 0]]
    # 数值积累会让模长偏离 1，判别器对朝向敏感，这里统一归一化
    norms = np.linalg.norm(root_rot, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("出现零四元数，重定向结果异常")
    root_rot = root_rot / norms
    dof_pos = qpos[:, 7:]
    return {"root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos}


def compute_local_body_pos(dof_pos: np.ndarray, robot: str,
                           device: str = "cpu") -> tuple[np.ndarray, list[str]]:
    """按项目要求算 local_body_pos：根节点置于原点、姿态为单位四元数。

        root position = [0, 0, 0]
        root rotation = identity
        joint position = dof_pos
                ↓  G1 forward kinematics
            local_body_pos

    这样得到的量只反映关节构型，与机器人走到哪、朝向哪无关:
    正是 AMP 判别器需要的"姿态风格"特征。
    """
    xml = ROBOT_XML.get(robot)
    if xml is None or not pathlib.Path(xml).exists():
        raise FileNotFoundError(f"找不到 {robot} 的 MJCF：{xml}")
    km = KinematicsModel(str(pathlib.Path(xml).resolve()), device=device)

    n = dof_pos.shape[0]
    if dof_pos.shape[1] != km._num_dof:
        raise ValueError(
            f"dof 维度不符：npz 给了 {dof_pos.shape[1]}，模型需要 {km._num_dof}")

    root_pos = torch.zeros(n, 3, dtype=torch.float, device=device)
    # 单位四元数，同样是 xyzw: GMR 的 torch_utils.quat_rotate 取 q[:, -1] 作 w，
    # 与 npz 规格一致，所以这里不需要再转换
    root_rot = torch.zeros(n, 4, dtype=torch.float, device=device)
    root_rot[:, 3] = 1.0
    dof = torch.as_tensor(dof_pos, dtype=torch.float, device=device)

    with torch.no_grad():
        body_pos, _ = km.forward_kinematics(root_pos, root_rot, dof)

    return body_pos.cpu().numpy().astype(np.float32), list(km.body_names)


def validate(data: dict) -> None:
    """落盘前自检: 这些条件一条不满足，实践 8 的加载器就会失败或静默学错。"""
    T = data["root_pos"].shape[0]
    checks = [
        ("fps 为正标量", np.ndim(data["fps"]) == 0 and float(data["fps"]) > 0),
        ("root_pos (T,3)", data["root_pos"].shape == (T, 3)),
        ("root_rot (T,4)", data["root_rot"].shape == (T, 4)),
        ("dof_pos (T,29)", data["dof_pos"].shape == (T, 29)),
        ("local_body_pos (T,N,3)",
         data["local_body_pos"].ndim == 3
         and data["local_body_pos"].shape[0] == T
         and data["local_body_pos"].shape[2] == 3),
        ("link_body_list 与第二维对应",
         len(data["link_body_list"]) == data["local_body_pos"].shape[1]),
        ("四元数已归一化",
         bool(np.allclose(np.linalg.norm(data["root_rot"], axis=1), 1.0, atol=1e-5))),
        ("无 NaN/Inf",
         all(np.isfinite(data[k]).all()
             for k in ("root_pos", "root_rot", "dof_pos", "local_body_pos"))),
        # 根节点在原点做的 FK，脚应当在 pelvis 下方: 用物理量兜住"算错了"
        ("局部坐标合理（最低点为负）",
         float(data["local_body_pos"][..., 2].min()) < 0.0),
    ]
    bad = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"    {'✅' if ok else '❌'} {name}")
    if bad:
        raise ValueError(f"自检未通过：{', '.join(bad)}")


def main() -> int:
    # 惰性导入：只有真正要跑重定向时才需要 SMPL-X 与 IK 依赖
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer
    from general_motion_retargeting.utils.smpl import (
        get_smplx_data_offline_fast,
        load_smplx_file,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smplx_file", required=True, help="SMPL-X 动作文件")
    ap.add_argument("--robot", default="unitree_g1",
                    choices=list(ROBOT_XML.keys()))
    ap.add_argument("--save_path", default=None, help="输出 .npz 路径")
    ap.add_argument("--no_save", action="store_true", help="只可视化不存盘")
    ap.add_argument("--record_video", action="store_true")
    ap.add_argument("--tgt_fps", type=int, default=30)
    ap.add_argument("--headless", action="store_true",
                    help="不开可视化窗口（批量/远程时用）")
    args = ap.parse_args()

    smplx_folder = HERE / ".." / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        args.smplx_file, smplx_folder)
    frames, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps)

    retarget = GMR(actual_human_height=human_height,
                   src_human="smplx", tgt_robot=args.robot)

    viewer = None
    if not args.headless:
        stem = pathlib.Path(args.smplx_file).stem
        viewer = RobotMotionViewer(
            robot_type=args.robot, motion_fps=fps,
            transparent_robot=0, record_video=args.record_video,
            video_path=f"videos/{args.robot}_{stem}.mp4")

    print(f"重定向 {len(frames)} 帧 @ {fps} fps …")
    qpos_list = []
    for i in range(len(frames)):
        qpos = retarget.retarget(frames[i])
        qpos_list.append(np.asarray(qpos, dtype=np.float64).copy())
        if viewer is not None:
            viewer.step(root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:],
                        human_motion_data=retarget.scaled_human_data,
                        rate_limit=True)
    if viewer is not None:
        viewer.close()

    if args.no_save:
        print("--no_save，仅可视化，未落盘")
        return 0

    save_path = args.save_path or (
        f"output/{pathlib.Path(args.smplx_file).stem}.npz")

    fields = qpos_to_fields(qpos_list)
    local_body_pos, link_body_list = compute_local_body_pos(
        fields["dof_pos"], args.robot)

    data = {
        "fps": np.float64(fps),
        "root_pos": fields["root_pos"],
        "root_rot": fields["root_rot"],
        "dof_pos": fields["dof_pos"],
        "local_body_pos": local_body_pos,
        "link_body_list": np.array(link_body_list, dtype="<U25"),
    }
    print(f"\n  自检 {save_path}")
    validate(data)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path, **data)
    print(f"\n✅ 已保存 {save_path}")
    print(f"   {len(qpos_list)} 帧 · {len(link_body_list)} 连杆 · {fps} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
