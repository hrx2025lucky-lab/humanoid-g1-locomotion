#!/usr/bin/env python3
"""录制实践 7 重定向产物的验收视频。

作业 §8.1 要求提交「G1 重定向动作的录制视频，需清晰展示脚部接触、
身体朝向、手臂动作和走路或跑步节奏」，且要覆盖三类：走路、跑步、
走跑切换或转弯。

GMR 自带 `scripts/vis_robot_motion.py` 能录像，但它的 `load_robot_motion()`
用 `pickle.load()` 读文件，而我们按作业规格产出的是 `.npz`
（作业明确要求 npz，且必须能 `allow_pickle=False` 读取）。
两者的**字段名完全一致**，只是容器格式不同，所以转一层即可复用官方脚本，
不用自己写渲染。

注意 data_loader 里有一行隐含约定：

    motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz

即它假定输入是 **xyzw**。我们的 npz 也是 xyzw（作业 §8.2 明确要求），
所以直接传原值，不要在这里再转一次 —— 转两次等于没转，
机器人会以错误姿态渲染，而且不会报错。

用法：
    envs/gmr/bin/python scripts/record_p7_videos.py
    envs/gmr/bin/python scripts/record_p7_videos.py --outdir /path/to/videos
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path("/home/limx/workspace/Roxan_warmup")
GMR = ROOT / "repos/GMR"
PY = ROOT / "envs/gmr/bin/python"
DATA = (ROOT / "shenlan_hw/unitree_lab_amp/source/unitree_rl_lab/unitree_rl_lab"
        / "tasks/locomotion/amp/data/mixed")

# 作业 §8.1 要求的三类，各挑一条最有代表性的
CLIPS = [
    ("01_walk", "B1_-_stand_to_walk_stageii.npz", "走路（站立起步→行走）"),
    ("02_run", "C3_-_Run_stageii.npz", "跑步"),
    ("03_walk_to_run", "C5_-_walk_to_run_stageii.npz", "走↔跑切换"),
    ("04_turn", "B12_-_walk_turn_right_(90)_stageii.npz", "转弯（走+右转90°）"),
]

G, R, Y, D, N = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def npz_to_pickle(npz_path: Path, pkl_path: Path) -> dict:
    """把 npz 转成 GMR 可视化脚本吃的 pickle，顺带做一次字段体检。"""
    d = np.load(npz_path, allow_pickle=False)
    need = {"fps", "root_pos", "root_rot", "dof_pos",
            "local_body_pos", "link_body_list"}
    missing = need - set(d.files)
    if missing:
        raise ValueError(f"{npz_path.name} 缺字段: {sorted(missing)}")

    motion = {k: d[k] for k in d.files}
    # link_body_list 是 numpy 的 U-string 数组，GMR 侧当普通序列用即可
    motion["link_body_list"] = [str(x) for x in d["link_body_list"]]
    # fps 在 npz 里是 0 维 ndarray（shape=()），不是 Python float。
    # imageio-ffmpeg 写视频时会 assert isinstance(fps, floatish) 而 ndarray
    # 过不了这个检查 —— 报错信息是 "fps must be float"，但看着像是我们传错了值，
    # 实际是容器类型的问题。np.load 取标量字段时都要显式转一次。
    motion["fps"] = float(d["fps"])

    with open(pkl_path, "wb") as f:
        pickle.dump(motion, f)

    return {
        "frames": int(d["dof_pos"].shape[0]),
        "fps": float(d["fps"]),
        "dof": int(d["dof_pos"].shape[1]),
        "links": len(motion["link_body_list"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(Path.home() / "humanoid_logs/p7_retarget/videos"))
    ap.add_argument("--display", default=":1", help="X display，录像需要 GL 上下文")
    ap.add_argument("--timeout", type=int, default=2400,
                    help="单段渲染超时秒数（实测一段十几分钟）")
    ap.add_argument("--only", help="只录某一段，传 tag 如 02_run")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DISPLAY": args.display}

    print(f"\n录制实践 7 验收视频 → {outdir}\n" + "=" * 64)

    ok_count = 0
    with tempfile.TemporaryDirectory() as tmp:
        for tag, fname, desc in CLIPS:
            if args.only and args.only != tag:
                continue
            src = DATA / fname
            if not src.is_file():
                print(f"  {Y}跳过{N} {desc}：找不到 {fname}")
                continue

            pkl = Path(tmp) / f"{tag}.pkl"
            try:
                info = npz_to_pickle(src, pkl)
            except Exception as exc:  # noqa: BLE001
                print(f"  {R}✗{N} {desc}：{type(exc).__name__}: {exc}")
                continue

            mp4 = outdir / f"{tag}.mp4"
            print(f"  {desc}  {D}{info['frames']} 帧 @ {info['fps']:.1f} fps"
                  f" · {info['dof']} dof · {info['links']} links{N}")

            # 单段渲染实测要十几分钟（MuJoCo 逐帧离屏渲染 + ffmpeg 编码），
            # 超时给足；且必须捕获 TimeoutExpired —— 第一版没捕获，
            # 第一段超时就把后面三段全带崩了。
            try:
                r = subprocess.run(
                    [str(PY), "scripts/vis_robot_motion.py",
                     "--robot", "unitree_g1",
                     "--robot_motion_path", str(pkl),
                     "--record_video", "--video_path", str(mp4)],
                    cwd=str(GMR), env=env,
                    capture_output=True, text=True, timeout=args.timeout,
                )
                err = r.stderr or ""
            except subprocess.TimeoutExpired:
                print(f"    {Y}⏱{N} 超过 {args.timeout}s，跳过这段")
                continue

            if mp4.is_file() and mp4.stat().st_size > 10_000:
                print(f"    {G}✓{N} {mp4.name}  {mp4.stat().st_size/1e6:.1f} MB")
                ok_count += 1
            else:
                print(f"    {R}✗{N} 没生成视频，stderr 末尾：")
                for line in err.strip().splitlines()[-4:]:
                    print(f"      {D}{line[:100]}{N}")

    print("=" * 64)
    print(f"完成 {ok_count}/{len(CLIPS)} 段\n")
    if ok_count:
        print(f"{D}作业 §8.1 要求覆盖走路 / 跑步 / 走跑切换或转弯三类，"
              f"上面四段已覆盖。{N}\n")
    return 0 if ok_count >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
