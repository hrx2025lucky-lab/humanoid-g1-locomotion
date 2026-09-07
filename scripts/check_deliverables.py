#!/usr/bin/env python3
"""核对每个实践的**提交材料**是否齐全。

与 audit_against_rubric.py 的分工：
  audit_against_rubric.py  查"做没做对"——代码实现、算法正确性
  这个脚本               查"交没交全"——作业明确列出的提交物

作业普遍要求：代码、测试输出截图、训练曲线、回放视频、checkpoint。
这些东西散落在各个目录，跑完训练容易忘记导出，等到打包时才发现缺。

用法：
    python3 scripts/check_deliverables.py
    python3 scripts/check_deliverables.py --practice 7
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

WS = Path("/home/limx/workspace/Roxan_warmup")
LOGS = Path.home() / "humanoid_logs"
CODE = WS / "motion control/humanoid_practice/course_code"
REPO = Path(__file__).resolve().parents[1]

G, R, Y, D, B, N = ("\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")


def log_shows_training(pattern: str) -> tuple[bool, str]:
    """训练日志不能只看"文件在不在"——失败的日志同样是个文件。

    实践 10 就栽在这：smoke 跑挂了（KeyError: '/World/ground'），
    日志文件照样生成，has_files 一路判 ✅，于是"根本没跑起来"被报成"材料齐全"。
    这里要求日志里真的出现训练循环，并且没有以 Traceback 收尾。
    """
    fs = [f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)]
    if not fs:
        return False, "没有日志文件"
    ok, bad = [], []
    for f in fs:
        try:
            t = Path(f).read_text(errors="ignore")
        except OSError:
            continue
        if re.search(r"Learning iteration \d+/", t):
            ok.append(f)
        elif "Traceback (most recent call last)" in t:
            err = re.findall(r"^(\w*(?:Error|Exception)):.*$", t, re.M)
            bad.append(f"{os.path.basename(f)}→{err[-1] if err else '异常'}")
    if ok:
        return True, f"{len(ok)} 份日志有训练循环，最新 {os.path.basename(max(ok, key=os.path.getmtime))}"
    return False, f"{len(fs)} 份日志都没跑起训练循环" + (f"（{'; '.join(bad[:2])}）" if bad else "")


def has_files(pattern: str, min_count: int = 1, min_size: int = 1000) -> tuple[bool, str]:
    """glob 匹配并检查文件确实有内容 —— 空文件比没文件更容易骗过检查。"""
    fs = [f for f in glob.glob(pattern, recursive=True)
          if os.path.isfile(f) and os.path.getsize(f) >= min_size]
    if len(fs) >= min_count:
        newest = max(fs, key=os.path.getmtime)
        return True, f"{len(fs)} 个，最新 {os.path.basename(newest)}"
    return False, f"只有 {len(fs)} 个（需 {min_count}）"


def has_checkpoint(pattern: str, min_runs: int = 1, min_iter: int = 100) -> tuple[bool, str]:
    """检查 checkpoint，但要求它真的是训练成果。

    为什么不能直接用 has_files：
      冒烟测试会留下 model_3.pt 这种 3 轮的存档，文件存在、大小也正常，
      has_files 一律判 ✅——于是"训练根本没跑完"会被报成"材料已交齐"。
      假阳性比漏报危险：漏报只是多看一眼，假阳性会让人以为做完了。

    两条额外约束：
      1. 从文件名 model_<N>.pt 解析轮数，N < min_iter 的当冒烟丢弃；
      2. 按 run 目录去重计数——消融实验要的是 3 次独立训练，
         而单次训练就能存出几十个 model_*.pt，按文件数会轻易凑够。
    """
    runs: dict[str, int] = {}
    for f in glob.glob(pattern, recursive=True):
        if not os.path.isfile(f) or os.path.getsize(f) < 1000:
            continue
        m = re.match(r"model_(\d+)\.pt$", os.path.basename(f))
        if not m or int(m.group(1)) < min_iter:
            continue
        d = os.path.dirname(f)
        runs[d] = max(runs.get(d, 0), int(m.group(1)))
    if len(runs) >= min_runs:
        best = max(runs.values())
        return True, f"{len(runs)} 次训练，最大 model_{best}.pt"
    smoke = len([f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)])
    extra = f"（另有 {smoke} 个文件但轮数 < {min_iter}，判为冒烟存档）" if smoke else ""
    return False, f"只有 {len(runs)} 次有效训练（需 {min_runs}）{extra}"


def latest_run(pattern: str) -> str | None:
    ds = [d for d in glob.glob(pattern, recursive=True) if os.path.isdir(d)]
    return max(ds, key=os.path.getmtime) if ds else None


# (实践号, 名称, [(材料名, 检查函数, 是否必需)])
def build_specs():
    return [
        ("1", "环境搭建与验证", [
            # 官方 §5：PDF 报告 3-6 页，含 4 类截图 + 项目理解 5 问 + 分析 3 问
            ("实验报告",
             lambda: (True, "docs/实践1_环境搭建与项目理解.md")
             if (REPO / "docs/实践1_环境搭建与项目理解.md").is_file() else (False, "缺"), True),
            ("环境启动截图（IsaacSim/IsaacLab/list_envs/train）",
             lambda: has_files(f"{LOGS}/p1_env/*.png", 1, 5000), True),
        ]),
        ("2", "粗糙地形行走", [
            ("训练 checkpoint",
             lambda: has_checkpoint(f"{WS}/repos/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_rough/**/model_*.pt", 1), True),
            ("训练曲线 tfevents",
             lambda: has_files(f"{WS}/repos/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_rough/**/events.out.tfevents.*"), True),
            ("回放视频",
             lambda: has_files(f"{WS}/repos/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_rough/**/*.mp4"), True),
            ("导出的 policy（sim2sim 用）",
             lambda: has_files(f"{WS}/repos/unitree_rl_lab/logs/rsl_rl/**/exported/policy.*"), False),
        ]),
        ("3", "HoST 增量动作空间", [
            # 官方 §8.3：视频保存至 outputs/simulation.mp4
            # 实践 3 的代码在 course_code/ 下而不是 shenlan_hw/，
            # 第一版按后者查报"全缺"，实际材料齐全 —— 假阴性
            ("sim2sim 仿真视频（作业指定 simulation.mp4）",
             lambda: has_files(f"{CODE}/shenlan_hw2_action_space/**/outputs/simulation.mp4", 1, 10000), True),
            ("消融对照视频（残差式 vs 增量式）",
             lambda: has_files(f"{CODE}/shenlan_hw2_action_space/**/outputs/ablation_*.mp4", 2, 10000), False),
            ("补全的部署脚本",
             lambda: has_files(f"{CODE}/shenlan_hw2_action_space/**/deploy_mujoco_host_student.py", 1, 500), True),
        ]),
        ("4", "蹲姿行走", [
            ("三组消融的 checkpoint",
             lambda: has_checkpoint(f"{WS}/shenlan_hw/hw4_mjlab/logs/rsl_rl/g1_velocity_height/*ablation*/model_*.pt", 3), True),
            ("训练曲线",
             lambda: has_files(f"{WS}/shenlan_hw/hw4_mjlab/logs/rsl_rl/g1_velocity_height/*ablation*/events.out.tfevents.*", 3), True),
            ("回放视频", lambda: has_files(f"{WS}/shenlan_hw/hw4_mjlab/logs/**/*.mp4"), True),
        ]),
        ("5", "分层导航", [
            ("checkpoint",
             lambda: has_checkpoint(f"{WS}/shenlan_hw/hw5_navigation/**/model_*.pt", 1), True),
            ("训练曲线", lambda: has_files(f"{WS}/shenlan_hw/hw5_navigation/**/events.out.tfevents.*"), True),
            ("回放视频", lambda: has_files(f"{WS}/shenlan_hw/hw5_navigation/**/*.mp4"), True),
        ]),
        ("6", "教师学生蒸馏", [
            ("两组 student checkpoint",
             lambda: has_checkpoint(f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_student_*/**/model_*.pt", 2), True),
            ("两组训练曲线",
             lambda: has_files(f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_student_*/**/events.out.tfevents.*", 2), True),
            ("回放视频", lambda: has_files(f"{WS}/shenlan_hw/hw6_distill/logs/**/*.mp4"), True),
        ]),
        ("7", "运动重定向", [
            ("≥3 条 npz 产物",
             lambda: has_files(f"{WS}/shenlan_hw/unitree_lab_amp/**/amp/data/mixed/*.npz", 3), True),
            ("可视化视频（三类动作）",
             lambda: has_files(f"{LOGS}/p7_retarget/videos/*.mp4", 3), True),
            ("导出脚本",
             lambda: has_files(f"{WS}/repos/GMR/scripts/smplx_to_robot*_npz.py", 2), True),
        ]),
        ("8", "AMP 拟人走跑", [
            ("checkpoint",
             lambda: has_checkpoint(f"{WS}/shenlan_hw/unitree_lab_amp/logs/rsl_rl_amp/**/model_*.pt", 1), True),
            ("训练曲线", lambda: has_files(f"{WS}/shenlan_hw/unitree_lab_amp/logs/**/events.out.tfevents.*"), True),
            ("FullPlay 回放视频", lambda: has_files(f"{WS}/shenlan_hw/unitree_lab_amp/logs/**/*.mp4"), True),
        ]),
        ("9", "轨迹跟踪", [
            ("P1 test 输出记录",
             lambda: (True, "已记录在文档") if "test" in (REPO / "docs/实践9_自适应采样与轨迹跟踪.md").read_text() else (False, "缺"), True),
            ("checkpoint",
             lambda: has_checkpoint(f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_teacher/**/model_*.pt", 1), True),
            ("训练曲线（官方点名 7 条）",
             lambda: has_files(f"{WS}/shenlan_hw/hw6_distill/logs/rsl_rl/g1_hw6_teacher/**/events.out.tfevents.*"), True),
            ("P2_play.mp4 回放视频",
             lambda: has_files(f"{WS}/shenlan_hw/hw6_distill/logs/**/*.mp4"), True),
        ]),
        ("10", "HOI 感知跟踪", [
            ("补全的 hoi_height_scan.py",
             lambda: has_files(f"{WS}/shenlan_hw/HOI_Mimic/**/mdp/hoi_height_scan.py"), True),
            ("补全的 tracking_env_cfg.py",
             lambda: has_files(f"{WS}/shenlan_hw/HOI_Mimic/**/*perceptive_raycast*/tracking_env_cfg.py"), True),
            ("smoke train 终端日志",
             lambda: log_shows_training(f"{LOGS}/p10_hoi/*.log"), True),
            ("RayCaster 命中点截图", lambda: has_files(f"{LOGS}/p10_hoi/*.png"), True),
            ("正式训练 checkpoint",
             lambda: has_checkpoint(f"{WS}/shenlan_hw/HOI_Mimic/logs/**/model_*.pt", 1), True),
        ]),
        ("11", "跑酷与 Sim2Sim", [
            ("深度管线验证脚本",
             lambda: has_files(f"{REPO}/sim2sim/verify_practice11_depth.py"), True),
            ("训练 checkpoint",
             lambda: has_checkpoint(f"{WS}/repos/instinctlab/logs/**/model_*.pt", 1), True),
            ("训练曲线", lambda: has_files(f"{WS}/repos/instinctlab/logs/**/events.out.tfevents.*"), True),
            ("sim2sim 回放视频/截图",
             lambda: has_files(f"{LOGS}/**/p11*.mp4") if glob.glob(f"{LOGS}/**/p11*.mp4", recursive=True) else (False, "未录"), True),
        ]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--practice")
    args = ap.parse_args()

    print(f"\n{B}提交材料清单核对{N}")
    print(f"{D}（查'交没交全'，与 audit_against_rubric.py 查'做没做对'分工）{N}")
    print("=" * 76)

    missing_required = []
    for pid, name, items in build_specs():
        if args.practice and args.practice != pid:
            continue
        print(f"\n{B}实践 {pid} · {name}{N}")
        for label, fn, required in items:
            try:
                ok, detail = fn()
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"检查出错 {type(exc).__name__}"
            if ok:
                print(f"  {G}✅{N} {label:<28}{D}{detail}{N}")
            elif required:
                print(f"  {R}❌{N} {label:<28}{D}{detail}{N}")
                missing_required.append((pid, name, label))
            else:
                print(f"  {Y}○{N}  {label:<28}{D}{detail}（可选）{N}")

    print("\n" + "=" * 76)
    if missing_required:
        print(f"\n{R}{B}还缺 {len(missing_required)} 项必需材料{N}")
        cur = None
        for pid, name, label in missing_required:
            if pid != cur:
                print(f"\n  实践 {pid} {name}：")
                cur = pid
            print(f"    · {label}")
        print(f"\n{D}录像命令见 docs/录像命令_按实践分列.md{N}")
    else:
        print(f"\n{G}{B}全部材料齐备{N}")
    print()
    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
