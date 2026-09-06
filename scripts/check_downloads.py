#!/usr/bin/env python3
"""检查三个网盘下载包是否放到了正确位置。

不只看文件在不在，还要打开来验证格式对不对 —— 放错一层目录、
下成了 HTML 错误页、解压不完整，这些都只有真读一遍才发现得了。

用法：
    python scripts/check_downloads.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path("/home/limx/workspace/Roxan_warmup")

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


class Result:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.state = "ok"          # ok / missing / broken

    def ok(self, msg: str) -> None:
        self.lines.append(f"  {GREEN}✅{RESET} {msg}")

    def bad(self, msg: str) -> None:
        self.lines.append(f"  {RED}❌{RESET} {msg}")
        self.state = "broken"

    def warn(self, msg: str) -> None:
        self.lines.append(f"  {YELLOW}⚠️{RESET}  {msg}")

    def hint(self, msg: str) -> None:
        self.lines.append(f"     {DIM}{msg}{RESET}")


def find_stray(
    name_pattern: str, search_roots: list[Path], max_depth: int = 3
) -> list[Path]:
    """在常见的下载/解压位置找找东西是不是放错地方了。

    必须限制深度：这些根目录里有 GMR/IsaacLab 这种几十万文件的仓库，
    无限递归会跑上好几分钟（第一版就是这么卡住的）。
    深度 3 足够覆盖"下载后随手一解压"的常见布局。
    """
    found: list[Path] = []
    seen: set[Path] = set()

    for root in search_roots:
        if not root.is_dir() or root in seen:
            continue
        seen.add(root)
        for depth in range(max_depth + 1):
            pattern = "/".join(["*"] * depth + [name_pattern]) if depth else name_pattern
            try:
                found.extend(p for p in root.glob(pattern) if p not in found)
            except (PermissionError, OSError):
                break
            if len(found) >= 8:
                return found[:8]
    return found[:8]


def check_smplx() -> Result:
    """SMPL-X: 三个 pkl 必须直接躺在 assets/body_models/smplx/ 下。

    smplx.create(path, "smplx") 会去 path/smplx/ 找模型，多套一层就找不到。
    """
    r = Result()
    target = ROOT / "repos/GMR/assets/body_models/smplx"
    wanted = ["SMPLX_NEUTRAL.pkl", "SMPLX_FEMALE.pkl", "SMPLX_MALE.pkl"]

    present = [n for n in wanted if (target / n).is_file()]
    if not present:
        r.state = "missing"
        r.lines.append(f"  {DIM}目标目录：{target}{RESET}")

        strays = find_stray(
            "SMPLX_*.pkl",
            [ROOT / "repos/GMR", Path.home() / "Downloads", Path("/tmp"), Path.home()],
        )
        if strays:
            r.warn("文件下载了但位置不对，检测到：")
            for s in strays[:4]:
                r.hint(str(s))
            r.hint(f"移动命令：mv <上面的 pkl> {target}/")
        return r

    for n in wanted:
        f = target / n
        if not f.is_file():
            # NEUTRAL 是必需的，另两个只在数据标了性别时才用得上
            (r.bad if n == "SMPLX_NEUTRAL.pkl" else r.warn)(
                f"{n} 缺失" + ("（必需）" if n == "SMPLX_NEUTRAL.pkl" else "（可选）")
            )
            continue

        size_mb = f.stat().st_size / 1e6
        # 真打开验证：下成 HTML 错误页、解压中断，都在这里现形
        try:
            with open(f, "rb") as fh:
                obj = pickle.load(fh, encoding="latin1")
        except Exception as exc:  # noqa: BLE001
            r.bad(f"{n} 打不开（{size_mb:.0f} MB）：{type(exc).__name__}")
            r.hint("多半是下载不完整或解压中断，重新下一次")
            continue

        keys = set(obj) if isinstance(obj, dict) else set()
        if "shapedirs" in keys and "J_regressor" in keys:
            r.ok(f"{n}  {size_mb:.0f} MB  关键字段齐全")
        else:
            r.bad(f"{n} 内容不像 SMPL-X 模型（字段：{sorted(keys)[:5]}）")

    return r


def check_accad() -> Result:
    """ACCAD 动作数据：实践 7 的动作素材，与 SMPL-X 模型缺一不可。

    只看目录非空不够 —— 得确认里面是 SMPL-X 格式的 npz。
    作业明确要求「不要下载 SMPL+H 版本」，两者文件名相似但字段不同，
    下错了要到重定向跑起来才报错。
    """
    r = Result()
    target = ROOT / "datasets/AMASS/ACCAD"

    npzs: list[Path] = []
    if target.is_dir():
        npzs = [p for p in target.rglob("*.npz")][:2000]

    if not npzs:
        r.state = "missing"
        r.lines.append(f"  {DIM}目标目录：{target}{RESET}")
        strays = find_stray("*_stageii.npz", [Path.home() / "Downloads", Path("/tmp")], 3)
        strays += find_stray("ACCAD", [Path.home() / "Downloads", Path("/tmp"), ROOT], 2)
        # 刚 mkdir 出来的空目标目录会被自己搜到，排掉
        strays = [s for s in dict.fromkeys(strays) if s.resolve() != target.resolve()]
        if strays:
            r.warn("检测到疑似位置：")
            for s in strays:
                r.hint(str(s))
            r.hint(f"移动到：{target}/")
        return r

    total_mb = sum(p.stat().st_size for p in npzs) / 1e6
    r.ok(f"{len(npzs)} 个 .npz，共 {total_mb:.0f} MB")

    # 真打开一个，确认是 SMPL-X 而不是 SMPL+H
    import numpy as np

    sample = npzs[0]
    try:
        with np.load(sample, allow_pickle=True) as d:
            keys = set(d.files)
    except Exception as exc:  # noqa: BLE001
        r.bad(f"{sample.name} 打不开：{type(exc).__name__}")
        return r

    # 判据要选真有区分度的字段。
    # `poses` 两种格式都有，拿它当 SMPL-X 标记会把 SMPL+H 误判成合格（踩过）。
    # 真正互斥的是：SMPL-X 把姿态拆成 pose_body/pose_jaw/pose_eye 等独立项，
    # SMPL+H 则有自己特有的 dmpls（动态软组织系数）。
    smplx_marks = {"pose_body", "pose_jaw", "pose_eye", "pose_hand"} & keys
    smplh_marks = {"dmpls"} & keys

    if smplx_marks:
        r.ok(f"格式确认 SMPL-X（{sample.name} 含 {sorted(smplx_marks)[:3]}）")
    elif smplh_marks:
        r.bad(f"这是 SMPL+H 不是 SMPL-X（含 {sorted(smplh_marks)}）")
        r.hint("作业 §3.2 明确要求 AMASS → SMPL-X → ACCAD，不要下 SMPL+H 版本")
        r.hint("重定向会跑到一半才失败，现在换掉省事")
    else:
        r.warn(f"认不出格式，字段：{sorted(keys)[:6]}")

    # 官方文档写"约 1.04 G"，但实测网盘里的 ACCAD 解压后就是 ~249 MB / 84 条。
    # 拿文档数字当阈值会一直报假警，改成检查动作类别是否覆盖作业要求。
    cats = {
        "走路": [p for p in npzs if re.search(r"walk", p.name, re.I)],
        "跑步": [p for p in npzs if re.search(r"run|jog", p.name, re.I)],
        "转弯/切换": [p for p in npzs if re.search(r"turn", p.name, re.I)],
    }
    missing = [k for k, v in cats.items() if not v]
    if missing:
        r.warn(f"缺少动作类别：{'、'.join(missing)}（作业 §7 要求三类都有）")
    else:
        detail = "  ".join(f"{k} {len(v)}条" for k, v in cats.items())
        r.ok(f"动作类别齐全 — {detail}")
    return r


def check_hoi_mimic() -> Result:
    """实践 10 代码：要能看到 scripts/rsl_rl/train.py 才算放对。"""
    r = Result()
    candidates = [
        ROOT / "shenlan_hw/HOI_Mimic",
        ROOT / "shenlan_hw/hoi_mimic",
    ]
    target = next((c for c in candidates if c.is_dir()), None)

    if target is None:
        r.state = "missing"
        r.lines.append(f"  {DIM}目标目录：{candidates[0]}{RESET}")
        strays = find_stray("HOI_Mimic", [Path.home() / "Downloads", Path("/tmp")])
        strays += find_stray("*HOI*", [Path.home() / "Downloads", Path("/tmp")], max_depth=1)
        strays += find_stray("HOI_Mimic", [ROOT], max_depth=2)
        if strays:
            r.warn("检测到疑似位置：")
            for s in dict.fromkeys(strays):
                r.hint(str(s))
        return r

    # 解压多套一层是最常见的问题，先自动识别
    if not (target / "scripts").is_dir():
        inner = [d for d in target.iterdir() if d.is_dir() and (d / "scripts").is_dir()]
        if inner:
            r.bad(f"多套了一层目录：{target.name}/{inner[0].name}/scripts/")
            r.hint(f"修正：mv {inner[0]}/* {target}/ && rmdir {inner[0]}")
            return r

    checks = [
        ("scripts/rsl_rl/train.py", "训练脚本", True),
        ("scripts/rsl_rl/play.py", "回放脚本", True),
        ("source", "源码目录", True),
        ("datasets", "数据集目录", False),
    ]
    for rel, desc, required in checks:
        p = target / rel
        if p.exists():
            extra = ""
            if p.is_dir() and rel == "datasets":
                n = sum(1 for _ in p.rglob("*") if _.is_file())
                extra = f"（{n} 个文件）"
                if n == 0:
                    r.warn(f"{desc} {rel} 是空的 —— 数据可能没解压全")
                    continue
            r.ok(f"{desc}  {rel}{extra}")
        elif required:
            r.bad(f"{desc} 缺失：{rel}")
        else:
            r.warn(f"{desc} 缺失：{rel}（可选）")

    return r


def check_unitree_usd() -> Result:
    """G1 USD：可选，当前实践 8 走 URDF 路线也能跑。"""
    r = Result()
    target = ROOT / "assets/unitree_model"
    if not target.is_dir():
        r.state = "missing"
        r.lines.append(f"  {DIM}目标目录：{target}{RESET}")
        r.hint("可选项 —— 实践 8 现在用 URDF spawn，不下也能跑")
        return r

    usds = list(target.rglob("*.usd")) + list(target.rglob("*.usda"))
    if not usds:
        r.bad(f"目录存在但没有 .usd 文件（{target}）")
        return r
    g1 = [u for u in usds if "g1" in u.name.lower()]
    r.ok(f"找到 {len(usds)} 个 USD" + (f"，含 G1：{g1[0].name}" if g1 else ""))
    if not g1:
        r.warn("没找到 G1 的 USD，检查是否下错了包")
    return r


def main() -> int:
    items = [
        ("① SMPL-X 人体模型", check_smplx, "实践 7 的人体骨架模板"),
        ("② ACCAD 动作数据", check_accad, "实践 7 的动作素材，与①缺一不可"),
        ("③ 实践 10 代码 HOI_Mimic", check_hoi_mimic, "解锁实践 10"),
        ("④ unitree_model USD", check_unitree_usd, "可选，实践 8 已有 URDF 替代"),
    ]

    print(f"\n{BOLD}下载包位置自检{RESET}")
    print("=" * 62)

    states: dict[str, str] = {}
    for title, fn, why in items:
        res = fn()
        icon = {"ok": GREEN + "就绪" + RESET,
                "missing": DIM + "未下载" + RESET,
                "broken": RED + "有问题" + RESET}[res.state]
        print(f"\n{BOLD}{title}{RESET}  [{icon}]  {DIM}{why}{RESET}")
        for line in res.lines:
            print(line)
        states[title] = res.state

    print("\n" + "=" * 62)
    print(f"{BOLD}下一步{RESET}\n")

    smplx_ok = states["① SMPL-X 人体模型"] == "ok"
    accad_ok = states["② ACCAD 动作数据"] == "ok"
    p10_ok = states["③ 实践 10 代码 HOI_Mimic"] == "ok"

    if smplx_ok and accad_ok:
        print(f"  {GREEN}▸{RESET} 实践 7 可以跑了：")
        print(f"    {DIM}cd {ROOT}/repos/GMR{RESET}")
        print(f"    {DIM}{ROOT}/envs/gmr/bin/python scripts/smplx_to_robot_dataset_npz.py \\{RESET}")
        print(f"    {DIM}    --src_folder {ROOT}/datasets/AMASS/ACCAD \\{RESET}")
        print(f"    {DIM}    --tgt_folder {ROOT}/datasets/g1_amp_npz \\{RESET}")
        print(f"    {DIM}    --robot unitree_g1 --num_cpus 1{RESET}")
        print(f"    {DIM}（先 --num_cpus 1 跑通一条，再加并行；我们的实现走 CPU，")
        print(f"      不占 GPU，可与训练并行）{RESET}")
        print(f"    实践 8 只缺 1 条跑步动作 —— 挑 ACCAD 里 Running 目录的即可\n")
    else:
        waiting = [n for n, ok in (("SMPL-X 模型", smplx_ok), ("ACCAD 数据", accad_ok)) if not ok]
        print(f"  {DIM}▸ 实践 7、8 还差：{'、'.join(waiting)}{RESET}\n")

    if p10_ok:
        print(f"  {GREEN}▸{RESET} 实践 10 可以开工了：3 组 TODO")
        print(f"    {DIM}metadata loader / RayCaster 配置 / 观测接线{RESET}")
        print(f"    技术点与实践 2 高度重合，可复用那边的实现\n")
    else:
        print(f"  {DIM}▸ 实践 10 等代码包{RESET}\n")

    # 三个必需包都到位才算"全部就绪"。
    # 漏掉 accad_ok 会出现"实践 7 还差 ACCAD"和"全部就绪"同时打印的矛盾。
    if smplx_ok and accad_ok and p10_ok:
        print(f"  {GREEN}全部就绪 —— 11 个实践的最后 3 个可以收尾了{RESET}\n")

    return 0 if not any(s == "broken" for s in states.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
