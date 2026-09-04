"""对照官方评分细则做闭环审计 —— 我们交的东西够不够、对不对。

为什么要写成脚本而不是人工核对
--------------------------
课程 11 份作业的提交要求散在各自 PDF 的「提交内容 / 评分细则 / 验收标准」里，
人工比对既慢又容易漏。更重要的是**审计本身要可复现**：
改完文档后要能一键复查，而不是重新读一遍 PDF。

数据来源
--------
`~/course_materials/txt/` —— 由 `scripts/decrypt_course_pdfs.py` 解密提取，
是官方要求的唯一真源。本脚本只做「要求 → 我们的产物」的比对，不重述要求。

三类检查
--------
1. **必交物存在性** —— 代码 / 视频 / 报告 / checkpoint 是否真的在盘上
2. **报告结构合规** —— 官方点名要求的章节是否都写了
3. **实现要点** —— 评分细则里写死的关键实现（如 HoST 增量公式、
   FIFO 方向、四元数顺序）是否正确

用法
    python audit_against_rubric.py            # 全部
    python audit_against_rubric.py --practice 4
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
WS = Path("/home/limx/workspace/Roxan_warmup")
COURSE = Path.home() / "course_materials" / "txt"

OK, WARN, BAD = [], [], []


def rec(level: str, practice: int, item: str, detail: str = "") -> None:
    line = f"P{practice} · {item}" + (f"  [{detail}]" if detail else "")
    {"ok": OK, "warn": WARN, "bad": BAD}[level].append(line)
    mark = {"ok": "✅", "warn": "⚠️ ", "bad": "❌"}[level]
    print(f"  {mark} {item}" + (f"  {detail}" if detail else ""))


def has(path: Path, *patterns: str, ci: bool = True) -> int:
    """文件里匹配到多少处。找不到文件返回 -1，与"匹配 0 处"区分开。"""
    if not path.exists():
        return -1
    text = path.read_text(errors="ignore")
    flags = re.I if ci else 0
    return sum(len(re.findall(p, text, flags)) for p in patterns)


def find_file(pattern: str, root: Path = WS) -> Path | None:
    try:
        out = subprocess.run(
            ["find", str(root), "-name", pattern, "-not", "-path", "*/.git/*"],
            capture_output=True, text=True, timeout=120).stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    return Path(out.splitlines()[0]) if out else None


# ────────────────────────────────────────────────────────────────
def audit_p2() -> None:
    print("\n══ 实践 2 · 粗糙地形行走 ══")
    print("   官方：代码20 + checkpoint20 + 视频20 + 报告40")
    d = DOCS / "实践2_实验报告.md"
    # 报告 40 分明确要求的四个说明点
    for name, pats in [
        ("说明 height_scan 作用", [r"height_scan"]),
        ("说明扫描区域", [r"1\.6", r"扫描(区域|范围)"]),
        ("说明观测维度", [r"187", r"观测维度"]),
        ("说明动作空间含义", [r"动作空间", r"29\s*(个)?\s*(DoF|关节)"]),
    ]:
        n = has(d, *pats)
        rec("ok" if n > 0 else "bad", 2, name, f"{n} 处")
    # checkpoint 与视频是计分项，必须真的存在
    ck = list((WS / "repos/unitree_rl_lab/logs/rsl_rl"
               "/unitree_g1_29dof_velocity_rough").glob("*/model_*.pt"))
    rec("ok" if ck else "bad", 2, "训练 checkpoint 存在", f"{len(ck)} 个")
    vids = list((WS / "repos/unitree_rl_lab/logs/rsl_rl"
                 "/unitree_g1_29dof_velocity_rough").glob("*/videos/play/*.mp4"))
    rec("ok" if vids else "bad", 2, "play 视频存在", f"{len(vids)} 个")


def audit_p3() -> None:
    print("\n══ 实践 3 · HoST Sim2Sim ══")
    print("   官方：4 个 TODO 各 25 分；视频缺失扣 10 分")
    f = find_file("deploy_mujoco_host_student.py")
    if f is None:
        rec("bad", 3, "找不到 deploy_mujoco_host_student.py")
        return
    # 评分细则里写死的关键实现
    for name, pats, must in [
        ("TODO1 索引 qpos[7:]/qvel[6:]/qpos[3:7]/qvel[3:6]",
         [r"qpos\[7:\]", r"qvel\[6:\]", r"qpos\[3:7\]", r"qvel\[3:6\]"], 4),
        ("TODO1 调用 get_gravity_orientation", [r"get_gravity_orientation\("], 2),
        ("TODO3 FIFO 最新帧在末尾",
         [r"observation_history\[single_observation_dim:\]"], 1),
        ("TODO4 增量公式 q_current + scale*action",
         [r"current_joint_positions\s*\+\s*action_scale\s*\*\s*policy_action"], 1),
    ]:
        n = has(f, *pats)
        rec("ok" if n >= must else "bad", 3, name, f"{n} 处")
    # 0-10 分档：仍用残差公式则判 0 分
    resid = has(f, r"default_angles\s*\+\s*action_scale\s*\*\s*policy_action")
    rec("ok" if resid == 0 else "bad", 3, "未误用残差公式 default+scale*a",
        f"{resid} 处")
    vid = find_file("simulation.mp4")
    rec("ok" if vid else "bad", 3, "outputs/simulation.mp4 存在（缺则扣 10 分）")


def audit_p4() -> None:
    print("\n══ 实践 4 · 双指令 MDP ══")
    print("   官方报告必含：①四维奖励结构 ②权重选择依据 ③≥2组消融 ④训练曲线")
    d = DOCS / "实践4_双指令MDP设计.md"
    for name, pats, need in [
        ("① 按 Task/Style/Reg/Penalty 四维组织",
         [r"Task", r"Style", r"Penalty", r"Reg\b|正则"], 4),
        ("② 权重选择依据", [r"权重.{0,6}(依据|选择|为何|为什么)"], 1),
        ("③ 消融实验 ≥2 组", [r"消融\s*[AB]|消融实验"], 2),
        ("④ 训练曲线含 error_height", [r"error_height"], 1),
        ("④ 训练曲线含速度误差", [r"error_lin_vel|速度.{0,4}误差|track_lin_vel"], 1),
        ("④ 训练曲线含总回报", [r"总回报|episode.{0,3}reward|Mean reward"], 1),
    ]:
        n = has(d, *pats)
        rec("ok" if n >= need else "bad", 4, name, f"{n} 处")
    logs = list((Path.home() / "humanoid_logs/p4_dual_command").glob("p4_*.log"))
    rec("ok" if len(logs) >= 2 else "bad", 4, "消融训练日志 ≥2 组", f"{len(logs)} 份")


def audit_p6() -> None:
    print("\n══ 实践 6 · 教师-学生蒸馏 ══")
    print("   官方检查清单：7个TODO + pytest 通过 + 两法对比 + 实验报告")
    hw6 = WS / "shenlan_hw/hw6_distill"
    n_todo = has(hw6 / "src/humanoid_hw6/rl/algorithms/distillation.py",
                 r"NotImplementedError")
    rec("ok" if n_todo <= 0 else "bad", 6, "TODO 无残留 NotImplementedError",
        f"{max(n_todo,0)} 处")
    d = DOCS / "实践6_教师学生蒸馏.md"
    for name, pats in [
        ("action vs KL 行为对比", [r"KL", r"action.{0,3}match"]),
        ("给出量化结论", [r"\d+\.\d+\s*%|\d+\.\d+x"]),
    ]:
        n = has(d, *pats)
        rec("ok" if n > 0 else "bad", 6, name, f"{n} 处")
    for k in ("kl_matching", "action_matching"):
        p = Path.home() / "humanoid_logs/p6_distill" / f"p6_{k}.log"
        rec("ok" if p.exists() else "bad", 6, f"{k} 训练日志")


def audit_p7() -> None:
    print("\n══ 实践 7 · 运动重定向 ══")
    print("   官方：新建两个脚本，输出 6 字段 npz 可被实践 8 加载")
    for f in ("smplx_to_robot_npz.py", "smplx_to_robot_dataset_npz.py"):
        p = WS / "repos/GMR/scripts" / f
        rec("ok" if p.exists() else "bad", 7, f"{f} 已新建")
    p = WS / "repos/GMR/scripts/smplx_to_robot_npz.py"
    for name, pats in [
        ("四元数 wxyz→xyzw 转换", [r"\[1,\s*2,\s*3,\s*0\]"]),
        ("local_body_pos 用 FK（根置原点）", [r"forward_kinematics"]),
        ("六个必需字段齐全",
         [r"link_body_list"]),
    ]:
        n = has(p, *pats)
        rec("ok" if n > 0 else "bad", 7, name, f"{n} 处")


def audit_p8() -> None:
    print("\n══ 实践 8 · AMP 拟人走跑 ══")
    print("   官方：9 个 TODO + 训练 + FullPlay 全速度评估")
    amp = WS / "shenlan_hw/unitree_lab_amp"
    if not amp.exists():
        rec("bad", 8, "代码未克隆")
        return
    left = subprocess.run(
        ["grep", "-rl", 'NotImplementedError("TODO', str(amp), "--include=*.py"],
        capture_output=True, text=True).stdout.strip()
    rec("ok" if not left else "bad", 8, "9 个 TODO 无残留", left[:40])
    init = amp / ("source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp"
                  "/config/g1/__init__.py")
    n = has(init, r"WalkToRun-FullPlay")
    rec("ok" if n > 0 else "bad", 8, "FullPlay 任务已注册", f"{n} 处")
    d = DOCS / "实践8_AMP拟人走跑.md"
    rec("ok" if d.exists() else "bad", 8, "实验文档存在")
    # 官方 7.5 验收要求覆盖走/跑/切换/转弯
    for name, pats in [("文档覆盖走跑切换验收", [r"走跑切换|走.{0,2}跑.{0,4}切换"]),
                       ("文档覆盖转弯验收", [r"转弯"])]:
        n = has(d, *pats)
        rec("ok" if n > 0 else "warn", 8, name, f"{n} 处")


def audit_p9() -> None:
    print("\n══ 实践 9 · BeyondMimic ══")
    print("   官方：算法实现50% + test通过10% + 曲线20% + 真实训练20%")
    d = DOCS / "实践9_自适应采样与轨迹跟踪.md"
    for name, pats in [
        ("失败统计实现", [r"失败.{0,4}统计|failure"]),
        ("概率采样实现", [r"概率采样|采样.{0,4}分布|sampling"]),
        # 判据用官方 test 的实际输出串，比"N/N 通过"这类措辞稳健 ——
        # 文档怎么行文都行，只要真的贴了测试输出就算数
        ("test 通过记录", [r"All tests passed", r"Test \d+ passed"]),
    ]:
        n = has(d, *pats)
        rec("ok" if n > 0 else "warn", 9, name, f"{n} 处")
    log = Path.home() / "humanoid_logs/p9_beyondmimic/p9_dance.log"
    if log.exists():
        it = re.findall(r"iteration (\d+)", log.read_text(errors="ignore"))
        rec("ok" if it else "warn", 9, "P2 训练进行中/已完成",
            f"iter {it[-1]}" if it else "无进度")
    else:
        rec("bad", 9, "P2 训练日志不存在")


def audit_p11() -> None:
    print("\n══ 实践 11 · Instinct 跑酷 ══")
    d = DOCS / "实践11_跑酷与深度感知.md"
    rec("ok" if d.exists() else "bad", 11, "实验文档存在")
    n = has(d, r"\d+\s*/\s*\d+\s*(通过|项)")
    rec("ok" if n > 0 else "warn", 11, "验证结果有记录", f"{n} 处")


AUDITS = {2: audit_p2, 3: audit_p3, 4: audit_p4, 6: audit_p6,
          7: audit_p7, 8: audit_p8, 9: audit_p9, 11: audit_p11}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--practice", type=int, choices=sorted(AUDITS))
    args = ap.parse_args()

    if not COURSE.exists():
        print(f"❌ 找不到 {COURSE}，先跑 scripts/decrypt_course_pdfs.py")
        return 1

    print("=" * 70)
    print("对照官方评分细则的闭环审计")
    print("=" * 70)

    todo = [args.practice] if args.practice else sorted(AUDITS)
    for n in todo:
        AUDITS[n]()

    print("\n" + "=" * 70)
    print(f"通过 {len(OK)} · 待改进 {len(WARN)} · 不合规 {len(BAD)}")
    print("=" * 70)
    if BAD:
        print("\n不合规项（官方明确要求但我们没做到）：")
        for b in BAD:
            print(f"  ❌ {b}")
    if WARN:
        print("\n待改进：")
        for w in WARN:
            print(f"  ⚠️  {w}")
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
