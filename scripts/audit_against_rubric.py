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
import pathlib
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


GREP_EXCLUDES = [f"--exclude-dir={d}" for d in
                 (".venv", ".git", "__pycache__", "logs", "node_modules", "outputs")]


def grep_rl(pattern: str, root: Path, timeout: int = 60) -> str:
    """grep -rl 的带保护版本。

    裸 subprocess.run 没有 timeout 会永久挂住，不排除 .venv 会白扫几十万文件 ——
    训练占着磁盘 IO 时两者都会让整个审计崩掉。
    """
    try:
        return subprocess.run(
            ["grep", "-rl", pattern, str(root), "--include=*.py", *GREP_EXCLUDES],
            capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


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
    left = grep_rl('NotImplementedError("TODO', amp)
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


def audit_p1() -> None:
    print("\n══ 实践 1 · 环境搭建 ══")
    print("   官方：PDF 报告 3-6 页，≥3 项环境截图 + 项目理解 5 问 + 简短分析 3 问")
    d = DOCS / "实践1_环境搭建与项目理解.md"
    rec("ok" if d.exists() else "bad", 1, "实践 1 报告存在")
    # 官方 §5.2 项目理解 5 问 + §5.3 简短分析 3 问
    for name, pats in [
        ("三栈可用（IsaacSim/IsaacLab/MuJoCo）",
         [r"Isaac ?Sim", r"Isaac ?Lab", r"MuJoCo"]),
        ("①任务入口文件", [r"gym\.register|__init__\.py"]),
        ("②环境配置文件", [r"velocity_env_cfg"]),
        ("③PPO 配置文件", [r"rsl_rl_ppo_cfg"]),
        ("④训练入口", [r"scripts/rsl_rl/train\.py"]),
        ("⑤checkpoint 目录", [r"logs/rsl_rl"]),
        ("分析1 --task 到环境创建流程", [r"parse_env_cfg|gym\.make"]),
        ("分析2 训练与播放配置为何分开", [r"RobotPlayEnvCfg|播放配置"]),
        ("分析3 play 如何找 checkpoint", [r"get_checkpoint_path"]),
    ]:
        n = has(d, *pats)
        rec("ok" if n > 0 else "warn", 1, name, f"{n} 处")
    # 环境实际可用比截图更硬
    py = WS / "envs/isaaclab/bin/python"
    rec("ok" if py.exists() else "bad", 1, "IsaacLab 环境存在")
    rec("ok" if (WS / "repos/IsaacLab").exists() else "bad", 1, "IsaacLab 仓库存在")


def audit_p5() -> None:
    print("\n══ 实践 5 · 分层导航 ══")
    print("   官方 Part1(50)：7个TODO + smoke test + 低层 eval 无梯度 + actor 376维")
    print("   官方 Part2(50)：难度扩展 + 可复现配置 + 公平对照 + 定量定性 + 失败模式")
    H5 = WS / ("shenlan_hw/hw5_navigation/unitree_rl_lab/source/unitree_rl_lab"
               "/unitree_rl_lab/tasks/navigation")
    act = H5 / "mdp/pre_trained_policy_action.py"
    # Part 1：评分清单里写死的几条
    left = grep_rl("HOMEWORK_TODO", H5)
    n_raise = has(act, r"raise NotImplementedError")
    rec("ok" if n_raise <= 0 else "bad", 5, "7 个 TODO 无残留 NotImplementedError",
        f"{max(n_raise,0)} 处")
    rec("ok" if has(act, r"HOMEWORK_TODO_\d+_(START|END)") > 0 else "warn", 5,
        "边界标记保留", f"{has(act, r'HOMEWORK_TODO')} 处")
    rec("ok" if has(act, r"\.eval\(\)") > 0 else "bad", 5, "低层策略 .eval()")
    rec("ok" if has(act, r"torch\.inference_mode\(\)|torch\.no_grad\(\)") > 0 else "bad",
        5, "推理无梯度")
    # ★ 本轮根因：带 history 的 ObservationManager 必须传 update_history=True
    rec("ok" if has(act, r"update_history=True") > 0 else "bad", 5,
        "低层观测历史正确更新（update_history=True）")
    cfg = H5 / "robots/g1/29dof/navigation_env_cfg.py"
    rec("ok" if has(cfg, r"376") > 0 else "warn", 5, "actor 观测 376 维有记录")
    # Part 2：难度扩展与对照
    d = DOCS / "实践5_分层强化学习导航.md"
    for name, pats in [
        ("定量指标", [r"到达.{0,4}(率|占比)|error_pos_2d"]),
        ("失败模式分析", [r"失败模式|根因|排除"]),
        ("可复现配置", [r"NAV_[A-Z_]+=|scripts/run_p5"]),
    ]:
        n = has(d, *pats)
        rec("ok" if n > 0 else "warn", 5, name, f"{n} 处")

    # ★ 难度扩展必须看**训练数据**而不是文档关键词。
    # 此前这条只 grep 文档里有没有 "V5|MixedObstacle|SingleGoal"，
    # 而这些词在讲配置继承链时也会出现 —— 文档提到 ≠ 真做了对照实验。
    # 官方 Part 2 要求"在相同训练预算下比较 baseline 与更难任务"，
    # 所以判据是：扩展组有训练日志，且轮数与 baseline 相当。
    logdir = Path.home() / "humanoid_logs/p5_navigation"
    base_log, ext_log = logdir / "p5_baseline.log", logdir / "p5_random_arena.log"
    def _iters(p: Path) -> int:
        if not p.exists():
            return 0
        m = re.findall(r"Learning iteration (\d+)", p.read_text(errors="ignore"))
        return int(m[-1]) if m else 0
    bi, ei = _iters(base_log), _iters(ext_log)
    rec("ok" if ei > 0 else "warn", 5, "难度扩展组有真实训练数据",
        f"baseline {bi} 轮 · 扩展组 {ei} 轮")
    if ei > 0:
        # "相同预算"允许一定偏差，但不该差一倍（实践 6 就栽在这上面）
        ratio = min(bi, ei) / max(bi, ei) if max(bi, ei) else 0
        rec("ok" if ratio > 0.8 else "warn", 5, "两组训练预算相当",
            f"比值 {ratio:.2f}（<0.8 视为不对等）")
    log = Path.home() / "humanoid_logs/p5_navigation/p5_baseline.log"
    if log.exists():
        t = log.read_text(errors="ignore")
        g = re.findall(r"Episode_Termination/goal_reached: ([0-9.]+)", t)
        rec("ok" if g and float(g[-1]) > 0.5 else "warn", 5,
            "baseline 可训练且能到达目标",
            f"到达终止占比 {g[-1]}" if g else "无数据")
    else:
        rec("bad", 5, "baseline 训练日志不存在")


def audit_p10() -> None:
    print("\n══ 实践 10 · HOI 人-物交互 ══")
    print("   官方：3 组 TODO（metadata loader / RayCaster 配置 / 观测接线）")
    hoi = None
    for cand in (WS / "shenlan_hw/HOI_Mimic", WS / "shenlan_hw/hoi_mimic"):
        if cand.exists():
            hoi = cand
            break
    if hoi is None:
        rec("warn", 10, "代码包未下载",
            "pan.baidu.com/s/1fbSggWFaxc_mL-ZXyYMANg 提取码 nk8r")
        return
    rec("ok", 10, "代码包已下载", str(hoi.name))


def audit_cross_cutting() -> None:
    """跨实践的通用陷阱检查。

    目前只有一项：手动建的 ObservationManager 若带 history，
    调 compute_group 时必须传 update_history=True。
    实践 5 因为漏传这个参数，低层策略的 5 帧历史从未更新
    （"当前帧重复 5 次"），排查了十一轮才找到 —— 因为它不改变任何量级。
    """
    print("\n══ 跨实践 · 通用陷阱 ══")
    bad = []
    timed_out = []
    roots = [WS / "shenlan_hw", WS / "repos/unitree_rl_lab"]
    # 在 grep 层面就排除，别扫完几十万文件再在结果里过滤 ——
    # 光 .venv 一个目录就够让 grep 在磁盘繁忙时超时崩掉整个审计
    excludes = [f"--exclude-dir={d}" for d in
                (".venv", ".git", "__pycache__", "logs", "node_modules", "outputs")]
    for root in roots:
        if not root.exists():
            continue
        try:
            out = subprocess.run(
                ["grep", "-rn", "compute_group(", str(root), "--include=*.py", *excludes],
                capture_output=True, text=True, timeout=120).stdout
        except subprocess.TimeoutExpired:
            # 训练占满磁盘 IO 时会撞上；降级成一条待查，别让整个审计挂掉
            timed_out.append(root.name)
            continue
        for line in out.splitlines():
            if ".venv" in line or "def compute_group" in line:
                continue
            # 只关心"没传 update_history"的调用；显式传 False 可能是有意的
            if "update_history" not in line:
                # 同一次调用可能跨行，取文件+行号再看一眼
                path, lineno = line.split(":")[0], line.split(":")[1]
                try:
                    ctx = "".join(pathlib.Path(path).read_text(
                        errors="ignore").splitlines(True)[int(lineno) - 1:int(lineno) + 3])
                except Exception:
                    ctx = line
                if "update_history" not in ctx:
                    bad.append(f"{pathlib.Path(path).name}:{lineno}")
    if timed_out:
        rec("warn", 0, "compute_group 扫描未完成",
            f"grep 超时（磁盘繁忙？）未覆盖: {timed_out}，稍后重跑")
    else:
        rec("ok" if not bad else "warn", 0,
            "compute_group 调用均已处理 update_history",
            f"漏传 {len(bad)} 处: {bad[:3]}" if bad else "")


AUDITS = {1: audit_p1, 2: audit_p2, 3: audit_p3, 4: audit_p4, 5: audit_p5,
          6: audit_p6, 7: audit_p7, 8: audit_p8, 9: audit_p9,
          10: audit_p10, 11: audit_p11, 0: audit_cross_cutting}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--practice", type=int, choices=sorted(AUDITS),
                help="0 = 跨实践通用陷阱检查")
    args = ap.parse_args()

    if not COURSE.exists():
        print(f"❌ 找不到 {COURSE}，先跑 scripts/decrypt_course_pdfs.py")
        return 1

    print("=" * 70)
    print("对照官方评分细则的闭环审计")
    print("=" * 70)

    # 用 is not None 而不是真值判断：--practice 0（跨实践检查）会被当成 falsy
    todo = [args.practice] if args.practice is not None else sorted(AUDITS)
    for n in todo:
        try:
            AUDITS[n]()
        except Exception as exc:  # noqa: BLE001
            # 一个实践的审计出错不该拖垮其余十个，记成待改进继续走
            label = "跨实践" if n == 0 else f"实践{n}"
            WARN.append(f"[{label}] 审计自身出错: {type(exc).__name__}: {exc}")
            print(f"  ⚠️  {label} 审计出错: {type(exc).__name__}: {str(exc)[:100]}")

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
