#!/usr/bin/env python3
"""验证 instinct_rl / instinctlab 能否在现有 envs/isaaclab 环境里跑（实践 11-C）。

背景：instinct_rl/setup.py 声明 python>=3.12 + torch==2.11.0 + numpy>=2，
本机 envs/isaaclab 是 3.11.15 + torch 2.7.0 + numpy 1.26.0。按声明看应该不兼容，
所以文档一度写着「必须另建 20GB 环境」。

但声明的依赖不等于真实需要的依赖。本脚本用三层可证伪的检验替代「读 setup.py 猜」：

  1. 语法层：用 3.11 编译全部源码，看有没有 3.12 专属语法
  2. API 层：提取代码里所有 torch.* 调用，在当前 torch 里逐个 getattr
  3. 运行层：真正 import 一遍（instinctlab 需先启动 SimulationApp 才有 pxr）

用法：
    envs/isaaclab/bin/python scripts/verify_p11_env.py           # 层 1+2，秒级
    envs/isaaclab/bin/python scripts/verify_p11_env.py --full    # 加层 3，需数分钟
"""
from __future__ import annotations

import argparse
import compileall
import contextlib
import io
import re
import sys
from pathlib import Path

REPOS = Path("/home/limx/workspace/Roxan_warmup/repos")
INSTINCT_RL = REPOS / "instinct_rl"
INSTINCT_LAB = REPOS / "instinctlab/source/instinctlab"

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)
OK, FAIL, WARN = f"{GREEN}✅{RESET}", f"{RED}❌{RESET}", f"{YELLOW}⚠️{RESET}"


def layer1_syntax() -> bool:
    """3.12 专属语法检测：能用 3.11 编译就说明没用。"""
    print(f"\n{BOLD}层 1 · 语法兼容{RESET}  {DIM}用当前 Python 编译全部源码{RESET}")
    print(f"  当前 Python: {sys.version.split()[0]}")

    all_ok = True
    for pkg in (INSTINCT_RL, INSTINCT_LAB):
        if not pkg.is_dir():
            print(f"  {WARN} 目录不存在：{pkg}")
            all_ok = False
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            ok = compileall.compile_dir(str(pkg), quiet=2, force=True)
        n = sum(1 for _ in pkg.rglob("*.py"))
        if ok:
            print(f"  {OK} {pkg.name}  {n} 个 .py 全部编译通过")
        else:
            print(f"  {FAIL} {pkg.name} 有语法错误：")
            print(f"    {DIM}{buf.getvalue()[:600]}{RESET}")
            all_ok = False
    return all_ok


def layer2_torch_api() -> bool:
    """torch 版本检测：代码用到的 API 在当前 torch 里存不存在。

    比对版本号没有意义 —— 真正决定能否跑的是 API 是否齐全。
    """
    import importlib

    import torch

    print(f"\n{BOLD}层 2 · torch API 覆盖{RESET}  {DIM}声明要 2.11.0，实测当前版本够不够{RESET}")
    print(f"  当前 torch: {torch.__version__}")

    # 结尾的点会把 torch.nn. 这类前缀混进来，统一去掉
    pattern = re.compile(r"torch\.[a-zA-Z_][a-zA-Z_0-9.]{1,40}")
    apis: set[str] = set()
    for f in INSTINCT_RL.rglob("*.py"):
        try:
            apis.update(m.rstrip(".") for m in pattern.findall(f.read_text(errors="ignore")))
        except OSError:
            continue

    missing = []
    for api in sorted(apis):
        obj = torch
        try:
            for part in api.split(".")[1:]:
                obj = getattr(obj, part)
        except AttributeError:
            # 可能是子模块而非属性，再试一次 import
            try:
                importlib.import_module(api)
            except Exception:  # noqa: BLE001
                missing.append(api)

    print(f"  提取到 {len(apis)} 个 torch.* 调用")
    if missing:
        print(f"  {FAIL} 当前 torch 缺失 {len(missing)} 个：")
        for m in missing[:10]:
            print(f"    {DIM}{m}{RESET}")
        return False
    print(f"  {OK} 全部存在 —— torch=={torch.__version__} 足够，无需升到 2.11")
    return True


def layer3_import() -> bool:
    """真 import。instinctlab 依赖 pxr，必须先启动 SimulationApp。

    坑：SimulationApp 会接管进程的 stdout/stderr，之后所有 print 都会被它吞掉。
    所以先 os.dup 复制一份原始 fd 保存起来，结果攒着，最后往那份 fd 上写。
    """
    import os

    print(f"\n{BOLD}层 3 · 实际 import{RESET}  {DIM}启动 IsaacSim，需数分钟{RESET}")
    sys.stdout.flush()
    saved_fd = os.dup(1)  # 必须在 SimulationApp 之前复制

    lines: list[str] = []

    def emit(msg: str) -> None:
        lines.append(msg)

    for p in (INSTINCT_RL, INSTINCT_LAB):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    ok = False
    app = None
    try:
        try:
            import instinct_rl  # noqa: F401
            emit(f"  {OK} instinct_rl")
        except Exception as exc:  # noqa: BLE001
            emit(f"  {FAIL} instinct_rl: {type(exc).__name__}: {exc}")
            return False

        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})

        import pxr  # noqa: F401
        emit(f"  {OK} pxr（由 IsaacSim 运行时注入，裸 python 里没有）")
        import instinctlab
        emit(f"  {OK} instinctlab  {DIM}{instinctlab.__file__}{RESET}")
        ok = True
    except Exception as exc:  # noqa: BLE001
        emit(f"  {FAIL} {type(exc).__name__}: {str(exc)[:300]}")
        if "pytorch_kinematics" in str(exc):
            emit(f"    {DIM}修复：envs/isaaclab/bin/pip install pytorch_kinematics{RESET}")
    finally:
        # 顺序很关键：SimulationApp.close() 会强制终止进程，
        # 放在它之后的任何输出都不会执行（第一版就栽在这里，
        # 日志停在 "Simulation App Shutting Down" 后什么都没有）。
        with contextlib.suppress(Exception):
            os.write(saved_fd, ("\n".join(lines) + "\n").encode())
            os.fsync(saved_fd)
            os.close(saved_fd)
        # 故意不在这里 app.close()，交给 main 打完结论后收尾
        globals()["_SIM_APP"] = app

    return ok


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="含层 3（启动 IsaacSim，数分钟）")
    args = ap.parse_args()

    print(f"{BOLD}实践 11-C 环境验证{RESET}")
    print("=" * 60)
    sys.stdout.flush()
    # 层 3 会启动 IsaacSim 接管 stdout，最终结论必须走这份原始 fd 才看得见
    saved_fd = os.dup(1)

    def say(msg: str = "") -> None:
        with contextlib.suppress(Exception):
            os.write(saved_fd, (msg + "\n").encode())

    results = [("语法兼容", layer1_syntax()), ("torch API", layer2_torch_api())]
    if args.full:
        results.append(("实际 import", layer3_import()))
    else:
        print(f"\n{DIM}层 3（实际 import）已跳过，加 --full 启用{RESET}")

    say("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    say(f"{BOLD}结论{RESET}：{passed}/{len(results)} 层通过\n")

    if all(ok for _, ok in results):
        say(f"  {GREEN}现有 envs/isaaclab 可直接用于实践 11-C，无需另建环境{RESET}")
        if args.full:
            say("\n  下一步（等 GPU 空闲）：")
            say(f"    {DIM}cd repos/instinctlab && PYTHONPATH=... \\{RESET}")
            say(f"    {DIM}  python scripts/instinct_rl/train.py \\{RESET}")
            say(f"    {DIM}  --task Instinct-Parkour-Target-Amp-G1-v0 --headless{RESET}")
        rc = 0
    else:
        for name, ok in results:
            if not ok:
                say(f"  {FAIL} {name} 未通过")
        rc = 1

    with contextlib.suppress(Exception):
        os.fsync(saved_fd)
        os.close(saved_fd)

    # 所有输出都落盘后，才允许 SimulationApp 关闭（它会强制终止进程）
    app = globals().get("_SIM_APP")
    if app is not None:
        with contextlib.suppress(Exception):
            app.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
