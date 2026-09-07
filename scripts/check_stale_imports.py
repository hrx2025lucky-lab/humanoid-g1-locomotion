#!/usr/bin/env python3
"""静态排查各框架 play/train 脚本里"已失效的上游导入"。

为什么需要这个脚本
------------------
2026-09-07 一晚上撞了三次同一个 bug：上游把
``isaaclab.utils.pretrained_checkpoint`` 挪到了 ``isaaclab_rl.utils``，
而实践 5、8、10 的 play.py 各有一份旧路径。

危险之处在于**只有回放路径会崩**：train.py 不引用这些模块，
所以训练全程正常，直到要录交付视频时才发现起不来——
那时往往已经是"训练都跑完了、就差材料"的阶段。

这个脚本在不启动 IsaacSim 的前提下，把这类问题一次找出来。

误报控制
--------
裸 Python 下有三类导入注定解析不到，它们不是问题：
  1. IsaacSim 运行时才注入的包（carb / omni / pxr / isaacsim …）
  2. 脚本同目录的兄弟模块（cli_args / helpers），靠运行时 sys.path 解析
  3. 项目自带 venv 里的包（mjlab 有独立 .venv）
所以要按项目挑对应的解释器，并跳过上述三类。

被 try/except ImportError 包住的导入也跳过——那是有意写的兼容分支。

用法
----
    envs/isaaclab/bin/python scripts/check_stale_imports.py
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

WS = pathlib.Path("/home/limx/workspace/Roxan_warmup")
LAB_PY = WS / "envs/isaaclab/bin/python"

# (脚本路径, 用哪个解释器, 运行时会加进 sys.path 的目录)
TARGETS = [
    ("shenlan_hw/hw5_navigation/unitree_rl_lab/scripts/rsl_rl/play.py", LAB_PY),
    ("shenlan_hw/unitree_lab_amp/scripts/rsl_rl/play.py", LAB_PY),
    ("shenlan_hw/HOI_Mimic/scripts/rsl_rl/play.py", LAB_PY),
    ("shenlan_hw/HOI_Mimic/scripts/rsl_rl/train.py", LAB_PY),
    ("repos/unitree_rl_lab/scripts/rsl_rl/play.py", LAB_PY),
    ("repos/instinctlab/scripts/instinct_rl/train.py", LAB_PY),
]

RUNTIME_ONLY = {
    "carb", "omni", "isaacsim", "pxr", "warp", "usd", "usdrt",
    # 这些子模块要 IsaacSim 起来后才注册
    "isaaclab_tasks", "isaacsim_core",
}


def toplevel_imports(path: pathlib.Path) -> list[tuple[str, int]]:
    """收集顶层 import（跳过 try/except 包住的兼容分支）。"""
    tree = ast.parse(path.read_text(errors="ignore"))
    out: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module, node.lineno))
    return out


def _search_paths(script: pathlib.Path) -> list[str]:
    """运行时这些脚本能看到的额外 sys.path。

    三个来源，缺一个就会误报：
      1. 脚本自己的目录 —— cli_args 这类兄弟模块；
      2. 上一级目录 —— HOI_Mimic 的 list_envs.py 在 scripts/ 而
         train.py 在 scripts/rsl_rl/；
      3. 项目根 —— rsl_rl_amp / instinctlab 都没有 pip 安装，
         靠 "必须在项目根目录运行" 进 sys.path（这一点在
         record_all_videos.sh 里也有注释）。
    """
    paths = [str(script.parent), str(script.parent.parent)]
    # 从脚本往上找项目根：有 scripts/ 或 source/ 的那一层
    p = script.parent
    for _ in range(4):
        p = p.parent
        if (p / "source").is_dir() or (p / "scripts").is_dir():
            paths.append(str(p))
            # source/<pkg>/ 这一层也要加：很多项目不做 pip 安装，
            # 而是在启动脚本里 export PYTHONPATH=...:$LAB/source/instinctlab
            # （见 run_p11_after_p5.sh:13）。这类路径只在运行时存在，
            # 静态检查不加进来就会把正常代码全判成"解析不到"。
            src = p / "source"
            if src.is_dir():
                paths += [str(d) for d in src.iterdir() if d.is_dir()]
    # instinct_rl 是独立仓库，和 instinctlab 平级
    paths.append(str(WS / "repos/instinct_rl"))
    return paths


def _root_location(interpreter: pathlib.Path, root: str, extra_paths: list[str]) -> str | None:
    """拿到顶层包在磁盘上的位置。

    只对**顶层包名**调 find_spec：查子模块会触发父包的 __init__ 执行，
    而 isaaclab 的 __init__ 需要 carb（IsaacSim 运行时才有），
    于是像 isaaclab.utils.dict 这种完全正常的模块也会被判成"解析不到"。
    """
    code = (
        "import sys, importlib.util as u\n"
        f"sys.path[:0] = {extra_paths!r}\n"
        "try:\n"
        f"    s = u.find_spec({root!r})\n"
        "    p = (s.submodule_search_locations[0] if s and s.submodule_search_locations\n"
        "         else (s.origin if s else ''))\n"
        "    print(p or '')\n"
        "except Exception:\n"
        "    print('')\n"
    )
    try:
        r = subprocess.run([str(interpreter), "-c", code],
                           capture_output=True, text=True, timeout=90)
        out = r.stdout.strip().splitlines()
        return out[-1] if out and out[-1] else None
    except Exception:
        return None


def resolvable(interpreter: pathlib.Path, module: str, extra_paths: list[str],
               cache: dict[str, str | None]) -> bool:
    """模块能否解析。顶层包用 find_spec 定位，子模块改查文件系统。"""
    root, *rest = module.split(".")
    if root not in cache:
        cache[root] = _root_location(interpreter, root, extra_paths)
    loc = cache[root]
    if loc is None:
        return False
    if not rest:
        return True
    base = pathlib.Path(loc)
    if base.is_file():          # 顶层是单文件模块，不可能有子模块
        return False
    # 逐段在磁盘上找：既可以是子包目录，也可以是 .py 文件
    for i, seg in enumerate(rest):
        pkg_dir = base / seg
        py_file = base / f"{seg}.py"
        if pkg_dir.is_dir():
            base = pkg_dir
        elif py_file.is_file() and i == len(rest) - 1:
            return True
        else:
            # 也可能是父模块里 re-export 的名字（from X import Y 形式）
            return False
    return True


def main() -> int:
    problems = 0
    for rel, interp in TARGETS:
        path = WS / rel
        if not path.is_file():
            print(f"⚠️  跳过（不存在）：{rel}")
            continue
        if not interp.is_file():
            print(f"⚠️  跳过（没有解释器 {interp}）：{rel}")
            continue

        paths = _search_paths(path)
        cache: dict[str, str | None] = {}
        bad: list[tuple[str, int]] = []
        for mod, lineno in toplevel_imports(path):
            root = mod.split(".")[0]
            if root in RUNTIME_ONLY:
                continue
            if not resolvable(interp, mod, paths, cache):
                bad.append((mod, lineno))

        if bad:
            problems += len(bad)
            print(f"❌ {rel}")
            for mod, lineno in bad:
                print(f"      :{lineno}  {mod}")
        else:
            print(f"✅ {rel}")

    print()
    if problems:
        print(f"发现 {problems} 处解析不到的顶层导入。")
        print("典型修法：模块被上游挪了位置 → try 新路径 / except 回退旧路径；")
        print("          函数被上游删除 → 可选导入，缺失时退化成 no-op。")
    else:
        print("所有目标脚本的顶层导入都能解析。")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
