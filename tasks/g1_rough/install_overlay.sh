#!/usr/bin/env bash
# 把 g1_rough/ 挂进 unitree_rl_lab 的任务树。
#
# 为什么用 symlink 而不是把文件直接放进 unitree_rl_lab：
#   unitree_rl_lab 是第三方仓库（本机是解压的，无 .git），不纳入本仓库版本控制。
#   symlink 让代码的唯一真实副本留在本仓库里，unitree_rl_lab 保持零改动，
#   git log 里 100% 是自己写的代码。
#
# 为什么这样能被自动发现：
#   IsaacLab 的 import_packages() 递归 import 所有**目录包**。
#   rough/ 是目录包 → 其 __init__.py 被 import → gym.register 执行。
#   pkgutil.iter_modules 对 symlink 透明（已用最小实验验证）。

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="${UNITREE_RL_LAB:-$HOME/workspace/Roxan_warmup/repos/unitree_rl_lab}"
DST="$LAB/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/rough"

if [[ ! -d "$LAB" ]]; then
    echo "找不到 unitree_rl_lab：$LAB" >&2
    echo "用 UNITREE_RL_LAB=/path/to/unitree_rl_lab $0 指定路径" >&2
    exit 1
fi

ln -sfn "$SRC" "$DST"
echo "已挂载: $DST"
echo "      -> $(readlink -f "$DST")"
echo
echo "验证（纯 Python，不启动 Isaac Sim）："
echo "  cd $LAB && python scripts/list_envs.py | grep Rough"
