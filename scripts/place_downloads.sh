#!/usr/bin/env bash
# 把 ~/Downloads 里下好的三个课程包自动放到正确位置。
#
# 用法：下载完（压缩包或已解压的文件夹都行）直接跑
#     bash scripts/place_downloads.sh
#
# 做完会自动调 check_downloads.py 验证。
set -uo pipefail

ROOT="/home/limx/workspace/Roxan_warmup"
SRC="${1:-$HOME/Downloads}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

G="\033[32m"; Y="\033[33m"; R="\033[31m"; D="\033[2m"; B="\033[1m"; N="\033[0m"

echo -e "\n${B}从 $SRC 归位课程下载包${N}"
echo "══════════════════════════════════════════════════"

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT

# 压缩包先解到临时目录，这样后面只用处理"目录树"一种情况
shopt -s nullglob nocaseglob
for f in "$SRC"/*.zip "$SRC"/*.tar.gz "$SRC"/*.tgz "$SRC"/*.rar "$SRC"/*.7z; do
    name=$(basename "$f")
    echo -e "  ${D}解压 $name …${N}"
    d="$work/$(basename "${name%.*}")"; mkdir -p "$d"
    case "$f" in
        *.zip)          unzip -q -o "$f" -d "$d" 2>/dev/null ;;
        *.tar.gz|*.tgz) tar xzf "$f" -C "$d" 2>/dev/null ;;
        *.rar)          command -v unrar >/dev/null && unrar x -inul "$f" "$d/" \
                          || echo -e "     ${Y}需要 unrar：sudo apt install unrar${N}" ;;
        *.7z)           command -v 7z >/dev/null && 7z x -o"$d" "$f" >/dev/null \
                          || echo -e "     ${Y}需要 p7zip：sudo apt install p7zip-full${N}" ;;
    esac
done
shopt -u nocaseglob

# 搜索范围 = 下载目录 + 刚解压出来的临时目录
roots=("$SRC" "$work")

find_in_roots() {  # 透传全部 find 表达式，不要只取 $1 $2 ——
                   # 少传一个 -name 就会退化成"匹配所有目录"，
                   # 第一个命中的是下载目录本身，会把整包搬错地方
    for r in "${roots[@]}"; do
        [ -d "$r" ] || continue
        find "$r" -maxdepth 6 "$@" 2>/dev/null
    done
}

moved=0

# ── ① SMPL-X：三个 pkl 必须直接落在 body_models/smplx/ 下 ──
dst="$ROOT/repos/GMR/assets/body_models/smplx"
mkdir -p "$dst"
mapfile -t pkls < <(find_in_roots -name "SMPLX_*.pkl" | sort -u)
if [ ${#pkls[@]} -gt 0 ]; then
    for p in "${pkls[@]}"; do
        [ -f "$dst/$(basename "$p")" ] && continue
        cp -n "$p" "$dst/" && echo -e "  ${G}✅${N} $(basename "$p") → body_models/smplx/" && moved=1
    done
else
    echo -e "  ${D}－ 没找到 SMPLX_*.pkl${N}"
fi

# ── ② ACCAD：整个目录树搬过去，保留子目录结构 ──
dst="$ROOT/datasets/AMASS/ACCAD"
mkdir -p "$dst"
# 认两种布局：解压出 ACCAD/ 目录，或散落的 *_stageii.npz
accad_dir=$(find_in_roots -type d -name "ACCAD" | grep -v "^$dst\$" | head -1)
# 保险：源必须真含动作 npz。上面那个 find 一旦写错就会返回下载目录本身，
# 直接 cp 会把整包（含 SMPL-X、HOI 代码）灌进 ACCAD —— 测试时踩过。
if [ -n "$accad_dir" ] && ! find "$accad_dir" -maxdepth 3 -name "*.npz" -print -quit \
     2>/dev/null | grep -q .; then
    echo -e "  ${Y}⚠${N}  $accad_dir 里没有 npz，跳过（疑似匹配错目录）"
    accad_dir=""
fi
if [ -n "$accad_dir" ]; then
    cp -rn "$accad_dir"/. "$dst/" 2>/dev/null \
      && echo -e "  ${G}✅${N} ACCAD 目录 → datasets/AMASS/ACCAD/" && moved=1
else
    mapfile -t npzs < <(find_in_roots -name "*_stageii.npz" | head -2000)
    if [ ${#npzs[@]} -gt 0 ]; then
        # 连同父目录一起搬，ACCAD 的动作是按 SubjectN/ 分组的
        parent=$(dirname "${npzs[0]}")
        cp -rn "$(dirname "$parent")"/. "$dst/" 2>/dev/null \
          && echo -e "  ${G}✅${N} ${#npzs[@]} 个动作 npz → datasets/AMASS/ACCAD/" && moved=1
    else
        echo -e "  ${D}－ 没找到 ACCAD 数据${N}"
    fi
fi

# ── ③ HOI_Mimic：认 scripts/rsl_rl/train.py 定位真正的根 ──
dst="$ROOT/shenlan_hw/HOI_Mimic"
train_py=$(find_in_roots -path "*scripts/rsl_rl/train.py" | grep -i "hoi" | head -1)
[ -z "$train_py" ] && train_py=$(find_in_roots -type d -name "HOI_Mimic" | head -1)/scripts/rsl_rl/train.py
if [ -f "$train_py" ]; then
    # train.py 在 <root>/scripts/rsl_rl/ 下，往上三层就是项目根
    src_root=$(cd "$(dirname "$train_py")/../.." && pwd)
    if [ -d "$dst" ] && [ -n "$(ls -A "$dst" 2>/dev/null)" ]; then
        echo -e "  ${Y}⚠${N}  $dst 已存在且非空，跳过（要覆盖请先删除）"
    else
        mkdir -p "$dst" && cp -rn "$src_root"/. "$dst/" 2>/dev/null \
          && echo -e "  ${G}✅${N} HOI_Mimic → shenlan_hw/HOI_Mimic/" && moved=1
    fi
else
    echo -e "  ${D}－ 没找到 HOI_Mimic${N}"
fi

# ── ④ unitree_model USD（可选）──
usd=$(find_in_roots -name "*.usd" | grep -i "g1" | head -1)
if [ -n "$usd" ]; then
    dst="$ROOT/assets/unitree_model"; mkdir -p "$dst"
    src_root=$(find_in_roots -type d -name "unitree_model" | head -1)
    [ -n "$src_root" ] && cp -rn "$src_root"/. "$dst/" 2>/dev/null \
      && echo -e "  ${G}✅${N} unitree_model → assets/unitree_model/" && moved=1
fi

echo
if [ "$moved" -eq 0 ]; then
    echo -e "${Y}没有搬动任何文件${N} —— 确认包已下载到 $SRC"
    echo -e "${D}也可以指定别的目录：bash scripts/place_downloads.sh /path/to/downloads${N}"
fi

echo "══════════════════════════════════════════════════"
python3 "$HERE/scripts/check_downloads.py"
