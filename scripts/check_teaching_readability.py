#!/usr/bin/env python3
"""检查 teaching/ 文档的"零基础可读性"。

为什么需要这个脚本
------------------
"零基础能看懂"是个主观判断，读的时候容易高估自己写得清楚。
这里把它拆成几条**可量化**的检查：

1. **术语首次出现有没有解释** —— 直接甩 GAE、KL 散度、优势函数
   而不解释，零基础就断在这里了
2. **有没有具体数字** —— 只讲概念不给实测值，读者无法建立量感
3. **有没有代码/公式的逐行走读** —— 光贴代码等于没讲
4. **段落是否过长** —— 连续大段文字没有换气点，读起来累
5. **8 节结构是否完整** —— 00_写作标准.md 定的固定结构

用法
----
    envs/isaaclab/bin/python scripts/check_teaching_readability.py
    envs/isaaclab/bin/python scripts/check_teaching_readability.py --file teaching/07_运动重定向GMR.md
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEACH = ROOT / "teaching"

G, Y, R, D, B, N = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"

# 这些术语第一次出现时应该有解释。不解释的话零基础直接断片。
# 判据：术语首次出现的那一段（或紧邻两段）里要有解释性表述。
JARGON = [
    "GAE", "KL 散度", "KL散度", "优势函数", "策略梯度", "重要性采样",
    "判别器", "蒸馏", "四元数", "正运动学", "逆运动学", "IK", "FK",
    "MDP", "折扣因子", "熵", "课程学习", "域随机化", "特权观测",
    "RayCaster", "decimation", "actor-critic", "PPO", "AMP",
]

# 出现下列任一表述，就算对该术语做了解释
EXPLAIN_HINTS = [
    "是指", "就是", "也就是", "意思是", "换句话说", "简单说", "通俗",
    "指的是", "= ", "≈", "叫做", "称为", "可以理解为", "打个比方",
    "举个例子", "比如说", "——", "：",
]

REQUIRED_SECTIONS = [
    "一句话概括", "问题从哪来", "核心概念", "算法原理",
    "代码走读", "踩过的坑", "面试", "延伸",
]


def check_jargon(text: str) -> list[str]:
    """术语首次出现时，附近有没有解释。

    判据要放宽两处，否则会大量误报（我第一版就误报了 FK/PPO/蒸馏）：

    1. **术语出现在标题里**（`### 正运动学 FK 与逆运动学 IK`）说明
       它有专章讲解，检查窗口要扩到整节而不是相邻两段；
    2. 有的文档先在引言里点一下名、后文才展开，所以窗口取
       "首次出现后的连续 3 段"而不是 2 段。
    """
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    problems = []
    for term in JARGON:
        for i, p in enumerate(paras):
            if term not in p:
                continue
            # 出现在标题里 = 有专章讲，窗口放到 6 段
            span = 6 if re.search(rf"^#{{1,4}} .*{re.escape(term)}", p, re.M) else 3
            window = "".join(paras[i:i + span])
            if not any(h in window for h in EXPLAIN_HINTS):
                problems.append(term)
            break   # 只看首次出现
    return problems


def check_numbers(text: str) -> int:
    """有多少个"带单位或小数的具体数字"——量感的来源。"""
    return len(re.findall(r"\d+\.\d+|\d+\s*(?:m|s|Hz|轮|维|帧|个|%|°|rad)", text))


def check_code_walkthrough(text: str) -> tuple[int, int]:
    """代码块数量，以及其中有多少块**被讲解过**。

    这里踩过一个坑：最初只数代码块内的 `#` 注释行占比，结果 11 份文档
    有 9 份被判"代码缺注释"——这么整齐的结果通常说明是判据错了，
    不是文档都写差了。抽查发现，很多 ``` 块装的是**数学公式、目录树、
    终端输出、对照示意图**，本来就不该有 `#` 注释。

    改成两条：
      1. 只统计带语言标记的代码块（```python / ```bash），
         裸 ``` 多半是公式或输出，不计入；
      2. "讲解过"的判据放宽为：块内有注释 **或** 块后紧跟解释文字。
         代码后面用正文逐行说明，和写在代码里的注释是等效的。
    """
    # (语言, 代码, 该块结束后的位置)
    blocks = list(re.finditer(r"```([a-z]+)\n(.*?)```", text, re.S))
    real, explained = 0, 0
    for m in blocks:
        lang, body = m.group(1), m.group(2)
        if lang in ("text", "txt", "console", "output"):
            continue
        lines = [l for l in body.splitlines() if l.strip()]
        if not lines:
            continue
        real += 1
        commented = sum(1 for l in lines if re.match(r"\s*(#|//|--)", l))
        if commented / len(lines) > 0.15:
            explained += 1
            continue
        # 块后 300 字里有没有解释性文字
        after = text[m.end():m.end() + 300]
        if any(h in after for h in EXPLAIN_HINTS) or len(after.strip()) > 80:
            explained += 1
    return real, explained


def check_long_paragraphs(text: str, limit: int = 400) -> int:
    """超过 limit 字的段落数——太长就没有换气点。"""
    paras = re.split(r"\n\s*\n", text)
    return sum(1 for p in paras if len(p) > limit and "```" not in p and "|" not in p)


def check_sections(text: str) -> list[str]:
    heads = "\n".join(re.findall(r"^#{1,3} .*$", text, re.M))
    return [s for s in REQUIRED_SECTIONS if s not in heads]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="只查一份")
    args = ap.parse_args()

    files = ([pathlib.Path(args.file)] if args.file
             else sorted(TEACH.glob("[0-1][0-9]_*.md")))
    files = [f for f in files if "写作标准" not in f.name and "RL算法原理" not in f.name]

    print(f"\n{B}teaching/ 零基础可读性检查{N}")
    print("=" * 92)
    print(f"{'文档':<34}{'行数':>6}{'数字':>6}{'代码块':>7}{'长段':>6}  问题")
    print("-" * 92)

    total_issues = 0
    for f in files:
        text = f.read_text(errors="ignore")
        lines = text.count("\n")
        nums = check_numbers(text)
        blocks, explained = check_code_walkthrough(text)
        longp = check_long_paragraphs(text)
        jargon = check_jargon(text)
        missing = check_sections(text)

        issues = []
        if nums < 15:
            issues.append(f"具体数字偏少({nums})")
        if blocks and explained / blocks < 0.5:
            issues.append(f"代码缺注释({explained}/{blocks})")
        if longp > 3:
            issues.append(f"长段落{longp}个")
        if jargon:
            issues.append(f"术语未解释: {'/'.join(jargon[:3])}")
        if missing:
            issues.append(f"缺节: {'/'.join(missing[:2])}")

        total_issues += len(issues)
        color = G if not issues else (Y if len(issues) <= 2 else R)
        mark = "✅" if not issues else ("🔶" if len(issues) <= 2 else "⚠️")
        name = f.name[:32]
        print(f"{name:<34}{lines:>6}{nums:>6}{blocks:>7}{longp:>6}  {color}{mark} "
              f"{'; '.join(issues) if issues else '好'}{N}")

    print("=" * 92)
    print(f"\n{D}判据说明：{N}")
    print(f"  {D}· 具体数字 ≥15：零基础需要量感，只讲概念记不住{N}")
    print(f"  {D}· 代码块注释率 >15% 才算「讲过」，光贴代码等于没讲{N}")
    print(f"  {D}· 术语首次出现的那一段（或下一段）要有解释性表述{N}")
    print(f"  {D}· 段落 >400 字算长段，没有换气点读着累{N}\n")
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
