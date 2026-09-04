"""课程 PDF 解密与提取 —— 把深蓝学院的加密 PDF 变成可读材料。

背景
----
课程 PDF 用 bash 工具（head/cat/dd/od）读出来是 `TSZ#` 开头的加密数据，
但用 **Python 或 Perl** 读同一个 inode 却得到正常的 `%PDF-1.7`。
机制未知（大概率是某个 DRM 驱动按进程做透明解密），但结果可复现：

    $ head -c 4 课件.pdf              → TSZ#
    $ python -c "print(open(...).read(4))"  → b'%PDF'

所以只要用 Python 把文件整读一遍另存，就得到一份可被 pdftotext /
pdftoppm 正常处理的真 PDF。

两类 PDF，处理方式不同
--------------------
1. **作业 PDF**（实践 1~11、作业讲解）—— 有文字层，直接 pdftotext
2. **课件 PDF**（第 2~8 章）—— **0 个字体**，每页是一张 1209×680 的图片，
   pdftotext 只能得到空白。必须转成 PNG 后用 OCR 或人/模型看图。

用法
----
    python decrypt_course_pdfs.py --scan          # 只体检，不产出
    python decrypt_course_pdfs.py                 # 解密 + 提文字
    python decrypt_course_pdfs.py --render 7      # 把第7章课件转成 PNG
    python decrypt_course_pdfs.py --render 7 --pages 1-10
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/home/limx/workspace/Roxan_warmup/motion control/humonoid ctrl")
OUT = Path.home() / "course_materials"
PDF_DIR = OUT / "pdf"       # 解密后的真 PDF
TXT_DIR = OUT / "txt"       # 有文字层的提取结果
IMG_DIR = OUT / "images"    # 图片型课件渲染出的 PNG


def decrypt(src: Path, dst: Path) -> bool:
    """用 Python 整读再写出 —— 这一步就是"解密"。

    不做增量判断以外的任何加工：读进来什么就写什么，
    保证产物与阅读器看到的内容完全一致。
    """
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False  # 已解密且大小一致，跳过
    data = src.read_bytes()
    if not data.startswith(b"%PDF"):
        raise RuntimeError(
            f"{src.name}: Python 读到的仍不是 PDF（前 8 字节 {data[:8]!r}）。"
            "解密假设不成立，不要继续。"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return True


def probe(pdf: Path) -> dict:
    """判断这份 PDF 有没有文字层。

    判据用 pdffonts 的字体数而不是"提取出的字符数" ——
    有些 PDF 能提取出少量页眉页脚文字，但正文仍是图片，
    只看字符数会把它误判为"可提取"。
    """
    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True)
    # 前两行是表头和分隔线
    n_fonts = max(len(fonts.stdout.strip().splitlines()) - 2, 0)
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    pages = 0
    for line in info.stdout.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split()[1])
    return {"fonts": n_fonts, "pages": pages, "has_text": n_fonts > 0}


def extract_text(pdf: Path, dst: Path) -> int:
    """提取文字层，返回非空行数。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftotext", "-layout", str(pdf), str(dst)],
                   capture_output=True)
    if not dst.exists():
        return 0
    return sum(1 for ln in dst.read_text(errors="ignore").splitlines() if ln.strip())


def render(pdf: Path, out_dir: Path, first: int | None, last: int | None,
           dpi: int = 150) -> int:
    """把图片型 PDF 渲染成 PNG，供 OCR 或直接看图。

    dpi=150 是权衡：原图内嵌分辨率是 192 ppi，再高只是插值放大不增信息；
    太低则中文小字糊掉。150 下单页约 200~400 KB。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["pdftoppm", "-png", "-r", str(dpi)]
    if first:
        cmd += ["-f", str(first)]
    if last:
        cmd += ["-l", str(last)]
    cmd += [str(pdf), str(out_dir / "page")]
    subprocess.run(cmd, check=True, capture_output=True)
    return len(list(out_dir.glob("page*.png")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", action="store_true", help="只体检，不产出文件")
    ap.add_argument("--render", metavar="关键词",
                    help="把文件名含该关键词的 PDF 渲染成 PNG（如 '第7章'）")
    ap.add_argument("--pages", help="渲染页范围，如 1-10")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    for tool in ("pdftotext", "pdffonts", "pdfinfo", "pdftoppm"):
        if not shutil.which(tool):
            print(f"❌ 缺少 {tool}（apt install poppler-utils）")
            return 1

    pdfs = sorted(SRC.glob("*.pdf"))
    if not pdfs:
        print(f"❌ {SRC} 下没有 PDF")
        return 1

    print(f"共 {len(pdfs)} 份 PDF\n")
    print(f"  {'文件':<34}{'页数':>5}{'字体':>5}  类型")
    print("  " + "-" * 62)

    text_pdfs, image_pdfs = [], []
    for src in pdfs:
        dec = PDF_DIR / src.name
        try:
            decrypt(src, dec)
        except RuntimeError as e:
            print(f"  ❌ {e}")
            continue
        info = probe(dec)
        kind = "文字层 ✅" if info["has_text"] else "图片型 ⚠️ 需 OCR"
        (text_pdfs if info["has_text"] else image_pdfs).append((src.name, info))
        print(f"  {src.name[:32]:<34}{info['pages']:>5}{info['fonts']:>5}  {kind}")

    print(f"\n  文字层 {len(text_pdfs)} 份 · 图片型 {len(image_pdfs)} 份")
    img_pages = sum(i["pages"] for _, i in image_pdfs)
    if image_pdfs:
        print(f"  图片型合计 {img_pages} 页，需要 OCR 或看图才能读")

    if args.scan:
        return 0

    if args.render:
        hits = [p for p in PDF_DIR.glob("*.pdf") if args.render in p.name]
        if not hits:
            print(f"\n❌ 没有文件名含 '{args.render}' 的 PDF")
            return 1
        first = last = None
        if args.pages:
            parts = args.pages.split("-")
            first = int(parts[0])
            last = int(parts[-1])
        for pdf in hits:
            d = IMG_DIR / pdf.stem
            n = render(pdf, d, first, last, args.dpi)
            print(f"\n  ✅ {pdf.name} → {n} 张 PNG")
            print(f"     {d}")
        return 0

    print("\n提取文字层 …")
    for name, _ in text_pdfs:
        pdf = PDF_DIR / name
        n = extract_text(pdf, TXT_DIR / (pdf.stem + ".txt"))
        print(f"  {name[:40]:<42} {n:>5} 行")

    print(f"\n产出目录：")
    print(f"  解密 PDF  {PDF_DIR}")
    print(f"  提取文本  {TXT_DIR}")
    if image_pdfs:
        print(f"\n图片型课件（{img_pages} 页）需要单独渲染，例如：")
        print(f"  python {Path(__file__).name} --render 第7章 --pages 1-10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
