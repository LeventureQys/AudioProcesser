# -*- coding: utf-8 -*-
"""把扫描版 PDF 的前 40 页渲染为 PNG，供视觉识别目录页。

用法: py -3.14 render_toc_pages.py
输出: Temp/toc_pages/page_001.png ... page_040.png
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.dirname(BASE)
PDF_NAME = "统计信号处理基础--估计与检测理论 (Steven M. Kay,  译者 罗鹏飞  张文明  刘忠  赵艳丽) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
PDF_PATH = os.path.join(PDF_DIR, PDF_NAME)
OUT = os.path.join(BASE, "toc_pages")
os.makedirs(OUT, exist_ok=True)

import fitz

doc = fitz.open(PDF_PATH)
N = min(40, doc.page_count)
for pno in range(N):
    pix = doc[pno].get_pixmap(dpi=150)
    out = os.path.join(OUT, f"page_{pno+1:03d}.png")
    pix.save(out)
    print(f"saved {out} ({pix.width}x{pix.height})")
print("done")
