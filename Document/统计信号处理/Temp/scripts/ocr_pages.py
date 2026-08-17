# -*- coding: utf-8 -*-
"""按 PDF 页码区间渲染并 OCR 指定页（供正文写作核对用）。

用法:
  py -3.14 ocr_pages.py "<PDF路径>" <起始PDF页> <结束PDF页> <输出标签>
输出:
  Temp/chapters_ocr/<标签>/page_XXX.png   （渲染页，dpi=200）
  Temp/chapters_ocr/<标签>/ocr_page_XXX.txt（OCR 行文本，XXX 为 PDF 页码）
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = sys.argv[1]
START = int(sys.argv[2])
END = int(sys.argv[3])
TAG = sys.argv[4]
OUT = os.path.join(BASE, "chapters_ocr", TAG)
os.makedirs(OUT, exist_ok=True)

doc = fitz.open(PDF)
engine = RapidOCR()
n_pages = doc.page_count
print(f"pdf {PDF} pages={n_pages}, ocr range {START}..{END} -> {OUT}", flush=True)

for pno in range(START, END + 1):  # PDF 页码，1-based
    if pno > n_pages:
        print(f"page {pno} 超出文档范围，停止", flush=True)
        break
    page = doc[pno - 1]
    pix = page.get_pixmap(dpi=200)
    png = os.path.join(OUT, f"page_{pno:03d}.png")
    pix.save(png)
    result, _ = engine(png)
    lines = []
    if result:
        items = sorted(
            result, key=lambda r: (round((r[0][0][1] + r[0][2][1]) / 2 / 20), r[0][0][0])
        )
        for box, text, score in items:
            lines.append(text)
    txt = os.path.join(OUT, f"ocr_page_{pno:03d}.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"page {pno}: {len(lines)} lines", flush=True)

print("done", flush=True)
