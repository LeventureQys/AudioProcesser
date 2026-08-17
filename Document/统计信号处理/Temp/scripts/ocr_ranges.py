# -*- coding: utf-8 -*-
"""批量 OCR：一次调用处理多个 (起始PDF页, 结束PDF页, 输出标签) 区间。

用法:
  py -3.14 ocr_ranges.py "PDF路径" 29-37:ch02 38-84:ch03 ...
每个区间输出到 Temp/chapters_ocr/<标签>/（page_XXX.png 与 ocr_page_XXX.txt，XXX 为 PDF 页码）
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz
from rapidocr_onnxruntime import RapidOCR

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = sys.argv[1]
SPECS = sys.argv[2:]

doc = fitz.open(PDF)
engine = RapidOCR()
n_pages = doc.page_count
print(f"pdf pages={n_pages}", flush=True)

for spec in SPECS:
    rng, tag = spec.rsplit(":", 1)
    start, end = (int(v) for v in rng.split("-"))
    out = os.path.join(BASE, "chapters_ocr", tag)
    os.makedirs(out, exist_ok=True)
    print(f"[{tag}] {start}..{end} -> {out}", flush=True)
    for pno in range(start, min(end, n_pages) + 1):
        page = doc[pno - 1]
        pix = page.get_pixmap(dpi=200)
        png = os.path.join(out, f"page_{pno:03d}.png")
        pix.save(png)
        result, _ = engine(png)
        lines = []
        if result:
            items = sorted(
                result, key=lambda r: (round((r[0][0][1] + r[0][2][1]) / 2 / 20), r[0][0][0])
            )
            for box, text, score in items:
                lines.append(text)
        txt = os.path.join(out, f"ocr_page_{pno:03d}.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[{tag}] page {pno}: {len(lines)} lines", flush=True)
    print(f"[{tag}] done", flush=True)

print("ALL DONE", flush=True)
