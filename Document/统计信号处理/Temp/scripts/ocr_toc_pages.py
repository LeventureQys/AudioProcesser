# -*- coding: utf-8 -*-
"""用 RapidOCR 识别扫描页 PNG，逐页输出文本。

用法: py -3.14 ocr_toc_pages.py
输入: Temp/toc_pages/page_*.png
输出: Temp/toc_pages/ocr_*.txt
"""
import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.path.join(BASE, "toc_pages")

from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
files = sorted(glob.glob(os.path.join(PAGES_DIR, "page_*.png")))
print(f"found {len(files)} pages")

for path in files:
    result, _ = engine(path)
    lines = []
    if result:
        # result: list of [box, text, score]; 按 box 的 y 中心排序还原行序
        items = sorted(result, key=lambda r: (round((r[0][0][1] + r[0][2][1]) / 2 / 15), r[0][0][0]))
        for box, text, score in items:
            lines.append(text)
    out = os.path.join(PAGES_DIR, "ocr_" + os.path.basename(path).replace(".png", ".txt"))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"ocr {os.path.basename(path)} -> {len(lines)} lines")

print("done")
