# -*- coding: utf-8 -*-
"""提取《统计信号处理基础--估计与检测理论》PDF 的目录信息。

用法: py -3.14 extract_toc.py
输出: 同目录上级的 toc_embedded.md 与 toc_printed_pages.md
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Temp/
PDF_DIR = os.path.dirname(BASE)  # 统计信号处理/
PDF_NAME = "统计信号处理基础--估计与检测理论 (Steven M. Kay,  译者 罗鹏飞  张文明  刘忠  赵艳丽) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
PDF_PATH = os.path.join(PDF_DIR, PDF_NAME)

import fitz  # PyMuPDF

doc = fitz.open(PDF_PATH)
print(f"pages={doc.page_count}")
print(f"metadata={json.dumps(doc.metadata, ensure_ascii=False)}")

# 1) 内置书签目录
toc = doc.get_toc(simple=True)
print(f"toc_entries={len(toc)}")
with open(os.path.join(BASE, "toc_embedded.md"), "w", encoding="utf-8") as f:
    f.write("# PDF 内置书签目录（提取结果）\n\n")
    f.write(f"- 总页数: {doc.page_count}\n- 书签条数: {len(toc)}\n\n")
    f.write("| 层级 | 标题 | 目标页(书内页码序号) |\n|---|---|---|\n")
    for lvl, title, page in toc:
        f.write(f"| {lvl} | {title.strip()} | {page} |\n")

if len(toc) <= 6:
    print("内置书签过少，尝试解析书内印刷目录页...")

# 2) 扫描前 40 页，寻找印刷版目录页（通常含"目录"字样或大量点线页码）
printed = []
for pno in range(min(40, doc.page_count)):
    text = doc[pno].get_text("text")
    head = text[:400].replace("\n", " ")
    printed.append(f"===== 第 {pno+1} 页（页码序号 pno={pno}）=====\n{head}\n")

with open(os.path.join(BASE, "printed_head40.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(printed))

print("done")
