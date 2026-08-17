# -*- coding: utf-8 -*-
"""plotutil 冒烟测试：字体可用性 + 干净图通过 + 故意重叠图被拦。"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotutil import setup_cn, check_figure

font = setup_cn()
print(f"chosen font: {font}")

# 1) 干净图应通过
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0, 1], [0, 1])
ax.set_title("干净图")
ax.text(0.5, 0.5, "居中的注释", ha="center")
try:
    problems = check_figure(fig, strict=False)
    print(f"clean figure problems: {len(problems)}")
    for p in problems:
        print("  ", p)
    check_figure(fig, strict=True)
    print("clean figure: PASS")
except RuntimeError as e:
    print("clean figure: FAIL")
    print(e)

# 2) 故意重叠的两条注释应被拦
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot([0, 1], [0, 1])
ax2.text(0.4, 0.4, "AAAAAAAAAAAA", fontsize=20)
ax2.text(0.41, 0.41, "BBBBBBBBBBBB", fontsize=20)
try:
    check_figure(fig2, strict=True)
    print("overlap figure: FAIL (未拦住)")
except RuntimeError as e:
    print("overlap figure: PASS (被拦住)")
    print("  拦下信息:", str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else e)

# 3) 越出 Axes 的注释应被拦
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot([0, 1], [0, 1])
ax3.text(1.4, 0.5, "越界注释", ha="left", transform=ax3.transAxes)
try:
    check_figure(fig3, strict=True)
    print("out-of-axes figure: FAIL (未拦住)")
except RuntimeError as e:
    print("out-of-axes figure: PASS (被拦住)")

print("smoke done")
