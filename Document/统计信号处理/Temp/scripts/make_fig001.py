# -*- coding: utf-8 -*-
"""Fig001 统计信号处理全景图：从物理世界到估计/检测的数据流。

用法: py -3.14 make_fig001.py
输出: Documents/figures/Fig001_统计信号处理全景图.png
设计要点: 三行方框 + 箭头，所有文字居中于框内，框间留足间距；绘制后经
plotutil.check_figure 程序化碰撞检测（重叠/穿框）通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(12.6, 6.2))
ax.set_xlim(0, 1.16)
ax.set_ylim(0, 1.0)
ax.axis("off")

C_BOX = "#E8F0FA"      # 数据流
C_MODEL = "#FDEBD0"    # 概率模型
C_EST = "#E8F8EC"      # 估计
C_DET = "#FDEDED"      # 检测
C_PERF = "#F4F4F4"     # 性能评价
EC = "#4A5568"         # 边框
AC = "#2B6CB0"         # 箭头

def box(x0, y0, x1, y1, text, fc, fs=10.5):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.006",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=1.2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="#1A202C",
            linespacing=1.45)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=AC, lw=1.8,
                                shrinkA=0, shrinkB=0))


# ---- 第一行：数据流 ----
box(0.02, 0.70, 0.22, 0.92, "物理世界\n温度 · 距离 · 语音", C_BOX)
box(0.30, 0.70, 0.50, 0.92, "传感器 + ADC\n采样", C_BOX)
box(0.58, 0.70, 0.98, 0.92, "数据\nx[0], x[1], …, x[N−1]", C_BOX)
arrow(0.22, 0.81, 0.30, 0.81)
arrow(0.50, 0.81, 0.58, 0.81)

# ---- 第二行：概率模型 → 估计 ----
box(0.02, 0.36, 0.22, 0.58, "概率模型\n$p(\\mathbf{x};\\theta)$\n噪声的统计假设", C_MODEL)
box(0.30, 0.36, 0.50, 0.58, "估计器\n$\\hat{\\theta}=g(\\mathbf{x})$\nθ 是多少？", C_EST)
box(0.58, 0.36, 0.98, 0.58, "估计的性能\n偏差 / 方差\n估得准不准？", C_PERF)
arrow(0.78, 0.70, 0.12, 0.58)   # 数据 → 概率模型（对角）
arrow(0.22, 0.47, 0.30, 0.47)
arrow(0.50, 0.47, 0.58, 0.47)

# ---- 第三行：检测 ----
box(0.02, 0.02, 0.22, 0.24, "检测器\n在 H0 / H1 之间选择\n信号在不在？", C_DET)
box(0.30, 0.02, 0.50, 0.24, "检测的性能\n虚警率 / 检测率\n判对没有？", C_PERF)
arrow(0.12, 0.36, 0.12, 0.24)   # 概率模型 → 检测器
arrow(0.22, 0.13, 0.30, 0.13)

# ---- 右侧卷标（用 transData 坐标，置于 x=1.01~1.14，xlim 已放宽到 1.16）----
ax.text(1.01, 0.81, "第一卷\n估计理论", ha="left", va="center",
        transform=ax.transData, fontsize=11, color="#2F855A", fontweight="bold")
ax.text(1.01, 0.47, "估计：给出\n参数的取值", ha="left", va="center",
        transform=ax.transData, fontsize=9.5, color="#276749")
ax.text(1.01, 0.13, "第二卷\n检测理论", ha="left", va="center",
        transform=ax.transData, fontsize=11, color="#C53030", fontweight="bold")

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig001_统计信号处理全景图.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
