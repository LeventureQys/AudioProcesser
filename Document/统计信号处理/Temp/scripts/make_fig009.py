# -*- coding: utf-8 -*-
"""Fig009 充分统计量信息压缩：数据 → T(x) 的无损压缩 + Neyman-Fisher 分解。

用法: py -3.14 make_fig009.py
输出: Documents/figures/Fig009_充分统计量信息压缩.png

设计要点（对应原书第 5 章）:
  (a) 左图: 压缩示意——N 个数据点 x[0..N-1] 汇入求和，压成 1 个统计量 T(x)=Σx[n]；
       关键是"无损"：给定 T(x) 后，条件分布 p(x|T(x)) 与参数 A 无关。
  (b) 右图: Neyman-Fisher 因子分解（定理 5.1）——p(x;A)=g(T(x),A)·h(x)，
       g 含 A 但只经 T 接触数据，h 不含 A → A 的全部信息都锁在 T(x) 里。

箭头用 FancyArrowPatch（patch 而非文本）。绘制后经 plotutil.check_figure
程序化碰撞检测（重叠/穿框）通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

C_DATA = "#2B6CB0"    # 数据
C_SUM = "#DD6B20"     # 求和/压缩
C_T = "#2F855A"       # 充分统计量
C_G = "#C53030"       # g(T,θ)
C_H = "#2B6CB0"       # h(x)
C_PDF = "#553C9A"     # 数据 PDF
EC = "#4A5568"
AC = "#4A5568"

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.6))
fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.03, wspace=0.26)


def box(ax, x0, y0, x1, y1, text, fc, fs=10.5):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.008",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=1.2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="#1A202C",
            linespacing=1.5)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 transform=ax.transAxes,
                                 arrowstyle="-|>", color=AC, lw=1.7,
                                 mutation_scale=14, shrinkA=0, shrinkB=0))


# ================= 左图 (a)：压缩示意 =================
ax_a.set_xlim(0, 1.0)
ax_a.set_ylim(0, 1.0)
ax_a.axis("off")

# 数据列
box(ax_a, 0.03, 0.78, 0.17, 0.88, "$x[0]$", C_DATA, fs=11)
box(ax_a, 0.03, 0.63, 0.17, 0.73, "$x[1]$", C_DATA, fs=11)
box(ax_a, 0.03, 0.48, 0.17, 0.58, "$x[2]$", C_DATA, fs=11)
ax_a.text(0.10, 0.385, "$\\vdots$", ha="center", va="center", transform=ax_a.transAxes,
          fontsize=15, color=EC)
box(ax_a, 0.03, 0.18, 0.17, 0.28, "$x[N{-}1]$", C_DATA, fs=11)

# 求和节点 + 充分统计量
box(ax_a, 0.40, 0.38, 0.56, 0.68, "求和\n$T(\\mathbf{x})=\\Sigma x[n]$", C_SUM, fs=10.5)
box(ax_a, 0.72, 0.42, 0.94, 0.64, "充分统计量\n$T(\\mathbf{x})$\n（1 个数）", C_T, fs=10.5)

arrow(ax_a, 0.17, 0.83, 0.40, 0.60)   # x[0] → Σ
arrow(ax_a, 0.17, 0.68, 0.40, 0.57)   # x[1] → Σ
arrow(ax_a, 0.17, 0.53, 0.40, 0.53)   # x[2] → Σ
arrow(ax_a, 0.17, 0.23, 0.40, 0.47)   # x[N-1] → Σ
arrow(ax_a, 0.56, 0.53, 0.72, 0.53)   # Σ → T

# 标注
ax_a.text(0.10, 0.945, "数据 $\\mathbf{x}$（N 个数）", ha="center", va="center",
          transform=ax_a.transAxes, fontsize=10, color="#1A202C")
ax_a.text(0.10, 0.09, "含 A 的全部信息", ha="center", va="center",
          transform=ax_a.transAxes, fontsize=9.5, color="#2F855A")
ax_a.text(0.48, 0.30, "压缩", ha="center", va="center", transform=ax_a.transAxes,
          fontsize=9.5, color=EC)

ax_a.set_title("(a) 数据 → 充分统计量：N 个数压成 1 个，信息不丢", fontsize=11)


# ================= 右图 (b)：Neyman-Fisher 分解 =================
ax_b.set_xlim(0, 1.0)
ax_b.set_ylim(0, 1.0)
ax_b.axis("off")

box(ax_b, 0.05, 0.74, 0.95, 0.94,
    "$p(\\mathbf{x};A)=(2\\pi\\sigma^2)^{-N/2}\\exp\\left[-\\frac{1}{2\\sigma^2}"
    "\\left(\\Sigma x^2[n]-2A\\,T+NA^2\\right)\\right]$",
    C_PDF, fs=10)
ax_b.text(0.5, 0.775, "其中 $T=\\Sigma x[n]$（数据之和）", ha="center", va="top",
          transform=ax_b.transAxes, fontsize=9.5, color=EC)

box(ax_b, 0.05, 0.32, 0.56, 0.60,
    "$g(T(\\mathbf{x}),A)=\\exp\\left[-\\frac{NA^2-2A\\,T}{2\\sigma^2}\\right]$\n"
    "含 $A$，但只经 $T$ 接触数据", C_G, fs=9.5)
box(ax_b, 0.61, 0.32, 0.95, 0.60,
    "$h(\\mathbf{x})=(2\\pi\\sigma^2)^{-N/2}\\exp\\left[-\\frac{\\Sigma x^2[n]}{2\\sigma^2}\\right]$\n"
    "不含 $A$（纯数据形状）", C_H, fs=9.5)

ax_b.text(0.585, 0.46, "×", ha="center", va="center", transform=ax_b.transAxes,
          fontsize=14, color=EC)

box(ax_b, 0.12, 0.05, 0.88, 0.22,
    "$\\Rightarrow T(\\mathbf{x})=\\Sigma x[n]$ 是 $A$ 的充分统计量（定理 5.1）\n"
    "$A$ 的全部信息都锁在 $T(\\mathbf{x})$ 里", C_T, fs=10)

arrow(ax_b, 0.30, 0.74, 0.30, 0.60)   # p → g
arrow(ax_b, 0.78, 0.74, 0.78, 0.60)   # p → h
arrow(ax_b, 0.30, 0.32, 0.40, 0.22)   # g → 结论
arrow(ax_b, 0.78, 0.32, 0.60, 0.22)   # h → 结论

ax_b.set_title("(b) Neyman-Fisher 分解：参数只藏在 $g(T,\\theta)$ 里", fontsize=11)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig009_充分统计量信息压缩.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
