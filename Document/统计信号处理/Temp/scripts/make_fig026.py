# -*- coding: utf-8 -*-
"""Fig026 估计器-相关器：数据先经维纳滤波估出信号，再与估计相关（两种实现）。

用法: py -3.14 make_fig026.py
输出: Documents/figures/Fig026_估计器相关器数据流.png

设计要点（对应原书第二卷第 5 章 §5.3，式 (5.5)(5.6)(5.9)，图 5.2/5.3，
        PDF 580~584 / 书内 565~569）:
  (a) 估计器-相关器直接形式：数据 x → 维纳滤波器 ŝ = C_s(C_s+σ²I)⁻¹x（先估信号）
      → 相关器 T(x) = xᵀŝ = Σ x[n]ŝ[n]（再与估计相关）→ 与门限比 → 判 H1。
      这是"检测器里长出一个估计器"的结构：判决统计量是观测与信号 MMSE 估计的内积。
  (b) 标准形式（特征分解）：先把数据用 C_s 的模态矩阵 V 去相关成 y = Vᵀx，
      再做加权能量检测 T(x) = Σ (λ_sn/(λ_sn+σ²)) y²[n]，权重 λ_sn/(λ_sn+σ²)
      即变换域上的维纳加权。同一个 T(x) 的两种实现。

绘制后经 plotutil.check_figure 程序化碰撞检测（重叠/穿框）通过才保存。
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

C_BOX = "#E8F0FA"     # 通用框
C_EST = "#F6E3CE"     # 估计器框（强调"检测器里长出了估计器"）
C_DEC = "#E2F0E6"     # 判决框
EC = "#4A5568"        # 框边
AC = "#2B6CB0"        # 箭头

fig = plt.figure(figsize=(13.0, 6.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0],
                      left=0.06, right=0.97, top=0.88, bottom=0.10, hspace=0.52)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0])


def box(ax, x0, y0, x1, y1, text, fc, fs=10.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.008",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=1.2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="#1A202C",
            linespacing=1.6)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=AC, lw=1.8,
                                shrinkA=0, shrinkB=0))


# ================= 上排 (a)：估计器-相关器直接形式 =================
ax_a.set_xlim(0, 1.0)
ax_a.set_ylim(0, 1.0)
ax_a.axis("off")
ax_a.set_title("(a) 估计器-相关器（直接形式）：先估信号，再与估计相关",
               fontsize=12.0, pad=10)

y0, y1 = 0.28, 0.80
box(ax_a, 0.010, y0, 0.200, y1,
    "数据\n$\\mathbf{x}=[x[0],\\ldots,x[N-1]]^T$\n（$N$ 点观测）", C_BOX)
box(ax_a, 0.260, y0, 0.545, y1,
    "维纳滤波器（先估信号）\n$\\hat{\\mathbf{s}}=C_s(C_s+\\sigma^2I)^{-1}\\mathbf{x}$\n"
    "$=$ 信号现实 $\\mathbf{s}$ 的 MMSE 估计", C_EST)
box(ax_a, 0.605, y0, 0.845, y1,
    "相关器（再与估计相关）\n$T(\\mathbf{x})=\\mathbf{x}^T\\hat{\\mathbf{s}}$\n"
    "$=\\sum_{n=0}^{N-1}x[n]\\,\\hat{s}[n]$", C_BOX)
box(ax_a, 0.895, y0, 0.995, y1,
    "判决\n$T(\\mathbf{x})>\\gamma'$\n判 $H_1$", C_DEC)

arrow(ax_a, 0.200, 0.54, 0.260, 0.54)
arrow(ax_a, 0.545, 0.54, 0.605, 0.54)
arrow(ax_a, 0.845, 0.54, 0.895, 0.54)

# ================= 下排 (b)：标准形式 =================
ax_b.set_xlim(0, 1.0)
ax_b.set_ylim(0, 1.0)
ax_b.axis("off")
ax_b.set_title("(b) 标准形式（特征分解）：去相关后按维纳权重做加权能量检测（同一个 $T(\\mathbf{x})$）",
               fontsize=12.0, pad=10)

box(ax_b, 0.010, y0, 0.285, y1,
    "去相关器\n$\\mathbf{y}=V^T\\mathbf{x}$\n（$V$：$C_s$ 的模态矩阵\n"
    "列是 $C_s$ 的特征矢量）", C_BOX)
box(ax_b, 0.355, y0, 0.815, y1,
    "加权能量检测器\n$T(\\mathbf{x})=\\sum_{n=0}^{N-1}\\frac{\\lambda_{s_n}}{\\lambda_{s_n}+\\sigma^2}\\,y^2[n]$\n"
    "权重 $=\\lambda_{s_n}/(\\lambda_{s_n}+\\sigma^2)$（变换域维纳加权）", C_EST)
box(ax_b, 0.880, y0, 0.995, y1,
    "判决\n$T(\\mathbf{x})>\\gamma'$\n判 $H_1$", C_DEC)

arrow(ax_b, 0.285, 0.54, 0.355, 0.54)
arrow(ax_b, 0.815, 0.54, 0.880, 0.54)

# 图底注：说明两种实现的关系（放在框下方空白区，避免与框/箭头重叠）
ax_b.text(0.5, 0.145,
          "两个结构算的是同一个 $T(\\mathbf{x})$：估计器-相关器把“检测”变成“先 MMSE 估信号、再求内积”；\n"
          "特征分解揭示其本质——把数据转到一个新坐标系，按各方向信噪比 $\\lambda_{s_n}/\\sigma^2$ 加权求能量。",
          ha="center", va="center", transform=ax_b.transAxes, fontsize=9.5,
          color="#2D3748", linespacing=1.6)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig026_估计器相关器数据流.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
