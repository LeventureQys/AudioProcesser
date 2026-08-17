# -*- coding: utf-8 -*-
"""Fig011 最小二乘正交原理：误差矢量正交于信号子空间（几何解释）。

用法: py -3.14 make_fig011.py
输出: Documents/figures/Fig011_最小二乘正交原理.png

设计要点（对应原书第 8 章 §8.5，图 8.2 的抽象化）:
  数据矢量 x 位于 N 维空间；信号矢量 s=Hθ 只能在 p 维子空间 S_p 内（由 H 的
  列 h1、h2 张成）。LSE 的几何本质：选 S_p 中欧氏距离最靠近 x 的矢量，即 x 在
  S_p 上的正交投影 ŝ=Px（P=H(HᵀH)⁻¹Hᵀ）。误差 e=x−ŝ 与子空间正交（正交原理）。

本图用 2D 透视"桌面"示意：平行四边形 = 信号子空间 S_p；h1、h2 为基矢量；
x 在"桌面"上方（含噪声、不在子空间内）；ŝ 是 x 的垂直投影；误差 e 是垂线。
矢量用 FancyArrowPatch（patch 而非文本）。绘制后经 plotutil.check_figure
程序化碰撞检测（重叠/穿框）通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

C_SUB = "#E8F0FA"   # 信号子空间填充
C_H = "#DD6B20"     # 基矢量 h1/h2
C_DATA = "#C53030"  # 数据矢量 x
C_PROJ = "#2B6CB0"  # 投影 ŝ
C_ERR = "#2F855A"   # 误差 e
EC = "#4A5568"

fig, ax = plt.subplots(figsize=(10.2, 7.0))
fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.02)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.6)
ax.axis("off")

# ---- 信号子空间 S_p（平行四边形"桌面"）----
O = (1.2, 0.8)
A = (6.4, 1.4)   # h1 方向
B = (2.2, 4.2)   # h2 方向
C = (7.4, 4.8)   # A + h2
subspace = Polygon([O, A, C, B], closed=True, facecolor=C_SUB,
                   edgecolor=EC, linewidth=1.4, zorder=1)
ax.add_patch(subspace)

# ---- 基矢量 h1、h2（沿子空间两条边）----
ax.add_patch(FancyArrowPatch(O, A, arrowstyle="-|>", color=C_H, lw=2.2,
                             mutation_scale=16, shrinkA=0, shrinkB=0, zorder=3))
ax.add_patch(FancyArrowPatch(O, B, arrowstyle="-|>", color=C_H, lw=2.2,
                             mutation_scale=16, shrinkA=0, shrinkB=0, zorder=3))

# ---- 投影点 ŝ 与数据点 x ----
s = (4.09, 2.94)   # O + 0.45*h1 + 0.55*h2，落在子空间内
x = (4.09, 6.4)    # 在 ŝ 正上方（视觉上垂直桌面）

# ---- 矢量：数据 x、投影 ŝ、误差 e ----
ax.add_patch(FancyArrowPatch(O, x, arrowstyle="-|>", color=C_DATA, lw=2.2,
                             mutation_scale=16, shrinkA=0, shrinkB=0, zorder=4))
ax.add_patch(FancyArrowPatch(O, s, arrowstyle="-|>", color=C_PROJ, lw=2.4,
                             mutation_scale=16, shrinkA=0, shrinkB=0, zorder=5))
ax.add_patch(FancyArrowPatch(s, x, arrowstyle="-|>", color=C_ERR, lw=2.0,
                             linestyle="--", mutation_scale=16,
                             shrinkA=0, shrinkB=0, zorder=5))

# ---- 关键点 ----
for pt, c in [(O, "#1A202C"), (s, C_PROJ), (x, C_DATA)]:
    ax.plot(*pt, marker="o", ms=6, color=c, mec="none", zorder=6)

# ---- 文字标注（散开布局，避开矢量走线）----
ax.text(0.82, 0.42, "O（原点）", ha="center", va="center", fontsize=10, color="#1A202C")
ax.text(6.4, 0.80, "$h_1$", ha="center", va="center", fontsize=12, color=C_H)
ax.text(1.82, 4.58, "$h_2$", ha="center", va="center", fontsize=12, color=C_H)
ax.text(5.5, 3.72, "信号子空间 $S_p$\n（$h_1$、$h_2$ 张成的平面）",
        ha="center", va="center", fontsize=10, color="#1A202C")
ax.text(4.72, 2.42, "投影 $\\hat{s}=Px$\n$=H(H^{T}H)^{-1}H^{T}x$",
        ha="left", va="center", fontsize=9.5, color=C_PROJ, linespacing=1.4)
ax.text(4.72, 4.72, "误差 $e = x - \\hat{s}$\n$\\perp S_p$（正交原理）",
        ha="left", va="center", fontsize=9.5, color=C_ERR, linespacing=1.4)
ax.text(4.72, 6.62, "数据矢量 $x$\n（含噪声，不在 $S_p$ 内）",
        ha="left", va="center", fontsize=9.5, color=C_DATA, linespacing=1.4)

# ---- 右上：正交原理公式框 ----
ax.add_patch(FancyBboxPatch((7.35, 5.95), 2.55, 1.95, boxstyle="round,pad=0.04",
                            facecolor="#FEFCBF", edgecolor=EC, linewidth=1.3, zorder=7))
ax.text(8.625, 7.35, "正交原理", ha="center", va="center", fontsize=10.5,
        color="#1A202C", fontweight="bold")
ax.text(8.625, 6.62, "$(x-H\\hat{\\theta})^{T}H=0^{T}$\n误差 ⊥ 子空间所有矢量",
        ha="center", va="center", fontsize=9.5, color="#1A202C", linespacing=1.5)

ax.set_title("最小二乘的几何本质：误差矢量正交于信号子空间（正交原理）", fontsize=12)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig011_最小二乘正交原理.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
