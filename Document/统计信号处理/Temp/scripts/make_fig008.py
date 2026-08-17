# -*- coding: utf-8 -*-
"""Fig008 线性模型数据流：x = Hθ + w 的结构与直线拟合例子。

用法: py -3.14 make_fig008.py
输出: Documents/figures/Fig008_线性模型数据流.png

设计要点（对应原书第 4 章）:
  (a) 左图: 线性模型的数据流——参数 θ 经观测矩阵 H 变成信号 Hθ，加噪声 w 得观测 x；
       再由观测经 θ̂=(HᵀH)⁻¹Hᵀx 反解参数（MVU、达 CRLB）。
  (b) 右图: 直线拟合的例子——H 的第 1 列全是 1、第 2 列是 n，θ=[A,B]ᵀ，
       于是 Hθ = A + Bn，即拟合直线 x[n]=A+Bn+w[n]。

箭头用 FancyArrowPatch（patch 而非文本，避免空字符串标注产生包围盒误撞）。
绘制后经 plotutil.check_figure 程序化碰撞检测（重叠/穿框）通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

C_THETA = "#2F855A"   # 参数
C_H = "#2B6CB0"       # 观测矩阵
C_SIG = "#DD6B20"     # 信号/噪声中间量
C_X = "#553C9A"       # 观测
C_EST = "#2F855A"     # 估计量
EC = "#4A5568"
AC = "#4A5568"

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.6))
fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.03, wspace=0.26)


# ================= 左图 (a)：线性模型数据流 =================
ax_a.set_xlim(0, 1.0)
ax_a.set_ylim(0, 1.0)
ax_a.axis("off")


def box_a(x0, y0, x1, y1, text, fc, fs=10.5):
    ax_a.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                  boxstyle="round,pad=0.008",
                                  transform=ax_a.transAxes,
                                  facecolor=fc, edgecolor=EC, linewidth=1.2))
    ax_a.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
              transform=ax_a.transAxes, fontsize=fs, color="#1A202C",
              linespacing=1.5)


def arrow_a(x0, y0, x1, y1):
    ax_a.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                   transform=ax_a.transAxes,
                                   arrowstyle="-|>", color=AC, lw=1.7,
                                   mutation_scale=14, shrinkA=0, shrinkB=0))


# 第一行：前向数据生成 θ → Hθ → x
box_a(0.03, 0.64, 0.18, 0.86, "参数 $\\theta$\n(p×1)", C_THETA)
box_a(0.26, 0.64, 0.44, 0.86, "观测矩阵 $\\mathbf{H}$\n(N×p，已知，秩 p)", C_H)
box_a(0.52, 0.64, 0.70, 0.86, "信号 $\\mathbf{H}\\theta$\n(N×1)", C_SIG)
box_a(0.78, 0.68, 0.90, 0.82, "＋", C_SIG, fs=13)
box_a(0.78, 0.42, 0.97, 0.62, "噪声 $\\mathbf{w}$\n(N×1)\n$\\sim\\mathcal{N}(0,\\sigma^2\\mathbf{I})$", C_SIG, fs=9.5)
box_a(0.34, 0.30, 0.54, 0.50, "观测 $\\mathbf{x}$\n$=\\mathbf{H}\\theta+\\mathbf{w}$\n(N×1)", C_X, fs=10)

# 第二行：反向估计
box_a(0.26, 0.04, 0.62, 0.24,
      "MVU 估计量\n$\\hat{\\theta}=(\\mathbf{H}^T\\mathbf{H})^{-1}\\mathbf{H}^T\\mathbf{x}$\n"
      "（无偏、达 CRLB、高斯）", C_EST, fs=10)

arrow_a(0.18, 0.75, 0.26, 0.75)   # θ → H
arrow_a(0.44, 0.75, 0.52, 0.75)   # H → Hθ
arrow_a(0.70, 0.75, 0.78, 0.75)   # Hθ → ＋
arrow_a(0.875, 0.62, 0.84, 0.68)  # w → ＋
arrow_a(0.84, 0.68, 0.44, 0.50)   # ＋ → x
arrow_a(0.44, 0.30, 0.44, 0.24)   # x → θ̂

ax_a.text(0.135, 0.55, "① 前向：数据是这样生成的", ha="left", va="center",
          transform=ax_a.transAxes, fontsize=9.5, color="#276749")
ax_a.text(0.135, 0.28, "② 反向：由数据反解参数", ha="left", va="center",
          transform=ax_a.transAxes, fontsize=9.5, color="#276749")

ax_a.set_title("(a) 线性模型的数据流：$\\mathbf{x}=\\mathbf{H}\\theta+\\mathbf{w}$", fontsize=11)


# ================= 右图 (b)：直线拟合的例子 =================
ax_b.set_xlim(-1.3, 5.8)
ax_b.set_ylim(-1.3, 5.8)
ax_b.axis("off")

CW, CH = 0.92, 0.72   # 单元格宽高
X0, Y0 = 0.0, 0.0     # 矩阵左下角
NROW = 6              # 示例 N=6

for i in range(NROW):
    ax_b.add_patch(Rectangle((X0, Y0 + i * CH), CW, CH, fill=True,
                             facecolor="#E8F0FA", edgecolor=EC, lw=1.1))
    ax_b.text(X0 + CW / 2, Y0 + (i + 0.5) * CH, "1", ha="center", va="center",
              fontsize=12, color="#1A202C")
    ax_b.add_patch(Rectangle((X0 + CW, Y0 + i * CH), CW, CH, fill=True,
                             facecolor="#FDEBD0", edgecolor=EC, lw=1.1))
    ax_b.text(X0 + CW + CW / 2, Y0 + (i + 0.5) * CH, str(i), ha="center", va="center",
              fontsize=12, color="#1A202C")

# 列头（矩阵上方留出"矩阵 H"标签的空间）
ax_b.text(X0 + CW / 2, Y0 + NROW * CH + 0.22, "1", ha="center", va="bottom",
          fontsize=12.5, color="#1A202C", fontweight="bold")
ax_b.text(X0 + CW + CW / 2, Y0 + NROW * CH + 0.22, "n", ha="center", va="bottom",
          fontsize=12.5, color="#1A202C", fontweight="bold")
ax_b.text(X0 + CW, Y0 + NROW * CH + 0.80, "矩阵 $\\mathbf{H}$（N×2，$n=0,\\dots,N-1$）",
          ha="center", va="bottom", fontsize=10, color=EC)

# 右侧 θ 向量
ax_b.text(2.30, Y0 + 3.52, "$\\theta = $", ha="left", va="center", fontsize=12.5, color="#1A202C")
ax_b.add_patch(Rectangle((2.80, Y0 + 3.16), 0.95, CH, fill=True, facecolor="#E8F8EC", edgecolor=EC, lw=1.1))
ax_b.text(2.80 + 0.95 / 2, Y0 + 3.16 + CH / 2, "A", ha="center", va="center", fontsize=12.5, color="#1A202C")
ax_b.add_patch(Rectangle((2.80, Y0 + 3.16 - CH), 0.95, CH, fill=True, facecolor="#E8F8EC", edgecolor=EC, lw=1.1))
ax_b.text(2.80 + 0.95 / 2, Y0 + 3.16 - CH / 2, "B", ha="center", va="center", fontsize=12.5, color="#1A202C")
ax_b.text(3.90, Y0 + 3.52, "截距", ha="left", va="center", fontsize=9.5, color=EC)
ax_b.text(3.90, Y0 + 2.80, "斜率", ha="left", va="center", fontsize=9.5, color=EC)

# 结论行（两行，居中，避免过宽穿出坐标轴）
ax_b.text(2.30, -0.45, "$\\mathbf{H}\\theta = A\\cdot\\mathbf{1} + B\\cdot n = A + Bn$",
          ha="center", va="center", fontsize=11, color="#1A202C")
ax_b.text(2.30, -1.00, "即数据模型 $x[n]=A+Bn+w[n]$（$w[n]$ 为 WGN）",
          ha="center", va="center", fontsize=11, color="#1A202C")

ax_b.set_title("(b) 例子：直线拟合的观测矩阵（第 3 章例 3.7 的延续）", fontsize=11)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig008_线性模型数据流.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
