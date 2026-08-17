# -*- coding: utf-8 -*-
"""Fig019 估计量选型决策流程：按"已知什么"选估计方法（对应原书第 14 章 §14.4，图 14.1）。

用法: py -3.14 make_fig019.py
输出: Documents/figures/Fig019_估计量选型决策流程.png

设计要点（对应原书 PDF 409~411 / 书内 394~396，图 14.1）:
  (a) 经典方法与贝叶斯方法（根决策）：先问"参数的先验知识（先验 PDF 或前二阶矩）可用？"
      是 → 贝叶斯方法（θ 是随机变量）；否 → 换数据模型 / 取更多数据 → 经典方法（θ 是确定常数）。
  (b) 贝叶斯方法：联合 PDF p(x,θ) 已知 → 后验均值 = MMSE / 后验最大 = MAP；
      否则（前二阶矩已知）→ LMMSE。
  (c) 经典方法：PDF p(x;θ) 已知 → 先查 CRLB 等号 → 有效(MVU)；再试完备充分统计量 → MVU；
      再试 MLE；再试矩方法；若 PDF 未知（噪声中的信号）→ 线性信号+噪声前二阶矩已知 → BLUE，
      否则 → LSE。

绘制后经 plotutil.check_figure 程序化碰撞检测（重叠/穿框）通过才保存。
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

EC = "#4A5568"
AC = "#2B6CB0"
C_ROOT = "#E2F0E6"    # 根/问题框（绿）
C_OPT = "#D6E4F5"     # 经典/贝叶斯路线（蓝）
C_LEAF = "#F6E3CE"    # 叶子（估计量，橙）
C_NOTE = "#EFEFEF"    # 注记（灰）

fig = plt.figure(figsize=(12.6, 11.4))
gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.0, 1.25],
                      left=0.03, right=0.985, top=0.96, bottom=0.03, hspace=0.34)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])


def setup_axis(ax):
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")


def box(ax, x0, y0, x1, y1, text, fc, fs=9.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.006",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=1.0))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="#1A202C",
            linespacing=1.32)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 transform=ax.transAxes,
                                 arrowstyle="-|>", color=AC, lw=1.5,
                                 mutation_scale=12, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, fs=8.0):
    ax.text(x, y, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color=AC)


# ================= (a) 经典方法 vs 贝叶斯方法（根决策） =================
setup_axis(ax1)
ax1.set_title("(a) 先分叉：参数有没有先验知识？（对应原书图 14.1(a)）", fontsize=11.5, pad=6)

box(ax1, 0.30, 0.70, 0.70, 0.92, "信号处理问题\n（起一个多维问题）", C_ROOT, fs=9.5)
box(ax1, 0.30, 0.40, 0.70, 0.62, "先验知识可用？\n（先验 PDF 或前二阶矩）", C_ROOT, fs=9.5)

box(ax1, 0.03, 0.05, 0.26, 0.27, "是 → 贝叶斯方法\n（θ 是随机矢量）", C_OPT, fs=8.8)
box(ax1, 0.74, 0.05, 0.98, 0.27, "否 → 换数据模型 / 取更多数据\n→ 经典方法（θ 是确定常数）", C_OPT, fs=8.2)

arrow(ax1, 0.50, 0.70, 0.50, 0.63)
arrow(ax1, 0.42, 0.40, 0.15, 0.40)
arrow(ax1, 0.58, 0.40, 0.86, 0.40)
arrow(ax1, 0.15, 0.40, 0.15, 0.275)
arrow(ax1, 0.86, 0.40, 0.86, 0.275)
label(ax1, 0.28, 0.445, "是")
label(ax1, 0.72, 0.445, "否")

# ================= (b) 贝叶斯方法 =================
setup_axis(ax2)
ax2.set_title("(b) 贝叶斯方法：MMSE / MAP / LMMSE（对应原书图 14.1(b)）", fontsize=11.5, pad=6)

box(ax2, 0.30, 0.82, 0.70, 0.96, "先验知识", C_ROOT, fs=9.5)
box(ax2, 0.30, 0.52, 0.70, 0.72, "联合 PDF $p(\\mathbf{x},\\boldsymbol{\\theta})$ 已知？", C_ROOT, fs=9.2)

box(ax2, 0.03, 0.52, 0.26, 0.72, "是 → 计算后验 PDF\n$p(\\boldsymbol{\\theta}|\\mathbf{x})$", C_OPT, fs=8.6)
box(ax2, 0.03, 0.22, 0.26, 0.42, "取后验均值\n$E(\\boldsymbol{\\theta}|\\mathbf{x})$\n→ MMSE 估计量", C_LEAF, fs=8.4)
box(ax2, 0.03, 0.02, 0.26, 0.18, "使后验最大\n→ MAP 估计量", C_LEAF, fs=8.4)

box(ax2, 0.74, 0.52, 0.98, 0.72, "否 → 只有前二阶矩\n（均值、协方差）", C_OPT, fs=8.4)
box(ax2, 0.74, 0.30, 0.98, 0.48, "限定线性 → LMMSE 估计量\n（= 维纳滤波器，第 12 章）", C_LEAF, fs=8.2)

arrow(ax2, 0.50, 0.82, 0.50, 0.725)
arrow(ax2, 0.42, 0.52, 0.145, 0.52)
arrow(ax2, 0.58, 0.52, 0.86, 0.52)
arrow(ax2, 0.145, 0.52, 0.145, 0.425)
arrow(ax2, 0.145, 0.22, 0.145, 0.185)
arrow(ax2, 0.86, 0.52, 0.86, 0.485)
label(ax2, 0.275, 0.575, "是")
label(ax2, 0.725, 0.575, "否")

# ================= (c) 经典方法 =================
setup_axis(ax3)
ax3.set_title("(c) 经典方法：MVU / MLE / 矩 / BLUE / LSE（对应原书图 14.1(c)）", fontsize=11.5, pad=6)

box(ax3, 0.30, 0.84, 0.70, 0.97, "PDF $p(\\mathbf{x};\\boldsymbol{\\theta})$ 已知？", C_ROOT, fs=9.5)

# 左支：PDF 已知
box(ax3, 0.03, 0.64, 0.45, 0.80, "是 → 求 MVU 的正规流程", C_OPT, fs=9.0)
box(ax3, 0.03, 0.47, 0.45, 0.60, "满足 CRLB 等号\n→ 有效（MVU）估计量", C_LEAF, fs=8.2)
box(ax3, 0.03, 0.28, 0.45, 0.43, "完备充分统计量存在\n→ 使之无偏 → MVU", C_LEAF, fs=8.2)
box(ax3, 0.03, 0.10, 0.45, 0.24, "否则 → 计算 MLE\n（数值方法可用）→ MLE", C_LEAF, fs=8.2)
box(ax3, 0.03, 0.02, 0.45, 0.06, "否则 → 矩方法", C_NOTE, fs=7.8)

# 右支：PDF 未知（噪声中的信号）
box(ax3, 0.52, 0.64, 0.98, 0.80, "否 → 噪声中的信号？\n（只知一、二阶矩 / 无概率假设）", C_OPT, fs=8.4)
box(ax3, 0.52, 0.42, 0.98, 0.60, "线性信号 + 噪声前二阶矩已知\n→ BLUE", C_LEAF, fs=8.2)
box(ax3, 0.52, 0.24, 0.98, 0.38, "否则 → LSE（可加权、\n可非线性最小二乘）", C_LEAF, fs=8.2)

# 注记
box(ax3, 0.52, 0.02, 0.98, 0.20,
    "若为高斯线性模型 $\\mathbf{x}=\\mathbf{H}\\boldsymbol{\\theta}+\\mathbf{w}$，\n"
    "上述多条经典路线收敛到同一个 MVU（表 14.1）", C_NOTE, fs=8.0)

arrow(ax3, 0.50, 0.84, 0.50, 0.805)
arrow(ax3, 0.42, 0.84, 0.24, 0.805)
arrow(ax3, 0.58, 0.84, 0.75, 0.805)
label(ax3, 0.32, 0.865, "是")
label(ax3, 0.67, 0.865, "否")
arrow(ax3, 0.24, 0.64, 0.24, 0.605)
arrow(ax3, 0.24, 0.47, 0.24, 0.435)
arrow(ax3, 0.24, 0.28, 0.24, 0.245)
arrow(ax3, 0.24, 0.10, 0.24, 0.065)
arrow(ax3, 0.75, 0.64, 0.75, 0.605)
arrow(ax3, 0.75, 0.42, 0.75, 0.385)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig019_估计量选型决策流程.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
