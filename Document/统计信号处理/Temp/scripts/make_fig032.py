# -*- coding: utf-8 -*-
"""Fig032 检测器选型决策流程：按"已知什么"选检测器（对应原书第二卷第 11 章 §11.4，图 11.1~11.3）。

用法: py -3.14 make_fig032.py
输出: Documents/figures/Fig032_检测器选型指南.png

设计要点（对应原书 PDF 792~809 / 书内 777~794）:
  (a) 最佳路线（图 11.1/11.2）：先问"先验概率是否已知"→ 贝叶斯路线（代价已知用贝叶斯风险、
      0-1 代价用 MAP、等先验用条件 ML）；否则走 NP 准则（固定 P_FA 使 P_D 最大），再问
      "数据 PDF 是否完全已知"→ 已知且线性模型用 LRT(16)，未知参数可指定先验用贝叶斯 LRT(7)。
  (b) 复合假设准最佳路线（图 11.3）：PDF 含未知参数且不能指定先验时，按"什么未知"分三支：
      只有信号参数未知 / 只有噪声参数未知 / 两者都未知，逐支落到 GLRT / Rao / LMP 的对应项。
  括号内数字是原书 §11.2/§11.3 的项目编号（如 17=具有未知参数的确定性信号 GLRT）。

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
C_OPT = "#D6E4F5"     # 最佳路线（蓝）
C_SUB = "#E8F0FA"     # 准最佳（浅蓝）
C_LEAF = "#F6E3CE"    # 叶子（检测器，橙）
C_WARN = "#F3E0E0"    # 警示/不能继续（红）

fig = plt.figure(figsize=(12.8, 9.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05],
                      left=0.03, right=0.985, top=0.93, bottom=0.04, hspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])


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
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 transform=ax.transAxes,
                                 arrowstyle="-|>", color=AC, lw=1.5,
                                 mutation_scale=13, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, fs=7.5):
    ax.text(x, y, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color=AC)


# ================= (a) 最佳路线 =================
setup_axis(ax1)
ax1.set_title("(a) 二元假设检验的最佳路线（对应原书图 11.1 / 11.2）", fontsize=11.5, pad=8)

box(ax1, 0.30, 0.88, 0.70, 0.985,
    "二元假设：$H_0$（只有噪声） vs $H_1$（信号 + 噪声）", C_ROOT, fs=10.0)
box(ax1, 0.30, 0.70, 0.70, 0.84,
    "先验概率 $P(H_0),\\ P(H_1)$ 已知？", C_ROOT, fs=10.0)

# 左支：贝叶斯
box(ax1, 0.03, 0.70, 0.25, 0.84, "是 → 贝叶斯路线", C_OPT, fs=9.5)
box(ax1, 0.03, 0.48, 0.25, 0.62, "代价 $C_{ij}$ 已知？", C_OPT, fs=9.5)
box(ax1, 0.03, 0.24, 0.25, 0.40, "任意代价\n→ 贝叶斯风险 (3)", C_LEAF, fs=8.5)
box(ax1, 0.03, 0.05, 0.25, 0.19, "$C_{00}=C_{11}=0,\\ C_{10}=C_{01}=1$\n→ MAP (2)；等先验 → 条件 ML (2)", C_LEAF, fs=8.0)

# 右支：NP
box(ax1, 0.75, 0.70, 0.98, 0.84, "否 → NP 准则\n（固定 $P_{FA}$ 使 $P_D$ 最大）", C_OPT, fs=8.8)
box(ax1, 0.75, 0.48, 0.98, 0.62, "数据 PDF $p(\\mathbf{x};H_0),\\ p(\\mathbf{x};H_1)$\n完全已知？", C_OPT, fs=8.6)
box(ax1, 0.75, 0.24, 0.98, 0.40, "已知 + 线性模型 → LRT (16)\n已知 + 非线性 → 一般 LRT (1)", C_LEAF, fs=8.2)
box(ax1, 0.75, 0.05, 0.98, 0.19, "未知参数可指定先验 → 贝叶斯 LRT (7)；\n否则 → 准最佳路线（下）", C_LEAF, fs=8.2)

# 箭头
arrow(ax1, 0.50, 0.88, 0.50, 0.845)
arrow(ax1, 0.42, 0.70, 0.14, 0.70)
arrow(ax1, 0.58, 0.70, 0.86, 0.70)
label(ax1, 0.28, 0.745, "是")
label(ax1, 0.72, 0.745, "否")
arrow(ax1, 0.14, 0.70, 0.14, 0.625)
arrow(ax1, 0.86, 0.70, 0.86, 0.625)
arrow(ax1, 0.14, 0.48, 0.14, 0.405)
arrow(ax1, 0.86, 0.48, 0.86, 0.405)

# ================= (b) 准最佳路线 =================
setup_axis(ax2)
ax2.set_title("(b) 复合假设的准最佳路线（PDF 含未知参数、不能指定先验；对应原书图 11.3）",
              fontsize=11.5, pad=8)

box(ax2, 0.28, 0.88, 0.72, 0.985, "什么未知？", C_ROOT, fs=10.0)

# 左支：只有信号参数未知
box(ax2, 0.02, 0.64, 0.34, 0.80, "只有信号参数未知", C_SUB, fs=9.0)
box(ax2, 0.02, 0.30, 0.34, 0.57,
    "高斯 + 线性模型 → GLRT (17)\n"
    "高斯 + 非线性 → GLRT (8/11)、\nRao (10/13)、LMP (14)\n"
    "IID 非高斯 + 线性 → Rao (21)", C_LEAF, fs=8.0)

# 中支：只有噪声参数未知
box(ax2, 0.36, 0.64, 0.58, 0.80, "只有噪声参数未知", C_SUB, fs=9.0)
box(ax2, 0.36, 0.40, 0.58, 0.57, "→ GLRT (6)", C_LEAF, fs=8.5)

# 右支：两者都未知
box(ax2, 0.60, 0.64, 0.98, 0.80, "信号与噪声参数都未知", C_SUB, fs=9.0)
box(ax2, 0.60, 0.14, 0.98, 0.57,
    "WGN 未知方差 + 线性 → GLRT (18)\n"
    "高斯非白（多噪声参数）+ 线性 → Rao (19)\n"
    "非高斯 / 任意模型 → GLRT (11)、Rao (13)", C_LEAF, fs=8.0)

# 底部注记
box(ax2, 0.02, 0.02, 0.58, 0.16,
    "Wald 检验(9/12)要求 $H_1$ 下 MLE，若可得则 $H_0$ 下 MLE 也可得，故检测中少用，GLRT 更佳",
    C_WARN, fs=8.0)

# 箭头
arrow(ax2, 0.50, 0.88, 0.50, 0.805)
arrow(ax2, 0.28, 0.76, 0.18, 0.72)
arrow(ax2, 0.50, 0.76, 0.47, 0.72)
arrow(ax2, 0.72, 0.76, 0.79, 0.72)
arrow(ax2, 0.18, 0.64, 0.18, 0.575)
arrow(ax2, 0.47, 0.64, 0.47, 0.575)
arrow(ax2, 0.79, 0.64, 0.79, 0.575)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig032_检测器选型指南.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
