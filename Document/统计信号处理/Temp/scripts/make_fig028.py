# -*- coding: utf-8 -*-
"""Fig028 未知参数确定性信号的两个检测器：相关器平方（未知幅度 GLRT）与能量检测器。

用法: py -3.14 make_fig028.py
输出: Documents/figures/Fig028_能量检测器.png

设计要点（对应原书第二卷第 7 章，式 (7.13)(7.3)，PDF 666 / 663，书内 651 / 648）:
  (a) 未知幅度信号的 GLRT = 相关器平方（式 7.13）：数据 x 与已知波形 s[n] 相关，
      取平方、除以信号能量 Σs²[n]，再与门限比——只丢了幅度符号，性能损失约 0.5 dB（低 P_FA）。
  (b) 能量检测器（式 7.3）：对信号一无所知时，逐样本平方再求和、与门限比。
      它丢掉了整个波形，处理增益只有 5log10N（匹配滤波器是 10log10N，见 §7.3）。
  (b) 是 (a) 的"知识归零"极限：当连 s[n] 都不知道时，相关器平方退化为能量检测器。

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
C_CORR = "#F6E3CE"    # 相关/平方框（强调"波形信息还在"）
C_SQ = "#F6E3CE"      # 平方/求和框
C_DEC = "#E2F0E6"     # 判决框
EC = "#4A5568"
AC = "#2B6CB0"

fig = plt.figure(figsize=(13.0, 6.8))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0],
                      left=0.06, right=0.97, top=0.86, bottom=0.12, hspace=0.55)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0])


def box(ax, x0, y0, x1, y1, text, fc, fs=9.5):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.008",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=1.2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="#1A202C",
            linespacing=1.55)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=AC, lw=1.8,
                                shrinkA=0, shrinkB=0))


# ================= 上排 (a)：未知幅度 → 相关器平方 =================
ax_a.set_xlim(0, 1.0)
ax_a.set_ylim(0, 1.0)
ax_a.axis("off")
ax_a.set_title("(a) 未知幅度信号的 GLRT = 相关器平方（式 7.13，波形 $s[n]$ 已知、幅度 $A$ 未知）",
               fontsize=11.5, pad=10)

y0, y1 = 0.26, 0.78
box(ax_a, 0.005, y0, 0.130, y1, "数据\n$\\mathbf{x}$\n（$N$ 点）", C_BOX)
box(ax_a, 0.180, y0, 0.440, y1,
    "相关器（波形还在）\n$u=\\sum_{n=0}^{N-1}x[n]s[n]$", C_CORR)
box(ax_a, 0.490, y0, 0.780, y1,
    "平方 + 归一化\n$T(\\mathbf{x})=u^2/\\sum_{n=0}^{N-1}s^2[n]$\n"
    "$=(\\hat A)^2$（幅度估计的平方）", C_SQ)
box(ax_a, 0.830, y0, 0.995, y1, "判决\n$T(\\mathbf{x})>\\gamma''$\n判 $H_1$", C_DEC)

arrow(ax_a, 0.130, 0.52, 0.180, 0.52)
arrow(ax_a, 0.440, 0.52, 0.490, 0.52)
arrow(ax_a, 0.780, 0.52, 0.830, 0.52)

# ================= 下排 (b)：一无所知 → 能量检测器 =================
ax_b.set_xlim(0, 1.0)
ax_b.set_ylim(0, 1.0)
ax_b.axis("off")
ax_b.set_title("(b) 能量检测器（式 7.3，对信号一无所知时，波形信息归零）",
               fontsize=11.5, pad=10)

box(ax_b, 0.005, y0, 0.130, y1, "数据\n$\\mathbf{x}$\n（$N$ 点）", C_BOX)
box(ax_b, 0.180, y0, 0.440, y1,
    "逐样本平方\n$x^2[n]$（丢波形、留功率）", C_CORR)
box(ax_b, 0.490, y0, 0.780, y1,
    "求和（能量）\n$T(\\mathbf{x})=\\sum_{n=0}^{N-1}x^2[n]$\n"
    "（= 估计器-相关器，$\\hat s[n]=x[n]$）", C_SQ)
box(ax_b, 0.830, y0, 0.995, y1, "判决\n$T(\\mathbf{x})>\\gamma'$\n判 $H_1$", C_DEC)

arrow(ax_b, 0.130, 0.52, 0.180, 0.52)
arrow(ax_b, 0.440, 0.52, 0.490, 0.52)
arrow(ax_b, 0.780, 0.52, 0.830, 0.52)

# 图底注：放在 (b) 面板底部空白区
ax_b.text(0.5, 0.115,
          "知识逐步归零：已知波形→匹配滤波器（处理增益 $10\\log_{10}N$）；只知波形不知幅度→相关器平方（损失约 0.5 dB）；\n"
          "连波形都不知道→能量检测器（处理增益只有 $5\\log_{10}N$，$N=1000$ 时比匹配滤波器多亏 11.5 dB，见原书图 7.1）。",
          ha="center", va="center", transform=ax_b.transAxes, fontsize=9.3,
          color="#2D3748", linespacing=1.6)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig028_能量检测器.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
