# -*- coding: utf-8 -*-
"""Fig021 检测问题全景图：四要素流程 + 判决几何（P_FA / P_D）。

用法: py -3.14 make_fig021.py
输出: Documents/figures/Fig021_检测问题全景图.png

设计要点（对应原书第二卷第 1 章，PDF 472~484 / 书内 457~469）:
  (a) 检测问题四要素流程：数据 → 两个假设(H0/H1) → 判决规则(T vs 门限) → 性能指标(P_FA/P_D)。
  (b) 判决几何：DC 电平 A=1、门限 γ=1/2、噪声方差 σ²=0.5（对应原书图 1.6(b) 的
      重叠情形）。H0 右尾(红色) = 虚警 P_FA，H1 左尾(蓝色) = 漏检 P_M = 1 − P_D。
      该图说明：门限把两条 PDF 切成两类错误，P_FA 与 P_D 由门限联动，此消彼长。

绘制后经 plotutil.check_figure 程序化碰撞检测（重叠/穿框）通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

C_BOX = "#E8F0FA"    # 四要素框底色
EC = "#4A5568"       # 框边
AC = "#2B6CB0"       # 箭头
C_H0 = "#2B6CB0"     # 只有噪声曲线
C_H1 = "#C53030"     # 信号+噪声曲线
C_FA = "#C53030"     # 虚警填充（红）
C_MISS = "#2B6CB0"   # 漏检填充（蓝）
C_TH = "#4A5568"     # 门限灰

fig = plt.figure(figsize=(12.6, 7.4))
gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.15],
                      left=0.08, right=0.97, top=0.92, bottom=0.08, hspace=0.34)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])


def box(ax, x0, y0, x1, y1, text, fc, fs=10.5):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.008",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=1.2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="#1A202C",
            linespacing=1.5)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=AC, lw=1.8,
                                shrinkA=0, shrinkB=0))


# ================= 上排 (a)：四要素流程 =================
ax1.set_xlim(0, 1.0)
ax1.set_ylim(0, 1.0)
ax1.axis("off")
ax1.set_title("(a) 检测问题的四要素：数据 → 假设 → 判决规则 → 性能指标",
              fontsize=11.5, pad=10)

y0, y1 = 0.16, 0.78
box(ax1, 0.02, y0, 0.24, y1, "数据\n$x[0], x[1], \\ldots, x[N-1]$\n（观测到手的 N 点）", C_BOX)
box(ax1, 0.28, y0, 0.50, y1, "两个假设\n$H_0$：只有噪声\n$H_1$：信号 + 噪声", C_BOX)
box(ax1, 0.54, y0, 0.76, y1, "判决规则\n$T(\\mathbf{x})$ 与门限 $\\gamma$ 比较\n$T>\\gamma$ 判 $H_1$，否则 $H_0$", C_BOX)
box(ax1, 0.80, y0, 0.98, y1, "性能指标\n$P_{FA}$：虚警率\n$P_D$：检测率", C_BOX)

arrow(ax1, 0.24, 0.47, 0.28, 0.47)
arrow(ax1, 0.50, 0.47, 0.54, 0.47)
arrow(ax1, 0.76, 0.47, 0.80, 0.47)

# ================= 下排 (b)：判决几何 =================
A = 1.0          # DC 电平幅度（原书 A=1）
gam = 0.5        # 门限（原书阈值 1/2）
var = 0.5        # 噪声方差 σ²（对应原书图 1.6(b)）
sd = np.sqrt(var)

x = np.linspace(-3.0, 4.0, 600)
p0 = 1.0 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0.0) / sd) ** 2)
p1 = 1.0 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - A) / sd) ** 2)

ax2.plot(x, p0, lw=2.2, color=C_H0, label="只有噪声（$H_0$）")
ax2.plot(x, p1, lw=2.2, color=C_H1, label="信号 + 噪声（$H_1$）")

ax2.axvline(gam, color=C_TH, ls="--", lw=1.4)

# 虚警区：H0 在门限右侧的尾部
xa = x[x >= gam]
ax2.fill_between(xa, 0.0, p0[x >= gam], color=C_FA, alpha=0.28, lw=0)
# 漏检区：H1 在门限左侧的尾部
xb = x[x <= gam]
ax2.fill_between(xb, 0.0, p1[x <= gam], color=C_MISS, alpha=0.28, lw=0)

ax2.set_xlim(-3.0, 4.0)
ax2.set_ylim(0.0, 0.66)
ax2.set_xlabel("观测值 $x[0]$")
ax2.set_ylabel("概率密度")
ax2.set_title("(b) 判决的几何：门限 $\\gamma=1/2$ 把两条 PDF 切成虚警 $P_{FA}$ 与漏检 $P_M$",
              fontsize=11.5, pad=10)

ax2.text(gam, 0.615, "门限 $\\gamma=1/2$", ha="center", va="bottom",
         fontsize=10, color=C_TH)
ax2.text(0.9, 0.13, "$P_{FA}$（虚警）", ha="center", va="center",
         fontsize=9.5, color=C_FA)
ax2.text(0.1, 0.13, "$P_M$（漏检，$=1-P_D$）", ha="center", va="center",
         fontsize=9.5, color=C_MISS)

ax2.legend(loc="upper right", fontsize=10, framealpha=0.95)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig021_检测问题全景图.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
