# -*- coding: utf-8 -*-
"""Fig023 接收机工作特性（ROC）：检测器"能力曲线"。

用法: py -3.14 make_fig023.py
输出: Documents/figures/Fig023_ROC曲线.png

设计要点（对应原书第二卷第 3 章 §3.4，图 3.8/3.9，PDF 526~528 / 书内 511~513）:
  ROC 是 P_D 对 P_FA 的曲线。对均值偏移高斯-高斯问题（DC 电平检测），
  由 (3.10) 式 P_D = Q( Q^{-1}(P_FA) - sqrt(d^2) )，其中 d^2 为偏移系数。
  本图画出一族 ROC（d^2 = 0.5, 1, 4, 9），并叠加：
    - 45° 线（d^2=0，掷硬币判决器，不看数据的下界）；
    - 左上角标注 d^2 -> ∞ 的理想 ROC（对任何 P_FA 都有 P_D=1）。
  结论：ROC 恒在 45° 线之上且为凹函数；d^2 越大曲线越凸向左上角（能力越强）。

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
from scipy.stats import norm

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)


def Q(x):
    return norm.sf(x)


def Qinv(p):
    return norm.ppf(1.0 - p)


fig, ax = plt.subplots(figsize=(8.0, 6.6))

# 轴范围留 5~10% 余量（ROC 惯例 0~1，这里向外扩一点点）
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)

pfa = np.linspace(0.0005, 0.9995, 800)

# 一族 ROC：d^2 越大越凸向左上角
d2s = [0.5, 1.0, 4.0, 9.0]
colors = ["#6B46C1", "#2B6CB0", "#2F855A", "#C53030"]
for d2, c in zip(d2s, colors):
    pd = Q(Qinv(pfa) - np.sqrt(d2))
    ax.plot(pfa, pd, lw=2.2, color=c, label=f"$d^2={d2}$")

# 45° 线（d^2 = 0，掷硬币判决器）
ax.plot([0, 1], [0, 1], ls="--", lw=1.6, color="#4A5568",
        label="45° 线（$d^2=0$，掷硬币）")

# 理想 ROC（d^2 -> infinity）：P_FA=0 处跳变，其余 P_D=1
ax.plot([0, 0], [0, 1], lw=1.4, color="#A0AEC0")
ax.plot([0, 1], [1, 1], lw=1.4, color="#A0AEC0")

ax.set_xlabel("虚警概率 $P_{FA}$")
ax.set_ylabel("检测概率 $P_D$")
ax.set_title("接收机工作特性（ROC）：$P_D=Q(Q^{-1}(P_{FA})-\\sqrt{d^2})$ 的曲线族",
             fontsize=12, pad=10)

# 标注：45° 线下方是"不看数据"的下界
ax.text(0.50, 0.36, "45° 线 = 掷硬币判决器\n（完全不看数据）",
        ha="center", va="center", fontsize=9.5, color="#4A5568")
# 标注：左上角是理想 ROC
ax.annotate("$d^2\\rightarrow\\infty$：理想 ROC\n（任何 $P_{FA}$ 都有 $P_D=1$）",
            xy=(0.015, 0.985), xytext=(0.20, 0.86),
            fontsize=9.5, color="#1A202C",
            arrowprops=dict(arrowstyle="->", color="#A0AEC0", lw=1.3,
                            shrinkA=0, shrinkB=4))

ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

# 网格辅助
ax.grid(True, ls=":", lw=0.6, color="#CBD5E0", alpha=0.7)
ax.set_aspect("equal", adjustable="box")

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig023_ROC曲线.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
