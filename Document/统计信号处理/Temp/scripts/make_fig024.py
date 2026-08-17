# -*- coding: utf-8 -*-
"""Fig024 Neyman-Pearson 准则示意：两假设 PDF、门限与虚警/漏检/检测区域。

用法: py -3.14 make_fig024.py
输出: Documents/figures/Fig024_NP准则示意.png

设计要点（对应原书第二卷第 3 章 §3.3，例 3.1，图 3.1~3.3，PDF 516~520 / 书内 501~505）:
  入门例子：单次观测 x[0] ~ N(0,1)（H0）或 N(1,1)（H1）。
  (a) 门限 γ'=1/2（中点）：H0 右尾 = 虚警 P_FA≈0.31，H1 左尾 = 漏检 P_M≈0.31，
      P_D=1-P_M≈0.69，两类错误大致折中。
  (b) 门限 γ'=3（NP 固定 P_FA=10^-3）：H0 右尾 P_FA=Q(3)≈10^-3，但检测概率
      P_D=Q(2)≈0.023 暴跌。这就是"固定虚警 → 检测率最大化"背后的此消彼长。
  结论：门限把每条 PDF 切成两类错误；门限一动，P_FA 与 P_D 反向变化。

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

C_H0 = "#2B6CB0"     # 只有噪声曲线
C_H1 = "#C53030"     # 信号+噪声曲线
C_FA = "#E53E3E"     # 虚警填充
C_MISS = "#4299E1"   # 漏检填充
C_TH = "#4A5568"     # 门限

fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.6))
ax_a, ax_b = axes

# ---------- 上 (a)：门限 1/2（中点） ----------
xa = np.linspace(-4.0, 5.0, 900)
p0a = norm.pdf(xa, 0.0, 1.0)
p1a = norm.pdf(xa, 1.0, 1.0)
gam_a = 0.5

ax_a.plot(xa, p0a, lw=2.2, color=C_H0, label="只有噪声（$H_0$：$N(0,1)$）")
ax_a.plot(xa, p1a, lw=2.2, color=C_H1, label="信号+噪声（$H_1$：$N(1,1)$）")
ax_a.axvline(gam_a, color=C_TH, ls="--", lw=1.5)

# 虚警：H0 在门限右侧
xa_r = xa[xa >= gam_a]
ax_a.fill_between(xa_r, 0.0, p0a[xa >= gam_a], color=C_FA, alpha=0.30, lw=0)
# 漏检：H1 在门限左侧
xa_l = xa[xa <= gam_a]
ax_a.fill_between(xa_l, 0.0, p1a[xa <= gam_a], color=C_MISS, alpha=0.30, lw=0)

ax_a.set_xlim(-4.0, 5.0)
ax_a.set_ylim(0.0, 0.46)
ax_a.set_ylabel("概率密度")
ax_a.set_title("(a) 门限 $\\gamma'=1/2$（中点）：$P_{FA}\\approx0.31$，$P_M\\approx0.31$，$P_D=1-P_M\\approx0.69$",
               fontsize=11.5, pad=9)

ax_a.text(gam_a, 0.445, "门限 $\\gamma'=1/2$", ha="center", va="top",
          fontsize=9.5, color=C_TH)
ax_a.text(0.95, 0.035, "$P_{FA}$（虚警）", ha="center", va="center",
          fontsize=9.5, color=C_FA)
ax_a.text(-0.75, 0.05, "$P_M$（漏检）", ha="center", va="center",
          fontsize=9.5, color=C_MISS)
ax_a.text(1.95, 0.155, "$P_D=1-P_M$\n（检测，未填色）", ha="center", va="center",
          fontsize=9.5, color=C_H1)
ax_a.legend(loc="upper left", fontsize=10, framealpha=0.95)

# ---------- 下 (b)：门限 3（NP 固定 P_FA=10^-3） ----------
xb = np.linspace(2.6, 5.6, 900)
p0b = norm.pdf(xb, 0.0, 1.0)
p1b = norm.pdf(xb, 1.0, 1.0)
gam_b = 3.0

ax_b.plot(xb, p0b, lw=2.2, color=C_H0, label="只有噪声（$H_0$）")
ax_b.plot(xb, p1b, lw=2.2, color=C_H1, label="信号+噪声（$H_1$）")
ax_b.axvline(gam_b, color=C_TH, ls="--", lw=1.5)

xb_r = xb[xb >= gam_b]
ax_b.fill_between(xb_r, 0.0, p0b[xb >= gam_b], color=C_FA, alpha=0.45, lw=0)
ax_b.fill_between(xb_r, 0.0, p1b[xb >= gam_b], color=C_H1, alpha=0.28, lw=0)

ax_b.set_xlim(2.6, 5.6)
ax_b.set_ylim(0.0, 0.105)
ax_b.set_xlabel("观测值 $x[0]$")
ax_b.set_ylabel("概率密度")
ax_b.set_title("(b) 门限 $\\gamma'=3$（NP 固定 $P_{FA}=10^{-3}$）：$P_{FA}=Q(3)\\approx10^{-3}$，但 $P_D=Q(2)\\approx0.023$ 暴跌",
               fontsize=11.5, pad=9)

ax_b.text(gam_b, 0.101, "门限 $\\gamma'=3$", ha="center", va="top",
          fontsize=9.5, color=C_TH)
ax_b.text(3.45, 0.011, "$P_{FA}$（虚警）", ha="center", va="center",
          fontsize=9.5, color=C_FA)
ax_b.text(3.45, 0.072, "$P_D$（检测）", ha="center", va="center",
          fontsize=9.5, color=C_H1)
ax_b.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

fig.tight_layout(rect=[0, 0, 1, 0.995])
check_figure(fig)
out = os.path.join(FIG_DIR, "Fig024_NP准则示意.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
