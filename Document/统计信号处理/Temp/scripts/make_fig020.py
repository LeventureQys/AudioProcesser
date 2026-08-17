# -*- coding: utf-8 -*-
"""Fig020 复高斯/实数化对应示意：循环对称 vs 非循环对称（对应原书第 15 章 §15.4）。

用法: py -3.14 make_fig020.py
输出: Documents/figures/Fig020_复高斯与实数化对应.png

设计要点（对应原书 PDF 420~421 / 书内 405~406）:
  复随机变量 z=u+jv 的复高斯 PDF（CN(μ,σ²)）要求实部 u、虚部 v 独立且等方差（各 N(μ/2, σ²/2)），
  等价于"伪方差" E[(z-μ)²]=0 —— 即循环对称（circularly symmetric）。
  (a) 循环对称：var(u)=var(v)，等概率线是圆，可写成 CN；
  (b) 非循环对称：var(u)≠var(v)，等概率线是椭圆，伪方差≠0，不能写成 CN。

自建数值实验（种子 20261515），只用于示意"圆 vs 椭圆"的几何差别，与原书无冲突。
绘制后经 plotutil.check_figure 程序化碰撞检测（重叠/穿框）通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

rng = np.random.default_rng(20261515)
N = 2200

# (a) 循环对称：u、v 独立同方差 σ²/2 = 0.5
u_a = rng.normal(0.0, np.sqrt(0.5), N)
v_a = rng.normal(0.0, np.sqrt(0.5), N)

# (b) 非循环对称：var(u)=0.5, var(v)=2
u_b = rng.normal(0.0, np.sqrt(0.5), N)
v_b = rng.normal(0.0, np.sqrt(2.0), N)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

for ax in (ax1, ax2):
    ax.set_xlim(-4.4, 4.4)
    ax.set_ylim(-4.4, 4.4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xlabel("实部 u")
    ax.set_ylabel("虚部 v")

# ---------- (a) 循环对称 ----------
ax1.scatter(u_a, v_a, s=3, alpha=0.35, color="#2B6CB0", linewidths=0)
ax1.add_patch(Circle((0, 0), 1.0, fill=False, ec="#C53030", lw=1.6, ls="--"))
ax1.add_patch(Circle((0, 0), 2.0, fill=False, ec="#C53030", lw=1.6, ls="--"))
ax1.set_title("(a) 循环对称（复高斯可表示）", fontsize=11.5, pad=6)
ax1.annotate("var(u)=var(v)=σ²/2\ncov(u,v)=0", xy=(0.05, 0.97), xycoords="axes fraction",
             ha="left", va="top", fontsize=9.5, color="#1A202C",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#9AA4B0", alpha=0.92))
ax1.annotate("等概率线是圆\n伪方差 E[(z−μ)²]=0", xy=(0.97, 0.03), xycoords="axes fraction",
             ha="right", va="bottom", fontsize=9.5, color="#C53030",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C53030", alpha=0.92))

# ---------- (b) 非循环对称 ----------
ax2.scatter(u_b, v_b, s=3, alpha=0.35, color="#DD6B20", linewidths=0)
ax2.add_patch(Ellipse((0, 0), 2.0, 4.0, fill=False, ec="#C53030", lw=1.6, ls="--"))
ax2.set_title("(b) 非循环对称（不能写成 CN）", fontsize=11.5, pad=6)
ax2.annotate("var(u)=0.5, var(v)=2\nvar(u)≠var(v)", xy=(0.05, 0.97), xycoords="axes fraction",
             ha="left", va="top", fontsize=9.5, color="#1A202C",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#9AA4B0", alpha=0.92))
ax2.annotate("等概率线是椭圆\n伪方差 E[(z−μ)²]≠0", xy=(0.97, 0.03), xycoords="axes fraction",
             ha="right", va="bottom", fontsize=9.5, color="#C53030",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C53030", alpha=0.92))

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig020_复高斯与实数化对应.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
