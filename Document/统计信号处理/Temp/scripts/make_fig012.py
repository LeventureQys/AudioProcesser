# -*- coding: utf-8 -*-
"""Fig012 序贯最小二乘更新示意：只修正、不重算（对应原书 §8.7）。

用法: py -3.14 make_fig012.py
输出: Documents/figures/Fig012_序贯最小二乘.png

设计要点（对应原书第 8 章 §8.7，图 8.9 / 8.10 的抽象化）:
  (a) 左图: 序贯更新的"预测—修正"流程——旧估计 θ̂[n-1] 与新数据 x[n] 合成
       预测误差（新息）e[n]=x[n]−hᵀ[n]θ̂[n-1]；再乘以增益 K[n] 修正旧估计，
       得到 θ̂[n]=θ̂[n-1]+K[n]e[n]。不重解最小二乘，只做一次修正。
  (b) 右图: 自建蒙特卡洛（真值 A=10、σ²=1，种子 20260815）——序贯 LSE Â[N]
       随样本数收敛到真值（蓝线），增益 K[N] 单调下降（红线，右轴）。

矢量/箭头用 FancyArrowPatch（patch 而非文本）。绘制后经 plotutil.check_figure
程序化碰撞检测通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

EC = "#4A5568"
C_OLD = "#DD6B20"   # 旧估计
C_NEW = "#2B6CB0"   # 新数据
C_ERR = "#C53030"   # 预测误差
C_K = "#553C9A"     # 增益
C_OUT = "#2F855A"   # 新估计

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.4))
fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.10, wspace=0.24)


def box(ax, x0, y0, x1, y1, text, fc, fs=10, color="#1A202C", bold=False):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.012",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=1.3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color=color,
            linespacing=1.5, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=EC, lw=1.7, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes,
                                 arrowstyle=style, color=color, lw=lw,
                                 mutation_scale=13, shrinkA=0, shrinkB=0,
                                 linestyle=ls))


# ================= 左图 (a)：序贯更新的"预测—修正"流程 =================
ax_a.set_xlim(0, 1.0)
ax_a.set_ylim(0, 1.0)
ax_a.axis("off")

box(ax_a, 0.03, 0.76, 0.28, 0.94, "新数据 $x[n]$", C_NEW, fs=11)
box(ax_a, 0.03, 0.30, 0.28, 0.48, "旧估计\n$\\hat{\\theta}[n{-}1]$", C_OLD, fs=10.5)

box(ax_a, 0.42, 0.72, 0.78, 0.94,
    "预测误差（新息）\n$e[n]=x[n]-h^{T}[n]\\,\\hat{\\theta}[n{-}1]$",
    C_ERR, fs=9.5)
box(ax_a, 0.42, 0.28, 0.66, 0.44, "增益 $K[n]$\n（由方差递推）", C_K, fs=10)
box(ax_a, 0.42, 0.04, 0.80, 0.20,
    "新估计 $\\hat{\\theta}[n]=\\hat{\\theta}[n{-}1]+K[n]\\,e[n]$",
    C_OUT, fs=10, bold=True)

arrow(ax_a, 0.28, 0.85, 0.42, 0.85)   # x[n] → 误差框
arrow(ax_a, 0.28, 0.39, 0.42, 0.80)   # 旧估计 → 误差框
arrow(ax_a, 0.42, 0.36, 0.42, 0.20)   # 增益 → 新估计（垂直，用下方起点）
arrow(ax_a, 0.78, 0.83, 0.80, 0.20)   # 误差框 → 新估计（右侧竖线）

ax_a.text(0.35, 0.885, "（只读）", ha="center", va="center", transform=ax_a.transAxes,
          fontsize=8.5, color=EC)
ax_a.text(0.355, 0.615, "旧估计用于预测", ha="center", va="center", transform=ax_a.transAxes,
          fontsize=8.5, color=EC)
ax_a.text(0.815, 0.55, "$\\times$", ha="center", va="center", transform=ax_a.transAxes,
          fontsize=13, color=C_K)

ax_a.set_title("(a) 序贯更新：旧估计 + 增益 × 预测误差，只修正不重算", fontsize=11)


# ================= 右图 (b)：自建蒙特卡洛收敛 =================
rng = np.random.default_rng(20260815)
N = 100
A_true = 10.0
sigma2 = 1.0
w = rng.standard_normal(N)
xseq = A_true + w

Ahat = np.zeros(N)
Kseq = np.zeros(N)
var = np.zeros(N)
Ahat[0] = xseq[0]
var[0] = sigma2
Kseq[0] = np.nan
for n in range(1, N):
    Kseq[n] = var[n - 1] / (var[n - 1] + sigma2)
    Ahat[n] = Ahat[n - 1] + Kseq[n] * (xseq[n] - Ahat[n - 1])
    var[n] = (1 - Kseq[n]) * var[n - 1]

ax_b.plot(range(N), Ahat, color=C_PROJ if False else "#2B6CB0", lw=1.6,
          label=None)
ax_b.axhline(A_true, color="#C53030", lw=1.4, ls="--")
ax_b.set_xlabel("当前样本 N", fontsize=10)
ax_b.set_ylabel("估计 $\\hat{A}[N]$", fontsize=10, color="#2B6CB0")
ax_b.set_ylim(8.8, 11.2)
ax_b.set_xlim(-2, 102)
ax_b.tick_params(labelsize=9)

ax_b2 = ax_b.twinx()
ax_b2.plot(range(1, N), Kseq[1:], color="#2F855A", lw=1.6)
ax_b2.set_ylabel("增益 $K[N]$", fontsize=10, color="#2F855A")
ax_b2.set_ylim(0.0, 0.55)
ax_b2.tick_params(labelsize=9, colors="#2F855A")

ax_b.text(56, 10.05, "估计 $\\hat{A}[N]$（左轴）", ha="left", va="center",
          fontsize=9, color="#2B6CB0")
ax_b.text(56, 9.55, "真值 $A=10$（虚线）", ha="left", va="center",
          fontsize=9, color="#C53030")
ax_b.text(4, 10.95, "增益 $K[N]$（右轴，单调下降）", ha="left", va="center",
          fontsize=9, color="#2F855A")

ax_b.set_title("(b) 自建蒙特卡洛：序贯 LSE 收敛、增益下降（$\\sigma^2{=}1$）", fontsize=11)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig012_序贯最小二乘.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("final Ahat[N-1] =", round(float(Ahat[-1]), 3), " K[1] =", round(float(Kseq[1]), 4))
