# -*- coding: utf-8 -*-
"""Fig013 矩方法示意：样本矩匹配理论矩（对应原书第 9 章 §9.3）。

用法: py -3.14 make_fig013.py
输出: Documents/figures/Fig013_矩方法示意.png

设计要点（对应原书第 9 章）:
  (a) 左图: 矩方法四步流程——数据 x → 样本矩 μ̂_k=(1/N)Σx^k[n] → 令样本矩
       等于理论矩 μ_k(θ) → 解出 θ̂=h⁻¹(μ̂_k)。不需要任何概率似然，只需矩方程。
  (b) 右图: 高斯混合 PDF 例（§9.3 例子的可视版）——理论矩 μ₂(ε)=(1−ε)σ₁²+εσ₂²
       是 ε 的直线（σ₁²=1、σ₂²=4 → μ₂=1+3ε）；样本矩 μ̂₂=(1/N)Σx²[n] 是与 ε
       无关的水平线；两者交点即矩估计 ε̂。自建数据（真值 ε=0.3，N=2000，种子
       20260816）。

箭头用 FancyArrowPatch（patch 而非文本）。绘制后经 plotutil.check_figure
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
C_DATA = "#2B6CB0"
C_MOM = "#DD6B20"
C_EQ = "#C53030"
C_THETA = "#2F855A"

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.4))
fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.10, wspace=0.24)


def box(ax, x0, y0, x1, y1, text, fc, fs=10, color="#1A202C"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.012",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=1.3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color=color, linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, color=EC, lw=1.7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes,
                                 arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=13, shrinkA=0, shrinkB=0))


# ================= 左图 (a)：矩方法四步流程 =================
ax_a.set_xlim(0, 1.0)
ax_a.set_ylim(0, 1.0)
ax_a.axis("off")

box(ax_a, 0.05, 0.80, 0.40, 0.96, "数据 $x[n]$（$n=0..N{-}1$）", C_DATA, fs=10.5)
box(ax_a, 0.05, 0.55, 0.40, 0.71,
    "样本矩\n$\\hat{\\mu}_k=\\frac{1}{N}\\sum_{n=0}^{N-1}x^k[n]$",
    C_MOM, fs=9.5)
box(ax_a, 0.05, 0.30, 0.40, 0.46,
    "令样本矩 = 理论矩\n$\\hat{\\mu}_k=\\mu_k(\\theta)$", C_EQ, fs=9.5)
box(ax_a, 0.05, 0.05, 0.40, 0.21,
    "解出估计量\n$\\hat{\\theta}=h^{-1}(\\hat{\\mu}_k)$", C_THETA, fs=10)

arrow(ax_a, 0.225, 0.80, 0.225, 0.71)
arrow(ax_a, 0.225, 0.55, 0.225, 0.46)
arrow(ax_a, 0.225, 0.30, 0.225, 0.21)

ax_a.text(0.60, 0.82, "不需要似然函数，\n只需矩方程",
          ha="center", va="center", transform=ax_a.transAxes,
          fontsize=9.5, color=EC, linespacing=1.5)
ax_a.text(0.60, 0.55, "理论矩 $\\mu_k(\\theta)=E[x^k]$：\n参数的函数（已知公式）",
          ha="center", va="center", transform=ax_a.transAxes,
          fontsize=9.5, color=C_MOM, linespacing=1.5)
ax_a.text(0.60, 0.28, "代价：通常达不到 CRLB\n（只是近似一致、近似高斯）",
          ha="center", va="center", transform=ax_a.transAxes,
          fontsize=9.5, color=C_EQ, linespacing=1.5)

ax_a.set_title("(a) 矩方法：用样本矩代替理论矩，解方程得估计", fontsize=11)


# ================= 右图 (b)：高斯混合例——交点即估计 =================
s1sq, s2sq = 1.0, 4.0
eps_true = 0.3
rng = np.random.default_rng(20260816)
N = 2000
mix = rng.uniform(0, 1, N) < eps_true
x = np.where(mix, rng.standard_normal(N) * np.sqrt(s2sq),
             rng.standard_normal(N) * np.sqrt(s1sq))
mu2_hat = np.mean(x ** 2)
eps_hat = (mu2_hat - s1sq) / (s2sq - s1sq)

eps = np.linspace(0, 1, 200)
mu2_theory = (1 - eps) * s1sq + eps * s2sq

ax_b.plot(eps, mu2_theory, color="#DD6B20", lw=2.0)
ax_b.axhline(mu2_hat, color="#2B6CB0", lw=1.8)
ax_b.plot([eps_hat], [mu2_hat], marker="o", ms=7, color="#C53030", mec="none", zorder=5)
ax_b.plot([eps_hat, eps_hat], [0, mu2_hat], color="#C53030", lw=1.2, ls=":")
ax_b.set_xlabel("混合参数 $\\varepsilon$", fontsize=10)
ax_b.set_ylabel("二阶矩 $\\mu_2$", fontsize=10)
ax_b.set_xlim(0, 1.0)
ax_b.set_ylim(0.5, 4.6)
ax_b.tick_params(labelsize=9)

ax_b.text(0.50, 3.85, "理论矩 $\\mu_2(\\varepsilon)=1+3\\varepsilon$",
          ha="left", va="center", fontsize=9.5, color="#DD6B20")
ax_b.text(0.045, mu2_hat + 0.16, "样本矩 $\\hat{\\mu}_2$（与 $\\varepsilon$ 无关）",
          ha="left", va="center", fontsize=9.5, color="#2B6CB0")
ax_b.text(eps_hat + 0.04, mu2_hat + 0.22, "$\\hat{\\varepsilon}$（交点）",
          ha="left", va="center", fontsize=9.5, color="#C53030")

ax_b.set_title("(b) 高斯混合例：$\\sigma_1^2{=}1,\\ \\sigma_2^2{=}4$，真值 $\\varepsilon{=}0.3$",
               fontsize=11)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig013_矩方法示意.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("mu2_hat =", round(float(mu2_hat), 3), " eps_hat =", round(float(eps_hat), 3))
