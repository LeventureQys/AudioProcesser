# -*- coding: utf-8 -*-
"""Fig004 MLE 数值确定方法示意：网格搜索 vs 牛顿迭代。

用法: py -3.14 make_fig004.py
输出: Documents/figures/Fig004_对数似然数值优化.png
模型: x[n] ~ N(0, σ²)，N=20，参数 λ = σ²。对数似然（差常数、按峰值平移）:
  ℓ(λ) - ℓ(λ̂) = -(N/2)·ln(λ/λ̂) - (S/2)·(1/λ - 1/λ̂),  S = Σx[n]²,  λ̂ = S/N。
(a) 网格搜索: λ ∈ [0.2, 2.2] 步长 0.2 共 11 次评估，最优格点离峰顶受步长限制；
(b) 牛顿迭代: 从 λ0=0.4 出发，切线 → 与横轴交点 → 下一步，数步内逼近 λ̂。
绘制后经 plotutil.check_figure 碰撞检测通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

rng = np.random.default_rng(20260814)
N = 20
x = rng.standard_normal(N)
S = float(np.sum(x**2))
lam_hat = S / N


def ell(lam):
    """ℓ(λ) - ℓ(λ̂)：峰值平移到 0。"""
    return -0.5 * N * np.log(lam / lam_hat) - 0.5 * S * (1.0 / lam - 1.0 / lam_hat)


def d1(lam):
    return -0.5 * N / lam + 0.5 * S / lam**2


def d2(lam):
    return 0.5 * N / lam**2 - S / lam**3


lam = np.linspace(0.16, 2.45, 900)
curve = ell(lam)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))

# ---- (a) 网格搜索 ----
ax1.plot(lam, curve, color="#2B6CB0", lw=1.8, label="对数似然 $\\ell(\\lambda)$")
grid = np.arange(0.2, 2.21, 0.2)
grid_vals = ell(grid)
ax1.plot(grid, grid_vals, "o", color="#C53030", ms=5, zorder=5)
best = grid[np.argmax(grid_vals)]
ax1.plot([best], [ell(best)], "s", color="#2F855A", ms=8, zorder=6)
ax1.set_xlabel("参数 λ = σ²")
ax1.set_ylabel("对数似然（峰值平移到 0）")
ax1.set_title("(a) 网格搜索：通用但慢", fontsize=11)
ax1.set_xlim(0.15, 2.45)
ax1.set_ylim(-26.0, 3.0)
ax1.text(0.24, 0.9, f"11 次评估，最优格点 λ = {best:.1f}\n峰顶在 $\\hat{{\\lambda}}$ ≈ {lam_hat:.2f}\n精度受步长限制",
         fontsize=9.5, ha="left", va="top",
         bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#4A5568", lw=0.8, alpha=0.9))

# ---- (b) 牛顿迭代 ----
ax2.plot(lam, curve, color="#2B6CB0", lw=1.8, label="对数似然 $\\ell(\\lambda)$")
lam0 = 0.4
lams = [lam0]
for _ in range(6):
    lams.append(lams[-1] - d1(lams[-1]) / d2(lams[-1]))
lams = np.array(lams)
ax2.plot(lams, ell(lams), "o", color="#C53030", ms=5, zorder=5)
# 前三条切线（显示"切线 → 与横轴交点 → 下一个迭代点"）
for k in range(3):
    seg = np.linspace(lams[k], lams[k + 1], 60)
    tan = ell(lams[k]) + d1(lams[k]) * (seg - lams[k])
    ax2.plot(seg, tan, color="#718096", lw=1.0, ls=":", zorder=4)
ax2.set_xlabel("参数 λ = σ²")
ax2.set_ylabel("对数似然（峰值平移到 0）")
ax2.set_title("(b) 牛顿迭代：快，但需要导数", fontsize=11)
ax2.set_xlim(0.15, 2.45)
ax2.set_ylim(-26.0, 3.0)
ax2.annotate("λ0 = 0.4", xy=(lams[0], ell(lams[0])), xytext=(0.30, -13.5),
             fontsize=9.5, ha="left", color="#C53030",
             arrowprops=dict(arrowstyle="-", color="#C53030", lw=0.8))
ax2.annotate(f"$\\hat{{\\lambda}}$ ≈ {lam_hat:.2f}", xy=(lam_hat, 0.0), xytext=(1.75, 1.6),
             fontsize=9.5, ha="center", color="#2F855A",
             arrowprops=dict(arrowstyle="-", color="#2F855A", lw=0.8))
ax2.text(1.72, -8.5, f"{len(lams) - 1} 步迭代\n每步只算一次 ℓ′ 和 ℓ″",
         fontsize=9.5, ha="left", va="center",
         bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#4A5568", lw=0.8, alpha=0.9))

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig004_对数似然数值优化.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("lam_hat =", round(lam_hat, 4))
print("newton iterates:", np.round(lams, 4))
