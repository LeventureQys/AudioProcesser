# -*- coding: utf-8 -*-
"""Fig015 MMSE 与 MAP 对比：同一后验的条件均值与峰（对应原书第 11 章 §11.3~11.5）。

用法: py -3.14 make_fig015.py
输出: Documents/figures/Fig015_MMSE与MAP对比.png

设计要点（对应原书图 11.2：不同代价函数的估计量）:
  (a) 左图: 一个偏斜的后验 PDF（对数正态，μ=0、σ=0.8）。三种代价函数给出三个
       不同估计量——众数=MAP（成功-失败代价）、中值（绝对误差代价）、均值=MMSE
       （二次型代价）。对数正态闭式值：众数 e^{-σ²}=0.527、中值 e^μ=1、
       均值 e^{μ+σ²/2}=1.377。三者位置不同。
  (b) 右图: 高斯后验 N(1, 0.04)——由对称性，均值=中值=众数，三者重合于 1，
       即 MMSE=MAP。这就是"高斯共轭下 MMSE=MAP"的几何原因。

自建数值（与原书无冲突）：对数正态 μ=0、σ=0.8；高斯 N(1, 0.04)。
绘制后经 plotutil.check_figure 程序化碰撞检测通过才保存。
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

C_MMSE = "#2B6CB0"   # 均值 = MMSE
C_MED = "#DD6B20"    # 中值
C_MAP = "#C53030"    # 众数 = MAP
C_POST = "#C53030"   # 右图重合点
EC = "#4A5568"


def lognormal_pdf(x, mu, sigma):
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(x > 0, np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))
                     / (x * sigma * np.sqrt(2 * np.pi)), 0.0)
    return p


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.0))
fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.12, wspace=0.24)

# ================= 左图 (a)：偏斜后验——三者不同 =================
mu, sigma = 0.0, 0.8
mode = np.exp(mu - sigma ** 2)          # 0.527
median = np.exp(mu)                     # 1.0
mean = np.exp(mu + sigma ** 2 / 2.0)    # 1.377

xs = np.linspace(0.0, 4.2, 800)
p = lognormal_pdf(xs, mu, sigma)

ax_a.plot(xs, p, color="#1A202C", lw=2.0, label="后验 $p(\\theta|\\mathbf{x})$")
ax_a.fill_between(xs, p, 0, color="#1A202C", alpha=0.06)

# 三条竖线 + 顶部标记
marks = [(mode, C_MAP, "众数 = MAP\n$e^{-\\sigma^2}{=}0.53$"),
         (median, C_MED, "中值\n$e^{\\mu}{=}1$"),
         (mean, C_MMSE, "均值 = MMSE\n$e^{\\mu+\\sigma^2/2}{=}1.38$")]
for xc, c, lab in marks:
    ax_a.plot([xc, xc], [0, lognormal_pdf(xc, mu, sigma)], color=c, lw=1.6, ls=":", alpha=0.9)
    ax_a.plot([xc], [lognormal_pdf(xc, mu, sigma)], marker="o", ms=6, color=c, mec="none", zorder=5)

ax_a.set_xlim(0.0, 4.2)
ax_a.set_ylim(0.0, 0.80)
ax_a.set_xlabel("参数 $\\theta$", fontsize=10)
ax_a.set_ylabel("后验 PDF $p(\\theta|\\mathbf{x})$", fontsize=10)
ax_a.tick_params(labelsize=9)

# 顶部标签（水平错开 + 垂直分层，避免碰撞）
ax_a.text(mode, 0.76, "众数 = MAP\n$e^{-\\sigma^2}{=}0.53$", ha="center", va="top",
          fontsize=9, color=C_MAP, fontweight="bold", linespacing=1.4)
ax_a.text(median + 0.06, 0.63, "中值\n$e^{\\mu}{=}1$", ha="center", va="top",
          fontsize=9, color=C_MED, linespacing=1.4)
ax_a.text(mean + 0.14, 0.50, "均值 = MMSE\n$e^{\\mu+\\sigma^2/2}{=}1.38$", ha="center", va="top",
          fontsize=9, color=C_MMSE, fontweight="bold", linespacing=1.4)

# 右下角：三种代价函数对应
ax_a.text(0.97, 0.97,
          "代价函数 → 估计量：\n"
          "$C(\\varepsilon)=\\varepsilon^2$ → 均值（MMSE）\n"
          "$C(\\varepsilon)=|\\varepsilon|$ → 中值\n"
          "成功-失败 → 众数（MAP）",
          ha="right", va="top", transform=ax_a.transAxes,
          fontsize=9, color="#1A202C", linespacing=1.6,
          bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7FAFC", edgecolor=EC, lw=0.9))

ax_a.set_title("(a) 偏斜后验：均值 ≠ 中值 ≠ 众数", fontsize=11)

# ================= 右图 (b)：高斯后验——三者重合 =================
mu2, s2 = 1.0, 0.2
xs2 = np.linspace(0.2, 1.8, 600)
p2 = np.exp(-(xs2 - mu2) ** 2 / (2 * s2 ** 2)) / (s2 * np.sqrt(2 * np.pi))

ax_b.plot(xs2, p2, color="#1A202C", lw=2.0)
ax_b.fill_between(xs2, p2, 0, color="#1A202C", alpha=0.06)
ax_b.plot([mu2, mu2], [0, p2.max()], color=C_POST, lw=1.6, ls=":", alpha=0.9)
ax_b.plot([mu2], [p2.max()], marker="o", ms=7, color=C_POST, mec="none", zorder=5)

ax_b.set_xlim(0.2, 1.8)
ax_b.set_ylim(0.0, 2.3)
ax_b.set_xlabel("参数 $\\theta$", fontsize=10)
ax_b.set_ylabel("后验 PDF $p(\\theta|\\mathbf{x})$", fontsize=10)
ax_b.tick_params(labelsize=9)

ax_b.text(mu2, 2.10, "均值 = 中值 = 众数 = 1\n（高斯后验对称）",
          ha="center", va="bottom", fontsize=9.5, color=C_POST,
          fontweight="bold", linespacing=1.5)
ax_b.text(mu2, 1.55, "MMSE = MAP", ha="center", va="center", fontsize=9,
          color=C_POST, linespacing=1.4)

ax_b.set_title("(b) 高斯后验：均值 = 中值 = 众数（MMSE = MAP）", fontsize=11)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig015_MMSE与MAP对比.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("mode =", round(float(mode), 3), " median =", round(float(median), 3),
      " mean =", round(float(mean), 3))
