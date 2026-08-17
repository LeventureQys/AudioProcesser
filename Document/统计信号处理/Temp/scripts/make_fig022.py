# -*- coding: utf-8 -*-
"""Fig022 重要分布速查：形状 + 关系树。

用法: py -3.14 make_fig022.py
输出: Documents/figures/Fig022_重要分布速查.png

设计要点（对应原书第二卷第 2 章，PDF 485~515 / 书内 470~500）:
  (a) 常用分布形状：高斯、中心 χ²(ν=2,6)、瑞利、莱斯。看三点：
      ① χ² 右偏、随 ν 增大趋近高斯；② χ²_2 即指数（单调下降）；
      ③ 瑞利/莱斯是"包络"，定义在 x≥0，莱斯比瑞利多一个非零均值使其右移。
  (b) 分布关系树：都从高斯出发——平方和 → χ² / 非中心 χ²；包络 → 瑞利 / 莱斯。

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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.special import gamma, iv

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

fig = plt.figure(figsize=(12.6, 7.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0],
                      left=0.07, right=0.98, top=0.90, bottom=0.09, hspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])


def gauss(x, mu, sd):
    return 1.0 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sd) ** 2)


def chi2_pdf(x, nu):
    return (1.0 / (2.0 ** (nu / 2.0) * gamma(nu / 2.0))) * x ** (nu / 2.0 - 1.0) * np.exp(-x / 2.0)


def rayleigh_pdf(x, s2):
    return (x / s2) * np.exp(-x ** 2 / (2.0 * s2))


def rician_pdf(x, alpha, s2):
    return (x / s2) * np.exp(-(x ** 2 + alpha ** 2) / (2.0 * s2)) * iv(0, alpha * x / s2)


# ================= 上 (a)：形状 =================
xg = np.linspace(-4.0, 12.0, 900)
xp = np.linspace(0.0, 12.0, 900)

ax1.plot(xg, gauss(xg, 0.0, 1.0), lw=2.0, color="#4A5568", label="高斯 $N(0,1)$")
ax1.plot(xp, chi2_pdf(xp, 2.0), lw=2.0, color="#2B6CB0", label="$\\chi^2_2$（=指数）")
ax1.plot(xp, chi2_pdf(xp, 6.0), lw=2.0, color="#6B46C1", label="$\\chi^2_6$（趋近高斯）")
ax1.plot(xp, rayleigh_pdf(xp, 1.0), lw=2.0, color="#C53030", label="瑞利（$\\sigma^2=1$）")
ax1.plot(xp, rician_pdf(xp, 1.0, 1.0), lw=2.0, color="#2F855A", label="莱斯（$\\alpha=1,\\sigma^2=1$）")

ax1.set_xlim(-4.0, 12.0)
ax1.set_ylim(0.0, 0.72)
ax1.set_xlabel("随机变量 $x$")
ax1.set_ylabel("概率密度")
ax1.set_title("(a) 常用分布的形状（自建演示，参数与 §2.2 一致）", fontsize=11)
ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)

# ================= 下 (b)：关系树 =================
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 1.0)
ax2.axis("off")
ax2.set_title("(b) 分布关系：都从高斯出发（§2.2）", fontsize=11, pad=8)

C_BOX = "#F0F4FA"
EC = "#4A5568"
AC = "#2B6CB0"


def tbox(x0, y0, x1, y1, text, fs=10):
    ax2.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                 boxstyle="round,pad=0.008",
                                 transform=ax2.transAxes,
                                 facecolor=C_BOX, edgecolor=EC, linewidth=1.1))
    ax2.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
             transform=ax2.transAxes, fontsize=fs, color="#1A202C",
             linespacing=1.5)


def tarrow(x0, y0, x1, y1):
    ax2.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                  transform=ax2.transAxes,
                                  arrowstyle="-|>", color=AC, lw=1.5,
                                  mutation_scale=11, shrinkA=0, shrinkB=0))


# 根
tbox(0.38, 0.76, 0.62, 0.95, "高斯（正态）\n$N(\\mu,\\sigma^2)$", fs=10.5)
# 两个构造
tbox(0.04, 0.47, 0.46, 0.66, "平方和\n$\\sum_i x_i^2$")
tbox(0.54, 0.47, 0.96, 0.66, "包络\n$\\sqrt{x_1^2+x_2^2}$")
# 四个结果
tbox(0.005, 0.18, 0.24, 0.37, "中心 $\\chi^2_\\nu$\n（$\\nu=2$ 时即指数）")
tbox(0.255, 0.18, 0.49, 0.37, "非中心 $\\chi^2_\\nu(\\lambda)$\n（$\\lambda=0$ 时即中心）")
tbox(0.505, 0.18, 0.74, 0.37, "瑞利\n（零均值包络）")
tbox(0.755, 0.18, 0.99, 0.37, "莱斯\n（非零均值，$\\alpha=0$ 退化瑞利）")

# 根 → 两个构造
tarrow(0.47, 0.76, 0.25, 0.66)
tarrow(0.53, 0.76, 0.75, 0.66)
# 平方和 → 两个 χ²
tarrow(0.12, 0.47, 0.12, 0.37)
tarrow(0.37, 0.47, 0.37, 0.37)
# 包络 → 瑞利 / 莱斯
tarrow(0.62, 0.47, 0.62, 0.37)
tarrow(0.87, 0.47, 0.87, 0.37)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig022_重要分布速查.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
