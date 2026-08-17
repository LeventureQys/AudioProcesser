# -*- coding: utf-8 -*-
"""Fig014 高斯先验×似然→后验（高斯共轭）：对应原书第 10 章 §10.4 例 10.1。

用法: py -3.14 make_fig014.py
输出: Documents/figures/Fig014_高斯先验与后验.png

设计要点（对应原书例 10.1：WGN 中 DC 电平、高斯先验 A~N(μ_A,σ_A²)）:
  (a) 左图: 先验 p(A)（宽）、似然 ∝p(x|A)（峰在样本均值 x̄）、后验 p(A|x)（最窄）
       三条曲线各自归一化到峰值 1。后验 = 先验×似然（再归一化），峰位落在先验
       均值 μ_A=0 与数据均值 x̄=1 之间，且后验更窄——说明"数据让不确定性减小"。
       取 σ_A²=1/16、σ²=1、N=16，故先验精度 1/σ_A²=16 与数据精度 N/σ²=16 相等，
       后验均值恰为 0.5（精度加权平均），后验方差 1/(16+16)=1/32。
  (b) 右图: 加权因子 α=σ_A²/(σ_A²+σ²/N)=N/(N+16) 随 N 上升（蓝，左轴），
       后验标准差 σ_{A|x}=1/√(1/σ_A²+N/σ²) 随 N 下降（红，右轴）。
       N 小则 α 小（信任先验）、N 大则 α→1（数据"淹没"先验）。

自建数值（与原书无冲突）：μ_A=0、σ_A²=1/16、σ²=1、样本均值 x̄=1（示意值）。
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

C_PRIOR = "#2F855A"   # 先验
C_LIKE = "#2B6CB0"    # 似然
C_POST = "#C53030"    # 后验
EC = "#4A5568"

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.2))
fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.11, wspace=0.22)


def gauss_pdf(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


# ================= 左图 (a)：三条归一化 PDF =================
muA = 0.0
sA2 = 1.0 / 16.0       # 先验方差
sig2 = 1.0
N = 16
xbar = 1.0
post_var = 1.0 / (1.0 / sA2 + N / sig2)   # = 1/32
alpha = sA2 / (sA2 + sig2 / N)            # = 0.5
post_mean = alpha * xbar + (1 - alpha) * muA  # = 0.5

xs = np.linspace(-0.9, 1.9, 600)
prior = gauss_pdf(xs, muA, sA2)
like = gauss_pdf(xs, xbar, sig2 / N)
post = gauss_pdf(xs, post_mean, post_var)

# 各自归一化到峰值 1（便于对比形状与峰位；乘积关系在正文图注说明）
prior /= prior.max()
like /= like.max()
post /= post.max()

ax_a.plot(xs, prior, color=C_PRIOR, lw=2.2, label="先验 $p(A)=\\mathcal{N}(0,\\frac{1}{16})$")
ax_a.plot(xs, like, color=C_LIKE, lw=2.0, ls="--", label="似然 $\\propto p(\\mathbf{x}|A)$（峰在 $\\bar{x}=1$）")
ax_a.plot(xs, post, color=C_POST, lw=2.6, label="后验 $p(A|\\mathbf{x})=\\mathcal{N}(0.5,\\frac{1}{32})$")

# 三条峰位竖线
for xc, c in [(muA, C_PRIOR), (xbar, C_LIKE), (post_mean, C_POST)]:
    ax_a.plot([xc, xc], [0, 1.0], color=c, lw=1.2, ls=":", alpha=0.85)

ax_a.set_xlim(-1.6, 1.9)
ax_a.set_ylim(0.0, 1.20)
ax_a.set_xlabel("参数 $A$", fontsize=10)
ax_a.set_ylabel("PDF（各自归一化到峰值 1）", fontsize=10)
ax_a.tick_params(labelsize=9)

# 后验峰位标注（关键结论：后验均值落在 0 与 1 之间）
ax_a.text(post_mean, 1.07, "后验均值 $0.5$", ha="center", va="bottom",
          fontsize=9.5, color=C_POST, fontweight="bold")

# 右下角：精度加权解释（空白区）
ax_a.text(0.985, 0.03,
          "后验均值 = 精度加权平均\n"
          "$=\\frac{16\\cdot0+16\\cdot1}{16+16}=0.5$\n"
          "后验方差 = $\\frac{1}{16+16}=\\frac{1}{32}$\n"
          "（先验信息 $1/\\sigma_A^2$ + 数据信息 $N/\\sigma^2$）",
          ha="right", va="bottom", transform=ax_a.transAxes,
          fontsize=9, color="#1A202C", linespacing=1.6,
          bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7FAFC", edgecolor=EC, lw=0.9))

ax_a.legend(loc="upper left", fontsize=8, framealpha=0.95)
ax_a.set_title("(a) 高斯先验 × 高斯似然 → 高斯后验（峰位折衷、更窄）", fontsize=11)

# ================= 右图 (b)：加权因子 α 与后验标准差随 N =================
Ns = np.logspace(0, np.log10(128), 300)
alpha_N = (Ns / sig2) / (Ns / sig2 + 1.0 / sA2)   # = N/(N+16)
post_std_N = 1.0 / np.sqrt(1.0 / sA2 + Ns / sig2)  # = 1/√(N+16)

ax_b.plot(Ns, alpha_N, color=C_LIKE, lw=2.2, label="加权因子 $\\alpha$（左轴）")
ax_b.axhline(1.0, color=C_LIKE, lw=0.9, ls=":", alpha=0.6)
ax_b.set_xscale("log")
ax_b.set_xlabel("数据长度 $N$（对数坐标）", fontsize=10)
ax_b.set_ylabel("加权因子 $\\alpha=\\dfrac{N}{N+16}$", fontsize=10, color=C_LIKE)
ax_b.set_xlim(1, 128)
ax_b.set_ylim(-0.03, 1.08)
ax_b.tick_params(labelsize=9, colors=C_LIKE)

ax_b2 = ax_b.twinx()
ax_b2.plot(Ns, post_std_N, color=C_POST, lw=2.0, ls="--",
           label="后验标准差 $\\sigma_{A|x}$（右轴）")
ax_b2.set_ylabel("后验标准差 $\\sigma_{A|x}$", fontsize=10, color=C_POST)
ax_b2.set_ylim(0.0, 0.26)
ax_b2.tick_params(labelsize=9, colors=C_POST)

# 关键点：N=16 处 α=0.5
ax_b.plot([16], [0.5], marker="o", ms=6, color="#C53030", mec="none", zorder=5)
ax_b.plot([16, 16], [0, 0.5], color="#C53030", lw=1.0, ls=":", alpha=0.7)
ax_b.text(19, 0.53, "$N{=}16:\\ \\alpha{=}0.5$", ha="left", va="bottom",
          fontsize=9, color="#C53030")

ax_b.text(1.6, 0.86, "数据越多，$\\alpha\\to1$：\n数据“淹没”先验", ha="left", va="center",
          fontsize=9, color=C_LIKE, linespacing=1.5)
ax_b.text(30, 0.18, "后验更窄：\n$\\sigma_{A|x}=\\frac{1}{\\sqrt{N+16}}$",
          ha="left", va="center", fontsize=9, color=C_POST, linespacing=1.5)

ax_b.set_title("(b) 加权因子与后验宽度随 $N$ 的变化", fontsize=11)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig014_高斯先验与后验.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("post_mean =", round(float(post_mean), 4), " post_var =", round(float(post_var), 4),
      " alpha(N=16) =", round(float(alpha), 4))
