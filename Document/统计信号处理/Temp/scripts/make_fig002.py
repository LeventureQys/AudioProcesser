# -*- coding: utf-8 -*-
"""Fig002 高斯白噪声中直流电平 A 的似然函数随 N 变尖的演示。

用法: py -3.14 make_fig002.py
输出: Documents/figures/Fig002_似然函数与N.png
模型: x[n] = A + w[n], w[n] ~ N(0, σ²), σ=1, 真值 A=1。
同一组数据（固定种子）分别取前 N=1 / 10 / 100 个样本，画出
  p(x; A) ∝ exp(-N/(2σ²)·(A - x̄_N)²)
三条曲线各自归一化到峰值 1，展示"数据越多，似然越尖"，宽度 ∝ σ/√N。
图例置于 Axes 右侧外（仍在 Figure 内），避免压线。绘制后经
plotutil.check_figure 碰撞检测通过才保存。
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

rng = np.random.default_rng(20260004)
SIGMA = 1.0
A_TRUE = 1.0

# 一次采样 100 点，N=1/10/100 用其前缀（同一数据，嵌套可比）
x = A_TRUE + SIGMA * rng.standard_normal(100)
N_list = [1, 10, 100]
colors = {"N = 1": "#C53030", "N = 10": "#2B6CB0", "N = 100": "#2F855A"}

A = np.linspace(0.0, 2.0, 801)
fig, ax = plt.subplots(figsize=(10.2, 5.0))
# 右侧留出约 28% 宽度给图例（图例整体保持在 Figure 内）
fig.subplots_adjust(left=0.08, right=0.72, top=0.92, bottom=0.12)

for N in N_list:
    xbar = np.mean(x[:N])
    lik = np.exp(-N / (2 * SIGMA**2) * (A - xbar) ** 2)
    lik /= lik.max()
    ax.plot(A, lik, lw=2.0, color=colors[f"N = {N}"],
            label=f"N = {N}，宽度 σ/√N ≈ {SIGMA / np.sqrt(N):.2f}")

ax.axvline(A_TRUE, color="#4A5568", ls="--", lw=1.2)
ax.set_xlabel("待估参数 A 的取值")
ax.set_ylabel("似然（各自归一化到峰值 1）")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.06, 1.12)

# 真值标注：贴虚线顶端，带白底框避免压曲线
ax.annotate("真值 A = 1", xy=(A_TRUE, 1.02), xytext=(0.62, 1.05),
            ha="left", va="center", fontsize=10, color="#4A5568",
            arrowprops=dict(arrowstyle="-", color="#4A5568", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="none", alpha=0.9))

# 结论框：右上空白区，白底
ax.text(1.30, 0.94, "N 越大，似然越尖\n峰 = 样本均值，且\n越来越贴近真值",
        fontsize=9.5, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#4A5568", lw=0.8, alpha=0.9))

# 图例放到 Axes 右侧外，仍位于 Figure 内
ax.legend(loc="upper left", bbox_to_anchor=(1.005, 0.99), fontsize=10,
          framealpha=0.95, title="数据长度")

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig002_似然函数与N.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("样本均值:", {N: round(float(np.mean(x[:N])), 4) for N in N_list})
