# -*- coding: utf-8 -*-
"""Fig033 模型变化检测：分段似然 + 动态规划（对应原书第二卷第 12 章，PDF 812~838 / 书内 797~823）。

用法: py -3.14 make_fig033.py
输出: Documents/figures/Fig033_模型变化检测.png

设计要点:
  (a) 变点信号与分段均值：DC 电平在 n0=50 从 A1=1 跳到 A2=4，WGN σ²=1（自建，种子 20260821）。
      跳变点把数据分成两段，各段用样本均值 Â1、Â2 拟合（分段似然的最小二乘口径，对应式 12.12/12.13）。
  (b) GLRT 统计量随候选跳变点 n0：2ln L_G(x) = (Â1-Â2)²/(σ²(1/n0 + 1/(N-n0)))（对应式 12.11），
      在真实 n0 附近出现峰值——"扫描所有 n0 取最大"就是未知跳变时间的 GLRT。
  (c) 动态规划求最短路径（对应原书图 12.5/12.6 的思路，边权为示意值）：
      从 A 到 D，DP 在每个中间节点只保留最短入边，剪掉冗余边（虚线），
      红粗线为最短路径 A→G→F→D（总距离 4）。

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
from matplotlib.patches import FancyArrowPatch

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

rng = np.random.default_rng(20260821)

EC = "#4A5568"
AC = "#2B6CB0"
C_SIG = "#2B6CB0"     # 信号
C_MEAN = "#C53030"    # 分段均值
C_NOISE = "#A0AEC0"   # 噪声
C_PEAK = "#C53030"    # 峰
C_SHORT = "#C53030"   # 最短路径（红）

fig = plt.figure(figsize=(12.8, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0],
                      width_ratios=[1.0, 1.0],
                      left=0.07, right=0.975, top=0.90, bottom=0.08,
                      hspace=0.42, wspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])   # (a) 信号 + 分段均值
ax2 = fig.add_subplot(gs[0, 1])   # (b) 统计量
ax3 = fig.add_subplot(gs[1, :])   # (c) DP 图

# ================= (a) 信号 + 分段均值 =================
N = 100
n0 = 50
A1, A2 = 1.0, 4.0
sigma = 1.0
n = np.arange(N)
x = np.where(n < n0, A1, A2) + sigma * rng.standard_normal(N)

ax1.plot(n, x, lw=0.9, color=C_NOISE, alpha=0.75, label="观测 $x[n]$（含噪声）")
true = np.where(n < n0, A1, A2)
ax1.step(np.append(n, N), np.append(true, A2), where="post",
         lw=2.0, color=C_SIG, label="真实电平（$A_1=1 \\to A_2=4$）")
a1 = np.mean(x[:n0])
a2 = np.mean(x[n0:])
ax1.axhline(a1, xmin=0, xmax=n0 / N, color=C_MEAN, ls="--", lw=1.6)
ax1.axhline(a2, xmin=n0 / N, xmax=1.0, color=C_MEAN, ls="--", lw=1.6)
ax1.axvline(n0, color=EC, ls=":", lw=1.2)

ax1.text(6, a1 - 0.55, "$\\hat{A}_1$（前段均值）", fontsize=9.5, color=C_MEAN)
ax1.text(n0 + 6, a2 + 0.35, "$\\hat{A}_2$（后段均值）", fontsize=9.5, color=C_MEAN)
ax1.text(n0 + 1, 6.9, "跳变点 $n_0=50$", fontsize=9.5, color=EC)

ax1.set_xlim(-2, N + 2)
ax1.set_ylim(-2.5, 8.0)
ax1.set_xlabel("样本 $n$")
ax1.set_ylabel("$x[n]$")
ax1.set_title("(a) 变点信号：跳变点把数据分成两段，各段用样本均值拟合", fontsize=10.5, pad=8)
ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

# ================= (b) 统计量 vs 候选跳变点 =================
n0_cand = np.arange(2, N - 1)
T = np.zeros_like(n0_cand, dtype=float)
for i, m in enumerate(n0_cand):
    a1 = np.mean(x[:m])
    a2 = np.mean(x[m:])
    T[i] = (a1 - a2) ** 2 / (sigma ** 2 * (1.0 / m + 1.0 / (N - m)))

ax2.plot(n0_cand, T, lw=1.8, color=AC)
ax2.axvline(n0, color=C_PEAK, ls=":", lw=1.4)
imax = n0_cand[np.argmax(T)]
ax2.scatter([imax], [T.max()], color=C_PEAK, s=36, zorder=5)
ax2.text(imax + 1.5, T.max() - 3.5, f"峰值在 $n_0={imax}$", fontsize=9.5, color=C_PEAK)
ax2.set_xlim(0, N)
ax2.set_ylim(0, T.max() * 1.15)
ax2.set_xlabel("候选跳变点 $n_0$")
ax2.set_ylabel("$2\\ln L_G(\\mathbf{x})$")
ax2.set_title("(b) GLRT 统计量随 $n_0$ 扫描：峰值逼近真实跳变点（式 12.11）", fontsize=10.5, pad=8)

# ================= (c) DP 最短路径 =================
ax3.set_xlim(0, 1.0)
ax3.set_ylim(0, 1.0)
ax3.axis("off")
ax3.set_title("(c) 动态规划：每个中间节点只保留最短入边（红粗=最短路径 A→G→F→D，虚线=被剪边）",
              fontsize=10.5, pad=8)

nodes = {"A": (0.04, 0.52), "B": (0.33, 0.78), "E": (0.33, 0.52), "G": (0.33, 0.26),
         "C": (0.62, 0.78), "F": (0.62, 0.52), "H": (0.62, 0.26), "D": (0.92, 0.52)}

edges = [
    ("A", "B", 3), ("A", "E", 1), ("A", "G", 2),
    ("B", "C", 4), ("B", "F", 5),
    ("E", "C", 2), ("E", "F", 3), ("E", "H", 5),
    ("G", "F", 1), ("G", "H", 3),
    ("C", "D", 3), ("F", "D", 1), ("H", "D", 4),
]

# 被 DP 剪掉的边（到达 C/F/H 的非最短入边）
pruned = {("B", "C"), ("B", "F"), ("E", "F"), ("E", "H")}
# 最短路径边
shortest = {("A", "G"), ("G", "F"), ("F", "D")}

# 每条边的权值标签位置（手工避开节点与其它标签）
label_pos = {
    ("A", "B"): (0.155, 0.665), ("A", "E"): (0.155, 0.475), ("A", "G"): (0.155, 0.345),
    ("B", "C"): (0.475, 0.815), ("B", "F"): (0.435, 0.665),
    ("E", "C"): (0.515, 0.665), ("E", "F"): (0.475, 0.475),
    ("E", "H"): (0.435, 0.355), ("G", "F"): (0.515, 0.355),
    ("G", "H"): (0.475, 0.225),
    ("C", "D"): (0.79, 0.665), ("F", "D"): (0.79, 0.475), ("H", "D"): (0.79, 0.355),
}

for a, b, w in edges:
    x0, y0 = nodes[a]
    x1, y1 = nodes[b]
    if (a, b) in shortest:
        color, lw, style = C_SHORT, 2.6, "-"
    elif (a, b) in pruned:
        color, lw, style = "#A0AEC0", 1.3, "--"
    else:
        color, lw, style = EC, 1.3, "-"
    ax3.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax3.transAxes,
                                  arrowstyle="-|>", color=color, lw=lw,
                                  mutation_scale=9, shrinkA=7, shrinkB=7,
                                  linestyle=style))
    wx, wy = label_pos[(a, b)]
    ax3.text(wx, wy, f"{w}", fontsize=8.5, color="#1A202C",
             ha="center", va="center", transform=ax3.transAxes)

for name, (x, y) in nodes.items():
    ax3.add_patch(plt.Circle((x, y), 0.032, transform=ax3.transAxes,
                              facecolor="#E8F0FA", edgecolor=EC, linewidth=1.2, zorder=3))
    ax3.text(x, y, name, fontsize=10.5, color="#1A202C", ha="center", va="center",
             transform=ax3.transAxes, zorder=4)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig033_模型变化检测.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
