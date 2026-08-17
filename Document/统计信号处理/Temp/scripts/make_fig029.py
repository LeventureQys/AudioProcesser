# -*- coding: utf-8 -*-
"""Fig029 未知参数随机信号检测：协方差含未知参数 θ 时的 GLRT 结构（时域 + 频域）。

用法: py -3.14 make_fig029.py
输出: Documents/figures/Fig029_未知参数随机信号检测.png

设计要点（对应原书第二卷第 8 章，式 (8.14)(8.19)，PDF 710~711，书内 695~696）:
  (a) 一般结构（§8.3）：数据 x → 先求协方差参数 θ 的 MLE θ̂（一般无闭式解，需数值优化）
      → 把 θ̂ 代入估计器-相关器 T(x)=xᵀC_s(θ̂)(C_s(θ̂)+σ²I)⁻¹x → 与门限比 → 判 H1。
      这正是第 5 章"估计器-相关器"在"协方差含未知参数"时的推广。
  (b) 大数据记录近似（§8.4，WSS 信号）：数据 x → 周期图 I(f) → 用 θ̂ 的维纳滤波
      H(f;θ̂)=P_ss(f;θ̂)/(P_ss(f;θ̂)+σ²) 加权 → 频域积分 → 与门限比 → 判 H1。
      它把矩阵求逆/行列式换成 FFT 可算的频域积分（回调第一卷 Ch07 渐近似然）。

绘制后经 plotutil.check_figure 程序化碰撞检测（重叠/穿框）通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

C_BOX = "#E8F0FA"     # 通用框
C_EST = "#F6E3CE"     # 估计器框（强调"检测器里长出了估计器"）
C_WIENER = "#EAD8F0"  # 维纳滤波/加权框
C_DEC = "#E2F0E6"     # 判决框
EC = "#4A5568"
AC = "#2B6CB0"

fig = plt.figure(figsize=(13.0, 6.8))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0],
                      left=0.06, right=0.97, top=0.86, bottom=0.12, hspace=0.55)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0])


def box(ax, x0, y0, x1, y1, text, fc, fs=9.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.008",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=1.2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="#1A202C",
            linespacing=1.6)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=AC, lw=1.8,
                                shrinkA=0, shrinkB=0))


# ================= 上排 (a)：一般 GLRT 结构 =================
ax_a.set_xlim(0, 1.0)
ax_a.set_ylim(0, 1.0)
ax_a.axis("off")
ax_a.set_title("(a) 协方差含未知参数 $\\boldsymbol{\\theta}$ 的 GLRT（式 8.14，一般无闭式解）",
               fontsize=11.5, pad=10)

y0, y1 = 0.24, 0.78
box(ax_a, 0.005, y0, 0.150, y1,
    "数据\n$\\mathbf{x}$\n（$N$ 点观测）", C_BOX)
box(ax_a, 0.208, y0, 0.455, y1,
    "估计协方差参数\n$\\hat{\\boldsymbol{\\theta}}=\\arg\\max_{\\boldsymbol{\\theta}}"
    "p(\\mathbf{x};\\boldsymbol{\\theta},H_1)$\n（MLE，一般需数值优化）", C_EST)
box(ax_a, 0.513, y0, 0.805, y1,
    "估计器-相关器（把 $\\hat{\\boldsymbol{\\theta}}$ 代入）\n"
    "$T(\\mathbf{x})=\\mathbf{x}^TC_s(\\hat{\\boldsymbol{\\theta}})"
    "(C_s(\\hat{\\boldsymbol{\\theta}})+\\sigma^2I)^{-1}\\mathbf{x}$\n"
    "（第 5 章结构在\"协方差未知\"时的推广）", C_BOX)
box(ax_a, 0.860, y0, 0.995, y1,
    "判决\n$T(\\mathbf{x})>\\gamma'$\n判 $H_1$", C_DEC)

arrow(ax_a, 0.150, 0.51, 0.208, 0.51)
arrow(ax_a, 0.455, 0.51, 0.513, 0.51)
arrow(ax_a, 0.805, 0.51, 0.860, 0.51)

# ================= 下排 (b)：大数据记录近似（频域） =================
ax_b.set_xlim(0, 1.0)
ax_b.set_ylim(0, 1.0)
ax_b.axis("off")
ax_b.set_title("(b) 大数据记录近似（WSS，式 8.16~8.19）：频域维纳滤波 + 周期图",
               fontsize=11.5, pad=10)

box(ax_b, 0.005, y0, 0.175, y1,
    "数据\n$\\mathbf{x}$\n（$N$ 点观测）", C_BOX)
box(ax_b, 0.235, y0, 0.415, y1,
    "周期图\n$I(f)=\\frac{1}{N}\\left|\\sum_n x[n]e^{-j2\\pi fn}\\right|^2$\n"
    "（FFT 计算）", C_BOX)
box(ax_b, 0.473, y0, 0.775, y1,
    "维纳加权\n$H(f;\\hat{\\boldsymbol{\\theta}})=\\frac{P_{ss}(f;\\hat{\\boldsymbol{\\theta}})}"
    "{P_{ss}(f;\\hat{\\boldsymbol{\\theta}})+\\sigma^2}$\n"
    "$T(\\mathbf{x})=N\\int H(f;\\hat{\\boldsymbol{\\theta}})\\,I(f)\\,df$", C_WIENER)
box(ax_b, 0.832, y0, 0.995, y1,
    "判决\n$T(\\mathbf{x})>\\gamma'$\n判 $H_1$", C_DEC)

arrow(ax_b, 0.175, 0.51, 0.235, 0.51)
arrow(ax_b, 0.415, 0.51, 0.473, 0.51)
arrow(ax_b, 0.775, 0.51, 0.832, 0.51)

# 图底注：放在 (b) 面板底部空白区
ax_b.text(0.5, 0.115,
          "频域形式把 (a) 里的 $N\\times N$ 矩阵求逆/行列式换成 FFT 可算的积分（回调第一卷 Ch07 §6.5 渐近似然）；\n"
          "周期随机信号（§8.6）是它的特例：$H(f)$ 变成等间隔的梳齿滤波器，检测器 = 对各谐波周期图求和比门限（式 8.31）。",
          ha="center", va="center", transform=ax_b.transAxes, fontsize=9.3,
          color="#2D3748", linespacing=1.6)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig029_未知参数随机信号检测.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
