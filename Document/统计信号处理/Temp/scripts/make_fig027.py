# -*- coding: utf-8 -*-
"""Fig027 GLRT 流程：把未知参数用 MLE 估计后代入似然比。

用法: py -3.14 make_fig027.py
输出: Documents/figures/Fig027_GLRT流程.png

设计要点（对应原书第二卷第 6 章 §6.4.2，式 (6.12)(6.14)，PDF 624~625 / 书内 609~610）:
  数据 x → 在 H0 下求 MLE θ̂0 → p(x;θ̂0,H0)；在 H1 下求 MLE θ̂1 → p(x;θ̂1,H1)
  → 广义似然比 L_G(x) = p(x;θ̂1,H1)/p(x;θ̂0,H0) → 与门限比 → 判 H1。
  GLRT 的核心动作是"先用 MLE 把未知参数估出来，再塞进似然比"（回调第一卷 Ch07 MLE）。

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
C_MLE = "#F6E3CE"     # MLE 框（强调"估计工具"）
C_LR = "#E2F0E6"      # 似然比框
C_DEC = "#D6E4F5"     # 判决框
EC = "#4A5568"
AC = "#2B6CB0"

fig = plt.figure(figsize=(12.6, 8.2))
ax = fig.add_axes([0.05, 0.06, 0.90, 0.88])
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.axis("off")
ax.set_title("GLRT：先用 MLE 把未知参数估出来，再代入似然比（回调第一卷 Ch07 的 MLE）",
             fontsize=12.5, pad=12)


def box(ax, x0, y0, x1, y1, text, fc, fs=10.0):
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


# 第 1 行：数据
box(ax, 0.38, 0.86, 0.62, 0.985,
    "数据 $\\mathbf{x}=[x[0],\\ldots,x[N-1]]^T$", C_BOX, fs=10.5)

# 第 2 行：两个 MLE 分支
box(ax, 0.03, 0.62, 0.40, 0.80,
    "$H_0$ 下的 MLE：$\\hat{\\boldsymbol{\\theta}}_0$\n"
    "使 $p(\\mathbf{x};\\boldsymbol{\\theta}_0,H_0)$ 最大", C_MLE)
box(ax, 0.60, 0.62, 0.97, 0.80,
    "$H_1$ 下的 MLE：$\\hat{\\boldsymbol{\\theta}}_1$\n"
    "使 $p(\\mathbf{x};\\boldsymbol{\\theta}_1,H_1)$ 最大", C_MLE)

# 第 3 行：似然比
box(ax, 0.22, 0.38, 0.78, 0.56,
    "广义似然比（把 MLE 代入似然比）\n"
    "$L_G(\\mathbf{x})=\\frac{p(\\mathbf{x};\\hat{\\boldsymbol{\\theta}}_1,H_1)}"
    "{p(\\mathbf{x};\\hat{\\boldsymbol{\\theta}}_0,H_0)}$\n"
    "$=\\frac{\\max_{\\boldsymbol{\\theta}_1}p(\\mathbf{x};\\boldsymbol{\\theta}_1,H_1)}"
    "{\\max_{\\boldsymbol{\\theta}_0}p(\\mathbf{x};\\boldsymbol{\\theta}_0,H_0)}$", C_LR, fs=10.0)

# 第 4 行：判决
box(ax, 0.28, 0.06, 0.72, 0.24,
    "判决：$L_G(\\mathbf{x})>\\gamma$ 判 $H_1$；\n"
    "大数据记录下 $2\\ln L_G(\\mathbf{x})$ 渐近 $\\sim\\chi_r^2$（$H_0$ 下）", C_DEC, fs=10.0)

# 箭头：数据 → 两个 MLE（箭头落点在 MLE 框顶边、避开框内文字）
arrow(ax, 0.42, 0.86, 0.21, 0.805)
arrow(ax, 0.58, 0.86, 0.79, 0.805)
# 两个 MLE → 似然比（箭头落点在似然比框顶边两角、避开框内公式）
arrow(ax, 0.21, 0.62, 0.25, 0.565)
arrow(ax, 0.79, 0.62, 0.75, 0.565)
# 似然比 → 判决（箭头落点在判决框顶边、避开框内文字）
arrow(ax, 0.50, 0.38, 0.50, 0.245)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig027_GLRT流程.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
