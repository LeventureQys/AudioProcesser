# -*- coding: utf-8 -*-
"""Fig035 第 1 章开头两个例子的几何示意：雷达测距 + 被动声纳测向。

用法: py -3.14 make_fig035.py
输出: Documents/figures/Fig035_雷达测距与声纳测向示意图.png

设计要点:
  (a) 雷达测距（transAxes 坐标）: 雷达站 → 发射脉冲 → 飞机 → 回波(衰减+噪声)；
      双程延迟 τ0 = 2R/c。右下角小轴画发射/回波波形示意。
  (b) 被动声纳测向（等比例数据坐标，保证角度真实）: 潜艇辐射噪声，波前以 β 角
      到达相距 d 的两个传感器；波前先到右侧传感器，再多走 d·cosβ 才到左侧，
      τ0 = d·cosβ/c → β = arccos(c·τ0/d)。
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
from matplotlib.patches import FancyBboxPatch, Arc

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

C_BOX = "#E8F0FA"
C_PLANE = "#FDEBD0"
C_SUB = "#FDEDED"
EC = "#4A5568"
AC = "#2B6CB0"
AC2 = "#C53030"

fig = plt.figure(figsize=(13.4, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0],
                      left=0.045, right=0.985, top=0.90, bottom=0.05,
                      wspace=0.14)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])

# ================= (a) 雷达测距 =================
axa.set_xlim(0, 1)
axa.set_ylim(0, 1)
axa.axis("off")
axa.set_title("(a) 雷达测距：测出双程延迟 $\\tau_0$，距离 $R$ 就到手", fontsize=11, pad=10)


def tbox(ax, x0, y0, x1, y1, text, fc, fs=10.5, ec=EC):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.008",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="#1A202C", linespacing=1.5)


# 雷达站（地面）与飞机（空中）——垂直布局，距离 R 一目了然
tbox(axa, 0.06, 0.08, 0.38, 0.24, "雷达站\n发射脉冲 · 接收回波", C_BOX)
tbox(axa, 0.06, 0.72, 0.38, 0.88, "飞机\n（目标）", C_PLANE)

# 发射（向上，实线）与回波（向下，虚线）
axa.annotate("", xy=(0.22, 0.70), xytext=(0.22, 0.24),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color=AC, lw=2.0))
axa.annotate("", xy=(0.32, 0.24), xytext=(0.32, 0.70),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color=AC2, lw=2.0, linestyle="--"))

# 左侧两个标签，白底防压线
axa.annotate("发射脉冲", xy=(0.225, 0.58), xytext=(0.015, 0.62),
             xycoords="axes fraction", textcoords="axes fraction",
             fontsize=9.5, color=AC, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=AC, lw=0.9),
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
axa.annotate("回波（衰减 + 噪声）", xy=(0.325, 0.42), xytext=(0.015, 0.38),
             xycoords="axes fraction", textcoords="axes fraction",
             fontsize=9.5, color=AC2, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=AC2, lw=0.9),
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
axa.annotate("距离 R", xy=(0.27, 0.47), xytext=(0.44, 0.56),
             xycoords="axes fraction", textcoords="axes fraction",
             fontsize=10, color="#1A202C", ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color="#1A202C", lw=0.9))

# 右上公式框
axa.text(0.56, 0.90, "回波比发射晚 $\\tau_0$ 秒\n$\\tau_0 = 2R/c$\n$\\Rightarrow R = c\\,\\tau_0/2$\n（$\\tau_0$ 只能从含噪波形里\u201c估\u201d出来）",
         fontsize=10.5, ha="left", va="top", transform=axa.transAxes,
         color="#1A202C", linespacing=1.7)

# 波形示意（小轴）
axw = fig.add_axes([0.545, 0.10, 0.425, 0.40])
t = np.arange(0, 11)
axw.set_xlim(-0.8, 10.8)
axw.set_ylim(-0.15, 1.15)
axw.spines["top"].set_visible(False)
axw.spines["right"].set_visible(False)
axw.plot([0, 0], [0, 1.0], color=AC, lw=2.2)
axw.plot([5, 5], [0, 0.38], color=AC2, lw=2.2)
# 回波上的噪声毛刺
rng = np.random.default_rng(35)
tn = np.linspace(4.0, 6.0, 12)
yn = 0.38 * np.ones_like(tn) + 0.06 * rng.standard_normal(len(tn))
axw.plot(tn, yn, color=AC2, lw=1.0, alpha=0.75)
axw.plot([0, 0], [1.0, 1.0], "o", color=AC, ms=3)
axw.set_xlabel("时间 $t$")
axw.set_ylabel("幅度")
axw.annotate("发射脉冲", xy=(0.0, 1.0), xytext=(0.6, 0.95),
             fontsize=9, color=AC, ha="left", va="top",
             arrowprops=dict(arrowstyle="-", color=AC, lw=0.8))
axw.annotate("回波（衰减 + 噪声）", xy=(5.0, 0.44), xytext=(5.6, 0.75),
             fontsize=9, color=AC2, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=AC2, lw=0.8))

# ================= (b) 被动声纳测向 =================
# 等比例坐标，保证几何角度真实
axb.set_xlim(0, 11.6)
axb.set_ylim(0, 6.8)
axb.set_aspect("equal")
axb.axis("off")
axb.set_title("(b) 被动声纳测向：测出传感器间时延 $\\tau_0$，方位 $\\beta$ 就到手",
              fontsize=11, pad=10)


def dbox(ax, cx, cy, w, h, text, fc, fs=10):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.06",
                                facecolor=fc, edgecolor=EC, linewidth=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color="#1A202C", linespacing=1.5)


# 传感器（基线 y=0.9，间距 d=4.0）
s1 = (2.0, 0.9)
s2 = (6.0, 0.9)
dbox(axb, *s1, 1.7, 0.7, "传感器 1", C_BOX, fs=9.5)
dbox(axb, *s2, 1.7, 0.7, "传感器 2", C_BOX, fs=9.5)
axb.plot([2.85, 5.15], [0.9, 0.9], color=EC, lw=1.6)
axb.text(4.0, 1.28, "间距 $d$", ha="center", va="center", fontsize=9.5, color=EC)

# 到达方向（与基线成 β≈54°），波前法线从传感器 2 指向潜艇
bdeg = np.deg2rad(54.0)
udir = np.array([np.cos(bdeg), np.sin(bdeg)])
sub_c = s2 + 5.6 * udir  # (9.29, 5.43)
dbox(axb, sub_c[0], sub_c[1], 2.1, 0.85, "潜艇\n（辐射噪声）", C_SUB, fs=9.5)
axb.plot([s2[0], sub_c[0]], [s2[1], sub_c[1]], color=EC, lw=1.4)

# 波前（垂直于到达方向）三根虚线
perp = np.array([-udir[1], udir[0]])
for k in (0.0, 1.15, 2.3):
    p0 = s2 + k * udir
    seg = np.array([p0 - 7.2 * perp, p0 + 3.2 * perp])
    axb.plot(seg[:, 0], seg[:, 1], ls="--", lw=1.1, color="#718096", alpha=0.9)

# 到达方向箭头（沿法线反方向，向下）
axb.annotate("", xy=(7.35, 2.85), xytext=(8.85, 4.9),
             arrowprops=dict(arrowstyle="-|>", color="#2F855A", lw=2.0))
axb.annotate("到达方向", xy=(8.0, 3.6), xytext=(9.75, 3.15),
             fontsize=9.5, color="#2F855A", ha="center", va="center",
             arrowprops=dict(arrowstyle="-", color="#2F855A", lw=0.9),
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

# 角度 β 弧与标注
arc = Arc(s2, 3.2, 3.2, angle=0.0, theta1=0.0, theta2=54.0,
          color=AC, lw=1.4)
axb.add_patch(arc)
mid = s2 + 1.6 * np.array([np.cos(bdeg / 2), np.sin(bdeg / 2)])
axb.text(mid[0] + 0.45, mid[1] + 0.22, "$\\beta$", fontsize=11, color=AC,
         ha="center", va="center")

# 路径差 d·cosβ（传感器 1 到最近波前的额外行程）
p1_wave = s1 + (4.0 * np.cos(bdeg)) * udir
axb.annotate("", xy=(p1_wave[0], p1_wave[1]), xytext=(s1[0], s1[1]),
             arrowprops=dict(arrowstyle="-|>", color=AC2, lw=1.8))
axb.text(3.35, 1.95, "$d\\cos\\beta$", fontsize=10, color=AC2,
         ha="left", va="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

# 底部结论式（单行斜杠形式，置于右下空白区）
axb.text(8.45, 0.42,
         "$\\tau_0 = d\\cos\\beta\\,/\\,c$   $\\Rightarrow$   "
         "$\\beta = \\arccos(c\\,\\tau_0 / d)$",
         fontsize=10.5, ha="center", va="center", color="#1A202C",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=EC, lw=0.8, alpha=0.95))

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig035_雷达测距与声纳测向示意图.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
