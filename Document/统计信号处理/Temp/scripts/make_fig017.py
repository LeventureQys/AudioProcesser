# -*- coding: utf-8 -*-
"""Fig017 卡尔曼滤波器"预测→更新"循环数据流（对应原书第 13 章 §13.4）。

用法: py -3.14 make_fig017.py
输出: Documents/figures/Fig017_卡尔曼预测更新循环.png

设计要点（对应原书标量卡尔曼滤波器式 (13.38)~(13.42) 与图 13.5 的方框图）:
  主链（左→右→下）：上一时刻滤波估计 → 预测（时间更新，用动态模型 a、σ_u²）
  → 新息 e[n]=x[n]−ŝ[n|n−1]（新观测减去预测）→ 修正（测量更新，用增益 K[n]）。
  增益 K[n] 由预测 MSE M[n|n−1] 与观测噪声 σ_n² 决定；修正后的 ŝ[n|n]、M[n|n]
  作为下一时刻的"上一时刻估计"，沿左侧/底部走线回到预测框，形成递推循环。
  看三件事：① 预测只用动态模型、不碰新数据；② 修正只把"新息"乘增益加回去，
  增益 = 预测不确定性与观测不确定性的折衷；③ 底部反馈线说明滤波器是"只存上一
  时刻状态"的 O(1) 递推，而不是每次全量重算。

矢量/箭头用 FancyArrowPatch（patch 而非文本）。绘制后经 plotutil.check_figure
程序化碰撞检测通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

EC = "#4A5568"
C_PREV = "#DD6B20"   # 上一时刻估计
C_PRED = "#2B6CB0"   # 预测
C_OBS = "#2F855A"    # 新观测
C_INN = "#C53030"    # 新息
C_GAIN = "#553C9A"   # 增益
C_UPD = "#1A202C"    # 修正（主输出）
C_LOOP = "#718096"   # 反馈线

fig, ax = plt.subplots(1, 1, figsize=(11.5, 6.4))
fig.subplots_adjust(left=0.02, right=0.99, top=0.90, bottom=0.02)

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.axis("off")


def box(x0, y0, x1, y1, text, fc, fs=10, color="#1A202C", bold=False, lw=1.3):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.012",
                                transform=ax.transAxes,
                                facecolor=fc, edgecolor=EC, linewidth=lw))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color=color,
            linespacing=1.55, fontweight="bold" if bold else "normal")


def arrow(x0, y0, x1, y1, color=EC, lw=1.8, style="-|>", ls="-", ms=13):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes,
                                 arrowstyle=style, color=color, lw=lw,
                                 mutation_scale=ms, shrinkA=0, shrinkB=0,
                                 linestyle=ls))


# ---- 主链六框 ----
box(0.03, 0.76, 0.30, 0.94, "上一时刻滤波估计\n$\\hat{s}[n{-}1\\,|\\,n{-}1]$\n$M[n{-}1\\,|\\,n{-}1]$",
    C_PREV, fs=10)
box(0.03, 0.40, 0.47, 0.66, "预测（时间更新）\n$\\hat{s}[n\\,|\\,n{-}1]=a\\,\\hat{s}[n{-}1\\,|\\,n{-}1]$\n$M[n\\,|\\,n{-}1]=a^2M[n{-}1\\,|\\,n{-}1]+\\sigma_u^2$",
    C_PRED, fs=9.6)
box(0.60, 0.80, 0.97, 0.94, "新观测\n$x[n]=s[n]+w[n]$", C_OBS, fs=10)
box(0.60, 0.56, 0.97, 0.74, "新息（预测误差）\n$e[n]=x[n]-\\hat{s}[n\\,|\\,n{-}1]$",
    C_INN, fs=10)
box(0.60, 0.36, 0.97, 0.50, "卡尔曼增益\n$K[n]=\\dfrac{M[n\\,|\\,n{-}1]}{M[n\\,|\\,n{-}1]+\\sigma_n^2}$",
    C_GAIN, fs=9.6)
box(0.42, 0.08, 0.97, 0.28,
    "修正（测量更新）\n$\\hat{s}[n\\,|\\,n]=\\hat{s}[n\\,|\\,n{-}1]+K[n]\\,e[n]$\n$M[n\\,|\\,n]=(1-K[n])\\,M[n\\,|\\,n{-}1]$",
    C_UPD, fs=10, bold=True)

# ---- 箭头（主链） ----
arrow(0.165, 0.76, 0.165, 0.66, color=C_PREV)                 # 上一时刻 → 预测
arrow(0.47, 0.53, 0.60, 0.65, color=C_PRED)                    # 预测 → 新息（ŝ 供新息）
arrow(0.47, 0.46, 0.60, 0.43, color=C_PRED)                    # 预测 → 增益（M 供增益）
arrow(0.785, 0.80, 0.785, 0.74, color=C_OBS)                   # 观测 → 新息
arrow(0.785, 0.56, 0.785, 0.28, color=C_INN)                   # 新息 → 修正
arrow(0.72, 0.36, 0.72, 0.28, color=C_GAIN)                    # 增益 → 修正

# ---- 反馈线（修正 → 上一时刻，沿左、下边缘正交走线） ----
arrow(0.42, 0.18, 0.012, 0.18, color=C_LOOP, lw=1.6)           # 底部向左
arrow(0.012, 0.18, 0.012, 0.85, color=C_LOOP, lw=1.6)          # 左缘向上
arrow(0.012, 0.85, 0.03, 0.85, color=C_LOOP, lw=1.6)           # 左缘向右入上一时刻框

# ---- 小注记 ----
ax.text(0.505, 0.60, "$\\times K[n]$", ha="center", va="center",
        transform=ax.transAxes, fontsize=9.5, color=C_GAIN)
ax.text(0.05, 0.105, "$n\\leftarrow n{+}1$（递推循环）", ha="left", va="center",
        transform=ax.transAxes, fontsize=9.5, color=C_LOOP)
ax.text(0.505, 0.565, "预测供新息", ha="left", va="center",
        transform=ax.transAxes, fontsize=8, color=EC)
ax.text(0.505, 0.415, "$M$ 供增益", ha="left", va="center",
        transform=ax.transAxes, fontsize=8, color=EC)

ax.set_title("卡尔曼滤波器：预测（时间更新）→ 新息 → 修正（测量更新）的递推循环", fontsize=12)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig017_卡尔曼预测更新循环.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
