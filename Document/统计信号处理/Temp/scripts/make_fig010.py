# -*- coding: utf-8 -*-
"""Fig010 BLUE 几何解释：线性类中的最优边界 + C^{-1} 预白化的几何意义。

用法: py -3.14 make_fig010.py
输出: Documents/figures/Fig010_BLUE几何解释.png

设计要点（对应原书第 6 章）:
  (a) 集合嵌套（原书图 6.1 的抽象化）：所有无偏估计量 ⊃ 线性无偏估计量；
      BLUE 是线性类中的最小方差者（★）；MVU 可能是非线性的（例 5.8 均匀噪声，
      取 (max+min)/2），落在线性类之外 → BLUE ≠ MVU，BLUE 只是准最佳；
      若 MVU 恰好线性（例 6.1 WGN 直流电平），则 BLUE = MVU。
  (b) 白化几何：正相关噪声（ρ=0.6）的 1σ 椭圆倾斜；信号 s=(1,0.4)；
      BLUE 权重 a_opt ∝ C^{-1}s = (1.1176, -0.2941) 不再与 s 同向，而是被 C^{-1}
      "反向扭转"——在正相关噪声下做"差分"以抵消共模噪声；var=1/(s^T C^{-1}s)。

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
from matplotlib.patches import FancyBboxPatch, Ellipse

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

EC = "#4A5568"
C_OUT = "#FDEBD0"   # 所有无偏估计量（外框）
C_LIN = "#E8F0FA"   # 线性无偏估计量（内框）
C_BLUE = "#2F855A"
C_MVU = "#C53030"
C_SIG = "#2B6CB0"
C_GRID = "#CBD5E0"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.3))
fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.13, wspace=0.26)

# ================= (a) 集合嵌套：BLUE 是"线性俱乐部"里的冠军 =================
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis("off")

ax1.add_patch(FancyBboxPatch((0.5, 0.5), 9.0, 9.0, boxstyle="round,pad=0.06",
                             facecolor=C_OUT, edgecolor=EC, linewidth=1.3))
ax1.text(5.0, 9.15, "所有无偏估计量", ha="center", va="center", fontsize=11.5,
         color="#1A202C", fontweight="bold")

ax1.add_patch(FancyBboxPatch((0.9, 1.6), 3.6, 5.4, boxstyle="round,pad=0.06",
                             facecolor=C_LIN, edgecolor=EC, linewidth=1.3))
ax1.text(2.7, 6.55, "线性无偏估计量\n$\\hat{\\theta}=\\sum_n a[n]x[n]$",
         ha="center", va="center", fontsize=10, color="#1A202C")

# BLUE 星标（用 plot 画星形，文字与星形分离，避免缺字形）
ax1.plot(2.7, 4.25, marker="*", ms=24, color=C_BLUE, mec="none", ls="none")
ax1.text(2.7, 3.25, "BLUE\n（线性类中方差最小）", ha="center", va="center",
         fontsize=9.5, color=C_BLUE)

# 例 6.1 注记（内框底部）
ax1.text(2.7, 2.02, "例 6.1：MVU 恰好线性\n→ BLUE = MVU", ha="center", va="center",
         fontsize=8.5, color="#276749")

# MVU（非线性）在外框右侧
ax1.plot(7.4, 5.6, marker="o", ms=11, color=C_MVU, mec="none", ls="none")
ax1.text(7.4, 4.45, "MVU（例 5.8 均匀噪声）\n非线性 → BLUE ≠ MVU\nBLUE 只是准最佳",
         ha="center", va="center", fontsize=9.5, color=C_MVU)

ax1.set_title("(a) BLUE 是\"线性俱乐部\"里的冠军，不一定是全体冠军", fontsize=11)

# ================= (b) C^{-1} 预白化的几何意义 =================
C = np.array([[1.0, 0.6], [0.6, 1.0]])
s = np.array([1.0, 0.4])
Cinv = np.linalg.inv(C)
a_opt = Cinv @ s / (s @ Cinv @ s)   # BLUE 权重（已满足 a^T s = 1）
var_blue = 1.0 / (s @ Cinv @ s)

ax2.set_xlim(-1.95, 2.75)
ax2.set_ylim(-1.85, 1.8)
ax2.set_aspect("equal")
ax2.axhline(0, color=C_GRID, lw=0.8)
ax2.axvline(0, color=C_GRID, lw=0.8)

# 1σ 噪声椭圆：eigh 升序返回特征值，最大特征值对应的特征向量是最后一列
w, V = np.linalg.eigh(C)
major = V[:, 1]
angle = np.degrees(np.arctan2(major[1], major[0]))
ell = Ellipse((0, 0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=angle,
              facecolor="#EDF2F7", edgecolor=EC, lw=1.3)
ax2.add_patch(ell)

# 信号 s 与 BLUE 权重 a_opt 箭头
ax2.annotate("", xy=(s[0], s[1]), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color=C_SIG, lw=2.4, shrinkA=0, shrinkB=0))
ax2.annotate("", xy=(a_opt[0], a_opt[1]), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.4, shrinkA=0, shrinkB=0))

ax2.text(s[0] + 0.06, s[1] + 0.13, "信号 s", ha="left", va="center",
         fontsize=10, color=C_SIG)
ax2.text(a_opt[0] + 0.06, a_opt[1] - 0.15, "BLUE 权重\n$a_{opt}\\propto C^{-1}s$",
         ha="left", va="center", fontsize=9, color=C_BLUE)

ax2.text(-1.4, 1.5, "噪声协方差 C 的 1σ 椭圆\n（样本相关 → 椭圆倾斜）",
         ha="left", va="top", fontsize=9, color=EC)

ax2.text(0.0, -1.5,
         "$\\mathrm{var}(\\hat{\\theta})=1/(s^T C^{-1}s)$：$C^{-1}$ 先\"转正\"椭圆（预白化），\n再沿白化后的信号方向匹配",
         ha="center", va="center", fontsize=9, color="#1A202C")

ax2.set_title("(b) $C^{-1}$ 预白化：$a_{opt}$ 反向扭转噪声的倾斜", fontsize=11)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig010_BLUE几何解释.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("a_opt =", np.round(a_opt, 4), " var =", round(float(var_blue), 4))
