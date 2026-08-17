# -*- coding: utf-8 -*-
"""Fig005 无偏与最小方差示意：无偏/有偏 PDF 对比 + MVU 存在性。

用法: py -3.14 make_fig005.py
输出: Documents/figures/Fig005_无偏与最小方差示意.png

设计要点（对应原书第 2 章）:
  (a) 无偏 = 居中、方差 = 宽度：同一真值 θ=0 下三条估计量 PDF——
      无偏且方差小（MVU 候选）、无偏但方差大、有偏（均值偏移 b）。
      虚线为真值；底部双箭头表示偏差 b（有偏的系统误差）。
  (b) 方差曲线随 θ 交叉（原书例 2.3 的数值 18/36、20/36、24/36、27/36）：
      θ<0 时 θ̂2 方差更小，θ≥0 时 θ̂1 方差更小，不存在一致最优 → MVU 不存在。

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

C_NARROW = "#2F855A"   # 无偏·方差小
C_WIDE = "#2B6CB0"     # 无偏·方差大
C_BIASED = "#C53030"   # 有偏
C_AXIS = "#4A5568"     # 灰：真值线 / 偏差箭头

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.16, wspace=0.30)


def gauss(x, mu, sd):
    return 1.0 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sd) ** 2)


# ================= 左图 (a)：无偏/有偏 PDF 对比 =================
x = np.linspace(-2.4, 2.9, 700)
ax1.plot(x, gauss(x, 0.0, 0.20), lw=2.2, color=C_NARROW, label="无偏·方差小（MVU 候选）")
ax1.plot(x, gauss(x, 0.0, 1.00), lw=2.0, color=C_WIDE, label="无偏·方差大")
ax1.plot(x, gauss(x, 0.9, 0.50), lw=2.0, color=C_BIASED, label="有偏（E[$\\hat{\\theta}$]=θ+b）")

ax1.axvline(0.0, color=C_AXIS, ls="--", lw=1.3)
ax1.set_xlim(-2.4, 2.9)
ax1.set_ylim(-0.38, 2.25)
ax1.set_xlabel("估计量 $\\hat{\\theta}$ 的取值")
ax1.set_ylabel("概率密度")
ax1.set_title("(a) 无偏 = 居中，方差 = 宽度；MVU = 最窄的居中者", fontsize=11)

# 真值标注：贴虚线顶端（窄峰约 1.99），白底避免压线
ax1.text(0.0, 2.06, "真值 θ", ha="center", va="bottom", fontsize=9.5,
         color=C_AXIS, bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec="none", alpha=0.9))

# 偏差双箭头：从真值 θ=0 到有偏估计均值 0.9（用 patch 画，不产生文字包围盒）
ax1.add_patch(FancyArrowPatch((0.0, -0.18), (0.9, -0.18),
             arrowstyle="<->", color=C_AXIS, lw=1.4, mutation_scale=12))
ax1.text(0.45, -0.285, "偏差 b = E[$\\hat{\\theta}$] − θ（有偏的系统误差）",
         ha="center", va="center", fontsize=9, color=C_AXIS)

ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)

# ================= 右图 (b)：方差曲线交叉 → MVU 不存在 =================
# 原书例 2.3 的数值：θ̂1 = (x[0]+x[1])/2，θ̂2 = (2/3)x[0]+(1/3)x[1]
ax2.hlines(27 / 36, -1.6, 0.0, color=C_WIDE, lw=2.4, label="$\\hat{\\theta}_1$ = (x[0]+x[1])/2")
ax2.hlines(18 / 36, 0.0, 1.6, color=C_WIDE, lw=2.4)
ax2.hlines(24 / 36, -1.6, 0.0, color=C_BIASED, lw=2.4, label="$\\hat{\\theta}_2$ = (2/3)x[0]+(1/3)x[1]")
ax2.hlines(20 / 36, 0.0, 1.6, color=C_BIASED, lw=2.4)

ax2.axvline(0.0, color=C_AXIS, ls="--", lw=1.2)
ax2.set_xlim(-1.9, 1.9)
ax2.set_ylim(0.28, 1.02)
ax2.set_xlabel("参数 θ")
ax2.set_ylabel("方差 var($\\hat{\\theta}$)")
ax2.set_title("(b) 两条方差曲线交叉：不存在一致最优，MVU 不存在", fontsize=11)

ax2.text(0.06, 0.315, "θ=0", ha="left", va="center", fontsize=9, color=C_AXIS)

# 左侧空区（θ̂₂ 段 y=24/36 下方）标出 θ<0 的更优者，箭头向上指向 θ̂₂
ax2.annotate("θ<0：$\\hat{\\theta}_2$ 更优\n（24/36 < 27/36）", xy=(-0.8, 24 / 36 + 0.008),
             xytext=(-0.8, 0.52), ha="center", va="bottom", fontsize=9,
             color=C_BIASED, arrowprops=dict(arrowstyle="-", color=C_BIASED, lw=0.9))

# 右侧空区（θ̂₁ 段 y=18/36 下方）标出 θ≥0 的更优者，箭头向上指向 θ̂₁
ax2.annotate("θ≥0：$\\hat{\\theta}_1$ 更优\n（18/36 < 20/36）", xy=(0.8, 18 / 36 + 0.008),
             xytext=(0.8, 0.40), ha="center", va="bottom", fontsize=9,
             color=C_WIDE, arrowprops=dict(arrowstyle="-", color=C_WIDE, lw=0.9))

ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig005_无偏与最小方差示意.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
