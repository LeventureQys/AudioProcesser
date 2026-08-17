# -*- coding: utf-8 -*-
"""Fig034 复矢量扩展及阵列处理：阵列信号模型与波束示意（对应原书第二卷第 13 章，PDF 839~879 / 书内 824~864）。

用法: py -3.14 make_fig034.py
输出: Documents/figures/Fig034_阵列处理.png

设计要点（全部为自建示意，对应原书图 13.8/13.10 的几何与图 13.9 的空域处理）:
  (a) 阵列信号模型：远场点源以到达角 β（自阵轴量起，β=90° 为侧面/宽边到达）辐射平面波，
      波前先到编号大的传感器。第 m 个传感器相对第 0 个的附加延迟（样本计）为
      n_m(β) = -m(d/(cΔ))cos β（均匀线阵，间距 d，采样间隔 Δ，声速/光速 c），
      对应相位差 -2π f1 n_m(β)。
  (b) 波束形成前：各传感器信号相位错开（M=4 个复相量各指一个方向），矢量相加部分抵消。
  (c) 波束形成后：乘以相位补偿 exp[j2π f1 n_m(β)] 使信号同相，相加后幅度 ×M；
      噪声相位随机、非同相，相加后功率 ×M —— 于是 SNR ×M，即阵列增益 10log10 M dB。

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

EC = "#4A5568"       # 结构/轴线
AC = "#2B6CB0"       # 主色（波前/相量）
C_SUM = "#C53030"    # 和矢量（红）
C_DASH = "#A0AEC0"   # 辅助

fig = plt.figure(figsize=(12.6, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95],
                      width_ratios=[1.0, 1.0],
                      left=0.06, right=0.975, top=0.915, bottom=0.07,
                      hspace=0.42, wspace=0.34)
ax_a = fig.add_subplot(gs[0, :])   # (a) 阵列几何
ax_b = fig.add_subplot(gs[1, 0])   # (b) 波束形成前
ax_c = fig.add_subplot(gs[1, 1])   # (c) 波束形成后

# ================= (a) 阵列信号模型 =================
M = 6
beta_deg = 60.0
beta = np.deg2rad(beta_deg)

ax_a.set_xlim(-0.8, 7.4)
ax_a.set_ylim(-2.0, 5.2)
ax_a.axis("off")
ax_a.set_title("(a) 阵列信号模型：远场平面波前以到达角 $\\beta$ 打到均匀线阵（$n_m(\\beta)=-m\\frac{d}{c\\Delta}\\cos\\beta$）",
               fontsize=10.5, pad=8)

# 阵轴（x 轴）与传感器
ax_a.plot([-0.4, 6.6], [0, 0], color=EC, lw=1.6, zorder=2)
for m in range(M):
    ax_a.add_patch(plt.Circle((m, 0), 0.13, facecolor="#E8F0FA", edgecolor=AC, lw=1.6, zorder=3))
    lbl = "0" if m == 0 else ("$M{-}1$" if m == M - 1 else str(m))
    ax_a.text(m, -0.44, lbl, fontsize=10, ha="center", va="top", color="#1A202C", zorder=4)

# 间距标注
ax_a.annotate("", xy=(1, -0.95), xytext=(0, -0.95),
              arrowprops=dict(arrowstyle="<->", color=EC, lw=1.1))
ax_a.text(0.5, -1.28, "间距 $d$", fontsize=9.5, ha="center", va="top", color=EC)

# 波前（等相位面）：垂直于传播方向。传播方向（源→阵）为 (-cosβ, -sinβ)，
# 波前线方向为 (sinβ, -cosβ)，斜率 -cosβ/sinβ = -cot β。
wavefront_slope = -np.cos(beta) / np.sin(beta)   # 相对 x 轴的斜率（负）
def wf_y(x, x0, y0):
    return y0 + wavefront_slope * (x - x0)

# 三条波前线（从左下到右上），穿过 y 轴上不同高度
wf_x0s = [1.6, 4.6, 7.0]
wf_y0s = [3.3, 1.7, 0.6]
for x0, y0 in zip(wf_x0s, wf_y0s):
    xs = np.array([x0 - 2.6, x0 + 1.2])
    ax_a.plot(xs, wf_y(xs, x0, y0), color=AC, lw=1.7, alpha=0.85, zorder=1)

# 传播方向箭头（源→阵），从右上方斜向左下方
p0 = np.array([5.0, 3.75])
pdir = np.array([-np.cos(beta), -np.sin(beta)])
ax_a.add_patch(FancyArrowPatch(p0, p0 + 1.5 * pdir, arrowstyle="-|>",
                               mutation_scale=14, color=C_SUM, lw=2.0, zorder=4))
ax_a.text(p0[0] + 0.12, p0[1] - 0.02, "传播方向", fontsize=9.5, color=C_SUM, va="bottom")

# 到达角 β 弧线（自 +x 轴到传播反方向 u）
arc_r = 0.85
th = np.linspace(0, np.pi - beta, 60)
ax_a.plot([0], [0], marker="o", ms=0)  # 占位
ax_a.plot(np.cos(np.pi - th) * arc_r + 4.9, np.sin(np.pi - th) * arc_r + 3.9,
          color=EC, lw=1.1, zorder=3)
ax_a.text(5.55, 3.42, "$\\beta$", fontsize=11, color=EC)

# 源标注
ax_a.text(6.0, 4.55, "远场源", fontsize=10, color="#1A202C", ha="center", va="bottom")
# 波前标注
ax_a.text(1.7, 3.42, "波前（等相位面）", fontsize=9.5, color=AC, va="bottom")

# 先到/后到说明
ax_a.text(5.3, -1.15, "编号大者先到（$n_m<0$：提前）", fontsize=9, color=EC,
          ha="center", va="center")

# ================= (b) 波束形成前 =================
ax_b.set_xlim(-2.6, 2.6)
ax_b.set_ylim(-2.6, 2.6)
ax_b.set_aspect("equal")
ax_b.axis("off")
ax_b.set_title("(b) 波束形成前：各传感器相位错开，矢量相加部分抵消", fontsize=10.5, pad=8)

# M=4 个错开相位的信号相量（单位长）
angles_b = np.deg2rad([40.0, -70.0, 130.0, -20.0])
qx = np.cos(angles_b)
qy = np.sin(angles_b)
for i in range(4):
    ax_b.quiver(0, 0, qx[i], qy[i], angles="xy", scale_units="xy", scale=1,
                color=AC, width=0.012, zorder=3)
sx, sy = qx.sum(), qy.sum()
ax_b.quiver(0, 0, sx, sy, angles="xy", scale_units="xy", scale=1,
            color=C_SUM, width=0.02, zorder=4)
ax_b.text(0.06, 0.10, "和", fontsize=9.5, color=C_SUM, va="bottom")
ax_b.text(-2.4, 2.3, "4 个相量各指一方，\n和长度 $\\approx 1.4 < M{=}4$",
          fontsize=9.5, color="#1A202C", va="top")

# ================= (c) 波束形成后 =================
ax_c.set_xlim(-0.6, 5.4)
ax_c.set_ylim(-2.6, 2.6)
ax_c.set_aspect("equal")
ax_c.axis("off")
ax_c.set_title("(c) 波束形成后（乘 $e^{j2\\pi f_1 n_m(\\beta)}$）：信号同相，噪声不同相", fontsize=10.5, pad=8)

# 4 个已对齐的信号相量（沿 +x，单位长）
for i in range(4):
    ax_c.quiver(0, 0, 1.0, 0.0, angles="xy", scale_units="xy", scale=1,
                color=AC, width=0.012, zorder=3)
ax_c.quiver(0, 0, 4.0, 0.0, angles="xy", scale_units="xy", scale=1,
            color=C_SUM, width=0.02, zorder=4)
ax_c.text(4.1, 0.24, "和", fontsize=9.5, color=C_SUM, va="bottom")

ax_c.text(0.05, -2.3,
          "信号同相叠加：幅度 $\\times M$；\n噪声相位随机：功率 $\\times M$\n"
          "$\\Rightarrow$ SNR $\\times M$，阵列增益 $10\\log_{10}M$ dB",
          fontsize=9.5, color="#1A202C", va="bottom")
ax_c.text(2.0, 0.95, "幅度 $= M = 4$", fontsize=9.5, color=C_SUM, ha="center")

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig034_阵列处理.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
