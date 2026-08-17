# -*- coding: utf-8 -*-
"""Fig007 变换参数 CRLB：非线性变换的统计线性化（切线近似 + 方差传播）。

用法: py -3.14 make_fig007.py
输出: Documents/figures/Fig007_变换参数CRLB.png

模型: x[n] = A + w[n], w[n] ~ N(0, σ²), σ=1, 真值 A=1。
  样本均值 Â 是 A 的有效估计量（var = σ²/N），但 g(Â) = Â²（估计功率 A²）是 A² 的有偏估计。
  (a) 左图: g(θ)=θ² 的曲线与 A=1 处的切线；宽区间（小 N）下曲线明显弯曲、切线失准，
      窄区间（大 N）下曲线≈切线 → 统计线性化成立；
  (b) 右图: var(Â²) 随 N 逼近渐近 CRLB = (2A)²σ²/N = 4/N（从上方收敛，多出一项 2σ⁴/N²）。
结论: 非线性变换破坏"有限样本有效"，只在 N→∞ 时（统计线性化成立时）渐近达到 CRLB。

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

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

A_TRUE = 1.0
SIGMA2 = 1.0
C_CURVE = "#2B6CB0"    # g(θ) 曲线
C_TAN = "#4A5568"      # 切线
C_WIDE = "#C53030"     # 小 N 宽区间
C_NARROW = "#2F855A"   # 大 N 窄区间
C_MC = "#C53030"       # 蒙特卡洛点
C_CRLB = "#4A5568"     # CRLB 线

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.6, 5.2))
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.14, wspace=0.30)

# ================= 左图 (a)：切线近似 =================
theta = np.linspace(0.0, 2.4, 500)
g = theta ** 2
tan = 2 * theta - 1  # 切线 g(A) + g'(A)(θ-A)，A=1，g'(1)=2

ax_a.plot(theta, g, lw=2.4, color=C_CURVE)
ax_a.plot(theta, tan, ls="--", lw=1.8, color=C_TAN)

# 宽区间（小 N）：±3σ = 1.0，[0, 2.0]
ax_a.axvspan(0.0, 2.0, color=C_WIDE, alpha=0.10)
# 窄区间（大 N）：±3σ = 0.2，[0.8, 1.2]
ax_a.axvspan(0.8, 1.2, color=C_NARROW, alpha=0.20)

ax_a.plot([A_TRUE], [A_TRUE ** 2], "o", ms=7, color=C_TAN, zorder=5)

ax_a.set_xlim(-0.15, 2.5)
ax_a.set_ylim(-1.4, 6.0)
ax_a.set_xlabel("参数 θ")
ax_a.set_ylabel("g(θ) = θ²")
ax_a.set_title("(a) 统计线性化：N 越大，$\\hat{\\theta}$ 越集中，曲线越像切线", fontsize=11)

# 曲线/切线文字（放左上空白区，避免压线）
ax_a.text(0.12, 5.6, "g(θ)=θ²（实线）", ha="left", va="center", fontsize=10, color=C_CURVE)
ax_a.text(0.12, 4.95, "切线（虚线，斜率 g′(A)=2）", ha="left", va="center",
          fontsize=9.5, color=C_TAN)
# 区间标签（放曲线下方、x 轴上方，单行避免越界）
ax_a.text(1.0, -0.55, "小 N：$\\hat{\\theta}$ 区间宽（±3σ=1.0）",
          ha="center", va="center", fontsize=9, color=C_WIDE)
ax_a.text(1.0, -1.0, "大 N：$\\hat{\\theta}$ 区间窄（±3σ=0.2），曲线≈切线",
          ha="center", va="center", fontsize=9, color=C_NARROW)
ax_a.text(A_TRUE, 1.28, "A=1", ha="center", va="bottom", fontsize=9.5, color=C_TAN)

# ================= 右图 (b)：方差传播（渐近达到 CRLB） =================
N_list = np.array([10, 20, 50, 100, 200, 500, 1000], dtype=float)
K = 5000
rng = np.random.default_rng(20260815)

var_mc = []
for N in N_list:
    data = A_TRUE + rng.standard_normal((K, int(N)))
    xbar2 = (data.mean(axis=1)) ** 2
    var_mc.append(xbar2.var(ddof=1))
var_mc = np.array(var_mc)
crlb = (2 * A_TRUE) ** 2 * SIGMA2 / N_list   # = 4/N
var_exact = (2 * A_TRUE) ** 2 * SIGMA2 / N_list + 2 * SIGMA2 ** 2 / N_list ** 2  # 4/N + 2/N²

ax_b.loglog(N_list, var_mc, "o-", color=C_MC, lw=1.6, ms=6,
            label="蒙特卡洛 var(Â²)（K=5000）")
ax_b.loglog(N_list, crlb, "--", color=C_CRLB, lw=1.8,
            label="CRLB(A²) = (2A)²σ²/N = 4/N")
ax_b.set_xlabel("数据长度 N（对数轴）")
ax_b.set_ylabel("方差 var(Â²)")
ax_b.set_title("(b) Â² 的方差从上方逼近 CRLB（渐近有效）", fontsize=11)
ax_b.set_ylim(3e-4, 1.0)
ax_b.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig007_变换参数CRLB.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("N       :", N_list.astype(int))
print("var_mc  :", np.round(var_mc, 6))
print("crlb    :", np.round(crlb, 6))
print("var_exact:", np.round(var_exact, 6))
