# -*- coding: utf-8 -*-
"""Fig006 CRLB 与估计量方差：无偏估计量的方差有一条不可逾越的下限。

用法: py -3.14 make_fig006.py
输出: Documents/figures/Fig006_CRLB与估计量方差.png

模型: x[n] = A + w[n], w[n] ~ N(0, σ²), σ=1, 真值 A=1；估计量 = 样本均值 Â（例 3.3 的有效估计量）。
实验: 对每个 N ∈ {5,10,20,50,100,200,500,1000} 做 K=5000 次独立实验，统计 var(Â)。
  (a) 左图: 样本均值 Â 的 PDF 随 N 变窄（方差 = σ²/N 逐一标注）；
  (b) 右图: 双对数下 var(Â) 的蒙特卡洛点贴着 CRLB = σ²/N 直线下降，
      直线下方阴影 = 无偏估计量的"禁区"（var < CRLB 不可能）。
结论: CRLB = σ²/N 是本题精度天花板，样本均值恰好摸到它（有效估计量）。

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

C_PDF1 = "#C53030"   # N=1 最宽
C_PDF2 = "#2B6CB0"   # N=10
C_PDF3 = "#2F855A"   # N=100 最窄
C_AXIS = "#4A5568"   # 灰：真值线 / CRLB 线
C_MC = "#C53030"     # 蒙特卡洛点

A_TRUE = 1.0
SIGMA2 = 1.0
K = 5000
N_list = np.array([5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)
rng = np.random.default_rng(20260815)

var_mc = []
for N in N_list:
    data = A_TRUE + rng.standard_normal((K, int(N)))
    Ahat = data.mean(axis=1)
    var_mc.append(Ahat.var(ddof=1))
var_mc = np.array(var_mc)
crlb = SIGMA2 / N_list

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.14, wspace=0.28)


def gauss(x, mu, sd):
    return 1.0 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sd) ** 2)


# ================= 左图 (a)：估计量 PDF 随 N 变窄 =================
x = np.linspace(-1.2, 3.2, 800)
ax1.plot(x, gauss(x, A_TRUE, 1.0), lw=2.2, color=C_PDF1,
         label="N=1，方差 = σ² = 1")
ax1.plot(x, gauss(x, A_TRUE, 1.0 / np.sqrt(10)), lw=2.0, color=C_PDF2,
         label="N=10，方差 = σ²/10 = 0.1")
ax1.plot(x, gauss(x, A_TRUE, 1.0 / np.sqrt(100)), lw=2.0, color=C_PDF3,
         label="N=100，方差 = σ²/100 = 0.01")

ax1.axvline(A_TRUE, color=C_AXIS, ls="--", lw=1.3)
ax1.set_xlim(-1.2, 3.2)
ax1.set_ylim(-0.35, 4.35)
ax1.set_xlabel("估计量 Â 的取值")
ax1.set_ylabel("概率密度")
ax1.set_title("(a) 数据越多，样本均值的 PDF 越窄（方差 = σ²/N）", fontsize=11)

ax1.text(A_TRUE, 2.6, "真值 A=1", ha="center", va="bottom", fontsize=9.5,
         color=C_AXIS, bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec="none", alpha=0.9))
ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

# ================= 右图 (b)：方差贴着 CRLB 直线 =================
ax2.loglog(N_list, var_mc, "o-", color=C_MC, lw=1.6, ms=6,
           label="蒙特卡洛方差 var(Â)（K=5000）")
ax2.loglog(N_list, crlb, "--", color=C_AXIS, lw=1.8,
           label="CRLB = σ²/N（无偏估计量的下限）")
ax2.fill_between(N_list, crlb * 0.05, crlb, color=C_MC, alpha=0.13, hatch="//",
                 edgecolor="none", label="var < CRLB：无偏估计量不可能（禁区）")

ax2.set_xlabel("数据长度 N（对数轴）")
ax2.set_ylabel("方差 var(Â)")
ax2.set_title("(b) 蒙特卡洛方差贴着 CRLB 直线下降（双对数）", fontsize=11)
ax2.set_ylim(5e-5, 0.6)
ax2.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig006_CRLB与估计量方差.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("N      :", N_list.astype(int))
print("var_mc :", np.round(var_mc, 6))
print("crlb   :", np.round(crlb, 6))
