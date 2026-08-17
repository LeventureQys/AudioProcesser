# -*- coding: utf-8 -*-
"""Fig030 未知噪声参数（CFAR）：门限随噪声电平自适应示意。

用法: py -3.14 make_fig030.py
输出: Documents/figures/Fig030_CFAR门限自适应.png

设计要点（对应原书第二卷第 9 章 §9.3，PDF 730~731，书内 715~716）:
  已知 DC 电平检测的门限 γ' = sqrt(σ²/N)·Q⁻¹(P_FA)（式 9.2）依赖噪声方差 σ²。
  (a) 固定门限（非 CFAR）：σ² 从 1 漂到 4，门限 γ' 不动，P_FA 从 0.10 漂到 0.26；
  (b) CFAR：门限随估计噪声电平 σ̂² 自适应（γ = sqrt(σ̂²/N)·Q⁻¹(P_FA)），
      两条 PDF 的右尾面积（P_FA）保持相等 = 0.10。
  这是"用参考噪声数据估计方差、门限跟着走"的几何本质。

绘制后经 plotutil.check_figure 程序化碰撞检测（重叠/穿框）通过才保存。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

# 参数
P_FA_TARGET = 0.10
Q_INV = norm.ppf(1 - P_FA_TARGET)          # Q⁻¹(P_FA) ≈ 1.2816
SIGMA1 = 1.0
SIGMA2 = 2.0
GAMMA_FIXED = Q_INV * SIGMA1               # 固定门限（按 σ²=1 标定）≈ 1.2816
GAMMA1 = Q_INV * SIGMA1                    # CFAR 下 σ²=1 的门限
GAMMA2 = Q_INV * SIGMA2                    # CFAR 下 σ²=4 的门限 ≈ 2.563

x = np.linspace(-5.0, 5.0, 1200)
p1 = norm.pdf(x, 0, SIGMA1)
p2 = norm.pdf(x, 0, SIGMA2)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
fig.subplots_adjust(left=0.065, right=0.975, top=0.84, bottom=0.14, wspace=0.26)

C1 = "#2B6CB0"   # σ²=1
C2 = "#D97706"   # σ²=4

for ax in axes:
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(0.0, 0.44)
    ax.set_xlabel("检验统计量 $T(\\mathbf{x})$（$H_0$ 下）", fontsize=11)
    ax.set_ylabel("概率密度 $p(T; H_0)$", fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.5)

# ============ 左图 (a)：固定门限，非 CFAR ============
axa = axes[0]
axa.set_title("(a) 固定门限：噪声电平一漂，$P_{FA}$ 就漂（非 CFAR）", fontsize=11.5, pad=10)
axa.plot(x, p1, color=C1, lw=2.0, label="$\\sigma^2=1$（$P_{FA}=0.10$）")
axa.plot(x, p2, color=C2, lw=2.0, label="$\\sigma^2=4$（$P_{FA}=0.26$）")
axa.axvline(GAMMA_FIXED, color="#4A5568", lw=1.4, ls="--")
# 右尾着色（固定门限）
xa1 = x[x >= GAMMA_FIXED]
axa.fill_between(xa1, norm.pdf(xa1, 0, SIGMA1), color=C1, alpha=0.30)
axa.fill_between(xa1, norm.pdf(xa1, 0, SIGMA2), color=C2, alpha=0.30)
axa.text(GAMMA_FIXED + 0.22, 0.235, "固定门限\n$\\gamma'=1.28$",
         ha="left", va="center", fontsize=9.5, color="#4A5568")
axa.legend(loc="upper right", fontsize=9.0, framealpha=0.9)

# ============ 右图 (b)：CFAR，门限随噪声电平缩放 ============
axb = axes[1]
axb.set_title("(b) CFAR：门限随 $\\hat{\\sigma}^2$ 自适应，$P_{FA}$ 恒定", fontsize=11.5, pad=10)
axb.plot(x, p1, color=C1, lw=2.0, label="$\\sigma^2=1$（$\\gamma_1=1.28$）")
axb.plot(x, p2, color=C2, lw=2.0, label="$\\sigma^2=4$（$\\gamma_2=2.56$）")
axb.axvline(GAMMA1, color=C1, lw=1.4, ls="--")
axb.axvline(GAMMA2, color=C2, lw=1.4, ls="--")
xb1 = x[x >= GAMMA1]
xb2 = x[x >= GAMMA2]
axb.fill_between(xb1, norm.pdf(xb1, 0, SIGMA1), color=C1, alpha=0.30)
axb.fill_between(xb2, norm.pdf(xb2, 0, SIGMA2), color=C2, alpha=0.30)
axb.text(GAMMA1 + 0.18, 0.245, "$\\gamma_1=1.28$\n$P_{FA}=0.10$",
         ha="left", va="center", fontsize=9.5, color=C1)
axb.text(GAMMA2 + 0.18, 0.150, "$\\gamma_2=2.56$\n$P_{FA}=0.10$",
         ha="left", va="center", fontsize=9.5, color=C2)
axb.legend(loc="upper right", fontsize=9.0, framealpha=0.9)

fig.suptitle("CFAR：门限必须随噪声电平缩放（式 9.2 $\\gamma'=\\sqrt{\\sigma^2/N}\\,Q^{-1}(P_{FA})$ 的两个实现）",
             fontsize=12.5, y=0.975)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig030_CFAR门限自适应.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
