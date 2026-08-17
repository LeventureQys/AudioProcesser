# -*- coding: utf-8 -*-
"""Fig003 MLE 渐近性能的蒙特卡洛验证。

用法: py -3.14 make_fig003.py
输出: Documents/figures/Fig003_MLE渐近性能.png
模型: x[n] = A + w[n], w[n] ~ N(0, σ²), σ=1, 真值 A=1；MLE = 样本均值。
实验: 对每个 N ∈ {5,10,20,50,100,200} 做 K=5000 次独立实验，
  (a) 左图: 偏差 bias_N = mean(Â) - A 随 N 的变化（半对数 x 轴），对照 0 参考线；
  (b) 右图: 方差 var(Â) 随 N 的变化（双对数），对照理论 CRLB = σ²/N。
结论: 偏差在 0 附近随机摆动（无偏），方差沿 1/N 直线下降、与 CRLB 重合。
绘制后经 plotutil.check_figure 碰撞检测通过才保存。
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

rng = np.random.default_rng(20260814)
A_TRUE = 1.0
SIGMA2 = 1.0
K = 5000
N_list = np.array([5, 10, 20, 50, 100, 200], dtype=float)

bias, var = [], []
for N in N_list:
    data = A_TRUE + rng.standard_normal((K, int(N)))
    Ahat = data.mean(axis=1)
    bias.append(Ahat.mean() - A_TRUE)
    var.append(Ahat.var(ddof=1))

bias = np.array(bias)
var = np.array(var)
crlb = SIGMA2 / N_list

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))

# ---- (a) 偏差 vs N ----
ax1.semilogx(N_list, bias, "o-", color="#2B6CB0", lw=1.6, ms=6)
ax1.axhline(0.0, color="#4A5568", ls="--", lw=1.0)
ax1.set_xlabel("数据长度 N（对数轴）")
ax1.set_ylabel("偏差  bias = mean(Â) − A")
ax1.set_title("(a) 偏差：围绕 0 随机摆动", fontsize=11)
ax1.set_ylim(-0.035, 0.035)

# ---- (b) 方差 vs N ----
ax2.loglog(N_list, var, "o-", color="#C53030", lw=1.6, ms=6, label="蒙特卡洛方差（K=5000）")
ax2.loglog(N_list, crlb, "--", color="#4A5568", lw=1.6, label="CRLB = σ²/N")
ax2.set_xlabel("数据长度 N（对数轴）")
ax2.set_ylabel("方差 var(Â)")
ax2.set_title("(b) 方差：沿 1/N 直线下降", fontsize=11)
ax2.legend(loc="lower left", fontsize=9.5, framealpha=0.9)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig003_MLE渐近性能.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("bias:", np.round(bias, 5))
print("var :", np.round(var, 5))
print("crlb:", np.round(crlb, 5))
