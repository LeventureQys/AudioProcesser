# -*- coding: utf-8 -*-
"""Fig025 匹配滤波器：时域镜像 + 输出在 n=N-1 达峰，SNR 最大化为 E/σ²。

用法: py -3.14 make_fig025.py
输出: Documents/figures/Fig025_匹配滤波器输出SNR.png

设计要点（对应原书第二卷第 4 章 §4.3，式 (4.3)(4.5)(4.10)，图 4.1~4.3，
        PDF 541~546 / 书内 526~531）:
  已知信号 s[n]（本图取 N=4 的非对称波形 [2,1,4,3]）。
  (a) 信号 s[n]，能量 E = Σ s²[n] = 2²+1²+4²+3² = 30。
  (b) 匹配滤波器冲激响应 h[n] = s[N-1-n] = [3,4,1,2]（信号的"镜像"）。
  (c) 输出 y[n] = s[n] * h[n]（自相关形状），在 n=N-1=3 处达峰 y[3]=E=30。
      在 n=N-1 采样：信号分量 = Σ s[k]h[N-1-k] = Σ s²[k] = E = 30；
      噪声分量方差 = σ² Σ h²[k] = σ² E，故输出 SNR = E²/(σ²E) = E/σ²。
      由 Cauchy-Schwarz 不等式，这是任何 FIR 滤波器在此时刻能达到的最大输出 SNR。
  结论：匹配滤波器把"信号能量"在采样时刻 n=N-1 对齐叠加，输出 SNR 达上界 E/σ²。

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

C_S = "#2B6CB0"    # 信号
C_H = "#C53030"    # 滤波器冲激响应
C_Y = "#2F855A"    # 输出
C_PEAK = "#4A5568"  # 峰值标记

s = np.array([2.0, 1.0, 4.0, 3.0])   # 已知信号（非对称，便于看出"镜像"）
N = len(s)
h = s[::-1]                          # h[n] = s[N-1-n]
y = np.convolve(s, h)                # 输出（自相关形状）
E = float(np.sum(s ** 2))            # 信号能量 = 30

fig, axes = plt.subplots(3, 1, figsize=(12.2, 8.4))
ax_a, ax_b, ax_c = axes

# ---------- (a) 信号 ----------
n = np.arange(N)
ax_a.stem(n, s, basefmt=" ", linefmt=C_S, markerfmt="o")
for ni, vi in zip(n, s):
    ax_a.annotate(f"{vi:g}", (ni, vi), textcoords="offset points",
                  xytext=(0, 7), ha="center", fontsize=10, color="#1A202C")
ax_a.set_xlim(-0.7, N - 1 + 0.7)
ax_a.set_ylim(0.0, 5.2)
ax_a.set_ylabel("$s[n]$")
ax_a.set_title("(a) 已知信号 $s[n]$（非对称波形，$N=4$），能量 $E=\\sum s^2[n]=2^2+1^2+4^2+3^2=30$",
               fontsize=11.5, pad=9)
ax_a.grid(True, axis="y", ls=":", lw=0.6, color="#CBD5E0", alpha=0.7)

# ---------- (b) 冲激响应 ----------
ax_b.stem(n, h, basefmt=" ", linefmt=C_H, markerfmt="o")
for ni, vi in zip(n, h):
    ax_b.annotate(f"{vi:g}", (ni, vi), textcoords="offset points",
                  xytext=(0, 7), ha="center", fontsize=10, color="#1A202C")
ax_b.set_xlim(-0.7, N - 1 + 0.7)
ax_b.set_ylim(0.0, 5.2)
ax_b.set_ylabel("$h[n]$")
ax_b.set_title("(b) 匹配滤波器冲激响应 $h[n]=s[N-1-n]=[3,4,1,2]$：把信号做镜像（时间反转）",
               fontsize=11.5, pad=9)
ax_b.grid(True, axis="y", ls=":", lw=0.6, color="#CBD5E0", alpha=0.7)

# ---------- (c) 输出 ----------
ny = np.arange(len(y))
ax_c.stem(ny, y, basefmt=" ", linefmt=C_Y, markerfmt="o")
ax_c.axvline(N - 1, color=C_PEAK, ls="--", lw=1.4)
ax_c.set_xlim(-0.7, len(y) - 1 + 0.7)
ax_c.set_ylim(0.0, 34.0)
ax_c.set_xlabel("$n$")
ax_c.set_ylabel("$y[n]$")
ax_c.set_title("(c) 输出 $y[n]=s[n]*h[n]$：在 $n=N-1=3$ 达峰 $y[3]=E=30$，此时输出 SNR 最大",
               fontsize=11.5, pad=9)
ax_c.grid(True, axis="y", ls=":", lw=0.6, color="#CBD5E0", alpha=0.7)

ax_c.annotate("采样时刻 $n=N-1$：\n信号分量 $=\\sum s^2[n]=E=30$\n噪声方差 $=\\sigma^2\\sum h^2[n]=\\sigma^2E$\n输出 SNR $=E^2/(\\sigma^2E)=E/\\sigma^2$（最大）",
              xy=(N - 1, E), xytext=(4.6, 26),
              ha="center", va="center", fontsize=9.5, color="#1A202C",
              arrowprops=dict(arrowstyle="->", color=C_PEAK, lw=1.3,
                              shrinkA=0, shrinkB=3))

fig.tight_layout(rect=[0, 0, 1, 0.995])
check_figure(fig)
out = os.path.join(FIG_DIR, "Fig025_匹配滤波器输出SNR.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
