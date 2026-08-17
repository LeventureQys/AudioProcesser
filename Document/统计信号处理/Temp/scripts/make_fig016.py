# -*- coding: utf-8 -*-
"""Fig016 维纳滤波前后对比：含噪信号 / 滤波输出 / 频响（对应原书第 12 章 §12.7）。

用法: py -3.14 make_fig016.py
输出: Documents/figures/Fig016_维纳滤波前后对比.png

设计要点（对应原书 §12.7 维纳平滑器，式 (12.53)(12.54) 与频域
H(f)=Pss(f)/(Pss(f)+Pww(f))）:
  (a) 左图: 自建 AR(1) 低通信号（s[n]=0.9 s[n-1]+u[n]，σ_u²=1）加 WGN（σ²=1），
       与频域非因果维纳平滑器的输出对比。看三点：真信号（灰）被噪声（浅灰点）
       淹没；维纳输出（蓝）把噪声起伏抹平、贴回真信号；但峰谷也被轻微平滑——
       这就是"降噪 vs 保真"的折衷（正文 §12.7 的代价）。
  (b) 右图: 频响 H(f)=η(f)/(η(f)+1)，η(f)=Pss(f)/Pww(f) 为"局部 SNR"。
       H(f) 落在 [0,1]；在信号 PSD 高（低频）处 H≈1（保留），PSD 低处 H≈0
       （衰减）——维纳平滑器是按"逐频段 SNR"做加权收缩的频域形式。

自建数值（与原书无冲突）：种子 20260916；AR(1) a[1]=-0.9、σ_u²=1；观测噪声 σ²=1；
长度 N=256。绘制后经 plotutil.check_figure 程序化碰撞检测通过才保存。
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

rng = np.random.default_rng(20260916)

# ---------------- 信号与噪声 ----------------
N = 256
a1 = -0.9                 # s[n] = -a1 s[n-1] + u[n] = 0.9 s[n-1] + u[n]（低通）
su2 = 1.0                 # 驱动噪声方差 σ_u²
sw2 = 1.0                 # 观测噪声方差 σ²

u = rng.standard_normal(N) * np.sqrt(su2)
s = np.zeros(N)
# 从稳态开始生成：先烧入 200 点再取尾部，避免暂态
burn = 200
sbuf = np.zeros(burn + N)
uall = rng.standard_normal(burn + N) * np.sqrt(su2)
for n in range(1, burn + N):
    sbuf[n] = 0.9 * sbuf[n - 1] + uall[n]
s = sbuf[burn:]
w = rng.standard_normal(N) * np.sqrt(sw2)
x = s + w

# ---------------- 频域非因果维纳平滑器 ----------------
# Pss(f) = σ_u² / |1 + a[1] e^{-j2πf}|² = σ_u² / |1 - 0.9 e^{-j2πf}|²
# （s[n] = -a[1] s[n-1] + u[n]，a[1]=a1=-0.9，故 a[1]=-0.9 → 低通 PSD）
freqs = np.fft.fftfreq(N, d=1.0)          # 周期 [-0.5, 0.5) 的离散频率
den = np.abs(1.0 + a1 * np.exp(-2j * np.pi * freqs)) ** 2
Pss = su2 / den
Pww = sw2 * np.ones_like(Pss)
H = Pss / (Pss + Pww)                      # 非因果维纳平滑器频响
X = np.fft.fft(x)
shat = np.fft.ifft(H * X).real            # 频域逐点收缩 + 逆变换

mse_before = float(np.mean((x - s) ** 2))
mse_after = float(np.mean((shat - s) ** 2))

# ---------------- 画图 ----------------
C_S = "#718096"   # 真信号（灰）
C_X = "#CBD5E0"   # 含噪观测（浅灰点）
C_HAT = "#2B6CB0" # 维纳输出（蓝）
C_H = "#C53030"   # 频响（红）

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.0))
fig.subplots_adjust(left=0.055, right=0.975, top=0.85, bottom=0.12, wspace=0.22)

# ---- (a) 时域：真信号 / 含噪 / 维纳输出 ----
show = 120
nn = np.arange(show)
ax_a.plot(nn, s[:show], color=C_S, lw=1.4, label="真信号 $s[n]$")
ax_a.plot(nn, x[:show], color=C_X, lw=0.9, alpha=0.9, label="含噪观测 $x[n]=s[n]+w[n]$")
ax_a.plot(nn, shat[:show], color=C_HAT, lw=1.8, label="维纳输出 $\\hat{s}[n]$")
ax_a.set_xlim(0, show)
ax_a.set_xlabel("样本 $n$", fontsize=10)
ax_a.set_ylabel("幅度", fontsize=10)
ax_a.tick_params(labelsize=9)
ax_a.legend(loc="upper right", fontsize=8.8, framealpha=0.9, ncol=1)

# 标注 MSE 前后对比（放左上空白区，避开曲线主峰）
ax_a.text(0.03, 0.96,
          f"MSE 滤波前 $=\\sigma^2=1$\n"
          f"MSE 滤波后（本现实）$={mse_after:.2f}$",
          ha="left", va="top", transform=ax_a.transAxes,
          fontsize=9, color="#1A202C", linespacing=1.5,
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7FAFC",
                    edgecolor="#4A5568", lw=0.9))

ax_a.set_title("(a) 时域：维纳平滑器把噪声抹平、贴回真信号", fontsize=11)

# ---- (b) 频域：维纳平滑器频响 + 归一化信号 PSD ----
# 频响在 [-0.5, 0.5) 上为偶函数；画升序排列以连续
idx = np.argsort(freqs)
f_asc = freqs[idx]
H_asc = H[idx]
Pss_asc = Pss[idx]
Pss_norm = Pss_asc / Pss_asc.max()

ax_b.plot(f_asc, H_asc, color=C_H, lw=1.8, label="频响 $H(f)=\\eta(f)/(\\eta(f){+}1)$")
ax_b.axhline(1.0, color="#4A5568", ls=":", lw=1.0, alpha=0.7)
ax_b.axhline(0.0, color="#4A5568", ls=":", lw=1.0, alpha=0.7)
ax_b.set_xlim(-0.5, 0.5)
ax_b.set_ylim(-0.04, 1.12)
ax_b.set_xlabel("频率 $f$（归一化）", fontsize=10)
ax_b.set_ylabel("频响 $H(f)$（左轴）", fontsize=10, color=C_H)
ax_b.tick_params(labelsize=9, colors=C_H)
ax_b.set_title("(b) 频域：逐频段 SNR 加权（$0\\leq H(f)\\leq1$）", fontsize=11)

ax_b2 = ax_b.twinx()
ax_b2.plot(f_asc, Pss_norm, color="#718096", lw=1.4, ls="--",
           label="归一化信号 PSD $P_{ss}(f)$")
ax_b2.set_ylabel("归一化 PSD（右轴）", fontsize=10, color="#718096")
ax_b2.set_ylim(-0.04, 1.12)
ax_b2.tick_params(labelsize=9, colors="#718096")

# 合并图例（放右下空白区）
lines1, labels1 = ax_b.get_legend_handles_labels()
lines2, labels2 = ax_b2.get_legend_handles_labels()
ax_b.legend(lines1 + lines2, labels1 + labels2, loc="lower center", fontsize=8.6,
            framealpha=0.9, ncol=1)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig016_维纳滤波前后对比.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("mse_before =", round(mse_before, 4), " mse_after =", round(mse_after, 4))
print("H[0] (DC) =", round(float(H[0]), 4), " H[0.5] =", round(float(H[N // 2]), 4))
