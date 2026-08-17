# -*- coding: utf-8 -*-
"""Fig018 标量卡尔曼滤波仿真：真值 / 观测 / 估计轨迹（对应原书第 13 章 §13.4~13.5）。

用法: py -3.14 make_fig018.py
输出: Documents/figures/Fig018_卡尔曼轨迹仿真.png

模型（自建，参数与原书 §13.5 一致，便于对照稳态数值）:
  状态方程 s[n] = a s[n-1] + u[n]，a=0.9，驱动噪声 σ_u²=1；
  观测方程 x[n] = s[n] + w[n]，观测噪声 σ_n²=1；
  初值 s[-1] ~ N(0,1)，滤波器初始化 ŝ[-1|-1]=0、M[-1|-1]=1。
  (a) 左图: 真信号 s[n]（灰）、观测 x[n]（浅灰点）、卡尔曼滤波 ŝ[n|n]（蓝），
      浅蓝带为 ±2√M[n|n]（滤波器自己报告的误差范围）。看三点：观测被噪声糊得
      很散；滤波输出平滑地贴住真值；误差带随时间收缩——卡尔曼"自带性能度量"。
  (b) 右图: 卡尔曼增益 K[n]（左轴）与最小 MSE M[n|n]（右轴）随 n 收敛到稳态。
      稳态值 K[∞]=M[∞]=0.5974（虚线），对应原书 §13.5 用 a=0.9、σ_u²=1、σ_n²=1
      解稳态 Riccati 方程得到的 K[∞]（M_p[∞]=1.4839、M[∞]=0.5974）。

种子: 20260913。绘制后经 plotutil.check_figure 程序化碰撞检测通过才保存。
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

rng = np.random.default_rng(20260913)

# ---------------- 参数 ----------------
a = 0.9
su2 = 1.0   # 驱动噪声方差 σ_u²
sn2 = 1.0   # 观测噪声方差 σ_n²
N = 60

# ---------------- 生成信号与观测 ----------------
s = np.zeros(N + 1)
s[0] = rng.standard_normal() * 1.0          # s[-1] ~ N(0,1) 记在 s[0]
u = rng.standard_normal(N) * np.sqrt(su2)
for n in range(1, N + 1):
    s[n] = a * s[n - 1] + u[n - 1]
s_true = s[1:]                               # n=0..N-1 的真信号
w = rng.standard_normal(N) * np.sqrt(sn2)
x = s_true + w                               # 观测

# ---------------- 标量卡尔曼滤波 ----------------
shat = np.zeros(N)       # 滤波估计 ŝ[n|n]
M = np.zeros(N)          # 最小 MSE M[n|n]
Mpred = np.zeros(N)      # 预测 MSE M[n|n-1]
Kseq = np.zeros(N)       # 增益 K[n]

s_prev = 0.0             # ŝ[-1|-1]
M_prev = 1.0             # M[-1|-1]
for n in range(N):
    M_pred = a * a * M_prev + su2            # (13.39)
    K = M_pred / (M_pred + sn2)              # (13.40)
    shat_n = a * s_prev + K * (x[n] - a * s_prev)   # (13.41) 预测 a*s_prev 再修正
    M_upd = (1.0 - K) * M_pred               # (13.42)
    shat[n] = shat_n
    M[n] = M_upd
    Mpred[n] = M_pred
    Kseq[n] = K
    s_prev = shat_n
    M_prev = M_upd

# 稳态值（§13.5：a=0.9, σ_u²=1, σ_n²=1）
K_ss = 0.5974
M_ss = 0.5974
Mp_ss = 1.4839

# ---------------- 画图 ----------------
C_S = "#718096"
C_X = "#CBD5E0"
C_HAT = "#2B6CB0"
C_BAND = "#2B6CB0"
C_K = "#2F855A"
C_M = "#C53030"

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(left=0.05, right=0.975, top=0.85, bottom=0.13, wspace=0.20)

nn = np.arange(N)

# ---- (a) 轨迹 ----
ax_a.fill_between(nn, shat - 2 * np.sqrt(M), shat + 2 * np.sqrt(M),
                  color=C_BAND, alpha=0.14, lw=0)
ax_a.plot(nn, s_true, color=C_S, lw=1.3, label="真信号 $s[n]$")
ax_a.plot(nn, x, color=C_X, lw=0.9, alpha=0.9, label="观测 $x[n]=s[n]+w[n]$")
ax_a.plot(nn, shat, color=C_HAT, lw=1.8, label="卡尔曼滤波 $\\hat{s}[n\\,|\\,n]$")
ax_a.set_xlim(0, N - 1)
ax_a.set_xlabel("样本 $n$", fontsize=10)
ax_a.set_ylabel("幅度", fontsize=10)
ax_a.tick_params(labelsize=9)
ax_a.legend(loc="upper left", fontsize=8.8, framealpha=0.9)
ax_a.set_title("(a) 轨迹：滤波输出平滑贴真值、误差带收缩", fontsize=11)

# ---- (b) 增益与 MSE 收敛 ----
ax_b.plot(nn, Kseq, color=C_K, lw=1.8, label="增益 $K[n]$（左轴）")
ax_b.axhline(K_ss, color=C_K, ls="--", lw=1.1, alpha=0.8)
ax_b.set_xlim(0, N - 1)
ax_b.set_xlabel("样本 $n$", fontsize=10)
ax_b.set_ylabel("卡尔曼增益 $K[n]$（左轴）", fontsize=10, color=C_K)
ax_b.tick_params(labelsize=9, colors=C_K)
ax_b.set_ylim(0.3, 0.75)
ax_b.set_title("(b) 增益与 MSE 收敛到稳态（$K[\\infty]{=}M[\\infty]{=}0.5974$）", fontsize=11)

ax_b2 = ax_b.twinx()
ax_b2.plot(nn, M, color=C_M, lw=1.6, label="最小 MSE $M[n\\,|\\,n]$（右轴）")
ax_b2.axhline(M_ss, color=C_M, ls="--", lw=1.1, alpha=0.8)
ax_b2.set_ylabel("最小 MSE $M[n\\,|\\,n]$（右轴）", fontsize=10, color=C_M)
ax_b2.tick_params(labelsize=9, colors=C_M)
ax_b2.set_ylim(0.3, 1.2)

lines1, labels1 = ax_b.get_legend_handles_labels()
lines2, labels2 = ax_b2.get_legend_handles_labels()
ax_b.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8.6,
            framealpha=0.9)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig018_卡尔曼轨迹仿真.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
print("K[N-1] =", round(float(Kseq[-1]), 4), "（稳态理论 0.5974）")
print("M[N-1] =", round(float(M[-1]), 4), "（稳态理论 0.5974）")
print("Mpred[N-1] =", round(float(Mpred[-1]), 4), "（稳态理论 1.4839）")
