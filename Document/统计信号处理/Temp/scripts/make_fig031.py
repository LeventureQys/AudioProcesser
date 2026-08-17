# -*- coding: utf-8 -*-
"""Fig031 非高斯噪声：重尾 PDF 与检测器的非线性（限幅/sign）。

用法: py -3.14 make_fig031.py
输出: Documents/figures/Fig031_非高斯噪声非线性.png

设计要点（对应原书第二卷第 10 章，图 10.1 / 10.8，PDF 766 / 781，书内 751 / 766）:
  (a) 相同方差 σ²=1 的高斯 PDF 与拉普拉斯 PDF（线性刻度）：拉普拉斯在 0 处更尖、
      拖尾更重 → 更容易出现"尖峰/野值"（图 10.1a、10.2）。
  (b) 广义高斯噪声下的归一化非线性 h(x)=|x|^{1/(1+β)}·sgn(x)（图 10.8）：
      β=0（高斯）→ 线性 h(x)=x（匹配滤波器）；
      β=1（拉普拉斯）→ 符号函数 h(x)=sgn(x)（无限限幅器/符号检测器）；
      β=0.5、0.75 → 介于两者之间，对大样本值起限幅作用，压低噪声尖峰。
  这对应"非高斯噪声下 NP/Rao 检测器长出非线性环节（限幅器）"的结论（10.4、10.5 节）。

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

from plotutil import setup_cn, check_figure

setup_cn()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "Documents", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
fig.subplots_adjust(left=0.07, right=0.975, top=0.84, bottom=0.14, wspace=0.28)

# ============ 左图 (a)：高斯 vs 拉普拉斯 PDF（σ²=1） ============
axa = axes[0]
axa.set_title("(a) 相同方差 $\\sigma^2=1$：拉普拉斯更尖、拖尾更重", fontsize=11.5, pad=10)
w = np.linspace(-4.5, 4.5, 1200)
p_gauss = np.exp(-w ** 2 / 2) / np.sqrt(2 * np.pi)
p_lap = np.exp(-np.sqrt(2) * np.abs(w)) / np.sqrt(2)
axa.plot(w, p_gauss, color="#2B6CB0", lw=2.0, label="高斯 $\\beta=0$")
axa.plot(w, p_lap, color="#D97706", lw=2.0, label="拉普拉斯 $\\beta=1$")
axa.set_xlim(-4.5, 4.5)
axa.set_ylim(0.0, 0.78)
axa.set_xlabel("噪声样本 $w[n]$", fontsize=11)
axa.set_ylabel("概率密度 $p(w[n])$", fontsize=11)
axa.grid(True, alpha=0.25, lw=0.5)
axa.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
axa.text(-4.35, 0.62, "拖尾重 →\n尖峰/野值多",
         ha="left", va="center", fontsize=9.5, color="#9C4221")

# ============ 右图 (b)：归一化非线性（限幅/sign） ============
axb = axes[1]
axb.set_title("(b) 广义高斯噪声的归一化非线性 $h(x)=|x|^{1/(1+\\beta)}\\,\\mathrm{sgn}(x)$",
              fontsize=11.5, pad=10)
x = np.linspace(-3.0, 3.0, 1200)


def h_of(xx, beta):
    return np.sign(xx) * np.abs(xx) ** (1.0 / (1.0 + beta))


colors = {"0": "#2B6CB0", "0.5": "#2F855A", "0.75": "#B83280", "1": "#D97706"}
for beta in ["0", "0.5", "0.75", "1"]:
    b = float(beta)
    lab = f"$\\beta={beta}$"
    if beta == "0":
        lab += "（高斯，线性）"
    elif beta == "1":
        lab += "（拉普拉斯，符号）"
    axb.plot(x, h_of(x, b), color=colors[beta], lw=2.0, label=lab)

# 标注"线性"与"限幅"两条分界
axb.axhline(0, color="#4A5568", lw=0.7)
axb.axvline(0, color="#4A5568", lw=0.7)
axb.set_xlim(-3.0, 3.0)
axb.set_ylim(-1.15, 1.15)
axb.set_xlabel("输入 $x$（变换后的数据 $y[n]=g(x[n])$ 的自变量）", fontsize=11)
axb.set_ylabel("$h(x)$（非线性输出）", fontsize=11)
axb.grid(True, alpha=0.25, lw=0.5)
axb.legend(loc="upper left", fontsize=9.0, framealpha=0.9)
axb.text(-2.92, -0.70, "大样本值被限幅，压住尖峰",
         ha="left", va="center", fontsize=9.5, color="#553C9A")

fig.suptitle("非高斯噪声：PDF 重尾是「因」，检测器长出非线性限幅器是「果」（§10.3~10.5）",
             fontsize=12.5, y=0.975)

check_figure(fig)
out = os.path.join(FIG_DIR, "Fig031_非高斯噪声非线性.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
