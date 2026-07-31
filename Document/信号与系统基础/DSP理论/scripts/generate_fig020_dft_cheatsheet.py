# -*- coding: utf-8 -*-
"""生成 Fig020（DFT 速查卡：定义 · 记号 · 计算 · 读法 · 性质 · 工程）。

用法:  python generate_fig020_dft_cheatsheet.py
输出:  ../figures/Fig020_DFT速查卡.png
依赖:  numpy, matplotlib
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _box(ax, x, y, w, h, text, fs=11, fc="#eef3fb", ec="0.35", tc="black",
         weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight)


def _arrow(ax, x1, y1, x2, y2, color="0.4", lw=1.3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def fig020():
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 0.88, 1.0],
                          height_ratios=[1.0, 1.0], hspace=0.48, wspace=0.30)

    # ---------------- (a) 定义公式 ----------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.1, 9.5, "(a) 定义：N 点信号 → N 个复数", fontsize=12, weight="bold")
    ax.text(5.0, 7.5, r"$X[k]=\sum_{n=0}^{N-1} x[n]\;e^{-j2\pi kn/N}$",
            ha="center", va="center", fontsize=17)
    ax.text(5.0, 6.3, "输入 $x[0..N-1]$（时域） → 输出 $X[0..N-1]$（频域）",
            ha="center", fontsize=10.5, color="0.25")
    ax.text(0.4, 4.9, "欧拉公式展开，每个 bin 拆成两条实数通道：",
            fontsize=10.5)
    ax.text(5.0, 3.9, r"$X[k]=\sum_n x[n]\cos(\omega_k n)\;-\;j\sum_n x[n]\sin(\omega_k n)$",
            ha="center", fontsize=12.5)
    ax.plot([4.6, 5.2], [4.25, 4.25], color="tab:blue", lw=2)
    ax.text(8.15, 4.28, r"$\mathrm{Re}$", fontsize=11, color="tab:blue",
            ha="center")
    ax.plot([6.0, 6.7], [3.15, 3.15], color="tab:green", lw=2)
    ax.text(8.15, 3.13, r"$\mathrm{Im}$", fontsize=11, color="tab:green",
            ha="center")
    ax.text(0.4, 2.2, "逆变换（拼回来）：", fontsize=10.5)
    ax.text(5.0, 1.2, r"$x[n]=\frac{1}{N}\sum_{k=0}^{N-1} X[k]\;e^{+j2\pi kn/N}$",
            ha="center", fontsize=13.5)

    # ---------------- (b) 记号速查 ----------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.1, 9.5, "(b) 记号速查", fontsize=12, weight="bold")
    lines = [
        (r"$N$", "帧长（采样点数），输入输出都含 N 个数"),
        (r"$n$", "时域序号，0 ~ N−1"),
        (r"$k$", "频率编号，0 ~ N−1（第 k 个 bin）"),
        (r"$\omega_k=\frac{2\pi k}{N}$", "每前进一个采样旋转的弧度"),
        (r"$f_k=\frac{k}{N}f_s$", "第 k 个 bin 的实际频率 (Hz)"),
        (r"$\Delta f=\frac{f_s}{N}$", "频率分辨率（bin 间距）"),
        (r"$k=0$", "直流；$k=N/2$ 奈奎斯特"),
        (r"$k$ 与 $N-k$", "实信号互为镜像（4.4 节）"),
    ]
    y = 8.55
    for sym, desc in lines:
        ax.text(0.3, y, sym, fontsize=12, color="tab:blue")
        ax.text(4.4, y, "——", fontsize=10, color="0.5")
        ax.text(5.0, y, desc, fontsize=10.5, va="center")
        y -= 1.02
    ax.text(0.1, 0.25,
            "候选频率个数 = 采样点数：一段 N 点信号只装得下 N 个独立数字。",
            fontsize=10, color="0.35")

    # ---------------- (c) 一个 bin 的计算流程 ----------------
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.1, 9.5, "(c) 每个 bin 是一台相关器", fontsize=12, weight="bold")
    _box(ax, 0.2, 4.3, 1.7, 1.0, r"$x[n]$", fc="#fdeaea", ec="0.45", fs=13)
    _box(ax, 3.0, 6.3, 2.2, 0.95, r"$\times\cos(\omega_k n)$", fs=11,
         fc="#eef3fb")
    _box(ax, 3.0, 3.3, 2.2, 0.95, r"$\times(-\sin(\omega_k n))$", fs=11,
         fc="#eef3fb")
    _box(ax, 5.9, 6.35, 1.1, 0.85, r"$\sum$", fs=14)
    _box(ax, 5.9, 3.35, 1.1, 0.85, r"$\sum$", fs=14)
    _box(ax, 7.5, 6.0, 2.2, 0.85, r"$\mathrm{Re}\{X[k]\}$", fs=12,
         fc="#fdeaea", ec="tab:blue")
    _box(ax, 7.5, 3.0, 2.2, 0.85, r"$\mathrm{Im}\{X[k]\}$", fs=12,
         fc="#fdeaea", ec="tab:green")
    _box(ax, 7.6, 0.9, 2.2, 1.0, r"$X[k]$", fs=14, fc="#f3fde8",
         ec="tab:red", weight="bold")
    _arrow(ax, 1.9, 4.8, 3.0, 6.8)
    _arrow(ax, 1.9, 4.8, 3.0, 3.8)
    _arrow(ax, 5.2, 6.8, 5.9, 6.8)
    _arrow(ax, 5.2, 3.8, 5.9, 3.8)
    _arrow(ax, 7.0, 6.8, 7.5, 6.45)
    _arrow(ax, 7.0, 3.8, 7.5, 3.45)
    _arrow(ax, 8.6, 6.0, 8.6, 1.9)
    _arrow(ax, 8.6, 3.0, 8.7, 1.9)
    ax.text(0.15, 0.25,
            "两路逐点相乘再求和：实部 = cos 匹配量，虚部 = −sin 匹配量。",
            fontsize=10, color="0.35")

    # ---------------- (d) 复数读法 ----------------
    ax = fig.add_subplot(gs[1, 0])
    ax.text(0.1, 9.5, "(d) 输出读法：一个复数，两种读法", fontsize=12,
            weight="bold", transform=ax.transAxes)
    ax.axhline(0, color="0.65", lw=0.8)
    ax.axvline(0, color="0.65", lw=0.8)
    re, im = 12, 9
    ax.annotate("", xy=(re, im), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2.2))
    ax.plot(re, im, "o", color="tab:red", ms=5)
    ax.plot([0, re], [0, 0], color="tab:blue", lw=1.4, ls="--")
    ax.plot([re, re], [0, im], color="tab:green", lw=1.4, ls="--")
    ax.annotate("", xy=(re * 0.55, im * 0.55), xytext=(re * 0.55, 0),
                arrowprops=dict(arrowstyle="-", color="0.5", lw=1))
    ax.text(2.6, 1.2, r"$\varphi=\angle X[k]$", fontsize=11, color="0.35")
    ax.text(re / 2, -2.4, r"$\mathrm{Re}=\sum x\cos$", fontsize=10.5,
            color="tab:blue", ha="center")
    ax.text(re + 1.4, im / 2, r"$\mathrm{Im}=-\sum x\sin$", fontsize=10.5,
            color="tab:green")
    ax.text(re * 0.6, im * 1.16, r"$|X[k]|$", fontsize=12, color="tab:red")
    ax.set(xlim=(-17, 17), ylim=(-14, 15))
    ax.set_aspect("equal")
    ax.text(-16.5, -13.2, r"$|X|=\sqrt{\mathrm{Re}^2+\mathrm{Im}^2}$",
            fontsize=11)
    ax.text(-16.5, -11.0, r"$\varphi=\mathrm{atan2}(\mathrm{Im},\mathrm{Re})$",
            fontsize=11)
    ax.text(2.6, 13.6, "幅度谱：多强；相位谱：从哪个角度起步",
            fontsize=10.5, color="0.25")
    ax.text(2.6, 12.0, "实余弦单边幅度 ≈ $2|X[k]|/N$", fontsize=10.5,
            color="0.25")

    # ---------------- (e) 常用性质 ----------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.1, 9.5, "(e) 常用性质", fontsize=12, weight="bold")
    props = [
        r"线性：$ax_1+bx_2$ → $aX_1+bX_2$",
        r"时移：$x[n-n_0]$ → $e^{-j\omega_k n_0}X[k]$",
        r"卷积定理：时域卷积 → 频域相乘",
        r"共轭对称（实信号）：$X[N-k]=X^*[k]$",
        r"只需前 $N/2+1$ 个 bin，其余是镜像",
        r"Parseval：$\sum_{n}|x[n]|^2$ → $\frac{1}{N}\sum_{k}|X[k]|^2$",
    ]
    y = 8.4
    for line in props:
        ax.text(0.3, y, line, fontsize=11)
        y -= 1.35
    ax.text(0.1, 0.3, "时移不改变幅度谱，只旋转相位——相位携带\"时间位置\"（4.2 节）。",
            fontsize=10, color="0.35")

    # ---------------- (f) 工程要点 ----------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.1, 9.5, "(f) 工程要点", fontsize=12, weight="bold")
    eng = [
        ("FFT", "快速算法，O(N log N) 替代 O(N²)"),
        ("归一化", "各库约定不同；2|X|/N 恢复实余弦幅度"),
        ("频谱泄漏", "周期不整时能量扩散到相邻 bin，加窗（Hann）缓解"),
        ("补零", "只插值频谱曲线，不提高真实分辨率"),
        ("实数输入", "用 rfft，只算 k=0..N/2，省一半"),
    ]
    y = 8.55
    for head, desc in eng:
        ax.text(0.3, y, head, fontsize=11.5, weight="bold", color="tab:blue")
        ax.text(2.0, y - 0.42, desc, fontsize=10.5, va="center", color="0.2")
        y -= 1.55
    ax.text(0.1, 0.55, "输出 X[k] 是复数：直角坐标干活，极坐标思考。",
            fontsize=10, color="0.35")

    fig.suptitle(
        "DFT 速查卡：定义 · 记号 · 计算 · 读法 · 性质 · 工程",
        fontsize=15)
    fig.subplots_adjust(top=0.92, bottom=0.07, left=0.05, right=0.97,
                        hspace=0.42, wspace=0.26)
    out = os.path.join(FIG_DIR, "Fig020_DFT速查卡.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    fig020()
