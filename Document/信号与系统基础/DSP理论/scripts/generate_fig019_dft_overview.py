# -*- coding: utf-8 -*-
"""生成 Fig019（DFT 全景图：输入 → 一排相关器 → 复数谱）。

用法:  python generate_fig019_dft_overview.py
输出:  ../figures/Fig019_DFT全景图.png
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


def fig019():
    """DFT 全景：输入 x[n] → 一排相关器 → 每个 bin 一个复数 X[k]。"""
    N, k3, k7 = 32, 3, 7
    n = np.arange(N)
    x = 1.0 * np.cos(2 * np.pi * k3 * n / N) \
        + 0.55 * np.cos(2 * np.pi * k7 * n / N + np.pi / 4)
    X = np.fft.fft(x)

    fig = plt.figure(figsize=(15.5, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0],
                          height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.30)

    # ---- (a) 输入：时域混合波形
    ax = fig.add_subplot(gs[0, 0])
    ax.stem(n, x, basefmt="0.8", linefmt="tab:blue", markerfmt="o")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel="采样序号 n", ylabel="x[n]", ylim=(-1.9, 1.9))
    ax.set_title("(a) 输入：两种频率混合的采样")

    # ---- (b) 相关器阵列：一排检测器，每个 bin 两条模板
    ax = fig.add_subplot(gs[0, 1:])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.05, 9.6, "(b) 内部的\"一排相关器\"：", fontsize=12, weight="bold")

    bins_show = [0, 1, 2, 3, 7, 15]
    for i, k in enumerate(bins_show):
        y = 8.2 - i * 1.32
        box = FancyBboxPatch((0.1, y - 0.48), 2.3, 0.96,
                             boxstyle="round,pad=0.06",
                             fc="#eef3fb", ec="0.35", lw=1.2)
        ax.add_patch(box)
        ax.text(1.25, y, f"bin $k={k}$", ha="center", va="center", fontsize=11)
        # 两条模板
        ax.text(2.75, y + 0.22, r"$\cos(\omega_k n)$", fontsize=10.5,
                color="tab:blue")
        ax.text(2.75, y - 0.24, r"$\sin(\omega_k n)$", fontsize=10.5,
                color="tab:green")
        # 输出箭头与复数
        ax.annotate("", xy=(5.6, y), xytext=(4.6, y),
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.2))
        ax.text(6.0, y, f"$X[{k}]={X[k].real:+.1f}{X[k].imag:+.1f}j$",
                fontsize=10.5, color="tab:red")
    ax.text(0.05, 0.32,
            "每检查一个候选频率，就输出一个复数：实部 = cos 模板的匹配量，虚部 = sin 模板的匹配量（差符号）。",
            fontsize=10.5, color="0.25", va="bottom")

    # ---- (c) 输出：幅度谱（哪个频率有多强）
    ax = fig.add_subplot(gs[1, 0])
    kk = np.arange(N // 2 + 1)
    ax.stem(kk, 2 * np.abs(X[:N // 2 + 1]) / N, basefmt="0.8",
            linefmt="tab:red", markerfmt="o")
    ax.annotate("k=3", xy=(3, 1.02), xytext=(5, 1.15),
                arrowprops=dict(arrowstyle="->"), fontsize=11)
    ax.annotate("k=7", xy=(7, 0.56), xytext=(10, 0.72),
                arrowprops=dict(arrowstyle="->"), fontsize=11)
    ax.set(xlabel="频率编号 k", ylabel="幅度", ylim=(-0.15, 1.5))
    ax.set_title("(c) 幅度谱：$|X[k]|$ 报告每个频率有多强")

    # ---- (d) 复数输出：k=3 和 k=7 两个 bin 的实部/虚部
    ax = fig.add_subplot(gs[1, 1])
    ax.axhline(0, color="0.65", lw=0.8)
    ax.axvline(0, color="0.65", lw=0.8)
    for k, c in [(3, "tab:blue"), (7, "tab:green")]:
        z = X[k]
        ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0))
        ax.plot(z.real, z.imag, "o", color=c, ms=6)
        ax.text(z.real * 1.15 + 0.6, z.imag * 1.15,
                f"$X[{k}]$", fontsize=12, color=c)
    ax.set(xlabel="实部（cos 匹配量）", ylabel="虚部（sin 匹配量）",
           xlim=(-19, 19), ylim=(-19, 19))
    ax.set_aspect("equal")
    ax.set_title("(d) 复数谱：每个 bin 是实虚部两点\n极坐标读法给出幅度和相位")

    # ---- (e) 实部 / 虚部 / 相位谱
    ax = fig.add_subplot(gs[1, 2])
    ax.stem(kk, X[:N // 2 + 1].real, basefmt="0.8", linefmt="tab:blue",
            markerfmt="o", label="实部")
    ax.stem(kk, X[:N // 2 + 1].imag, basefmt="0.8", linefmt="tab:green",
            markerfmt="s", label="虚部")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel="频率编号 k", ylabel="数值", ylim=(-19, 19))
    ax.legend(fontsize=9)
    ax.set_title("(e) 实部虚部各是一张谱：\n它们一起才完整描述信号")

    fig.suptitle(
        "DFT 全景：时域波形 → 一排相关器（每个 bin 用 cos/sin 两条模板）→ 每个 bin 交回一个复数",
        fontsize=15)
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.065, right=0.97,
                        hspace=0.45, wspace=0.28)
    out = os.path.join(FIG_DIR, "Fig019_DFT全景图.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    fig019()
