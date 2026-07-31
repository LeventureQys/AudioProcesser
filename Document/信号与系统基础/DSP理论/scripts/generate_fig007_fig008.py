# -*- coding: utf-8 -*-
"""生成 Fig007（希尔伯特变换与解析信号）、Fig008（IQ解调）。

用法:  python generate_fig007_fig008.py
输出:  ../figures/Fig007_希尔伯特变换与解析信号.png
       ../figures/Fig008_IQ解调.png
依赖:  numpy, scipy, matplotlib
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.signal import butter, filtfilt, hilbert

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------- Fig007
def fig007():
    """解析信号：包络、单边谱、瞬时频率。"""
    fs, dur, fc = 1000, 1.0, 50.0
    t = np.arange(int(fs * dur)) / fs
    m = 1 + 0.6 * np.cos(2 * np.pi * 3 * t)          # 缓变包络（3Hz 调幅）
    x = m * np.cos(2 * np.pi * fc * t)               # 实信号
    xa = hilbert(x)                                  # 解析信号 x + jH{x}

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.6))

    # (a) 时域：实信号 + 包络
    axes[0].plot(t, x, color="0.55", lw=0.8, label="实信号 $x(t)$")
    axes[0].plot(t, np.abs(xa), color="tab:red", lw=2.0,
                 label=r"包络 $|x_a(t)|=\sqrt{x^2+\hat{x}^2}$")
    axes[0].plot(t, -np.abs(xa), color="tab:red", lw=2.0)
    axes[0].plot(t, m, "--", color="tab:green", lw=1.2, label="真实包络 $m(t)$")
    axes[0].set_xlim(0, dur)
    axes[0].set_xlabel("时间（s）")
    axes[0].set_ylabel("幅值")
    axes[0].legend(loc="upper right", fontsize=9, ncol=3)
    axes[0].set_title("(a) 50Hz 载波 × 3Hz 调幅包络：解析信号的模精确还原包络")

    # (b) 频域：双边谱 vs 单边谱
    f = np.fft.fftshift(np.fft.fftfreq(len(t), 1 / fs))
    X = np.fft.fftshift(np.abs(np.fft.fft(x))) / len(t)
    Xa = np.fft.fftshift(np.abs(np.fft.fft(xa))) / len(t)
    axes[1].plot(f, X, color="tab:blue", lw=1.4, label="实信号谱 |X(f)|（正负对称）")
    axes[1].plot(f, Xa, color="tab:red", lw=1.4, alpha=0.85,
                 label="解析信号谱 |Xa(f)|（负频率清零，正频率×2）")
    axes[1].axvline(0, color="0.6", lw=0.8)
    axes[1].set_xlim(-120, 120)
    axes[1].set_xlabel("频率（Hz）")
    axes[1].set_ylabel("幅度")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].set_title("(b) 解析信号 = 把镜像的负频率删掉，只留正频率")

    # (c) 瞬时频率
    fi = np.diff(np.unwrap(np.angle(xa))) * fs / (2 * np.pi)
    axes[2].plot(t[1:], fi, color="tab:purple", lw=1.4)
    axes[2].axhline(fc, color="0.5", ls="--", lw=1.0)
    axes[2].text(0.83, fc + 4, f"载波频率 {fc:.0f}Hz", fontsize=10, color="0.35")
    axes[2].set_xlim(0, dur)
    axes[2].set_ylim(fc - 25, fc + 25)
    axes[2].set_xlabel("时间（s）")
    axes[2].set_ylabel("瞬时频率（Hz）")
    axes[2].set_title(r"(c) 瞬时频率 $f_i(t)=\frac{1}{2\pi}\frac{d}{dt}\angle x_a(t)$"
                      "：有了相位才谈得上“每一刻的频率”（两端为边界效应）")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig007_希尔伯特变换与解析信号.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


# ---------------------------------------------------------------- Fig008
def _draw_iq_diagram(ax):
    """用 matplotlib 画 IQ 下变频框图。"""
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, text):
        ax.add_patch(Rectangle((x, y), w, h, fc="#eef3fb", ec="0.3", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    def mixer(x, y):
        ax.add_patch(Circle((x, y), 0.32, fc="white", ec="0.3", lw=1.2))
        ax.text(x, y - 0.02, "×", ha="center", va="center", fontsize=15)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color="0.2", lw=1.3))

    ax.text(0.4, 3.0, "实信号\n$x(t)$", ha="center", va="center", fontsize=11)
    arrow(1.0, 3.0, 2.2, 3.0)
    ax.plot([2.2, 2.2], [1.5, 4.5], color="0.2", lw=1.3)   # 分路
    arrow(2.2, 4.5, 3.38, 4.5)
    arrow(2.2, 1.5, 3.38, 1.5)
    mixer(3.7, 4.5)
    mixer(3.7, 1.5)
    ax.text(3.7, 5.45, r"$2\cos(2\pi f_c t)$", ha="center", fontsize=11,
            color="tab:blue")
    arrow(3.7, 5.25, 3.7, 4.85)
    ax.text(3.7, 0.55, r"$-2\sin(2\pi f_c t)$", ha="center", fontsize=11,
            color="tab:red")
    arrow(3.7, 0.75, 3.7, 1.15)
    arrow(4.02, 4.5, 5.2, 4.5)
    arrow(4.02, 1.5, 5.2, 1.5)
    box(5.2, 4.0, 1.8, 1.0, "低通滤波")
    box(5.2, 1.0, 1.8, 1.0, "低通滤波")
    arrow(7.0, 4.5, 8.2, 4.5)
    arrow(7.0, 1.5, 8.2, 1.5)
    ax.text(9.0, 4.5, "$I(t)$ 同相分量", ha="center", va="center",
            fontsize=12, color="tab:blue")
    ax.text(9.0, 1.5, "$Q(t)$ 正交分量", ha="center", va="center",
            fontsize=12, color="tab:red")
    ax.set_title("(a) I/Q 下变频：一对正交的本振把实信号拆成两路基带")


def fig008():
    """IQ 解调：从实带通信号恢复缓变的幅度与相位。"""
    fs, dur, fc = 10000, 1.0, 1000.0
    t = np.arange(int(fs * dur)) / fs
    amp = 1 + 0.5 * np.cos(2 * np.pi * 5 * t)        # 缓变幅度（5Hz）
    ph = (np.pi / 3) * np.sin(2 * np.pi * 3 * t)     # 缓变相位（3Hz）
    x = amp * np.cos(2 * np.pi * fc * t + ph)        # 实带通信号

    # 混频 + 低通（5 阶 Butterworth，截止 100Hz，零相位滤波）
    b, a = butter(5, 100 / (fs / 2))
    i_sig = filtfilt(b, a, x * 2 * np.cos(2 * np.pi * fc * t))
    q_sig = filtfilt(b, a, x * -2 * np.sin(2 * np.pi * fc * t))
    # 理论值：I = amp·cos(ph),  Q = amp·sin(ph)

    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    ax_diag = fig.add_subplot(gs[0, :])
    _draw_iq_diagram(ax_diag)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, i_sig, color="tab:blue", lw=1.6, label="$I(t)$ 实测")
    ax1.plot(t, q_sig, color="tab:red", lw=1.6, label="$Q(t)$ 实测")
    ax1.plot(t, amp * np.cos(ph), "--", color="k", lw=0.9,
             label=r"理论 $a(t)\cos\varphi(t)$ / $a(t)\sin\varphi(t)$")
    ax1.plot(t, amp * np.sin(ph), "--", color="k", lw=0.9)
    ax1.set_xlabel("时间（s）")
    ax1.set_ylabel("幅值")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title("(b) 解调出的 I/Q 与理论值重合\n（载波 1kHz，幅度 5Hz 缓变，相位 3Hz 缓变）")

    ax2 = fig.add_subplot(gs[1, 1])
    sl = slice(int(0.05 * fs), int(0.95 * fs))       # 去掉滤波边界段
    pts = ax2.scatter(i_sig[sl], q_sig[sl], c=t[sl], s=2, cmap="viridis")
    ax2.axhline(0, color="0.7", lw=0.8)
    ax2.axvline(0, color="0.7", lw=0.8)
    ax2.set_aspect("equal")
    ax2.set_xlabel("I（实部）")
    ax2.set_ylabel("Q（虚部）")
    cb = fig.colorbar(pts, ax=ax2, shrink=0.85)
    cb.set_label("时间（s）")
    ax2.set_title("(c) 复基带轨迹 $I+jQ$：\n到原点的距离 = 瞬时幅度，辐角 = 瞬时相位")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig008_IQ解调.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    fig007()
    fig008()
