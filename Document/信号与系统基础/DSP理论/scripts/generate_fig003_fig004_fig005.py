# -*- coding: utf-8 -*-
"""生成 Fig003（DFT相关器）、Fig004（同幅不同相）、Fig005（共轭对称）。

用法:  python generate_fig003_fig004_fig005.py
输出:  ../figures/Fig003_DFT相关器.png
       ../figures/Fig004_同幅不同相.png
       ../figures/Fig005_共轭对称.png
依赖:  numpy, matplotlib
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------- Fig003
def fig003():
    """DFT 的一个 bin = 与 cos/sin 两个模板做相关。"""
    N, k = 32, 3
    phi = np.radians(60)
    n = np.arange(N)
    x = np.cos(2 * np.pi * k * n / N + phi)
    bc = np.cos(2 * np.pi * k * n / N)
    bs = np.sin(2 * np.pi * k * n / N)
    pc, ps = x * bc, x * bs
    sc, ss = pc.sum(), ps.sum()          # sc = (N/2)cosφ,  ss = -(N/2)sinφ

    # 与 FFT 核对
    X = np.fft.fft(x)
    assert np.allclose([X[k].real, X[k].imag], [sc, -ss], atol=1e-9)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    nf = np.linspace(0, N - 1, 400)

    axes[0].stem(n, x, basefmt="0.7")
    axes[0].plot(nf, np.cos(2 * np.pi * k * nf / N + phi), color="0.75", lw=1.0)
    axes[0].set_ylabel("x(n)")
    axes[0].set_title(
        f"(a) 被分析信号：$x(n)=\\cos(2\\pi\\cdot{k}n/{N} + 60°)$ —— "
        "频率已知，相位未知")

    axes[1].plot(nf, np.cos(2 * np.pi * k * nf / N), color="tab:blue",
                 lw=1.2, alpha=0.6, label="cos 模板")
    mask = pc >= 0
    axes[1].bar(n[mask], pc[mask], color="tab:red", alpha=0.75, width=0.6)
    axes[1].bar(n[~mask], pc[~mask], color="tab:blue", alpha=0.75, width=0.6)
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set_ylabel(r"$x(n)\cos(2\pi kn/N)$")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].set_title(
        f"(b) 与 cos 模板逐点相乘再求和：$\\sum = {sc:.2f}"
        f" = \\frac{{N}}{{2}}\\cos 60° \\;\\Rightarrow\\; "
        f"\\mathrm{{Re}}\\,X[{k}]={X[k].real:.2f}$")

    mask = ps >= 0
    axes[2].plot(nf, np.sin(2 * np.pi * k * nf / N), color="tab:green",
                 lw=1.2, alpha=0.6, label="sin 模板")
    axes[2].bar(n[mask], ps[mask], color="tab:red", alpha=0.75, width=0.6)
    axes[2].bar(n[~mask], ps[~mask], color="tab:blue", alpha=0.75, width=0.6)
    axes[2].axhline(0, color="0.6", lw=0.8)
    axes[2].set_ylabel(r"$x(n)\sin(2\pi kn/N)$")
    axes[2].set_xlabel("采样序号 n")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].set_title(
        f"(c) 与 sin 模板逐点相乘再求和：$\\sum = {ss:.2f}"
        f" \\;\\Rightarrow\\; \\mathrm{{Im}}\\,X[{k}] = -\\sum = {X[k].imag:.2f}$")

    mag, ang = np.abs(X[k]), np.degrees(np.angle(X[k]))
    fig.suptitle(
        f"DFT 第 {k} 号 bin 是一对相关器：  "
        f"$X[{k}] = {X[k].real:.2f} + j\\,{X[k].imag:.2f}"
        f" = {mag:.1f}\\,e^{{j\\,{ang:.0f}°}}$ —— 恰好还原出 60° 初相位",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, "Fig003_DFT相关器.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


# ---------------------------------------------------------------- Fig004
def fig004():
    """同一幅度、不同相位：实部虚部此消彼长，模不变。"""
    phis = np.radians([0, 45, 90, 135])
    colors = plt.cm.plasma(np.linspace(0.05, 0.75, len(phis)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    t = np.linspace(0, 2, 600)
    for phi, c in zip(phis, colors):
        ax1.plot(t, np.cos(2 * np.pi * t + phi), color=c, lw=1.8,
                 label=f"$\\varphi={np.degrees(phi):.0f}°$")
    ax1.axhline(0, color="0.8", lw=0.8)
    ax1.set_xlabel("时间（周期数）")
    ax1.set_ylabel("幅值")
    ax1.set_ylim(-1.35, 1.65)
    ax1.legend(ncol=4, loc="upper center", fontsize=10)
    ax1.set_title("(a) 四条正弦波：幅度相同、只差初相位（波形整体平移）")

    circ = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(circ), np.sin(circ), "--", color="0.7", lw=1.0)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.axvline(0, color="0.6", lw=0.8)
    for phi, c in zip(phis, colors):
        z = np.exp(1j * phi)
        ax2.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0))
        ax2.text(1.13 * z.real - 0.06, 1.13 * z.imag,
                 f"{np.degrees(phi):.0f}°", color=c, fontsize=11)
    # 用 45° 那个向量演示直角坐标分量
    z = np.exp(1j * phis[1])
    ax2.plot([z.real, z.real], [0, z.imag], ":", color="0.4", lw=1.4)
    ax2.plot([0, z.real], [z.imag, z.imag], ":", color="0.4", lw=1.4)
    ax2.text(z.real + 0.03, z.imag / 2 - 0.06, r"$A\sin\varphi$",
             fontsize=10, color="0.3")
    ax2.text(z.real / 2 - 0.15, -0.13, r"$A\cos\varphi$", fontsize=10, color="0.3")
    ax2.set_xlim(-1.45, 1.45)
    ax2.set_ylim(-1.45, 1.45)
    ax2.set_aspect("equal")
    ax2.set_xlabel("实部")
    ax2.set_ylabel("虚部")
    ax2.set_title("(b) 频域视角：四个点都在同一个圆上\n模（幅度）不变，实部虚部随相位此消彼长")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig004_同幅不同相.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


# ---------------------------------------------------------------- Fig005
def fig005():
    """任意实信号的 DFT 共轭对称性。"""
    rng = np.random.default_rng(42)
    N = 64
    x = rng.standard_normal(N)
    x = np.convolve(x, np.ones(3) / 3, mode="same")  # 轻微平滑，非必需
    X = np.fft.fft(x)
    k = np.arange(N)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex=True)
    panels = [
        (X.real, "实部 Re X[k]", "偶对称：Re X[N-k] = Re X[k]", "tab:blue"),
        (X.imag, "虚部 Im X[k]", "奇对称：Im X[N-k] = -Im X[k]", "tab:red"),
        (np.abs(X), "幅度 |X[k]|", "偶对称：|X[N-k]| = |X[k]|", "tab:green"),
        (np.angle(X), "相位 ∠X[k]（rad）", "奇对称：∠X[N-k] = -∠X[k]", "tab:purple"),
    ]
    for ax, (val, ylab, note, c) in zip(axes.flat, panels):
        ml, sl, bl = ax.stem(k, val, basefmt="0.8")
        plt.setp(ml, color=c, markersize=3.5)
        plt.setp(sl, color=c, linewidth=1.0)
        ax.axvline(N / 2, color="0.3", lw=1.2, ls="--")
        ax.axvspan(N / 2, N - 1, color="0.85", alpha=0.45)
        ax.set_ylabel(ylab)
        ax.set_title(note, fontsize=11)
        ax.text(N * 0.74, ax.get_ylim()[1] * 0.82, "镜像区\n（无新信息）",
                fontsize=9, color="0.35", ha="center")
    axes[1, 0].set_xlabel("频率序号 k")
    axes[1, 1].set_xlabel("频率序号 k")
    fig.suptitle(
        "一段 64 点实数随机信号的 DFT：$X[N-k]=X^*[k]$ —— "
        "右半边永远是左半边的镜像（随机种子 42）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, "Fig005_共轭对称.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    fig003()
    fig004()
    fig005()
