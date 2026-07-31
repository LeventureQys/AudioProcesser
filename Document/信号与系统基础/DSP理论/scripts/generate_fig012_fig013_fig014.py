# -*- coding: utf-8 -*-
"""生成 2.0 节的说明图。

用法: python generate_fig012_fig013_fig014.py
输出: ../figures/Fig012_从直角坐标到极坐标.png
      ../figures/Fig013_旋转乘法与指数记号.png
      ../figures/Fig014_DSP中的逐采样旋转.png
      ../figures/Fig015_30度乘45度等于旋转75度.png
依赖: numpy, matplotlib
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def setup_complex_plane(ax, limit):
    ax.axhline(0, color="0.65", lw=0.9)
    ax.axvline(0, color="0.65", lw=0.9)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="实轴", ylabel="虚轴")
    ax.set_aspect("equal")


def vector(ax, value, color, label, offset=(0.06, 0.06)):
    ax.annotate("", xy=(value.real, value.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
    ax.text(value.real + offset[0], value.imag + offset[1], label,
            color=color, fontsize=12)


def fig012():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    value = 1 + 1j * np.sqrt(3)
    radius, theta = abs(value), np.angle(value)

    setup_complex_plane(axes[0], 2.35)
    vector(axes[0], value, "tab:red", r"$z=1+j\sqrt{3}$")
    axes[0].plot([1, 1], [0, np.sqrt(3)], "--", color="tab:blue", lw=1.5)
    axes[0].plot([0, 1], [np.sqrt(3), np.sqrt(3)], "--", color="tab:blue", lw=1.5)
    axes[0].text(0.5, -0.22, r"$a=1$", color="tab:blue", fontsize=12, ha="center")
    axes[0].text(1.10, 0.85, r"$b=\sqrt{3}$", color="tab:blue", fontsize=12)
    axes[0].set_title(r"(a) 直角坐标：横向走 $a$，纵向走 $b$")

    setup_complex_plane(axes[1], 2.35)
    circle = np.linspace(0, 2 * np.pi, 300)
    axes[1].plot(radius * np.cos(circle), radius * np.sin(circle), "--",
                 color="0.78", lw=1.0)
    vector(axes[1], value, "tab:red", r"$z=2e^{j\pi/3}$")
    axes[1].add_patch(Arc((0, 0), 1.25, 1.25, theta1=0,
                          theta2=np.degrees(theta), color="tab:green", lw=1.8))
    axes[1].text(0.62, 0.25, r"$\theta=\pi/3$", color="tab:green", fontsize=12)
    axes[1].text(0.28, 1.02, r"$r=2$", color="tab:red", fontsize=12, rotation=60)
    axes[1].text(0, -2.18, r"$a=r\cos\theta$，$b=r\sin\theta$",
                 fontsize=11, ha="center", color="0.25")
    axes[1].set_title(r"(b) 极坐标：长度为 $r$，方向为 $\theta$")

    fig.suptitle(r"同一个点，两套读法：$a+jb=r(\cos\theta+j\sin\theta)=re^{j\theta}$",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Fig012_从直角坐标到极坐标.png")


def fig013():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    alpha, beta = np.radians(35), np.radians(55)
    first, result = np.exp(1j * alpha), np.exp(1j * (alpha + beta))

    setup_complex_plane(axes[0], 1.55)
    circle = np.linspace(0, 2 * np.pi, 300)
    axes[0].plot(np.cos(circle), np.sin(circle), "--", color="0.75", lw=1.0)
    vector(axes[0], first, "tab:blue", r"$U(\alpha)$", (0.06, -0.13))
    vector(axes[0], result, "tab:red", r"$U(\alpha+\beta)$")
    axes[0].add_patch(Arc((0, 0), 0.7, 0.7, theta1=0, theta2=35,
                          color="tab:blue", lw=1.7))
    axes[0].text(0.39, 0.11, r"$\alpha$", color="tab:blue", fontsize=12)
    axes[0].add_patch(FancyArrowPatch(
        (1.18 * first.real, 1.18 * first.imag),
        (1.18 * result.real, 1.18 * result.imag),
        connectionstyle="arc3,rad=0.28", arrowstyle="-|>", mutation_scale=14,
        color="tab:red", lw=1.7))
    axes[0].text(0.62, 1.18, r"乘 $U(\beta)$：再转 $\beta$",
                 color="tab:red", fontsize=11, ha="center")
    axes[0].set_title(r"(a) 先转 $\alpha$，再转 $\beta$，角度会相加")

    axes[1].axis("off")
    axes[1].text(0.5, 0.88, "旋转的运算规律", fontsize=15, ha="center", weight="bold")
    formula_box(axes[1], 0.70, r"$U(\alpha)U(\beta)=U(\alpha+\beta)$",
                "#E8F3FF", "tab:blue")
    axes[1].annotate("结构完全相同", xy=(0.5, 0.40), xytext=(0.5, 0.55),
                     ha="center", fontsize=12,
                     arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.4))
    axes[1].text(0.5, 0.27, "指数函数的运算规律", fontsize=15,
                 ha="center", weight="bold")
    formula_box(axes[1], 0.09, r"$e^{j\alpha}e^{j\beta}=e^{j(\alpha+\beta)}$",
                "#FFF0E8", "tab:red")
    axes[1].set_title("(b) 指数记号天然把角度相加变成乘法")

    fig.suptitle(r"$e^{j\theta}$ 是“旋转 $\theta$”的代数记号，不是新的物理量",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Fig013_旋转乘法与指数记号.png")


def formula_box(ax, y, text, facecolor, edgecolor):
    ax.text(0.5, y, text, fontsize=19, ha="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=facecolor,
                      edgecolor=edgecolor))


def fig014():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    omega, phase = np.pi / 4, np.pi / 8
    samples = np.arange(8)
    values = np.exp(1j * (omega * samples + phase))
    colors = plt.cm.viridis(np.linspace(0.05, 0.90, len(samples)))

    setup_complex_plane(axes[0], 1.45)
    circle = np.linspace(0, 2 * np.pi, 300)
    axes[0].plot(np.cos(circle), np.sin(circle), "--", color="0.72", lw=1.0)
    for sample, value, color in zip(samples, values, colors):
        axes[0].plot(value.real, value.imag, "o", color=color, ms=7)
        axes[0].text(1.15 * value.real, 1.15 * value.imag, f"{sample}",
                     color=color, fontsize=10, ha="center", va="center")
    for start, end in zip(values[:-1], values[1:]):
        axes[0].add_patch(FancyArrowPatch(
            (1.04 * start.real, 1.04 * start.imag),
            (1.04 * end.real, 1.04 * end.imag),
            connectionstyle="arc3,rad=0.20", arrowstyle="-|>", mutation_scale=10,
            color="0.45", lw=1.0))
    axes[0].text(0, -1.34, r"每一步都乘 $e^{j\omega}$，即再转 $\omega=\pi/4$",
                 ha="center", fontsize=11)
    axes[0].set_title(r"(a) $z[n+1]=e^{j\omega}z[n]$：离散时间就是逐步旋转")

    dense_n = np.linspace(0, 7, 500)
    axes[1].plot(dense_n, np.cos(omega * dense_n + phase), color="0.78", lw=1.4,
                 label=r"连续投影 $\cos(\omega n+\varphi)$")
    axes[1].stem(samples, values.real, linefmt="tab:red", markerfmt="o", basefmt="0.5",
                 label=r"采样值 $\operatorname{Re}\{z[n]\}$")
    for sample, value, color in zip(samples, values.real, colors):
        axes[1].plot(sample, value, "o", color=color, ms=7, zorder=4)
    axes[1].axhline(0, color="0.65", lw=0.8)
    axes[1].set(xlabel="采样序号 n", ylabel="实部投影", ylim=(-1.35, 1.35))
    axes[1].grid(alpha=0.18)
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].set_title("(b) 旋转向量在实轴上的影子，就是实际余弦采样")

    fig.suptitle(r"DSP 读法：$z[n]=e^{j(\omega n+\varphi)}$ 保存完整旋转，实信号只取它的投影",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Fig014_DSP中的逐采样旋转.png")


def fig015():
    """用 U(30°)U(45°)=U(75°) 展示旋转角度相加。"""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.3))
    alpha, beta = np.radians(30), np.radians(45)
    first = np.exp(1j * alpha)
    multiplier = np.exp(1j * beta)
    product = first * multiplier
    circle = np.linspace(0, 2 * np.pi, 300)

    setup_complex_plane(axes[0], 1.45)
    axes[0].plot(np.cos(circle), np.sin(circle), "--", color="0.75", lw=1.0)
    vector(axes[0], first, "tab:blue", r"$U(30°)$", (0.06, -0.13))
    vector(axes[0], multiplier, "tab:green", r"$U(45°)$", (0.06, 0.06))
    axes[0].add_patch(Arc((0, 0), 0.65, 0.65, theta1=0, theta2=30,
                          color="tab:blue", lw=1.6))
    axes[0].add_patch(Arc((0, 0), 0.95, 0.95, theta1=0, theta2=45,
                          color="tab:green", lw=1.6))
    axes[0].text(-1.36, -1.30,
                 r"$U(30°)=0.866+j0.500$" + "\n" +
                 r"$U(45°)=0.707+j0.707$",
                 fontsize=11, color="0.25")
    axes[0].set_title("(a) 两个复数各自代表一个旋转角度")

    setup_complex_plane(axes[1], 1.45)
    axes[1].plot(np.cos(circle), np.sin(circle), "--", color="0.75", lw=1.0)
    vector(axes[1], first, "tab:blue", r"起点 $U(30°)$", (0.05, -0.14))
    vector(axes[1], product, "tab:red", r"终点 $U(75°)$", (0.05, 0.05))
    axes[1].add_patch(FancyArrowPatch(
        (1.12 * first.real, 1.12 * first.imag),
        (1.12 * product.real, 1.12 * product.imag),
        connectionstyle="arc3,rad=0.28", arrowstyle="-|>", mutation_scale=14,
        color="tab:green", lw=2.0))
    axes[1].text(0.78, 1.10, r"乘 $U(45°)$" + "\n" + r"$=$ 再转 $45°$",
                 color="tab:green", fontsize=11, ha="center")
    axes[1].text(0, -1.30, r"原角度 $30°$ $+$ 新旋转 $45°$ $=$ 结果 $75°$",
                 fontsize=11, ha="center", color="0.25")
    axes[1].set_title("(b) 乘法的含义：把第一个向量再旋转 45°")

    axes[2].axis("off")
    axes[2].text(0.5, 0.92, "把坐标真的乘一遍", fontsize=15,
                 ha="center", weight="bold")
    axes[2].text(
        0.5, 0.68,
        r"$(0.866+j0.500)(0.707+j0.707)$" + "\n\n" +
        r"$=(0.866\times0.707-0.500\times0.707)$" + "\n" +
        r"$\quad+j(0.866\times0.707+0.500\times0.707)$" + "\n\n" +
        r"$=0.259+j0.966$",
        fontsize=14, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#F5F5F5",
                  edgecolor="0.55"))
    axes[2].text(
        0.5, 0.25,
        r"$0.259=\cos75°$，$0.966=\sin75°$" + "\n\n" +
        r"所以结果 $=U(75°)$",
        fontsize=15, ha="center", color="tab:red",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF0E8",
                  edgecolor="tab:red"))
    axes[2].set_title("(c) 坐标相乘后的终点，确实落在 75°")

    fig.suptitle(r"具体例子：$U(30°)U(45°)=U(30°+45°)=U(75°)$",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "Fig015_30度乘45度等于旋转75度.png")


def save(fig, filename):
    output = os.path.join(FIG_DIR, filename)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print("saved:", output)


if __name__ == "__main__":
    fig012()
    fig013()
    fig014()
    fig015()
