# -*- coding: utf-8 -*-
"""生成第三章 DFT 入门配图。

用法: python generate_fig016_fig017_fig018.py
输出: ../figures/Fig016_DFT要回答什么.png
      ../figures/Fig017_相关器如何识别频率.png
      ../figures/Fig018_复指数模板拆成两个相关器.png
依赖: numpy, matplotlib
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


def save(fig, filename):
    output = os.path.join(FIG_DIR, filename)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print("saved:", output)


def fig016():
    """DFT 将混合采样转换为各候选频率的系数。"""
    sample_count = 32
    sample_index = np.arange(sample_count)
    low = 1.0 * np.cos(2 * np.pi * 3 * sample_index / sample_count)
    high = 0.55 * np.cos(2 * np.pi * 7 * sample_index / sample_count + np.pi / 4)
    signal = low + high
    spectrum = np.fft.rfft(signal)
    amplitudes = 2 * np.abs(spectrum) / sample_count
    amplitudes[0] /= 2

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    axes[0, 0].stem(sample_index, low, basefmt="0.7")
    axes[0, 0].set_title("(a) 成分 1：32 点内转 3 圈，幅度 1.0")
    axes[0, 1].stem(sample_index, high, basefmt="0.7", linefmt="tab:green",
                    markerfmt="o")
    axes[0, 1].set_title("(b) 成分 2：32 点内转 7 圈，幅度 0.55，相位 45°")
    axes[1, 0].stem(sample_index, signal, basefmt="0.7", linefmt="tab:purple",
                    markerfmt="o")
    axes[1, 0].set_title("(c) 实际输入：两种频率相加后，时域波形不再容易辨认")
    axes[1, 1].stem(np.arange(len(amplitudes)), amplitudes, basefmt="0.7",
                    linefmt="tab:red", markerfmt="o")
    axes[1, 1].annotate("第 3 号频率：幅度 1.0", xy=(3, amplitudes[3]),
                        xytext=(5, 1.12), arrowprops=dict(arrowstyle="->"), fontsize=10)
    axes[1, 1].annotate("第 7 号频率：幅度 0.55", xy=(7, amplitudes[7]),
                        xytext=(9, 0.75), arrowprops=dict(arrowstyle="->"), fontsize=10)
    axes[1, 1].set(xlabel="频率编号 k", ylabel="幅度", ylim=(0, 1.3))
    axes[1, 1].set_title("(d) DFT 的答案：逐个报告每个候选频率含有多少")
    for axis in axes.flat[:3]:
        axis.set(xlabel="采样序号 n", ylabel="采样值")
        axis.axhline(0, color="0.7", lw=0.8)
    fig.suptitle("DFT 的目的：把难读的混合波形，转换成一张频率成分清单", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Fig016_DFT要回答什么.png")


def fig017():
    """同频模板乘积累积，异频模板乘积抵消。"""
    sample_count = 32
    sample_index = np.arange(sample_count)
    signal_bin = 3
    signal = np.cos(2 * np.pi * signal_bin * sample_index / sample_count)
    correct_template = np.cos(2 * np.pi * 3 * sample_index / sample_count)
    wrong_template = np.cos(2 * np.pi * 5 * sample_index / sample_count)
    correct_product = signal * correct_template
    wrong_product = signal * wrong_template

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4), sharex=True)
    axes[0, 0].plot(sample_index, signal, "o-", color="0.35", label="输入 k=3")
    axes[0, 0].plot(sample_index, correct_template, "--", color="tab:blue",
                    label="测试模板 k=3")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_title("(a) 频率相同：波峰波谷基本对齐")
    axes[1, 0].bar(sample_index, correct_product, color="tab:red", alpha=0.75)
    axes[1, 0].axhline(0, color="0.5", lw=0.8)
    axes[1, 0].set_title(f"(c) 逐点相乘后大多为正，求和 = {correct_product.sum():.1f}")

    axes[0, 1].plot(sample_index, signal, "o-", color="0.35", label="输入 k=3")
    axes[0, 1].plot(sample_index, wrong_template, "--", color="tab:green",
                    label="测试模板 k=5")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_title("(b) 频率不同：有时同号，有时异号")
    product_colors = np.where(wrong_product >= 0, "tab:red", "tab:blue")
    axes[1, 1].bar(sample_index, wrong_product, color=product_colors, alpha=0.75)
    axes[1, 1].axhline(0, color="0.5", lw=0.8)
    axes[1, 1].set_title(f"(d) 正负面积互相抵消，求和 ≈ {wrong_product.sum():.1f}")
    for axis in axes.flat:
        axis.set_ylabel("数值")
    axes[1, 0].set_xlabel("采样序号 n")
    axes[1, 1].set_xlabel("采样序号 n")
    fig.suptitle("相关检测的核心：与模板逐点相乘再求和；越像，累积结果越大", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Fig017_相关器如何识别频率.png")


def fig018():
    """将一个 DFT bin 的复指数模板拆成 cos 与 -sin 两路。"""
    sample_count, frequency_bin = 16, 3
    sample_index = np.arange(sample_count)
    angle = 2 * np.pi * frequency_bin * sample_index / sample_count
    phase = np.radians(50)
    signal = np.cos(angle + phase)
    cosine_template = np.cos(angle)
    negative_sine_template = -np.sin(angle)
    real_terms = signal * cosine_template
    imaginary_terms = signal * negative_sine_template
    coefficient = np.sum(signal * np.exp(-1j * angle))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6), sharex=True)
    axes[0, 0].stem(sample_index, signal, basefmt="0.7")
    axes[0, 0].set_title("(a) 输入：含有 k=3，但初相位不是 0°")
    axes[0, 1].plot(sample_index, cosine_template, "o-", color="tab:blue",
                    label="实部模板 cos")
    axes[0, 1].plot(sample_index, negative_sine_template, "o-", color="tab:green",
                    label="虚部模板 -sin")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_title(r"(b) $e^{-j\theta}=\cos\theta-j\sin\theta$：一条复模板就是两条实模板")
    axes[1, 0].bar(sample_index - 0.18, real_terms, width=0.36, color="tab:blue",
                   label=f"求和 = {real_terms.sum():.2f}")
    axes[1, 0].bar(sample_index + 0.18, imaginary_terms, width=0.36,
                   color="tab:green", label=f"求和 = {imaginary_terms.sum():.2f}")
    axes[1, 0].axhline(0, color="0.5", lw=0.8)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_title("(c) 两路分别逐点相乘并求和，得到两个坐标")

    axes[1, 1].axhline(0, color="0.65", lw=0.8)
    axes[1, 1].axvline(0, color="0.65", lw=0.8)
    axes[1, 1].annotate("", xy=(coefficient.real, coefficient.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2.4))
    axes[1, 1].plot(coefficient.real, coefficient.imag, "o", color="tab:red")
    axes[1, 1].text(coefficient.real + 0.25, coefficient.imag,
                    f"X[3] = {coefficient.real:.2f} + j{coefficient.imag:.2f}\n"
                    f"模 = {abs(coefficient):.1f}，角度 = {np.degrees(np.angle(coefficient)):.0f}°",
                    fontsize=11, color="tab:red")
    axes[1, 1].set(xlabel="cos 相关结果（实部）", ylabel="-sin 相关结果（虚部）",
                   xlim=(-9, 9), ylim=(-9, 9))
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("(d) 把两路结果装进一个复数：同时保存强度和相位")
    for axis in axes.flat[:3]:
        axis.set_ylabel("数值")
    axes[1, 0].set_xlabel("采样序号 n")
    fig.suptitle("为什么按欧拉公式拆 DFT：复指数模板本来就由 cos 与 -sin 两个坐标组成", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Fig018_复指数模板拆成两个相关器.png")


if __name__ == "__main__":
    fig016()
    fig017()
    fig018()
