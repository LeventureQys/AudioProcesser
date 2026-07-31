# -*- coding: utf-8 -*-
"""生成 Fig006（幅度相位交换实验）。

用真实语音做实验：取仓库 Test_Audio 中两段 16kHz 语音，整段 FFT 后
互换幅度谱与相位谱再逆变换，对比语谱图，验证"相位携带时间结构"。

用法:  python generate_fig006_phase_swap.py
输入:  ../../../../Test_Audio/AudioSample-16000hz/1_01.wav（语音 A，只读）
       ../../../../Test_Audio/AudioSample-16000hz/2_01.wav（语音 B，只读）
输出:  ../figures/Fig006_幅度相位交换实验.png
       ../outputs/A幅度_B相位.wav, ../outputs/B幅度_A相位.wav（可试听）
依赖:  numpy, scipy, matplotlib
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
OUT_DIR = os.path.join(HERE, "..", "outputs")
AUDIO_DIR = os.path.join(HERE, "..", "..", "..", "..",
                         "Test_Audio", "AudioSample-16000hz")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SEG_SECONDS = 4.0  # 分析时长
B_OFFSET_SECONDS = 2.0  # 语音 B 从 2s 处开始取，避免两段音频的静音/发声段恰好对齐


def load_mono(path, n_samples, offset=0):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float64)
    x = x / (np.max(np.abs(x)) + 1e-12)
    return fs, x[offset:offset + n_samples]


def main():
    fs_probe, _ = wavfile.read(os.path.join(AUDIO_DIR, "1_01.wav"))
    n = int(SEG_SECONDS * fs_probe)
    fs, a = load_mono(os.path.join(AUDIO_DIR, "1_01.wav"), n)
    fs2, b = load_mono(os.path.join(AUDIO_DIR, "2_01.wav"), n,
                       offset=int(B_OFFSET_SECONDS * fs_probe))
    assert fs == fs2, "两段音频采样率必须一致"
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    # 整段 FFT，互换幅度与相位
    A, B = np.fft.fft(a), np.fft.fft(b)
    hyb_a_mag = np.fft.ifft(np.abs(A) * np.exp(1j * np.angle(B))).real  # A幅度+B相位
    hyb_b_mag = np.fft.ifft(np.abs(B) * np.exp(1j * np.angle(A))).real  # B幅度+A相位

    for name, sig in [("A幅度_B相位.wav", hyb_a_mag), ("B幅度_A相位.wav", hyb_b_mag)]:
        pcm = (sig / (np.max(np.abs(sig)) + 1e-12) * 32000).astype(np.int16)
        wavfile.write(os.path.join(OUT_DIR, name), fs, pcm)

    # 四张语谱图统一色标
    panels = [
        (a, "(a) 语音 A 原始"),
        (b, "(b) 语音 B 原始"),
        (hyb_a_mag, "(c) 合成：|A| 的幅度 + ∠B 的相位"),
        (hyb_b_mag, "(d) 合成：|B| 的幅度 + ∠A 的相位"),
    ]
    specs = []
    for sig, _ in panels:
        f, t, S = spectrogram(sig, fs, nperseg=512, noverlap=384)
        specs.append((f, t, 10 * np.log10(S + 1e-12)))
    vmax = max(s[2].max() for s in specs)
    vmin = vmax - 80

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=True, sharey=True)
    for ax, (f, t, Sdb), (_, title) in zip(axes.flat, specs, panels):
        pcm = ax.pcolormesh(t, f / 1000, Sdb, vmin=vmin, vmax=vmax,
                            shading="gouraud", cmap="magma")
        ax.set_title(title, fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("频率（kHz）")
    for ax in axes[1, :]:
        ax.set_xlabel("时间（s）")
    cbar = fig.colorbar(pcm, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("功率（dB）")
    fig.suptitle(
        "整段 FFT 幅度/相位交换实验（16kHz 语音，A 取 0–4s，B 取 2–6s）："
        "合成信号的时间结构跟着相位提供者走", fontsize=13)
    out = os.path.join(FIG_DIR, "Fig006_幅度相位交换实验.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
