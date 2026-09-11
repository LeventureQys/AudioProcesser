# -*- coding: utf-8 -*-
"""Generate figures for the article:
"一个基本问题 - 当我们谈论移植的时候，我们聊的是 ONNX 还是自定义算子?"

Run:  python make_figures.py
Outputs PNG files next to this script.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))


def rbox(ax, x, y, w, h, text, fc="#ffffff", ec="#37474f", fs=11, tc="#212121", lw=1.4, weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.45, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, color="#455a64", lw=1.6, style="-|>", ls="solid"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                        lw=lw, linestyle=ls, mutation_scale=14, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


# ----------------------------------------------------------------------------
# Figure 1: ONNX as the hub (N x M  ->  N + M)
# ----------------------------------------------------------------------------
def fig_onnx_hub():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(6.5, 9.55, "ONNX：把  N × M  的对接问题变成  N + M",
            ha="center", va="center", fontsize=15, fontweight="bold")

    frameworks = ["PyTorch", "TensorFlow / Keras", "PaddlePaddle", "JAX / Others"]
    engines = ["ONNX Runtime\n(CPU / CUDA / DML)", "TensorRT", "OpenVINO",
               "CoreML", "QNN / Hexagon\n(各家 NPU SDK)"]

    # left column
    for i, name in enumerate(frameworks):
        y = 7.3 - i * 1.55
        rbox(ax, 0.7, y - 0.55, 2.9, 1.1, name, fc="#e3f2fd", ec="#1565c0")
        arrow(ax, 3.65, y, 5.05, 5.05)

    # right column
    for i, name in enumerate(engines):
        y = 7.9 - i * 1.5
        rbox(ax, 9.4, y - 0.55, 3.1, 1.1, name, fc="#ede7f6", ec="#5e35b1", fs=10)
        arrow(ax, 7.95, 5.05, 9.35, y)

    # center hub
    rbox(ax, 5.05, 3.4, 2.9, 3.3,
         "ONNX\n\n标准算子集\n(opset)\n计算图格式\n(protobuf)\n类型与布局规范",
         fc="#fff8e1", ec="#e65100", fs=11, lw=2.2, weight="bold")

    ax.text(6.5, 2.55, "一个 .onnx 文件 ≈ 模型世界的 ELF：\n协议只有一份，生产者与消费者互不感知",
            ha="center", va="center", fontsize=10.5, color="#5d4037", linespacing=1.6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "onnx_hub.png"), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: operator fusion, before / after
# ----------------------------------------------------------------------------
def fig_graph_fusion():
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.6))

    # ---- top: before fusion ---------------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("融合前：逐算子独立执行，中间结果反复访存", fontsize=13, fontweight="bold", pad=8)

    ops = ["Conv", "BatchNorm", "ReLU", "Gemm", "Add"]
    w, h = 1.75, 0.95
    x0, ymid = 2.6, 1.15
    rbox(ax, 0.35, ymid - 0.35, 1.05, 0.7, "x", fc="#eceff1", ec="#607d8b", fs=11)
    centers = []
    for i, op in enumerate(ops):
        x = x0 + i * (w + 0.55)
        rbox(ax, x, ymid - h / 2, w, h, op, fc="#e3f2fd", ec="#1565c0")
        centers.append((x, x + w))
    rbox(ax, 13.6, ymid - 0.35, 1.05, 0.7, "y", fc="#eceff1", ec="#607d8b", fs=11)

    arrow(ax, 1.45, ymid, x0, ymid)
    for i in range(len(ops) - 1):
        x1 = centers[i][1]
        x2 = centers[i + 1][0]
        arrow(ax, x1, ymid, x2, ymid)
    arrow(ax, centers[-1][1], ymid, 13.6, ymid)

    # DRAM round trips
    for i in range(len(ops) - 1):
        xm = (centers[i][1] + centers[i + 1][0]) / 2
        arrow(ax, xm, ymid + 0.62, xm, ymid + 1.28, color="#c62828", lw=1.4)
        arrow(ax, xm, ymid + 1.28, xm, ymid + 0.62, color="#c62828", lw=1.4)
    ax.text((centers[0][1] + centers[-1][0]) / 2, ymid + 1.62,
            "红↑↓：每个中间张量都要写回内存、再读回来 —— 访存开销随算子数量线性增长",
            ha="center", va="center", fontsize=10, color="#c62828")

    # ---- bottom: after fusion --------------------------------------------
    ax = axes[1]
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("融合后：中间结果留在寄存器 / L1，一次写出", fontsize=13, fontweight="bold", pad=8)

    ymid2 = 1.35
    rbox(ax, 0.35, ymid2 - 0.35, 1.05, 0.7, "x", fc="#eceff1", ec="#607d8b", fs=11)
    rbox(ax, 2.6, ymid2 - 0.75, 4.6, 1.5,
         "Fused(Conv + BN + ReLU)\n单 kernel", fc="#e8f5e9", ec="#2e7d32", fs=11.5, lw=2.0)
    rbox(ax, 8.5, ymid2 - 0.75, 3.6, 1.5,
         "Fused(Gemm + Add)\n单 kernel", fc="#e8f5e9", ec="#2e7d32", fs=11.5, lw=2.0)
    rbox(ax, 13.6, ymid2 - 0.35, 1.05, 0.7, "y", fc="#eceff1", ec="#607d8b", fs=11)

    arrow(ax, 1.45, ymid2, 2.6, ymid2)
    arrow(ax, 7.25, ymid2, 8.5, ymid2, color="#2e7d32", lw=2.0)
    arrow(ax, 12.15, ymid2, 13.6, ymid2)

    ax.text(7.87, ymid2 + 0.72, "寄存器传递", ha="center", fontsize=10, color="#2e7d32")
    ax.text(7.5, 0.42,
            "BN 推理态可折叠为逐通道 scale/shift，scale 并入 Conv 权重、shift 并入偏置 —— 三层在数学上严格等价于一层",
            ha="center", fontsize=10, color="#455a64")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "graph_fusion.png"), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: magnitude comparison (cold start / binary size / runtime memory)
# ----------------------------------------------------------------------------
def fig_deploy_metrics():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    colors = ["#1565c0", "#2e7d32"]

    def bar(ax, labels, values, unit, fmts, title):
        xs = np.arange(len(values))
        ax.bar(xs, values, color=colors, width=0.55, edgecolor="white")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=10.5)
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.set_axisbelow(True)
        for x, v, f in zip(xs, values, fmts):
            ax.text(x, v * 1.35, f, ha="center", fontsize=11, fontweight="bold")

    bar(axes[0], ["ONNX Runtime\nSessionInit", "手写 C++\n启动 + 载权重"],
        [200.0, 1.0], "毫秒 (log)", ["~10² ms", "~1 ms"],
        "冷启动 / 首帧延迟")
    bar(axes[1], ["ONNX Runtime\n(含引擎库)", "手写 C++\n代码段 + 权重"],
        [30.0, 0.05], "MB (log)", ["数 MB ~ 数十 MB", "几十 KB"],
        "包体积")
    bar(axes[2], ["ONNX Runtime\n模型 + Arena", "手写 C++\n静态平铺缓冲"],
        [150.0, 10.0], "MB (log)", ["模型 + 内存池", "≈ 权重 + 零散头寸"],
        "运行时内存占用")

    fig.suptitle("两条路线的工程量级对比（示意数值，注意对数轴）", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "deploy_metrics.png"), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: scenario-driven decision map
# ----------------------------------------------------------------------------
def fig_tradeoff_map():
    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    # gradient background: top-left (handwritten) -> bottom-right (onnx)
    cmap = LinearSegmentedColormap.from_list("route", ["#2e7d32", "#9e9e9e", "#1565c0"])
    grad = np.zeros((200, 200, 4))
    yy, xx = np.mgrid[0:200, 0:200]
    v = (xx / 199 * 0.65 + (1 - yy / 199) * 0.35)  # 0 = handwritten corner
    for i in range(200):
        for j in range(200):
            c = cmap(v[i, j])
            grad[i, j] = [c[0], c[1], c[2], 0.16]
    ax.imshow(grad, extent=[0, 10, 0, 10], origin="lower", aspect="auto", zorder=0)

    # dividing line
    xs = np.linspace(0, 10, 100)
    ax.plot(xs, 10 - xs, ls="--", color="#616161", lw=1.8, zorder=1)

    pts = [
        (1.8, 8.2, "端上实时小模块\n拉起即出首帧", "#1b5e20"),
        (2.6, 6.2, "固定流水线信号处理\n(如音频前处理)", "#1b5e20"),
        (8.0, 2.4, "服务端大模型 / LLM\n持续迭代", "#0d47a1"),
        (8.9, 4.2, "多硬件分发\nCPU / GPU / NPU", "#0d47a1"),
    ]
    for x, y, s, c in pts:
        ax.scatter([x], [y], s=90, color=c, zorder=3, edgecolor="white", lw=1.2)
        ax.text(x, y + 0.45, s, ha="center", fontsize=10.5, color=c, linespacing=1.4, zorder=3)

    # middle zone
    ax.scatter([5.05], [5.05], s=110, color="#e65100", marker="D", zorder=3,
               edgecolor="white", lw=1.2)
    ax.text(5.05, 5.55, "混合：主干 ONNX +\n手写前后处理 / Custom Op", ha="center",
            fontsize=10.5, color="#e65100", linespacing=1.4, zorder=3)

    ax.text(1.15, 9.35, "手写算子占优", fontsize=13, fontweight="bold", color="#1b5e20")
    ax.text(8.85, 0.65, "ONNX / 引擎路线占优", fontsize=13, fontweight="bold",
            color="#0d47a1", ha="right")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([0.5, 5, 9.5])
    ax.set_xticklabels(["网络小\nshape 固定", "中等规模", "网络大\nshape 多变 · 迭代频繁"], fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel("网络规模 / shape 多样性 / 迭代与分发需求  →", fontsize=11.5)
    ax.set_ylabel("冷启动 · 体积 · 内存 敏感度  →", fontsize=11.5)
    ax.set_title("没有通吃的方案：场景决定路线", fontsize=14, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tradeoff_map.png"), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_onnx_hub()
    fig_graph_fusion()
    fig_deploy_metrics()
    fig_tradeoff_map()
    print("figures saved to", OUT)
