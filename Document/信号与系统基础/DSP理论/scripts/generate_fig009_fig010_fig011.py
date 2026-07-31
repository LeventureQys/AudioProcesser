# -*- coding: utf-8 -*-
"""生成 Fig009（乘j等于旋转90度）、Fig010（e指数的两种跑法）、Fig011（复数变换几何词典）。

用法:  python generate_fig009_fig010_fig011.py
输出:  ../figures/Fig009_乘j等于旋转90度.png
       ../figures/Fig010_e指数的两种跑法.png
       ../figures/Fig011_复数变换几何词典.png
依赖:  numpy, matplotlib
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _vec(ax, z, color, lw=2.0, ls="-"):
    """从原点画一个复数向量。"""
    ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0))


def _unit_circle(ax, r=1.0):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(th), r * np.sin(th), "--", color="0.75", lw=1.0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_aspect("equal")


# ---------------------------------------------------------------- Fig009
def fig009():
    """乘 j = 逆时针旋转 90°。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.6))

    # ---- (a) 1 → j → -1 → -j 的四步循环
    _unit_circle(ax1)
    pts = [1 + 0j, 1j, -1 + 0j, -1j]
    labels = ["$1$", "$j$", "$-1$", "$-j$"]
    offs = [(0.10, -0.02), (0.06, 0.10), (-0.24, -0.02), (0.06, -0.16)]
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, 4))
    for z, lab, off, c in zip(pts, labels, offs, colors):
        _vec(ax1, z, c)
        ax1.text(z.real + off[0], z.imag + off[1], lab, fontsize=14, color=c)
    # 相邻两点间的弧形箭头，标注 ×j
    for a, b in zip(pts, pts[1:] + pts[:1]):
        arr = FancyArrowPatch((1.22 * a.real, 1.22 * a.imag),
                              (1.22 * b.real, 1.22 * b.imag),
                              connectionstyle="arc3,rad=0.35",
                              arrowstyle="-|>", mutation_scale=13,
                              color="tab:red", lw=1.4)
        ax1.add_patch(arr)
        mid = 1.45 * (a + b) / abs(a + b)
        ax1.text(mid.real - 0.10, mid.imag - 0.05, r"$\times j$",
                 fontsize=11, color="tab:red")
    ax1.set_xlim(-1.75, 1.75)
    ax1.set_ylim(-1.75, 1.75)
    ax1.set_xlabel("实轴")
    ax1.set_ylabel("虚轴")
    ax1.set_title("(a) 连乘四次 $j$ 转回原点：$j^2=-1$ 就是“转两次90°=转180°”")

    # ---- (b) 任意向量乘 j：原地左转 90°，长度不变
    _unit_circle(ax2, r=abs(1.15 + 0.55j))
    z = 1.15 + 0.55j
    jz = 1j * z
    _vec(ax2, z, "tab:blue")
    _vec(ax2, jz, "tab:red")
    ax2.text(z.real + 0.06, z.imag, r"$z = a+jb$", fontsize=12, color="tab:blue")
    ax2.text(jz.real - 1.05, jz.imag - 0.05, r"$jz = -b+ja$",
             fontsize=12, color="tab:red")
    # 直角记号
    u, v = z / abs(z), jz / abs(jz)
    corner = 0.17 * (u + v)
    ax2.plot([0.17 * u.real, corner.real, 0.17 * v.real],
             [0.17 * u.imag, corner.imag, 0.17 * v.imag],
             color="0.35", lw=1.2)
    arr = FancyArrowPatch((1.12 * z.real, 1.12 * z.imag),
                          (1.12 * jz.real, 1.12 * jz.imag),
                          connectionstyle="arc3,rad=0.30",
                          arrowstyle="-|>", mutation_scale=13,
                          color="tab:red", lw=1.4)
    ax2.add_patch(arr)
    ax2.text(0.30, 1.28, r"$\times j$", fontsize=12, color="tab:red")
    ax2.set_xlim(-1.75, 1.75)
    ax2.set_ylim(-1.75, 1.75)
    ax2.set_xlabel("实轴")
    ax2.set_ylabel("虚轴")
    ax2.set_title("(b) 对任意 $z$：乘 $j$ 都是原地左转90°，长度不变\n（横纵坐标互换并变号：$a+jb \\to -b+ja$）")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig009_乘j等于旋转90度.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


# ---------------------------------------------------------------- Fig010
def fig010():
    """e 的两种跑法：实指数沿射线逃逸，虚指数绕圈。"""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))

    # ---- (a) e^{at}, a 为正实数：速度与位置同向
    ax = axes[0]
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ts = np.array([0.0, 0.7, 1.4])
    xs = np.exp(0.8 * ts)
    ax.plot(xs, np.zeros_like(xs), "o", color="tab:blue", ms=7, zorder=3)
    for x in xs:
        arr = FancyArrowPatch((x, 0), (x + 0.45 * 0.8 * x, 0),
                              arrowstyle="-|>", mutation_scale=15,
                              color="tab:red", lw=1.8)
        ax.add_patch(arr)
    for x, t in zip(xs, ts):
        ax.text(x, -0.28, f"$t={t:.1f}$", fontsize=9,
                color="0.35", ha="center")
    ax.text(1.6, 0.75, "速度 = $a\\,\\times$位置\n始终同方向\n→ 越跑越快，一去不回",
            fontsize=11, color="tab:red")
    ax.set_xlim(-0.5, 4.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("实轴")
    ax.set_ylabel("虚轴")
    ax.set_title("(a) $e^{at}$（$a$ 为正实数）：沿实轴指数逃逸")

    # ---- (b) e^{jθ}：速度垂直位置
    ax = axes[1]
    _unit_circle(ax)
    thetas = np.radians([15, 75, 150, 210, 285])
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(thetas)))
    for th, c in zip(thetas, colors):
        z = np.exp(1j * th)
        _vec(ax, z, "0.55", lw=1.3)
        v = 1j * z * 0.45                       # 速度 = jz，画短一点
        arr = FancyArrowPatch((z.real, z.imag),
                              (z.real + v.real, z.imag + v.imag),
                              arrowstyle="-|>", mutation_scale=15,
                              color=c, lw=2.0)
        ax.add_patch(arr)
        ax.plot(z.real, z.imag, "o", color=c, ms=6, zorder=3)
    ax.text(0.0, -1.30, "速度 = $j\\,\\times$位置，始终垂直半径 → 半径锁死，只能转圈",
            fontsize=10.5, color="0.2", ha="center", va="top")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    ax.set_xlabel("实轴")
    ax.set_ylabel("虚轴")
    ax.set_title("(b) $e^{j\\theta}$：速度处处垂直位置 → 匀速圆周")

    # ---- (c) (1+jθ/N)^N 的折线逼近
    ax = axes[2]
    _unit_circle(ax)
    theta = 2 * np.pi / 3                        # 120°
    for N, c in zip([4, 12, 48], ["tab:red", "tab:orange", "tab:green"]):
        step = 1 + 1j * theta / N
        zs = step ** np.arange(N + 1)            # 1, step, step², ...
        ax.plot(zs.real, zs.imag, "o-", color=c, ms=3.5, lw=1.4,
                label=f"$N={N}$，终点模 {abs(zs[-1]):.3f}")
    target = np.exp(1j * theta)
    ax.plot(target.real, target.imag, "*", color="k", ms=15, zorder=4,
            label=r"极限 $e^{j\,120°}$（模 1）")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    ax.set_xlabel("实轴")
    ax.set_ylabel("虚轴")
    ax.set_title("(c) $(1+j\\theta/N)^N$：每步小转一点\n$N$ 越大越贴住单位圆")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig010_e指数的两种跑法.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


# ---------------------------------------------------------------- Fig011
def fig011():
    """常见复数变换的几何词典。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.0))

    # ---- (a) 乘 e^{jφ}：纯旋转
    ax = axes[0, 0]
    z = 1.25 * np.exp(1j * np.radians(20))
    phi = np.radians(55)
    w = z * np.exp(1j * phi)
    _unit_circle(ax, r=abs(z))
    _vec(ax, z, "tab:blue")
    _vec(ax, w, "tab:red")
    ax.text(z.real + 0.05, z.imag - 0.10, "$z$", fontsize=13, color="tab:blue")
    ax.text(w.real - 0.02, w.imag + 0.10, r"$e^{j\varphi}z$", fontsize=13,
            color="tab:red")
    arr = FancyArrowPatch((1.12 * z.real, 1.12 * z.imag),
                          (1.12 * w.real, 1.12 * w.imag),
                          connectionstyle="arc3,rad=0.25",
                          arrowstyle="-|>", mutation_scale=13,
                          color="tab:red", lw=1.4)
    ax.add_patch(arr)
    ax.text(1.05, 0.95, r"$\varphi$", fontsize=12, color="tab:red")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_title(r"(a) 乘 $e^{j\varphi}$：纯旋转，长度不变")

    # ---- (b) 乘 M=|M|e^{jφ}：缩放+旋转一步完成
    ax = axes[0, 1]
    X = 1.4 * np.exp(1j * np.radians(15))
    M = 0.6 * np.exp(1j * np.radians(50))
    _unit_circle(ax, r=abs(X))
    _unit_circle(ax, r=abs(M * X))
    _vec(ax, X, "tab:blue")
    _vec(ax, M * X, "tab:red")
    ax.text(X.real + 0.05, X.imag - 0.08, "$X$", fontsize=13, color="tab:blue")
    ax.text((M * X).real - 0.15, (M * X).imag + 0.13, r"$MX$", fontsize=13,
            color="tab:red")
    ax.text(-1.68, -1.45,
            r"$M=|M|e^{j\theta_M}$：模相乘、角相加" + "\n"
            r"$MX = |M||X|\,e^{j(\angle X+\theta_M)}$",
            fontsize=11, color="0.25")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_title(r"(b) 乘复数 $M$：缩放 $|M|$ 倍并旋转 $\theta_M$")

    # ---- (c) 共轭：关于实轴镜像
    ax = axes[1, 0]
    z = 1.3 * np.exp(1j * np.radians(40))
    _unit_circle(ax, r=abs(z))
    _vec(ax, z, "tab:blue")
    _vec(ax, np.conj(z), "tab:green")
    ax.plot([z.real, z.real], [z.imag, -z.imag], ":", color="0.4", lw=1.3)
    ax.text(z.real + 0.05, z.imag, r"$z = re^{j\theta}$", fontsize=13,
            color="tab:blue")
    ax.text(z.real + 0.05, -z.imag - 0.10, r"$z^* = re^{-j\theta}$",
            fontsize=13, color="tab:green")
    ax.text(-1.68, -1.45, r"$z\,z^* = r^2 = |z|^2$：旋转正反抵消，只剩长度平方",
            fontsize=11, color="0.25")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_title(r"(c) 共轭 $z^*$：关于实轴翻镜像 = 反向旋转")

    # ---- (d) 幂：棣莫弗定理
    ax = axes[1, 1]
    _unit_circle(ax)
    th0 = np.radians(40)
    ns = np.arange(0, 8)
    colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(ns)))
    for n, c in zip(ns, colors):
        z = np.exp(1j * n * th0)
        _vec(ax, z, c, lw=1.5)
        ax.text(1.14 * z.real - 0.06, 1.14 * z.imag - 0.03, f"$n={n}$",
                fontsize=9, color=c)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_title(r"(d) 幂 $(e^{j\theta})^n = e^{jn\theta}$：转 $n$ 次 $\theta$ 就是转 $n\theta$"
                 "\n（图中 $\\theta=40°$，转 8 次回不到起点但均匀爬圆）")

    for ax in axes.flat:
        ax.set_xlabel("实轴")
        ax.set_ylabel("虚轴")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig011_复数变换几何词典.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    fig009()
    fig010()
    fig011()
