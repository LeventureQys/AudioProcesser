# -*- coding: utf-8 -*-
"""生成 Fig001（复平面与旋转向量）与 Fig002（欧拉公式与螺旋线）。

用法:  python generate_fig001_fig002.py
输出:  ../figures/Fig001_复平面与旋转向量.png
       ../figures/Fig002_欧拉公式与螺旋线.png
依赖:  numpy, matplotlib
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------- Fig001
def fig001():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # ---- 左：同一个复数的两种读法 ----
    a, b = 1.2, 0.9
    r = np.hypot(a, b)
    theta = np.degrees(np.arctan2(b, a))

    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.axvline(0, color="0.6", lw=0.8)
    # 单位刻度网格淡化
    ax1.annotate(
        "", xy=(a, b), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2.2),
    )
    ax1.plot([a, a], [0, b], "--", color="tab:blue", lw=1.4)
    ax1.plot([0, a], [b, b], "--", color="tab:blue", lw=1.4)
    ax1.plot(a, b, "o", color="tab:red", ms=7, zorder=5)
    ax1.add_patch(
        Arc((0, 0), 0.9, 0.9, angle=0, theta1=0, theta2=theta, color="tab:green", lw=1.8)
    )
    ax1.text(0.50, 0.14, r"$\theta=\arctan\frac{b}{a}$", color="tab:green", fontsize=12)
    ax1.text(a / 2 - 0.12, -0.16, r"实部 $a$", color="tab:blue", fontsize=12)
    ax1.text(a + 0.05, b / 2, r"虚部 $b$", color="tab:blue", fontsize=12)
    ax1.text(a / 2 - 0.30, b / 2 + 0.14, r"$r=\sqrt{a^2+b^2}$", color="tab:red",
             fontsize=12, rotation=theta)
    ax1.text(a + 0.06, b + 0.08, r"$z=a+jb=r\,e^{j\theta}$", fontsize=13)
    ax1.set_xlim(-0.35, 2.0)
    ax1.set_ylim(-0.35, 1.45)
    ax1.set_aspect("equal")
    ax1.set_xlabel("实轴")
    ax1.set_ylabel("虚轴")
    ax1.set_title("(a) 直角坐标 $(a,b)$ 与极坐标 $(r,\\theta)$：同一个点的两种读法")

    # ---- 右：匀速旋转向量的投影是正弦波 ----
    cx, R = -2.2, 1.0  # 圆心与半径
    circ = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(cx + R * np.cos(circ), R * np.sin(circ), color="0.55", lw=1.2)
    ax2.axhline(0, color="0.75", lw=0.8)

    t = np.linspace(0, 4 * np.pi, 400)
    ax2.plot(t, np.sin(t), color="tab:red", lw=1.8)

    marks = np.array([0, 60, 120, 210, 300]) * np.pi / 180
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(marks)))
    for th, c in zip(marks, colors):
        px, py = cx + R * np.cos(th), R * np.sin(th)
        ax2.plot([cx, px], [0 if th == 0 else py * 0 + 0, py * 0], lw=0)  # 占位
        ax2.annotate("", xy=(px, py), xytext=(cx, 0),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
        ax2.plot([px, th], [py, py], "--", color=c, lw=1.0, alpha=0.85)
        ax2.plot(th, np.sin(th), "o", color=c, ms=6, zorder=5)
    # 旋转方向
    ax2.add_patch(Arc((cx, 0), 2.7, 2.7, angle=0, theta1=25, theta2=95,
                      color="0.35", lw=1.4))
    ax2.annotate("", xy=(cx + 1.35 * np.cos(np.radians(100)), 1.35 * np.sin(np.radians(100))),
                 xytext=(cx + 1.35 * np.cos(np.radians(92)), 1.35 * np.sin(np.radians(92))),
                 arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.4))
    ax2.text(cx - 0.45, 1.45, r"角速度 $\omega$", fontsize=11, color="0.25")
    ax2.text(2 * np.pi, 1.25, r"$\sin(\omega t)$：旋转向量在虚轴上的投影",
             fontsize=11, color="tab:red", ha="center")
    ax2.set_xlim(cx - 1.7, 4 * np.pi + 0.3)
    ax2.set_ylim(-1.6, 1.75)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"相位角 $\omega t$（rad）")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_title("(b) 匀速旋转的向量，其投影就是正弦波")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig001_复平面与旋转向量.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


# ---------------------------------------------------------------- Fig002
def fig002():
    fig = plt.figure(figsize=(12.5, 5.4))

    # ---- 左：e^{jωt} 是三维空间里的螺旋线 ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    t = np.linspace(0, 3, 600)  # 3 个周期
    w = 2 * np.pi
    re, im = np.cos(w * t), np.sin(w * t)
    ax1.plot(t, re, im, color="tab:red", lw=2.0, label=r"$e^{j\omega t}$ 螺旋线")
    ax1.plot(t, re, np.full_like(t, -1.6), color="tab:blue", lw=1.3,
             label=r"实部投影 $\cos\omega t$")
    ax1.plot(t, np.full_like(t, 1.6), im, color="tab:green", lw=1.3,
             label=r"虚部投影 $\sin\omega t$")
    # t=0 处的单位圆截面
    circ = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.zeros_like(circ), np.cos(circ), np.sin(circ), color="0.6", lw=0.9)
    ax1.set_xlabel("时间 t（周期数）")
    ax1.set_ylabel("实部")
    ax1.set_zlabel("虚部")
    ax1.set_ylim(-1.6, 1.6)
    ax1.set_zlim(-1.6, 1.6)
    ax1.view_init(elev=20, azim=-55)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("(a) $e^{j\\omega t}=\\cos\\omega t+j\\sin\\omega t$：\n螺旋线与它的两个影子")

    # ---- 右：实数余弦 = 一对反向旋转的复向量 ----
    ax2 = fig.add_subplot(1, 2, 2)
    th = np.radians(50)
    v1 = 0.5 * np.exp(1j * th)   # 正频率分量
    v2 = 0.5 * np.exp(-1j * th)  # 负频率分量
    s = v1 + v2                  # 和：落在实轴上

    circ = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(0.5 * np.cos(circ), 0.5 * np.sin(circ), "--", color="0.75", lw=0.9)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.axvline(0, color="0.6", lw=0.8)

    for v, c, lbl in [(v1, "tab:red", r"$\frac{1}{2}e^{+j\theta}$（正频率）"),
                      (v2, "tab:blue", r"$\frac{1}{2}e^{-j\theta}$（负频率）")]:
        ax2.annotate("", xy=(v.real, v.imag), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0))
        ax2.text(v.real + 0.04, v.imag + (0.03 if v.imag > 0 else -0.09), lbl,
                 color=c, fontsize=12)
    # 平行四边形虚线
    ax2.plot([v1.real, s.real], [v1.imag, s.imag], "--", color="0.5", lw=1.0)
    ax2.plot([v2.real, s.real], [v2.imag, s.imag], "--", color="0.5", lw=1.0)
    ax2.annotate("", xy=(s.real, s.imag), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2.6))
    ax2.plot(s.real, s.imag, "o", color="tab:green", ms=7, zorder=5)
    ax2.text(s.real - 0.10, -0.14, r"和 $=\cos\theta$（纯实数）",
             color="tab:green", fontsize=12)
    ax2.text(-0.62, 0.55,
             r"$\cos\theta=\frac{1}{2}e^{j\theta}+\frac{1}{2}e^{-j\theta}$",
             fontsize=13)
    ax2.text(-0.62, 0.42, "两个向量的虚部方向相反，恰好抵消", fontsize=10, color="0.35")
    ax2.set_xlim(-0.7, 0.85)
    ax2.set_ylim(-0.62, 0.72)
    ax2.set_aspect("equal")
    ax2.set_xlabel("实轴")
    ax2.set_ylabel("虚轴")
    ax2.set_title("(b) 实数余弦 = 一对反向旋转的复向量之和")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig002_欧拉公式与螺旋线.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    fig001()
    fig002()
