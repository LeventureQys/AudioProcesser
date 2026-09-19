# -*- coding: utf-8 -*-
"""生成正文图 8、图 9：慢态跟踪的解析形态与它对显示的后果。

约定与 make_fast_state_figs.py 一致：曲线用与正文公式同构的显式欧拉递推
（dt=0.01 s，dt 截断 [0,0.1]）积分，零点 z0 置零以隔离两个状态；图按设计
含参数数值，属于"推导图"系列，不进入无量纲概念图口径。

输出：fig08_slow_tracking_model.png
      fig09_slow_tracking_effect.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 130
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.size"] = 11

R1, TC1, TR1 = 0.12, 8.0, 6.0                 # 快态
R2MAX, TR2 = 0.35, 150.0                      # 慢态上限 / 空载恢复
TAU_SLOPE, G_SLOPE, C_MAX = 3.0, 0.02, 0.01   # 低通导数 / 沿门 / 速率上限
DT = 0.01

OUT = os.path.dirname(os.path.abspath(__file__))
C_X1, C_X2, C_DIS, C_IN = "#1f6feb", "#ae3ec9", "#c0392b", "#9aa0a6"
C_GATE, C_OK, C_NOTE = "#e67e22", "#27ae60", "#c0392b"


def simulate(v_fun, t_end, x1_0=0.0, x2_0=0.0, v_lp_0=None,
             subtract_dx1=True, dt=DT):
    """与正文公式同构的递推；z0 置零，只保留 x1 / x2 / v_lp。"""
    n = int(round(t_end / dt))
    ts = np.arange(n) * dt
    x1, x2 = float(x1_0), float(x2_0)
    v_lp = v_fun(0.0) if v_lp_0 is None else float(v_lp_0)
    V = np.empty(n); X1 = np.empty(n); X2 = np.empty(n)
    SL = np.empty(n); E = np.empty(n)
    for k, t in enumerate(ts):
        v = v_fun(t)
        y = v
        e_now = max(y - x1 - x2, 0.0)
        dx1_rate = (R1 * e_now - x1) / TC1 if e_now > 0 else -x1 / TR1
        x1 = max(x1 + dt * dx1_rate, 0.0)
        slope = (v - v_lp) / TAU_SLOPE
        v_lp += (dt / TAU_SLOPE) * (v - v_lp)
        e = max(y - x1 - x2, 0.0)
        if e > 0.0:
            if abs(slope) < G_SLOPE * e:
                dx2 = slope - dx1_rate if subtract_dx1 else slope
                dx2 = float(np.clip(dx2, -C_MAX * e, C_MAX * e))
                x2 += dt * dx2
            x2 = float(np.clip(x2, 0.0, R2MAX * e))
        else:
            x2 = max(x2 - dt * x2 / TR2, 0.0)
        V[k], X1[k], X2[k], SL[k], E[k] = v, x1, x2, slope, e
    return ts, V, X1, X2, SL, E


def v_creep(t, t0=2.0, slow_rate=0.0020, fast=R1, tau_fast=8.0):
    """阶跃 + 快相蠕变 + 恒速慢漂移（归一化载荷）。"""
    if t < t0:
        return 0.0
    return 1.0 + fast * (1.0 - np.exp(-(t - t0) / tau_fast)) + slow_rate * (t - t0)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.normpath(p))


ts0, V0, X10, X20, SL0, E0 = simulate(lambda t: v_creep(t, slow_rate=0.0020), 120.0)
blocked = SL0 > G_SLOPE * E0
t_open = ts0[np.argmax(~blocked & (ts0 > 2.0))]
BLOCK_S = t_open - 2.0

# ================================================================ 图 8
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

# ---- (a) 低通导数的形态 -------------------------------------------------
ax = axes[0]
vdot = np.gradient(V0, DT)
ax.semilogy(ts0, np.maximum(SL0, 1e-6), lw=2.4, color=C_X2,
            label=r"$\mathrm{slope}=(v-v_{lp})/\tau_{\mathrm{slope}}$")
ax.semilogy(ts0, np.maximum(vdot, 1e-6), lw=1.8, ls="--", color=C_IN,
            label=r"真实斜率 $\dot v$")
ax.axvline(2.0, ls=":", lw=1.1, color="#888888")
ax.annotate(r"台阶瞬间：$\mathrm{slope}$ 跳到 $1/\tau_{\mathrm{slope}}=0.33$ 倍载荷/s",
            xy=(2.05, 0.32), xytext=(12.0, 0.30), fontsize=10, color=C_NOTE,
            arrowprops=dict(arrowstyle="->", color=C_NOTE, lw=1.1))
ax.annotate(r"慢相段：$\mathrm{slope}$ 追上真实速率" "\n"
            r"（滞后 $\tau_{\mathrm{slope}}=3$ s）",
            xy=(70, 0.0022), xytext=(28.0, 0.0007), fontsize=10, color=C_OK,
            arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.1))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("速率（相对载荷 / s）")
ax.set_title("(a) 低通导数的形态", fontsize=12)
ax.set_ylim(5e-5, 1.2)
ax.legend(loc="upper right", fontsize=9.5)

# ---- (b) 构造性钉平（稳态起步，门内且未触限）---------------------------
ax = axes[1]
T_B = 60.0


def v_lin(t):
    return 1.0 + 0.0020 * t


def v_acc(t):
    return 1.0 + 0.10 * (1.0 - np.exp(-t / 30.0))


for vf, col, lab in ((v_lin, C_OK, r"恒速蠕变：漂移率 $\to0$（$\tau_{\mathrm{slope}}$ 尺度）"),
                     (v_acc, C_IN, r"加速蠕变：残余 $\approx\tau_{\mathrm{slope}}\ddot v$")):
    tb, Vb, X1b, X2b, SLb, Eb = simulate(vf, T_B, x1_0=R1 / (1 + R1) * 1.0, v_lp_0=vf(0.0))
    disp = Vb - X1b - X2b
    true_rate = np.gradient(disp, DT)
    theo = np.gradient(Vb, DT) - SLb
    ax.semilogy(tb[1:], np.maximum(np.abs(true_rate[1:]), 1e-7), lw=2.4, color=col, label=lab)
    ax.semilogy(tb[1:], np.maximum(np.abs(theo[1:]), 1e-7), lw=1.2, ls=":", color="#555555")
ax.annotate(r"点线：$\dot v-\mathrm{slope}$（门内且未触限时与实线重合）",
            xy=(18.0, 3e-4), xytext=(20.0, 2.2e-3), fontsize=9.5, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("显示漂移率（相对载荷 / s）")
ax.set_title(r"(b) 构造性钉平：漂移率 $=\dot v-\mathrm{slope}$", fontsize=12)
ax.set_ylim(1e-7, 1e-2)
ax.legend(loc="lower left", fontsize=9)

# ---- (c) 扣除 dx1 的作用 ------------------------------------------------
ax = axes[2]
tse, Ve, X1e, X2e, _, _ = simulate(lambda t: v_creep(t, slow_rate=0.0020), 90.0, subtract_dx1=True)
tsn, Vn, X1n, X2n, _, _ = simulate(lambda t: v_creep(t, slow_rate=0.0020), 90.0, subtract_dx1=False)
k = len(tse) - 1
ax.plot(tse, X1e, lw=2.2, color=C_X1, label=r"$x_1$（快态，两份一致）")
ax.plot(tse, X2e, lw=2.4, color=C_X2, label=r"$x_2$：扣除 $\dot x_1$（本文）")
ax.plot(tsn, X2n, lw=2.2, ls="--", color=C_NOTE, label=r"$x_2$：不扣除 $\dot x_1$")
ax.fill_between(tsn, X2e, X2n, color=C_NOTE, alpha=0.15)
ax.annotate("不扣 $\\dot x_1$ 时慢态多吃一份快相速率：\n"
            "本算例 90 s 处多计 %.3f（约载荷的 %.1f%%），\n"
            "显示被多扣同样的量（沿门与速率上限已挡住大部分）"
            % (X2n[k] - X2e[k], (X2n[k] - X2e[k]) * 100),
            xy=(80.0, (X2n[k] + X2e[k]) / 2), xytext=(14.0, 0.235), fontsize=9.5,
            color=C_NOTE, arrowprops=dict(arrowstyle="->", color=C_NOTE, lw=1.1))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("状态量（相对载荷）")
ax.set_title(r"(c) 为什么要扣除 $\dot x_1$", fontsize=12)
ax.set_ylim(-0.01, 0.36)
ax.legend(loc="lower right", fontsize=9.5)

fig.tight_layout()
save(fig, "fig08_slow_tracking_model.png")

# ================================================================ 图 9
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

# ---- (d) 沿门 ----------------------------------------------------------
ax = axes[0]
gate = G_SLOPE * E0
ax.plot(ts0, SL0, lw=2.2, color=C_X2, label=r"$\mathrm{slope}$")
ax.plot(ts0, gate, lw=2.0, color=C_GATE, label=r"沿门 $g_{\mathrm{slope}}e$")
ax.fill_between(ts0, gate, SL0, where=blocked, color=C_NOTE, alpha=0.18,
                label="被挡住：这一帧不积分")
ax.annotate("台阶后约 %.1f s 才重新开门" % BLOCK_S,
            xy=(t_open, gate[int(t_open / DT)]), xytext=(11.0, 0.17),
            fontsize=10, color=C_NOTE,
            arrowprops=dict(arrowstyle="->", color=C_NOTE, lw=1.1))
ax.annotate("峰值与门限相差约 %.0f 倍" % ((1.0 / TAU_SLOPE) / G_SLOPE),
            xy=(2.05, 0.30), xytext=(6.5, 0.38), fontsize=10, color="#555555")
ax.set_xlabel("时间 (s)")
ax.set_ylabel("速率（相对载荷 / s）")
ax.set_title("(d) 沿门：台阶增量不进慢态", fontsize=12)
ax.set_ylim(0, 0.45)
ax.legend(loc="upper right", fontsize=9.5)

# ---- (e) 速率上限 ------------------------------------------------------
ax = axes[1]


def v_rate(t, rate, t0=2.0):
    if t < t0:
        return 0.0
    return 1.0 + R1 * (1.0 - np.exp(-(t - t0) / 8.0)) + rate * (t - t0)


for rate, col, lab in ((0.005, C_OK, r"速率在限内（$0.5\%e$/s）"),
                       (0.012, C_DIS, r"速率超限（$1.2\%e$/s）")):
    tsx, Vx, X1x, X2x, _, _ = simulate(lambda t, r=rate: v_rate(t, r), 60.0)
    ax.plot(tsx, Vx, lw=1.3, ls="--", color=col, alpha=0.6)
    ax.plot(tsx, Vx - X1x - X2x, lw=2.4, color=col, label=lab + "：显示（虚线为读数）")
ax.annotate(r"超限时慢态最多以 $c_{\max}e=1\%e$/s 积累，" "\n"
            r"显示按 $\dot v-c_{\max}e$ 线性漂移",
            xy=(52.0, 1.24), xytext=(12.0, 1.33), fontsize=10, color=C_DIS,
            arrowprops=dict(arrowstyle="->", color=C_DIS, lw=1.1))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("相对载荷")
ax.set_title("(e) 速率上限：判错也有限速", fontsize=12)
ax.set_ylim(0.98, 1.48)
ax.legend(loc="upper left", fontsize=9)

# ---- (f) 卸载后双态的恢复 ----------------------------------------------
ax = axes[2]
tt = np.linspace(0, 40, 800)
ax.plot(tt, np.exp(-tt / TR1), lw=2.4, color=C_X1, label=r"快态 $x_1$：$\tau_{r1}=6$ s")
ax.plot(tt, np.exp(-tt / TR2), lw=2.4, color=C_X2, label=r"慢态 $x_2$：$\tau_{r2}=150$ s")
ax.axvline(20.0, ls=":", lw=1.1, color="#888888")
for tau, col in ((TR1, C_X1), (TR2, C_X2)):
    ax.plot([20.0], [np.exp(-20.0 / tau)], "o", color=col, ms=6)
    ax.annotate("20 s 后残留 %.1f%%" % (np.exp(-20.0 / tau) * 100),
                xy=(20.0, np.exp(-20.0 / tau)),
                xytext=(21.0, np.exp(-20.0 / tau) + 0.05), fontsize=10, color=col)
ax.annotate("跨卸载连续性主要由慢态承载：" "\n"
            "239 s 的受载历史绝大部分长在 $x_2$ 上，" "\n"
            "而卸载期间它几乎不掉",
            xy=(26.0, 0.75), xytext=(4.5, 0.30), fontsize=10, color="#333333")
ax.set_xlabel("卸载后时间 (s)")
ax.set_ylabel("归一化残余状态")
ax.set_title("(f) 卸载后谁来扛连续性", fontsize=12)
ax.set_ylim(-0.03, 1.12)
ax.legend(loc="center right", fontsize=9.5)

fig.tight_layout()
save(fig, "fig09_slow_tracking_effect.png")

# ================================================================ 数值核对
print("[解析] 阶跃峰值 slope = 1/tau_slope = %.4f；门限 = %.4f；比值 %.1f 倍"
      % (1.0 / TAU_SLOPE, G_SLOPE, (1.0 / TAU_SLOPE) / G_SLOPE))
print("[数值] 沿门遮挡时长 = %.2f s" % BLOCK_S)
tb, Vb, X1b, X2b, SLb, _ = simulate(v_lin, T_B, x1_0=R1 / (1 + R1), v_lp_0=1.0)
tr = np.gradient(Vb - X1b - X2b, DT)
th = np.gradient(Vb, DT) - SLb
print("[核对] 恒速蠕变：|实际漂移率 - (vdot-slope)| 最大 %.2e（t>1 s）；末值 %.2e"
      % (np.abs(tr[100:] - th[100:]).max(), abs(tr[-1])))
print("[核对] 恒速蠕变：t=60 s 漂移率 %.2e（初值 %.2e）" % (abs(tr[-1]), abs(tr[1])))
tse, Ve, X1e, X2e, _, _ = simulate(lambda t: v_creep(t, slow_rate=0.0020), 90.0, subtract_dx1=True)
tsn, Vn, X1n, X2n, _, _ = simulate(lambda t: v_creep(t, slow_rate=0.0020), 90.0, subtract_dx1=False)
print("[数值] t=90 s：x2(扣)=%.4f  x2(不扣)=%.4f  多计 %.4f（载荷的 %.2f%%）"
      % (X2e[-1], X2n[-1], X2n[-1] - X2e[-1], (X2n[-1] - X2e[-1]) * 100))
print("[解析] 卸载 20 s 残留：x1 %.2f%%，x2 %.2f%%"
      % (np.exp(-20 / TR1) * 100, np.exp(-20 / TR2) * 100))
