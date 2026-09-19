# -*- coding: utf-8 -*-
"""生成正文图 6、图 7：快态固定模型的解析形态与它作用在显示上的三种面貌。

对应正文 §2.2 的有载/空载两分支：
    e>0 :  dx1/dt = (r1*e - x1)/tau_c1
    e=0 :  dx1/dt = -x1/tau_r1
本脚本把慢态 x2 与零点 z0 置零，只观察快态本身，并用与产品实现一致的
显式欧拉递推（dt=0.01 s）积分，以保证图上的曲线与代码行为同源。
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 中文与样式
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 130
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.size"] = 11

# ---------------------------------------------------------------- 参数（§3）
R1, TC1, TR1 = 0.12, 8.0, 6.0          # 快态幅度比 / 收敛 tau / 恢复 tau
TAU_EFF = TC1 / (1.0 + R1)             # 闭环有效时间常数 = 7.14 s
X1_INF = R1 / (1.0 + R1)               # 闭环饱和值（相对载荷）= 0.1071
DT = 0.01

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def simulate(y_fun, t_end, x1_0=0.0, r1=R1, tc1=TC1, tr1=TR1, dt=DT):
    """与产品实现同构的显式欧拉递推（x2=z0=0，隔离快态）。"""
    n = int(round(t_end / dt))
    ts = np.arange(n) * dt
    x1 = float(x1_0)
    x1s = np.empty(n)
    es = np.empty(n)
    for k, t in enumerate(ts):
        y = y_fun(t)
        e = y - x1
        if e > 0.0:
            x1 += dt * (r1 * e - x1) / tc1
        else:
            e = 0.0
            x1 -= dt * x1 / tr1
        x1s[k] = x1
        es[k] = e
    return ts, x1s, es


def box(ax, text, loc="upper left", color="#333333"):
    xy = {"upper left": (0.03, 0.97, "left", "top"),
          "upper right": (0.97, 0.97, "right", "top"),
          "lower left": (0.03, 0.04, "left", "bottom"),
          "lower right": (0.97, 0.04, "right", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=10, color=color,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fbfbfb", ec="#bbbbbb", lw=0.8))


# ================================================================ 图 A
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

# ---- (a) 有载分支：把 e 当外部输入，就是一阶惯性环节 --------------------
ax = axes[0]
t = np.linspace(0, 30, 600)
x1_open = R1 * 1.0 * (1.0 - np.exp(-t / TC1))
ax.plot(t, x1_open, lw=2.4, color="#1f6feb", label=r"$x_1(t)=r_1 e\,(1-e^{-t/\tau_{c1}})$")
ax.axhline(R1, ls="--", lw=1.3, color="#888888")
ax.text(0.6, R1 + 0.004, r"饱和值 $r_1 e=0.12\,e$（幅度与载荷成正比）",
        fontsize=10, color="#555555")
for tt, lab in [(TC1, r"$\tau_{c1}=8$ s → 63%"), (3 * TC1, r"$3\tau\to95\%$")]:
    ax.axvline(tt, ls=":", lw=1.1, color="#c0392b")
    ax.annotate(lab, xy=(tt, R1 * (1 - np.exp(-tt / TC1))),
                xytext=(tt + 0.6, R1 * (1 - np.exp(-tt / TC1)) + 0.018),
                fontsize=10, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))
ax.axhline(0, lw=1.0, color="#cccccc")
box(ax, "这是把 $e$ 当成固定输入看到的形状：\n一阶惯性环节，无延迟、无振荡、无超调。", "lower right")
ax.set_xlabel("时间 (s)")
ax.set_ylabel(r"$x_1$（相对载荷）")
ax.set_title("(a) 有载分支：开环一阶滞后", fontsize=12)
ax.set_ylim(-0.01, 0.16)
ax.legend(loc="center right", fontsize=10)

# ---- (b) 同一过程的闭环真相：e 自己也会被 x1 吃掉 -----------------------
ax = axes[1]
ts, x1s, es = simulate(lambda t: 1.0, 30.0)
ax.plot(ts, x1s, lw=2.4, color="#1f6feb", label=r"$x_1$（快态＝扣掉的量）")
ax.plot(ts, es, lw=2.4, color="#e67e22", label=r"$e=y-x_1$（弹性估计＝显示－零点）")
ax.plot(t, x1_open, lw=1.6, ls="--", color="#1f6feb", alpha=0.55,
        label=r"若 $e$ 不下降（(a) 的开环曲线）")
ax.axhline(X1_INF, ls=":", lw=1.2, color="#1f6feb")
ax.axhline(1 - X1_INF, ls=":", lw=1.2, color="#e67e22")
ax.annotate(r"$x_1\to\dfrac{r_1}{1+r_1}=10.7\%$", xy=(18, X1_INF),
            xytext=(9.5, 0.055), fontsize=11, color="#1f6feb",
            arrowprops=dict(arrowstyle="->", color="#1f6feb", lw=1.1))
ax.annotate(r"$e\to\dfrac{1}{1+r_1}=89.3\%$", xy=(22, 1 - X1_INF),
            xytext=(6.0, 0.74), fontsize=11, color="#e67e22",
            arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.1))
ax.annotate(r"耦合后更快：$\tau_{\rm eff}=\dfrac{\tau_{c1}}{1+r_1}=7.14$ s",
            xy=(TAU_EFF, 0.5), xytext=(11.0, 0.30), fontsize=10.5, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))
box(ax, "公式里 $e=y-x_1$ 自己含 $x_1$：\n快态长得越快，$e$ 掉得越快，目标也随之变小。", "lower right")
ax.set_xlabel("时间 (s)")
ax.set_ylabel("相对载荷")
ax.set_title("(b) 闭环真相：$e$ 不是外部输入", fontsize=12)
ax.set_ylim(0, 1.08)
ax.legend(loc="center right", fontsize=9.5)

# ---- (c) 空载分支：指数恢复，且不清零 ----------------------------------
ax = axes[2]
t = np.linspace(0, 20, 600)
x1_rec = X1_INF * np.exp(-t / TR1)
ax.plot(t, x1_rec, lw=2.4, color="#27ae60", label=r"本算法：$x_1(t)=x_1(0)e^{-t/\tau_{r1}}$")
ax.axhline(0, lw=2.4, color="#c0392b", ls="-", label="上一代：卸载即清零")
for tt in (1.0, 3.0, TR1):
    frac = np.exp(-tt / TR1)
    ax.plot([tt], [X1_INF * frac], "o", color="#27ae60", ms=5)
    ax.annotate(f"{tt:g} s → 残留 {frac*100:.0f}%", xy=(tt, X1_INF * frac),
                xytext=(tt + 0.5, X1_INF * frac + 0.008), fontsize=10, color="#1e7a45")
ax.axvline(TR1, ls=":", lw=1.1, color="#c0392b")
box(ax, "恢复比生长更快（6 s < 7.14 s），\n但不清零：短暂卸载后残留还在，\n重载时正是这笔残留把显示顶住。", "upper right")
ax.set_xlabel("卸载后时间 (s)")
ax.set_ylabel(r"$x_1$（相对载荷）")
ax.set_title("(c) 空载分支：衰减但不清零", fontsize=12)
ax.set_ylim(-0.008, 0.135)
ax.legend(loc="lower left", fontsize=10)

fig.tight_layout()
p1 = os.path.join(OUT_DIR, "fig06_fast_state_model.png")
fig.savefig(p1, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================================================================ 图 B
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

# ---- (d) 加载—保持—短卸载—重载 的完整时序 ------------------------------
ax = axes[0]
T_LOAD1, T_HOLD1, T_UNLOAD, T_RELOAD, T_END = 2.0, 20.0, 1.5, 3.5, 45.0


def y_seq(t):
    if t < T_LOAD1:
        return 0.0
    if t < T_LOAD1 + T_HOLD1:
        return 1.0
    if t < T_LOAD1 + T_HOLD1 + T_UNLOAD:
        return 0.0
    return 1.0


ts, x1s, es = simulate(y_seq, T_END)
ys = np.array([y_seq(t) for t in ts])
disp = ys - x1s
t_unload_end = T_LOAD1 + T_HOLD1 + T_UNLOAD
i_re = int(round(t_unload_end / DT))
disp_inf = 1.0 - X1_INF
band = 0.05 * disp_inf

ax.fill_between(ts, disp_inf - band, disp_inf + band, color="#f1c40f", alpha=0.22,
                label=r"终值 $\pm5\%$ 带")
ax.plot(ts, ys, lw=1.6, color="#9aa0a6", label=r"读数 $y$（纯弹性阶跃）")
ax.plot(ts, x1s, lw=2.2, color="#1f6feb", label=r"快态 $x_1$（扣掉的量）")
ax.plot(ts, disp, lw=2.4, color="#c0392b", label=r"显示 $=y-x_1$")
ax.axhline(disp_inf, ls=":", lw=1.1, color="#c0392b")
ax.annotate("首次加载：显示从 1.00 沉到 0.893\n（−10.7%），约 6.2 s 才落进带内",
            xy=(T_LOAD1 + 4.0, 0.955), xytext=(6.0, 1.03), fontsize=10,
            color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))
ax.annotate(f"重载：起点 {disp[i_re]:.3f} 距终值只差 "
            f"{(disp[i_re]-disp_inf)/disp_inf*100:.1f}%\n→ 一开始就在 $\\pm5\\%$ 带内",
            xy=(t_unload_end + 0.4, disp[i_re]), xytext=(26.0, 1.02), fontsize=10,
            color="#1e7a45", arrowprops=dict(arrowstyle="->", color="#1e7a45", lw=1.1))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("相对载荷")
ax.set_title("(d) 加载—保持—卸载 1.5 s—重载", fontsize=12)
ax.set_ylim(-0.05, 1.16)
ax.legend(loc="lower left", fontsize=9.5, ncol=2)

# ---- (e) 材料确有蠕变（幅度比与模型一致）→ 显示被钉平 -------------------
ax = axes[1]


def y_creep(t):
    if t < T_LOAD1:
        return 0.0
    return 1.0 + R1 * (1.0 - np.exp(-(t - T_LOAD1) / 8.0))


ts, x1s, es = simulate(y_creep, 60.0)
ys = np.array([y_creep(t) for t in ts])
disp = ys - x1s
ax.plot(ts, ys, lw=2.0, color="#9aa0a6", label=r"读数 $y$（阶跃 + 快相蠕变，自身在爬）")
ax.plot(ts, x1s, lw=2.0, color="#1f6feb", label=r"快态 $x_1$（跟着爬，把它扣掉）")
ax.plot(ts, disp, lw=2.6, color="#c0392b", label=r"显示 $=y-x_1$（钉在弹性电平上）")
ax.axhline(1.0, ls=":", lw=1.1, color="#c0392b")
ax.axhline(1.0 + R1, ls=":", lw=1.1, color="#9aa0a6")
ax.annotate(r"终值恰为 $\dfrac{y_\infty}{1+r_1}=1.00$：材料蠕变幅度\n"
            r"等于 $r_1$ 时，扣除量正好等于蠕变量",
            xy=(40, 1.0), xytext=(20.0, 0.985), fontsize=10, color="#c0392b")
ax.annotate("读数爬 12%", xy=(30, 1.09), xytext=(24.0, 1.105),
            fontsize=10, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("相对载荷")
ax.set_title("(e) 材料有蠕变 → 显示被钉平", fontsize=12)
ax.set_ylim(0.96, 1.135)
ax.legend(loc="lower right", fontsize=9.5)

# ---- (f) 材料纯弹性（没有蠕变）→ 显示被误伤 -----------------------------
ax = axes[2]
ts, x1s, es = simulate(lambda t: 0.0 if t < T_LOAD1 else 1.0, 30.0)
ys = np.array([0.0 if t < T_LOAD1 else 1.0 for t in ts])
disp = ys - x1s
ax.plot(ts, ys, lw=1.8, color="#9aa0a6", label=r"读数 $y$（纯弹性阶跃，自己不爬）")
ax.plot(ts, x1s, lw=2.2, color="#1f6feb", label=r"快态 $x_1$（仍然长出来）")
ax.plot(ts, disp, lw=2.6, color="#c0392b", label=r"显示 $=y-x_1$（被扣掉 10.7%）")
ax.axhline(1.0, ls=":", lw=1.1, color="#9aa0a6")
ax.annotate("显示下沉 −10.7%：\n材料其实没有蠕变，\n这是模型按时间尺度\n强行分账的结果",
            xy=(12.0, 0.905), xytext=(13.5, 0.955), fontsize=10, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("相对载荷")
ax.set_title("(f) 材料无蠕变 → 显示被误伤", fontsize=12)
ax.set_ylim(0.86, 1.06)
ax.legend(loc="lower left", fontsize=9.5)

fig.tight_layout()
p2 = os.path.join(OUT_DIR, "fig07_fast_state_display.png")
fig.savefig(p2, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================================================================ 数值核对
print("[参数] r1=%.2f tau_c1=%.1f tau_r1=%.1f" % (R1, TC1, TR1))
print("[解析] tau_eff      = %.4f s" % TAU_EFF)
print("[解析] x1 饱和 r1/(1+r1) = %.4f   e 稳态 1/(1+r1) = %.4f" % (X1_INF, 1 - X1_INF))
ts, x1s, es = simulate(lambda t: 1.0, 30.0)
print("[数值] t=7.14 s  x1=%.4f  e=%.4f" % (x1s[int(TAU_EFF / DT)], es[int(TAU_EFF / DT)]))
print("[数值] t=30 s    x1=%.4f  e=%.4f  (闭式 %.4f)" % (x1s[-1], es[-1], X1_INF))
disp_inf = 1.0 - X1_INF
x1_need = X1_INF - 0.05 * disp_inf          # 显示落进 ±5% 带所需的 x1
idx = int(np.argmax(x1s >= x1_need))
print("[数值] 纯弹性阶跃下首次加载进 ±5%% 带：x1 需 >= %.4f，实现于 t = %.2f s"
      % (x1_need, ts[idx]))
print("[数值] 卸载 1.5 s 后 x1 = %.4f（残留 %.1f%%）"
      % (X1_INF * np.exp(-1.5 / TR1), np.exp(-1.5 / TR1) * 100))
print("已输出:\n  %s\n  %s" % (p1, p2))
