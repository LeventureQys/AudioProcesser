# -*- coding: utf-8 -*-
"""
卡尔曼滤波工程手记 配图与仿真脚本
运行:  python gen_figures.py
依赖:  numpy, matplotlib
输出:  ../figures/Fig01..Fig05 .png
      sim_results.json （正文实测数字来源）
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20240501)  # 固定种子，结果可复现

results = {}


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ============================================================
# Fig01 为什么需要卡尔曼滤波：静态平均在动态系统上失效
# 模型: 匀加速运动 x'' = a, 只能观测含噪位置 z = x + v, v~N(0,R)
# ============================================================
N = 200
dt = 0.1
a_true = 0.5
t = np.arange(N) * dt
x_true = 0.5 * a_true * t**2  # 初值 0, 初速 0
R_true = 4.0
z = x_true + rng.normal(0, np.sqrt(R_true), N)

# 滑动平均（窗口 11）作为"传统静态平均"代表
win = 11
kernel = np.ones(win) / win
x_ma = np.convolve(z, kernel, mode="same")
x_ma[: win // 2] = x_ma[win // 2]
x_ma[-(win // 2):] = x_ma[-(win // 2)]

# 1D 匀加速 KF（状态 = [位置,速度,加速度], 仅观测位置）
Fm = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])
Hm = np.array([[1.0, 0, 0]])
Q1 = 1e-6 * np.eye(3)
x_hat = np.zeros(3)
P = np.eye(3) * 100.0
x_kf = []
k_kf = []
for k in range(N):
    x_hat = Fm @ x_hat
    P = Fm @ P @ Fm.T + Q1
    yk = z[k] - (Hm @ x_hat)[0]
    Sk = (Hm @ P @ Hm.T)[0, 0] + R_true
    Kk = (P @ Hm.T) / Sk
    x_hat = x_hat + Kk.flatten() * yk
    P = (np.eye(3) - Kk @ Hm) @ P
    x_kf.append(x_hat[0])
    k_kf.append(float(Kk[0, 0]))
x_kf = np.array(x_kf)
k_kf = np.array(k_kf)

results["fig1_rmse_measure"] = rmse(z, x_true)
results["fig1_rmse_ma"] = rmse(x_ma, x_true)
results["fig1_rmse_kf"] = rmse(x_kf, x_true)
# 滞后量: 滑动平均相对真值的平均偏差（后段，排除边缘）
results["fig1_ma_mean_lag"] = float(np.mean(x_ma[20:-20] - x_true[20:-20]))
results["fig1_kf_mean_bias"] = float(np.mean(x_kf[20:-20] - x_true[20:-20]))

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(t, x_true, "k-", lw=2, label="真实位置（不可见）")
ax.plot(t, z, color="0.75", marker=".", ms=4, ls="none", label=f"含噪测量（RMSE={results['fig1_rmse_measure']:.2f}）")
ax.plot(t, x_ma, "g-", lw=1.8,
        label=f"滑动平均(窗11)（RMSE={results['fig1_rmse_ma']:.2f}, 平均滞后 {results['fig1_ma_mean_lag']:.2f}）")
ax.plot(t, x_kf, "r-", lw=1.8, label=f"卡尔曼滤波（RMSE={results['fig1_rmse_kf']:.2f}）")
ax.set_xlabel("时间 / s")
ax.set_ylabel("位置 / m")
ax.set_title("动态系统中：静态平均滞后，卡尔曼滤波跟住", fontsize=12)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.set_xlim(0, t[-1] * 1.02)
ax.set_ylim(min(z) - 1, max(z) + 1)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig01_动态系统中静态平均失效.png"), dpi=150)
plt.close(fig)

# ============================================================
# Fig02 预测-更新循环流程图（手工坐标 + 碰撞避让）
# ============================================================
fig, ax = plt.subplots(figsize=(9.6, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 68)
ax.axis("off")


def box(x, y, w, h, text, fc="#eaf2fb", ec="#2c6fa8", fs=10.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                       fc=fc, ec=ec, lw=1.6)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, text="", rad=0.0, dx=0, dy=0, fs=9):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        connectionstyle=f"arc3,rad={rad}",
                        arrowstyle="-|>", mutation_scale=14, lw=1.5, color="#444")
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text,
                ha="center", va="center", fontsize=fs,
                bbox=dict(fc="white", ec="none", pad=1.2))


# 预测组
ax.text(26, 64.5, "预测步（拿模型猜）", ha="center", fontsize=12, weight="bold", color="#2c6fa8")
box(6, 50, 40, 9, "状态预测  $\\hat{x}_{k|k-1}=F_k\\hat{x}_{k-1}$\n（按运动模型外推）")
box(6, 35, 40, 9, "协方差预测  $P_{k|k-1}=F_kP_{k-1}F_k^{T}+Q_k$\n（不确定性变大：模型噪声 $Q_k$ 记账）")
arrow(26, 50, 26, 44)

# 更新组
ax.text(74, 64.5, "更新步（拿测量纠偏）", ha="center", fontsize=12, weight="bold", color="#a8442c")
box(56, 50, 38, 9, "新息  $y_k=z_k-H_k\\hat{x}_{k|k-1}$\n（测量与预测的差）", fc="#fdeee6", ec="#a8442c")
box(56, 35, 38, 9, "增益  $K_k=P_{k|k-1}H_k^{T}(H_kP_{k|k-1}H_k^{T}+R_k)^{-1}$\n（信模型还是信测量）", fc="#fdeee6", ec="#a8442c")
box(56, 20, 38, 9, "状态更新  $\\hat{x}_k=\\hat{x}_{k|k-1}+K_ky_k$\n协方差更新  $P_k=(I-K_kH_k)P_{k|k-1}$", fc="#fdeee6", ec="#a8442c")
arrow(74, 50, 74, 44.4)
arrow(74, 35, 74, 29.4)

# 输入与输出
box(31, 6, 16, 8, "测量 $z_k$", fc="#f2f2f2", ec="#666")
box(75, 6, 18, 8, "输出 $\\hat{x}_k$", fc="#eef7ee", ec="#3a7d3a")
arrow(39, 14, 66, 49.4, "每来一个新测量", rad=-0.25, dy=3)
arrow(74, 20, 84, 14)
arrow(26, 35, 26, 29.4)
box(6, 20, 40, 9, "循环：$k\\to k+1$（递推、无需存历史）", fc="#fbf6e9", ec="#a8932c")
arrow(26, 20, 26, 14.4)
arrow(35, 10, 20, 6.6, "初值 $\\hat{x}_0,\\,P_0$", rad=0.0, dy=-3)
arrow(46, 24.5, 56, 24.5, "预测交给更新", dy=2.6)
fig.suptitle("卡尔曼滤波：预测—更新递推循环", fontsize=13, y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig02_预测更新循环.png"), dpi=150)
plt.close(fig)

# ============================================================
# Fig03 增益行为：噪声大时信模型、噪声小时信测量
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
axes[0].plot(t, z, color="0.78", marker=".", ms=4, ls="none", label="含噪测量")
axes[0].plot(t, x_true, "k-", lw=2, label="真值")
axes[0].plot(t, x_kf, "r-", lw=1.6, label="KF 估计")
axes[0].set_ylabel("位置 / m")
axes[0].legend(loc="upper left", fontsize=9)
axes[0].set_title("KF 估计与卡尔曼增益（R=4, Q=1e-6·I）", fontsize=12)
axes[1].plot(t, k_kf, "b-", lw=1.8)
axes[1].set_ylabel("增益 $K_k$（位置分量）")
axes[1].set_xlabel("时间 / s")
axes[1].set_ylim(0, 1.05)
axes[1].axhline(np.sqrt(1 / (1 + 1)) * 0, color="0.6", ls=":")
results["fig3_k_steady"] = float(np.mean(k_kf[100:]))
results["fig3_k_initial"] = float(k_kf[1])
ax_txt = ("增益曲线读法：初始 $P_0$ 大 → 增益接近 1 → 几乎照单全收测量；\n"
          f"收敛后增益稳定在 {results['fig3_k_steady']:.3f}（本例），按噪声比例分配信任。")
axes[1].text(0.02, 0.62, ax_txt, transform=axes[1].transAxes, fontsize=9.5,
             bbox=dict(fc="#f5f8fb", ec="#9bb"))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig03_卡尔曼增益行为.png"), dpi=150)
plt.close(fig)

# ============================================================
# Fig04 Q、R 失配敏感性（实测热力图）
# 模型同 Fig01，Q 的名义值 1e-6·I3，R 名义值 4.0
# ============================================================
q_list = [1e-8, 1e-6, 1e-4, 1e-2, 1e0]
r_list = [0.5, 1.0, 2.0, 4.0, 16.0]
grid = np.zeros((len(q_list), len(r_list)))


def run_kf(Qs, Rs):
    xh = np.zeros(3)
    Pm = np.eye(3) * 100.0
    out = []
    for k in range(N):
        xh = Fm @ xh
        Pm = Fm @ Pm @ Fm.T + Qs * np.eye(3)
        yk = z[k] - xh[0]
        Sk = Pm[0, 0] + Rs
        Kk = Pm[:, 0] / Sk
        xh = xh + Kk * yk
        Pm = Pm - np.outer(Kk, Pm[0, :])
        out.append(xh[0])
    return np.array(out)


for i, q in enumerate(q_list):
    for j, r in enumerate(r_list):
        grid[i, j] = rmse(run_kf(q, r)[20:], x_true[20:])

results["fig4_best"] = float(grid.min())
results["fig4_nominal"] = float(grid[q_list.index(1e-6), r_list.index(4.0)])
results["fig4_worst"] = float(grid.max())

fig, ax = plt.subplots(figsize=(7.6, 5.4))
im = ax.imshow(grid, cmap="viridis", origin="lower", aspect="auto")
ax.set_xticks(range(len(r_list)), [f"{r:g}" for r in r_list])
ax.set_yticks(range(len(q_list)), [f"{q:g}" for q in q_list])
ax.set_xlabel("设计参数 $R$（名义 4.0）")
ax.set_ylabel("设计参数 $Q$（名义 1e-6）")
ax.set_title(f"$Q$/$R$ 失配下的稳态 RMSE（真值 $R$=4, 真实 $Q$≈0）", fontsize=12)
for i in range(len(q_list)):
    for j in range(len(r_list)):
        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                color="white" if grid[i, j] > grid.mean() else "black", fontsize=9)
fig.colorbar(im, label="RMSE / m（后 180 步）")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig04_QR失配敏感性.png"), dpi=150)
plt.close(fig)

# ============================================================
# Fig05 非线性问题：EKF vs UKF（经典 z = x^2 观测混叠案例）
# 状态: 随机游走 x_{k+1} = x_k + w,  真值从 -1.0 出发
# 观测: z_k = x_k^2 + v   —— x 与 -x 观测值相同（符号混叠）
# 初值故意给错符号: x0 = +1.0, P0 = 2.0
# ============================================================
Nx = 200
q_nl, R_nl = 1e-3, 0.05
rng_nl = np.random.default_rng(7)
x_nl = -1.0
x_nl_true = []
for k in range(Nx):
    x_nl = x_nl + rng_nl.normal(0, np.sqrt(q_nl))
    x_nl_true.append(x_nl)
x_nl_true = np.array(x_nl_true)
z_nl = x_nl_true**2 + rng_nl.normal(0, np.sqrt(R_nl), Nx)


def run_ekf(x0, P0):
    xh, P = x0, P0
    out = []
    for k in range(Nx):
        P = P + q_nl                      # 状态方程线性且为单位映射
        Hj = 2 * xh                       # 观测雅可比在"当前估计"处取值
        S = Hj * P * Hj + R_nl
        Kk = P * Hj / S
        xh = xh + Kk * (z_nl[k] - xh**2)
        P = (1 - Kk * Hj) * P
        out.append(xh)
    return np.array(out)


def run_ukf(x0, P0):
    # 对称采样（lam=0）：仅 2 个 sigma 点，均值/协方差权重均为 0.5
    xh, P = x0, P0
    out = []
    for k in range(Nx):
        P = P + q_nl
        pts = np.array([xh + np.sqrt(P), xh - np.sqrt(P)])
        Z = pts**2                        # 直接让真非线性函数"投票"
        zm = 0.5 * (Z[0] + Z[1])
        Pzz = 0.5 * ((Z[0] - zm) ** 2 + (Z[1] - zm) ** 2) + R_nl
        Pxz = 0.5 * ((pts[0] - xh) * (Z[0] - zm) + (pts[1] - xh) * (Z[1] - zm))
        Kk = Pxz / Pzz
        xh = xh + Kk * (z_nl[k] - zm)
        P = P - Kk * Pxz
        out.append(xh)
    return np.array(out)


x_ekf = run_ekf(1.0, 2.0)
x_ukf = run_ukf(1.0, 2.0)
results["fig5_rmse_ekf"] = rmse(x_ekf[20:], x_nl_true[20:])
results["fig5_rmse_ukf"] = rmse(x_ukf[20:], x_nl_true[20:])
results["fig5_rmse_ekf_goodinit"] = rmse(run_ekf(-1.0, 2.0)[20:], x_nl_true[20:])

tt = np.arange(Nx)
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(tt, x_nl_true, "k-", lw=2, label="真值（从 -1.0 随机游走）")
ax.plot(tt, x_ekf, "r--", lw=1.5, label=f"EKF（初值 +1.0, RMSE={results['fig5_rmse_ekf']:.2f}，发散）")
ax.plot(tt, x_ukf, "b-", lw=1.5, label=f"UKF（同初值, RMSE={results['fig5_rmse_ukf']:.3f}，收敛）")
ax.axhline(0, color="0.85", lw=0.8)
ax.set_xlabel("步数 $k$")
ax.set_ylabel("状态 $x$")
ax.set_title("非线性观测 $z=x^2$：EKF 线性化在错误初值下发散，UKF 采样收敛", fontsize=12)
ax.legend(loc="lower left", fontsize=9)
ax.set_xlim(0, Nx - 1)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig05_EKF与UKF非线性对比.png"), dpi=150)
plt.close(fig)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim_results.json"),
          "w", encoding="utf-8") as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
