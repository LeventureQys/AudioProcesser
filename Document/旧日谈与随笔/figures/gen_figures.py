"""
生成文章配图。运行方式：
    cd Document/旧日谈/figures
    python gen_figures.py
"""
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

fs = 48000
fc = 1000

# ============================================================
# 图 1: FIR vs IIR 结构框图
# ============================================================
def fig1_structure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # FIR 结构
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 4)
    ax1.set_aspect('equal')
    ax1.set_title('FIR 滤波器结构（无反馈）', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # 延迟链
    for i in range(4):
        x = 1.5 + i * 2
        rect = FancyBboxPatch((x, 2.2), 1.2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x + 0.6, 2.6, f'$z^{{-{i}}}$', ha='center', va='center', fontsize=12)

    # 乘法器
    for i in range(4):
        x = 1.8 + i * 2
        ax1.plot(x, 1.6, 'o', color='#D32F2F', markersize=18, zorder=5)
        ax1.text(x, 1.6, f'$b_{i}$', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # 加法器
    ax1.plot(9, 1.6, 'o', color='#2E7D32', markersize=22, zorder=5)
    ax1.text(9, 1.6, '$\\Sigma$', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    # 输入输出
    ax1.annotate('', xy=(0.8, 2.6), xytext=(0.2, 2.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    ax1.text(0.0, 2.6, '$x[n]$', ha='right', va='center', fontsize=13, fontweight='bold')

    ax1.annotate('', xy=(9.8, 1.6), xytext=(9.3, 1.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    ax1.text(9.9, 1.6, '$y[n]$', ha='left', va='center', fontsize=13, fontweight='bold')

    # 连线
    for i in range(4):
        x = 1.8 + i * 2
        ax1.plot([x, x], [2.2, 1.8], color='#333', lw=1.5)
        if i < 3:
            ax1.plot([x + 0.6, x + 1.4], [2.6, 2.6], color='#333', lw=1.5)
        ax1.plot([x + 0.2, 8.8], [1.6, 1.6], color='#333', lw=1, alpha=0.3)

    ax1.text(5, 0.5, '信号单向流动，无反馈环路', ha='center', fontsize=11,
             style='italic', color='#555')

    # IIR 结构
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.set_aspect('equal')
    ax2.set_title('IIR 滤波器结构（有反馈）', fontsize=13, fontweight='bold')
    ax2.axis('off')

    # 前馈部分
    for i in range(3):
        x = 1 + i * 2
        rect = FancyBboxPatch((x, 3.2), 1.2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x + 0.6, 3.6, f'$z^{{-{i}}}$', ha='center', va='center', fontsize=12)

    # 反馈部分
    for i in range(1, 3):
        x = 1 + i * 2
        rect = FancyBboxPatch((x, 0.5), 1.2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x + 0.6, 0.9, f'$z^{{-{i}}}$', ha='center', va='center', fontsize=12)

    # 乘法器 - 前馈
    for i in range(3):
        x = 1.3 + i * 2
        ax2.plot(x, 2.6, 'o', color='#D32F2F', markersize=16, zorder=5)
        ax2.text(x, 2.6, f'$b_{i}$', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # 乘法器 - 反馈
    for i in range(1, 3):
        x = 1.3 + i * 2
        ax2.plot(x, 1.7, 'o', color='#D32F2F', markersize=16, zorder=5)
        ax2.text(x, 1.7, f'$a_{i}$', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # 加法器
    ax2.plot(7.5, 2.6, 'o', color='#2E7D32', markersize=22, zorder=5)
    ax2.text(7.5, 2.6, '$\\Sigma$', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    # 输入输出
    ax2.annotate('', xy=(0.5, 3.6), xytext=(0.1, 3.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    ax2.text(-0.1, 3.6, '$x[n]$', ha='right', va='center', fontsize=13, fontweight='bold')

    ax2.annotate('', xy=(9.5, 2.6), xytext=(7.8, 2.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    ax2.text(9.6, 2.6, '$y[n]$', ha='left', va='center', fontsize=13, fontweight='bold')

    # 反馈环路
    ax2.annotate('', xy=(8.5, 0.9), xytext=(7.8, 2.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#E65100',
                               connectionstyle='arc3,rad=0.3'))
    ax2.annotate('', xy=(1.3, 1.5), xytext=(1.3, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='#E65100'))

    # 连线
    for i in range(3):
        x = 1.3 + i * 2
        ax2.plot([x, x], [3.2, 2.8], color='#333', lw=1.5)
        if i < 2:
            ax2.plot([x + 0.6, x + 1.4], [3.6, 3.6], color='#333', lw=1.5)
    ax2.plot([1.3, 7.3], [2.6, 2.6], color='#333', lw=1, alpha=0.3)

    ax2.text(5, -0.1, '输出反馈回来，再次参与计算', ha='center', fontsize=11,
             style='italic', color='#E65100')

    plt.tight_layout()
    plt.savefig('fig1_structure.png', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print("fig1_structure.png done")


# ============================================================
# 图 2: 线性相位 vs 非线性相位 示意图
# ============================================================
def fig2_phase_illustration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    t = np.linspace(0, 4e-3, 1000)
    # 两个频率叠加
    f1, f2 = 1000, 3000
    x = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)

    # 线性相位：两个频率延迟相同
    delay_s = 0.2e-3
    y_linear = np.sin(2*np.pi*f1*(t - delay_s)) + 0.5*np.sin(2*np.pi*f2*(t - delay_s))

    # 非线性相位：两个频率延迟不同
    d1 = 0.2e-3
    d2 = 0.05e-3
    y_nonlinear = np.sin(2*np.pi*f1*(t - d1)) + 0.5*np.sin(2*np.pi*f2*(t - d2))

    ax1.plot(t*1000, x, 'k', alpha=0.4, label='原始信号')
    ax1.plot(t*1000, y_linear, 'b', linewidth=2, label='线性相位后')
    ax1.set_title('线性相位：整体平移，波形不变', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间 (ms)')
    ax1.set_ylabel('幅度')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 4])

    ax2.plot(t*1000, x, 'k', alpha=0.4, label='原始信号')
    ax2.plot(t*1000, y_nonlinear, 'r', linewidth=2, label='非线性相位后')
    ax2.set_title('非线性相位：各频率延迟不同，波形畸变', fontsize=12, fontweight='bold')
    ax2.set_xlabel('时间 (ms)')
    ax2.set_ylabel('幅度')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 4])

    plt.tight_layout()
    plt.savefig('fig2_phase_illustration.png', bbox_inches='tight')
    plt.close()
    print("fig2_phase_illustration.png done")


# ============================================================
# 图 3: IIR 频率响应（幅度、相位、群延迟）
# ============================================================
def fig3_iir_response():
    iir_b, iir_a = signal.butter(2, fc, btype='high', fs=fs)

    w = np.linspace(20, 20000, 4096)
    _, h = signal.freqz(iir_b, iir_a, worN=w, fs=fs)
    _, gd = signal.group_delay((iir_b, iir_a), w=w, fs=fs)

    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax1.semilogx(w, mag_db, 'b', linewidth=2)
    ax1.axhline(-3, color='gray', linestyle='--', alpha=0.5, label='-3dB')
    ax1.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('幅度 (dB)')
    ax1.set_title('二阶巴特沃斯高通 IIR 滤波器频率响应\n($f_c$=1kHz, $f_s$=48kHz)', fontsize=13, fontweight='bold')
    ax1.set_ylim([-60, 5])
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    ax2.semilogx(w, phase_deg, 'b', linewidth=2)
    ax2.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('相位 (°)')
    ax2.grid(True, which='both', alpha=0.3)

    ax3.semilogx(w, gd, 'b', linewidth=2)
    ax3.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    ax3.set_ylabel('群延迟 (采样点)')
    ax3.set_xlabel('频率 (Hz)')
    ax3.grid(True, which='both', alpha=0.3)
    ax3.set_xlim([20, 20000])

    plt.tight_layout()
    plt.savefig('fig3_iir_response.png', bbox_inches='tight')
    plt.close()
    print("fig3_iir_response.png done")


# ============================================================
# 图 4: FIR vs IIR 频率响应对比
# ============================================================
def fig4_fir_vs_iir():
    iir_b, iir_a = signal.butter(2, fc, btype='high', fs=fs)
    fir_b = signal.firwin(215, fc, pass_zero=False, fs=fs)

    w = np.linspace(20, 20000, 4096)
    _, h_iir = signal.freqz(iir_b, iir_a, worN=w, fs=fs)
    _, h_fir = signal.freqz(fir_b, 1, worN=w, fs=fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    mag_iir = 20 * np.log10(np.abs(h_iir) + 1e-12)
    mag_fir = 20 * np.log10(np.abs(h_fir) + 1e-12)

    ax1.semilogx(w, mag_iir, 'r', linewidth=2, label='IIR 二阶巴特沃斯')
    ax1.semilogx(w, mag_fir, 'b', linewidth=2, label='FIR 215阶线性相位')
    ax1.axhline(-3, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('幅度 (dB)')
    ax1.set_title('IIR vs FIR 高通滤波器幅度响应对比', fontsize=13, fontweight='bold')
    ax1.set_ylim([-80, 5])
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    phase_iir = np.unwrap(np.angle(h_iir)) * 180 / np.pi
    phase_fir = np.unwrap(np.angle(h_fir)) * 180 / np.pi

    ax2.semilogx(w, phase_iir, 'r', linewidth=2, label='IIR')
    ax2.semilogx(w, phase_fir, 'b', linewidth=2, label='FIR')
    ax2.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('相位 (°)')
    ax2.set_xlabel('频率 (Hz)')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_xlim([20, 20000])

    plt.tight_layout()
    plt.savefig('fig4_fir_vs_iir.png', bbox_inches='tight')
    plt.close()
    print("fig4_fir_vs_iir.png done")


# ============================================================
# 图 5: 群延迟对比曲线
# ============================================================
def fig5_group_delay():
    iir_b, iir_a = signal.butter(2, fc, btype='high', fs=fs)
    fir_b = signal.firwin(215, fc, pass_zero=False, fs=fs)

    w = np.linspace(20, 20000, 4096)
    _, gd_iir = signal.group_delay((iir_b, iir_a), w=w, fs=fs)
    _, gd_fir = signal.group_delay((fir_b, 1), w=w, fs=fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.semilogx(w, gd_iir, 'r', linewidth=2, label='IIR 二阶巴特沃斯')
    ax1.semilogx(w, gd_fir, 'b', linewidth=2, label='FIR 215阶线性相位')
    ax1.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('群延迟 (采样点)')
    ax1.set_title('IIR vs FIR 群延迟对比', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    # 放大通带部分
    ax2.semilogx(w, gd_iir, 'r', linewidth=2, label='IIR')
    ax2.axhline(107, color='b', linewidth=2, linestyle='--', alpha=0.5, label='FIR 常数群延迟 (107)')
    ax2.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('群延迟 (采样点)')
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylim([0, 15])
    ax2.set_xlim([500, 20000])
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    # 标注关键点
    test_freqs = [1000, 2000, 5000, 10000]
    for f in test_freqs:
        w_pt = 2 * np.pi * f / fs
        _, gd_pt = signal.group_delay((iir_b, iir_a), w=[w_pt], fs=fs)
        ax2.plot(f, gd_pt[0], 'ro', markersize=8, zorder=5)
        ax2.annotate(f'{gd_pt[0]:.2f}', xy=(f, gd_pt[0]),
                    xytext=(f*1.1, gd_pt[0]+1.5),
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))

    plt.tight_layout()
    plt.savefig('fig5_group_delay.png', bbox_inches='tight')
    plt.close()
    print("fig5_group_delay.png done")


# ============================================================
# 图 6: 方波实验
# ============================================================
def fig6_square_wave():
    iir_b, iir_a = signal.butter(2, fc, btype='high', fs=fs)
    fir_b = signal.firwin(215, fc, pass_zero=False, fs=fs)

    t = np.arange(0, 0.006, 1/fs)
    x_sq = (np.sin(2*np.pi*1000*t) +
            np.sin(2*np.pi*3000*t)/3 +
            np.sin(2*np.pi*5000*t)/5 +
            np.sin(2*np.pi*7000*t)/7 +
            np.sin(2*np.pi*9000*t)/9)

    y_iir = signal.lfilter(iir_b, iir_a, x_sq)
    y_fir = signal.lfilter(fir_b, 1, x_sq)
    y_fir_comp = np.roll(y_fir, -107)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    t_ms = t * 1000

    axes[0].plot(t_ms, x_sq, 'k', linewidth=1.5)
    axes[0].set_title('原始 1kHz 方波（含 5 次谐波）', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('幅度')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 6])

    axes[1].plot(t_ms, y_iir, 'r', linewidth=1.5, label='IIR 高通后')
    axes[1].plot(t_ms, y_fir_comp, 'b', linewidth=1.5, linestyle='--',
                label='FIR 高通后（延迟补偿）')
    axes[1].set_title('IIR vs FIR 滤波结果对比', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('幅度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 6])

    diff = y_iir - y_fir_comp
    axes[2].fill_between(t_ms, diff, 0, color='#4CAF50', alpha=0.5)
    axes[2].plot(t_ms, diff, 'g', linewidth=1)
    axes[2].set_title(f'波形差（IIR - FIR），最大差异: {np.max(np.abs(diff)):.4f}',
                      fontsize=12, fontweight='bold')
    axes[2].set_xlabel('时间 (ms)')
    axes[2].set_ylabel('差值')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 6])

    plt.tight_layout()
    plt.savefig('fig6_square_wave.png', bbox_inches='tight')
    plt.close()
    print("fig6_square_wave.png done")


# ============================================================
# 图 7: 零相位滤波原理示意
# ============================================================
def fig7_zero_phase():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    t = np.linspace(0, 5e-3, 1000)
    x = np.sin(2*np.pi*1000*t) + 0.6*np.sin(2*np.pi*3000*t)

    iir_b, iir_a = signal.butter(2, fc, btype='high', fs=fs)

    # 正向滤波
    y1 = signal.lfilter(iir_b, iir_a, x)
    # 反转
    y1_rev = y1[::-1]
    # 反向滤波
    y2 = signal.lfilter(iir_b, iir_a, y1_rev)
    # 再反转
    y_zp = y2[::-1]

    t_ms = t * 1000

    axes[0].plot(t_ms, x, 'k', alpha=0.5, label='原始')
    axes[0].plot(t_ms, y1, 'r', label='正向滤波后')
    axes[0].set_title('第一步：正向 IIR 滤波', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('时间 (ms)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ms, x, 'k', alpha=0.5, label='原始')
    axes[1].plot(t_ms, y1, 'r', alpha=0.4, label='正向')
    axes[1].plot(t_ms, y_zp, 'b', linewidth=2, label='零相位结果')
    axes[1].set_title('第二步：反转 + 再滤波 + 再反转', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('时间 (ms)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 幅度响应对比
    w = np.linspace(20, 20000, 2048)
    _, h_orig = signal.freqz(iir_b, iir_a, worN=w, fs=fs)
    h_zp = h_orig * np.conj(h_orig)  # = |H|^2，零相位

    axes[2].semilogx(w, 20*np.log10(np.abs(h_orig) + 1e-12), 'r',
                     linewidth=2, label='单次 IIR')
    axes[2].semilogx(w, 20*np.log10(np.abs(h_zp) + 1e-12), 'b',
                     linewidth=2, label='零相位 (两次)')
    axes[2].set_title('幅度响应对比', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('频率 (Hz)')
    axes[2].set_ylabel('dB')
    axes[2].set_ylim([-80, 5])
    axes[2].legend()
    axes[2].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig7_zero_phase.png', bbox_inches='tight')
    plt.close()
    print("fig7_zero_phase.png done")


# ============================================================
# 图 8: IIR 通带内群延迟变化（热力图风格）
# ============================================================
def fig8_phase_error_heatmap():
    iir_b, iir_a = signal.butter(2, fc, btype='high', fs=fs)

    test_freqs = np.array([1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000])
    w_pts = [2*np.pi*f/fs for f in test_freqs]
    _, gd = signal.group_delay((iir_b, iir_a), w=w_pts, fs=fs)

    # 以 5kHz 为基准计算相位误差
    ref_idx = np.argmin(np.abs(test_freqs - 5000))
    ref_gd = gd[ref_idx]
    delta_samples = gd - ref_gd
    delta_us = delta_samples / fs * 1e6
    phase_errors = 360 * test_freqs * delta_us * 1e-6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：群延迟
    ax1.bar(range(len(test_freqs)), gd, color='#2196F3', alpha=0.8)
    ax1.axhline(ref_gd, color='red', linestyle='--', alpha=0.7, label=f'基准 ({ref_gd:.2f})')
    ax1.set_xticks(range(len(test_freqs)))
    ax1.set_xticklabels([f'{f/1000:.0f}k' for f in test_freqs], rotation=45)
    ax1.set_xlabel('频率 (Hz)')
    ax1.set_ylabel('群延迟 (采样点)')
    ax1.set_title('IIR 通带内各频率的群延迟', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图：相位误差
    colors = ['#F44336' if abs(e) > 10 else '#FF9800' if abs(e) > 5 else '#4CAF50'
              for e in phase_errors]
    bars = ax2.bar(range(len(test_freqs)), phase_errors, color=colors, alpha=0.8)
    ax2.axhline(0, color='gray', linewidth=1)
    ax2.set_xticks(range(len(test_freqs)))
    ax2.set_xticklabels([f'{f/1000:.0f}k' for f in test_freqs], rotation=45)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('相位误差 (°)')
    ax2.set_title(f'IIR 通带内相位误差（基准: 5kHz）', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, phase_errors)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5*np.sign(val),
                f'{val:.1f}°', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig8_phase_error_heatmap.png', bbox_inches='tight')
    plt.close()
    print("fig8_phase_error_heatmap.png done")


# ============================================================
# 图 9: 计算量对比柱状图
# ============================================================
def fig9_computation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    categories = ['乘法', '加法', '总乘加']
    iir_vals = [5, 4, 9]
    fir_vals = [215, 214, 429]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, iir_vals, width, label='IIR 二阶', color='#2196F3', alpha=0.8)
    bars2 = ax1.bar(x + width/2, fir_vals, width, label='FIR 215阶', color='#FF9800', alpha=0.8)

    ax1.set_ylabel('每采样点运算次数')
    ax1.set_title('IIR vs FIR 每采样点计算量', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{int(bar.get_height())}', ha='center', fontsize=11, fontweight='bold')
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{int(bar.get_height())}', ha='center', fontsize=11, fontweight='bold')

    # 右图：倍数关系
    ratios = [43, 53.5, 47.7]
    bars3 = ax2.bar(categories, ratios, color=['#2196F3', '#4CAF50', '#F44336'], alpha=0.8)
    ax2.set_ylabel('FIR / IIR 倍数')
    ax2.set_title('FIR 相对 IIR 的计算量倍数', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars3, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}x', ha='center', fontsize=14, fontweight='bold', color='#D32F2F')

    plt.tight_layout()
    plt.savefig('fig9_computation.png', bbox_inches='tight')
    plt.close()
    print("fig9_computation.png done")


# ============================================================
# 运行所有
# ============================================================
if __name__ == '__main__':
    fig1_structure()
    fig2_phase_illustration()
    fig3_iir_response()
    fig4_fir_vs_iir()
    fig5_group_delay()
    fig6_square_wave()
    fig7_zero_phase()
    fig8_phase_error_heatmap()
    fig9_computation()
    print("\n全部图片生成完毕。")
