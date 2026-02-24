import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 设置参数
omega_c = 1.0  # 截止频率 (rad/s)
omega = np.logspace(-2, 2, 1000)  # 频率范围 (0.01 到 100 rad/s)

# 计算一阶巴特沃斯滤波器响应
def first_order(omega, omega_c):
    return 1 / np.sqrt(1 + (omega/omega_c)**2)

# 计算二阶巴特沃斯滤波器响应
def second_order(omega, omega_c):
    return 1 / np.sqrt(1 + (omega/omega_c)**4)

# 计算传递函数
sys_first = signal.TransferFunction([1], [1, 1])  # 一阶: 1/(s+1)
sys_second = signal.TransferFunction([1], [1, np.sqrt(2), 1])  # 二阶: 1/(s²+√2s+1)

# 计算频率响应
w_first, mag_first = signal.freqresp(sys_first, omega)
w_second, mag_second = signal.freqresp(sys_second, omega)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制幅度响应 (对数坐标)
plt.semilogx(omega, 20*np.log10(first_order(omega, omega_c)), 
             label='1st Order (n=1)', linewidth=2, color='blue')
plt.semilogx(omega, 20*np.log10(second_order(omega, omega_c)), 
             label='2nd Order (n=2)', linewidth=2, color='red')

# 标注关键点
plt.axvline(x=omega_c, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=-3, color='black', linestyle=':', linewidth=1)
plt.text(omega_c*1.2, -25, f'Cutoff ($\omega_c$ = {omega_c} rad/s)', rotation=90, va='bottom')
plt.text(0.05, -2.8, '-3 dB Point', ha='left', va='bottom')

# 添加理论斜率标记
plt.text(10, -20, '-20 dB/decade', rotation=-20, color='blue', ha='center')
plt.text(10, -40, '-40 dB/decade', rotation=-40, color='red', ha='center')

# 设置图形属性
plt.title('Butterworth Filter Frequency Response', fontsize=14)
plt.xlabel('Frequency [rad/s] (log scale)', fontsize=12)
plt.ylabel('Magnitude [dB]', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.ylim(-60, 5)
plt.xlim(omega[0], omega[-1])

# 显示图形
plt.tight_layout()
plt.show()