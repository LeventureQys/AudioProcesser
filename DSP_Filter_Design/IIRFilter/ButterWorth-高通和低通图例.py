import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

# 定义巴特沃斯滤波器频率响应
def butterworth_lp(omega):
    return 1 / np.sqrt(1 + omega**4)  # 二阶巴特沃斯低通幅度响应

def butterworth_hp(omega):
    return omega**2 / np.sqrt(1 + omega**4)  # 二阶巴特沃斯高通幅度响应

# 创建频率范围（对数尺度更清晰）
omega = np.logspace(-2, 2, 1000)  # 0.01 到 100 rad/s

# 计算幅度响应
mag_lp = butterworth_lp(omega)
mag_hp = butterworth_hp(omega)

# 创建图形
plt.figure(figsize=(12, 6))

# 低通滤波器幅度响应
plt.subplot(1, 2, 1)
plt.semilogx(omega, 20*np.log10(mag_lp), 'b', linewidth=2)
plt.axvline(1, color='r', linestyle='--', label='Cutoff (ω=1)')
plt.axhline(-3, color='g', linestyle=':', label='-3 dB')
plt.title('Butterworth Low-Pass Filter\n(2nd Order)')
plt.xlabel('Frequency [rad/s] (log scale)')
plt.ylabel('Magnitude [dB]')
plt.grid(which='both', linestyle=':', alpha=0.7)
plt.legend()

# 添加说明箭头
plt.annotate('Passband: Low frequencies\nare passed', xy=(0.1, -0.5), 
             xytext=(0.1, -10), arrowprops=dict(arrowstyle="->"))
plt.annotate('Stopband: High frequencies\nare attenuated', xy=(10, -40), 
             xytext=(3, -30), arrowprops=dict(arrowstyle="->"))

# 高通滤波器幅度响应
plt.subplot(1, 2, 2)
plt.semilogx(omega, 20*np.log10(mag_hp), 'r', linewidth=2)
plt.axvline(1, color='b', linestyle='--', label='Cutoff (ω=1)')
plt.axhline(-3, color='g', linestyle=':', label='-3 dB')
plt.title('Butterworth High-Pass Filter\n(2nd Order)')
plt.xlabel('Frequency [rad/s] (log scale)')
plt.ylabel('Magnitude [dB]')
plt.grid(which='both', linestyle=':', alpha=0.7)
plt.legend()

# 添加说明箭头
plt.annotate('Stopband: Low frequencies\nare blocked', xy=(0.1, -40), 
             xytext=(0.3, -30), arrowprops=dict(arrowstyle="->"))
plt.annotate('Passband: High frequencies\nare passed', xy=(10, -0.5), 
             xytext=(3, -10), arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.show()