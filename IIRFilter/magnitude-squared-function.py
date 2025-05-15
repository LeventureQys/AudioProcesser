import numpy as np
import matplotlib.pyplot as plt

# 定义二阶低通滤波器的幅度平方函数
def magnitude_squared(omega, omega_n=1.0, zeta=0.707):
    """计算二阶系统的幅度平方函数 |H(jω)|²
    参数:
        omega: 角频率 (rad/s)
        omega_n: 自然频率 (默认1.0)
        zeta: 阻尼比 (默认0.707，临界阻尼)
    """
    return 1 / ((1 - (omega/omega_n)**2)**2 + (2 * zeta * (omega/omega_n))**2)

# 生成频率范围 (0.1到10倍自然频率)
omega = np.logspace(-1, 1, 500)  # 对数坐标更清晰
mag_squared = magnitude_squared(omega)

# 绘制幅度平方函数
plt.figure(figsize=(10, 6))
plt.semilogx(omega, mag_squared, label=r'$|H(j\omega)|^2$', linewidth=2)
plt.axvline(x=1.0, color='red', linestyle='--', label=r'$\omega_n$ (Natural Frequency)')

# 标注关键点
plt.title('Magnitude-Squared Function of a 2nd-Order Low-Pass Filter', fontsize=12)
plt.xlabel('Frequency (rad/s)', fontsize=12)
plt.ylabel(r'$|H(j\omega)|^2$', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.show()