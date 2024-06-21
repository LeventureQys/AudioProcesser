import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

def low_shelf_eq(fc, fs, gain, Q):
    """
    计算Low Shelf EQ滤波器的系数
    :param fc: 中心频率
    :param fs: 采样率
    :param gain: 增益
    :param Q: Q值
    :return: b, a 系数
    """
    A = 10 ** (gain / 40)
    omega = 2 * np.pi * fc / fs
    alpha = np.sin(omega) / (2 * Q)
    
    b0 = A * ((A + 1) - (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega))
    b2 = A * ((A + 1) - (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(omega))
    a2 = (A + 1) + (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha
    
    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1, a1 / a0, a2 / a0]
    
    return b, a

def plot_frequency_response(fs, b, a):
    """
    绘制频率响应曲线
    :param fs: 采样率
    :param b: 滤波器的b系数
    :param a: 滤波器的a系数
    """
    w, h = freqz(b, a, worN=20000)
    freq = w * fs / (2 * np.pi)
    response = 20 * np.log10(abs(h))
    
    plt.figure(figsize=(10, 6))
    plt.plot(freq, response, label='Low Shelf Filter Frequency Response')
    plt.xscale('log')
    plt.title('Low Shelf Filter Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(which='both', linestyle='--')
    plt.axhline(0, color='black', lw=2)
    plt.legend()
    plt.show()

# 输入参数
fc = 1000      # 中心频率
fs = 96000     # 采样率
gain = 6       # 增益
Q = 1          # Q值

# 计算滤波器系数
b, a = low_shelf_eq(fc, fs, gain, Q)

# 绘制频率响应曲线
plot_frequency_response(fs, b, a)
