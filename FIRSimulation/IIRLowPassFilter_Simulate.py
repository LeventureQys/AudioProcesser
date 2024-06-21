import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

def low_pass_eq(fc, fs, Q):
    """
    计算Low Pass EQ滤波器的系数
    :param fc: 中心频率
    :param fs: 采样率
    :param Q: Q值
    :return: b, a 系数
    """
    omega = 2 * np.pi * fc / fs
    alpha = np.sin(omega) / (2 * Q)
    
    b0 = (1 - np.cos(omega)) / 2
    b1 =  1 - np.cos(omega)
    b2 = (1 - np.cos(omega)) / 2
    a0 =  1 + alpha
    a1 = -2 * np.cos(omega)
    a2 =  1 - alpha
    
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
    plt.plot(freq, response, label='Low Pass Filter Frequency Response')
    plt.xscale('log')
    plt.title('Low Pass Filter Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(which='both', linestyle='--')
    plt.axhline(0, color='black', lw=2)
    plt.legend()
    plt.show()

# 输入参数
fc = 1000      # 中心频率
fs = 96000     # 采样率
Q = 1          # Q值

# 计算滤波器系数
b, a = low_pass_eq(fc, fs, Q)

# 绘制频率响应曲线
plot_frequency_response(fs, b, a)
