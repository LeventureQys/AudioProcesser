import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def design_peaking_eq(f0, fs, Q, gain, N):
    """
    设计一个Peaking EQ类型的FIR滤波器
    
    参数:
    f0 (float): 中心频率 (Hz)
    fs (float): 采样率 (Hz)
    Q (float): Q值
    gain (float): 增益 (dB)
    N (int): 滤波器长度
    
    返回:
    h (numpy array): 滤波器系数
    """
    
    # 将增益转换为线性标度
    A = 10**(gain / 40)
    
    # 计算归一化中心频率和带宽
    f0_norm = f0 / fs
    BW = f0 / Q
    BW_norm = BW / fs

    # 理想脉冲响应
    n = np.arange(N)
    alpha = (N - 1) / 2
    h_d = A * (np.sinc(2 * BW_norm * (n - alpha)) * 
               np.cos(2 * np.pi * f0_norm * (n - alpha)))
    
    # 汉明窗口
    w = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    
    # 应用窗口函数
    h = h_d * w
    
    return h

# 参数设置
f0 = 1000  # 中心频率 (Hz)
fs = 8000  # 采样率 (Hz)
Q = 2  # Q值
gain = 6  # 增益 (dB)
N = 101  # 滤波器长度

# 设计滤波器
h = design_peaking_eq(f0, fs, Q, gain, N)

# 频率响应
H = np.fft.fft(h, 1024)
H = np.fft.fftshift(H)  # 移动零频到中心
H_dB = 20 * np.log10(np.abs(H))

# 绘制时域和频域响应
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.stem(np.arange(N), h, use_line_collection=True)
plt.title('时域脉冲响应')
plt.xlabel('样本点')
plt.ylabel('幅度')

plt.subplot(2, 1, 2)
f = np.linspace(-0.5, 0.5, len(H))
plt.plot(f, H_dB)
plt.title('频域响应')
plt.xlabel('归一化频率')
plt.ylabel('幅度 (dB)')
plt.grid()
plt.show()
