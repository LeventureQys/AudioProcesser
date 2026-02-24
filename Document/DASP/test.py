import numpy as np
from scipy.signal import get_window

def firwin(numtaps, cutoff, window='hamming', pass_zero=True, fs=2.0):
    # 计算归一化频率
    cutoff = np.asarray(cutoff) / (0.5 * fs)
    
    # 如果是高通滤波器，处理 pass_zero 参数
    if not pass_zero:
        cutoff = 1 - cutoff
    
    # 生成理想脉冲响应
    m = np.arange(0, numtaps) - (numtaps - 1) / 2
    h = np.sinc(cutoff * m)
    
    # 生成窗函数
    win = get_window(window, numtaps, fftbins=False)
    
    # 应用窗函数
    h = h * win
    
    # 归一化滤波器系数
    if pass_zero:
        h /= np.sum(h)
    else:
        h /= -np.sum(h)
        h[int((numtaps - 1) / 2)] += 1
    
    return h

# 测试 firwin 函数
numtaps = 101
cutoff = 100  # 归一化截止频率
fs = 1000

coefficients = firwin(numtaps, cutoff, fs=fs)
print(coefficients)