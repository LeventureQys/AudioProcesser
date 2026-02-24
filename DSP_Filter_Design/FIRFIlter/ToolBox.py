#频谱绘制
import numpy as np
import matplotlib.pyplot as plt

def PrintFrequencyResponse(signal, fs,title = ""):
    #计算傅里叶变换得到频率曲线，并绘制
    #value : 1. 频率坐标， 2.频率对应的幅度 3.曲线名称，如果有的话，则会插入到图表中去，如果没有，则不插入
    T = 1 / fs  # 采样间隔
    t = np.linspace(0, T, len(signal), endpoint=False)  # 时间向量

    N = len(signal)  # 信号长度

    fft_values = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_values) / N  # 幅度归一化
    frequencies = np.fft.fftfreq(N, T)  # 频率坐标
    half_N = N // 2  # 取正频率部分
    frequencies = frequencies[:half_N]
    fft_magnitude = fft_magnitude[:half_N]
    if(title != ""):
        plt.plot(frequencies,fft_magnitude,label = title)
    return frequencies,fft_magnitude






