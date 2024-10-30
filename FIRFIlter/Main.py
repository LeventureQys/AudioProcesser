import ToolBox as tb
import numpy as np
import matplotlib.pyplot as plt
# 参数设置
fs = 1000  # 采样频率
T = 1 / fs  # 采样间隔
t = np.linspace(0, 1, fs, endpoint=False)  # 1秒钟的时间向量

# 创建测试信号：一个包含低频和高频成分的信号
freq_low = 5  # 低频成分
freq_high = 100  # 高频成分
signal = np.sin(2 * np.pi * freq_low * t) + 0.5 * np.sin(2 * np.pi * freq_high * t)

tb.PrintFrequencyResponse(signal,fs,'Test Signal')

plt.legend()
plt.show()
