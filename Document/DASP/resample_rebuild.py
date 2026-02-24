import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
def sinc_interp(x, s, u):
    """ Sinc interpolation """
    if len(x) != len(s):
        raise ValueError('x 和 s 必须长度相同')
    
    # Calculate the sinc matrix
    sinc_matrix = np.tile(u, (len(x), 1)) - np.tile(x[:, np.newaxis], (1, len(u)))
    sinc_matrix = np.sinc(sinc_matrix / (x[1] - x[0]))  # Normalize by the sample spacing
    
    # Perform the sinc interpolation
    return np.dot(s, sinc_matrix)

# 定义正弦信号的参数
frequency = 10  # 10Hz的正弦信号
duration = 1  # 持续时间为1秒
sampling_rate1 = 40  # 采样率为20Hz
sampling_rate2 = 400  # 采样率为400Hz

# 生成时间序列
t1 = np.linspace(0, duration, int(sampling_rate1 * duration), endpoint=False)
t2 = np.linspace(0, duration, int(sampling_rate2 * duration), endpoint=False)

# 生成正弦信号
signal1 = np.sin(2 * np.pi * frequency * t1)
signal2 = np.sin(2 * np.pi * frequency * t2)

# 生成用于重构的高分辨率时间序列，确保在t1范围内
t_high_res = np.linspace(t1[0], t1[-1], 1000, endpoint=False)

# 使用线性插值进行重构
linear_interp = interp1d(t1, signal1, kind='linear')
linear_interp2 = interp1d(t2,signal2,kind='linear')
signal1_linear_reconstructed = linear_interp(t_high_res)
signal2_linear_reconstructed2 = linear_interp2(t_high_res)
# 使用Sinc插值进行重构
signal1_sinc_reconstructed = sinc_interp(t1, signal1, t_high_res)
signal1_sinc_reconstructed2 = sinc_interp(t2,signal2,t_high_res)
# 绘制信号
plt.figure(figsize=(12, 12))

# 绘制原始信号
plt.subplot(3, 1, 1)
plt.plot(t1,signal1,'-', label='原始信号采样1 (40Hz)')
plt.plot(t2, signal2, '-', label='原始信号采样2 (400Hz)')
plt.title('采样信号')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 绘制线性重构信号
plt.subplot(3, 1, 2)
plt.plot(t_high_res, signal2_linear_reconstructed2, 'o', label='线性重构信号2')
plt.plot(t_high_res, signal1_linear_reconstructed, '-', label='线性重构信号1')
plt.title('线性重构信号组')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 绘制Sinc重构信号
plt.subplot(3, 1, 3)
plt.plot(t_high_res, signal1_sinc_reconstructed2, 'o', label='Sinc重构信号2')
plt.plot(t_high_res, signal1_sinc_reconstructed, '-', label='Sinc重构信号1')
plt.title('Sinc重构信号')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
