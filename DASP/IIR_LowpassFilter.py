import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
import io
import scipy.io.wavfile as wav
#本文来自https://blog.csdn.net/Andius/article/details/140151606?spm=1001.2014.3001.5502
# 在本文中，我将导入一个音频，并将一个低通滤波器作用于这个音频之上，并最终返回低通滤波器的频率响应曲线以及这个IIR滤波器的频率响应曲线
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 读取MP3文件并转换为音频信号,听我最喜欢的women on the hills
audio = AudioSegment.from_mp3(r'../Audio/mp3/2.mp3')

# 使用 BytesIO 作为中间缓冲区
with io.BytesIO() as buf:
    audio.export(buf, format="wav")
    buf.seek(0)
    sampling_rate, signal = wav.read(buf)

# 如果音频是立体声，转换为单声道
if len(signal.shape) > 1:
    signal = signal.mean(axis=1)

# 计算滤波器系数
Filter_w0 = 500  # 滤波器中心频率
Filter_Q = 1   # 滤波器峰值信噪比
Filter_T = 1 / sampling_rate  # 滤波器周期
omega_0 = 2 * np.pi * Filter_w0 / sampling_rate  # 归一化角频率
alpha = np.sin(omega_0) / (2 * Filter_Q)

coeff_b0 = (1 - np.cos(omega_0)) / 2
coeff_b1 = 1 - np.cos(omega_0)
coeff_b2 = (1 - np.cos(omega_0)) / 2
coeff_a0 = 1 + alpha
coeff_a1 = -2 * np.cos(omega_0)
coeff_a2 = 1 - alpha

# 归一化滤波器系数
b = [coeff_b0 / coeff_a0, coeff_b1 / coeff_a0, coeff_b2 / coeff_a0]
a = [1, coeff_a1 / coeff_a0, coeff_a2 / coeff_a0]

# 使用IIR滤波器
filtered_signal = lfilter(b, a, signal)

# 计算原始信号和滤波后信号的频域响应
N = len(signal)
yf_signal = fft(signal)
yf_filtered_signal = fft(filtered_signal)
xf = fftfreq(N, 1 / sampling_rate)

# 只取前一半频率，因为FFT是对称的
xf = xf[:N // 2]
yf_signal = 2.0 / N * np.abs(yf_signal[:N // 2])
yf_filtered_signal = 2.0 / N * np.abs(yf_filtered_signal[:N // 2])

# 转换为dB值
yf_signal_db = 20 * np.log10(yf_signal + np.finfo(float).eps)  # 添加一个很小的值以避免log(0)
yf_filtered_signal_db = 20 * np.log10(yf_filtered_signal + np.finfo(float).eps)

# 假设 signal, filtered_signal, xf, yf_signal_db, yf_filtered_signal_db, b, a, sampling_rate 已经定义

# 绘制原始信号和滤波后的信号
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# 绘制原始信号
axs[0, 0].plot(signal)
axs[0, 0].set_title('原始信号')

# 绘制滤波后的信号
axs[0, 1].plot(filtered_signal)
axs[0, 1].set_title('滤波后的信号')
axs[0, 1].set_xlabel('时间 (样本)')

# 绘制频域响应（dB）
axs[1, 0].plot(xf, yf_signal_db, label='原始信号 (dB)')
axs[1, 0].plot(xf, yf_filtered_signal_db, label='滤波后的信号 (dB)')
axs[1, 0].set_title('频域响应 (dB)')
axs[1, 0].set_xlabel('频率 (Hz)')
axs[1, 0].set_ylabel('幅度 (dB)')
axs[1, 0].legend()

# 计算并绘制IIR滤波器的频率响应
w, h = freqz(b, a, worN=8000)
axs[1, 1].plot(0.5 * sampling_rate * w / np.pi, 20 * np.log10(np.abs(h)), 'b')
axs[1, 1].set_xlim(0, 2000)
axs[1, 1].set_ylim(-24, 12)
axs[1, 1].set_title('IIR滤波器频率响应')
axs[1, 1].set_xlabel('频率 (Hz)')
axs[1, 1].set_ylabel('幅度 (dB)')
axs[1, 1].grid()

plt.tight_layout()
plt.show()