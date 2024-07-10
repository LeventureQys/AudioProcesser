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

# 读取MP3文件并转换为音频信号
audio = AudioSegment.from_mp3(r'D:\\WorkShop\\CurrentWork\\FIRFilter_Venture\\Audio\\mp3\\2.mp3')

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

# 绘制原始信号和滤波后的信号
plt.figure(figsize=(12, 12))
plt.subplot(3, 1, 1)
plt.plot(signal)
plt.title('原始信号')
plt.subplot(3, 1, 2)
plt.plot(filtered_signal)
plt.title('滤波后的信号')
plt.xlabel('时间 (样本)')

# 绘制频域响应（dB）
plt.subplot(3, 1, 3)
plt.plot(xf, yf_signal_db, label='原始信号 (dB)')
plt.plot(xf, yf_filtered_signal_db, label='滤波后的信号 (dB)')
plt.title('频域响应 (dB)')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度 (dB)')
plt.legend()

plt.tight_layout()
plt.show()

# 计算并绘制IIR滤波器的频率响应
w, h = freqz(b, a, worN=8000)
plt.figure(figsize=(8, 6))
plt.plot(0.5 * sampling_rate * w / np.pi, 20 * np.log10(np.abs(h)), 'b')
plt.xlim(0,2000)
plt.ylim(-24,12)
plt.title('IIR滤波器频率响应')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度 (dB)')
plt.grid()
plt.show()
