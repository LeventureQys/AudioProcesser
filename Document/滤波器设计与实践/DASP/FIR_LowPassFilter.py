import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
import io
import scipy.io.wavfile as wav
#本文来自https://blog.csdn.net/Andius/article/details/140151606?spm=1001.2014.3001.5502
# 在本文中，我将导入一个音频，并将一个低通滤波器作用于这个音频之上，并最终返回低通滤波器的频率响应曲线以及这个FIR滤波器的频率响应曲线
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

#获得窗函数结果
def get_window(window, Nx, fftbins=True):
    if isinstance(window, str):
        window = window.lower()
        if window == 'hamming':
            return hamming_window(Nx)
        elif window == 'hann':
            return hann_window(Nx)
        elif window == 'blackman':
            return blackman_window(Nx)
        else:
            raise ValueError("Unknown window type.")
    else:
        raise ValueError("Window type must be a string.")
# 汉明窗
def hamming_window(Nx):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(Nx) / (Nx - 1))
# 汉窗
def hann_window(Nx):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(Nx) / (Nx - 1)))
# 布莱克曼窗
def blackman_window(Nx):
    return 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(Nx) / (Nx - 1)) + 0.08 * np.cos(4 * np.pi * np.arange(Nx) / (Nx - 1))


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

# 读取MP3文件并转换为音频信号
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
cutoff_hz = 500  # 截止频率
numtaps = 101  # 滤波器阶数

# 设计FIR低通滤波器

fir_coeff = firwin(numtaps, cutoff_hz, fs=sampling_rate)

# 使用FIR滤波器
#lfilter函数通过求解线性时不变系统的差分方程来进行滤波操作，其中fir_coeff为分子参数，1.0为分母的参数
filtered_signal = lfilter(fir_coeff, 1.0, signal)

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

# 计算并绘制FIR滤波器的频率响应
w, h = freqz(fir_coeff, worN=8000)
plt.figure(figsize=(8, 6))
plt.plot(0.5 * sampling_rate * w / np.pi, 20 * np.log10(np.abs(h)), 'b')
plt.xlim(0, 2000)
plt.ylim(-60, 5)
plt.title('FIR滤波器频率响应')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度 (dB)')
plt.grid()
plt.show()
