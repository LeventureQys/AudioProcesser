import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fftpack import fft, ifft

# 1. 读取纯噪声信号，建立噪声频谱模型
noise_rate, noise_data = wav.read('./AudioSource/Noisy.wav')
# 如果噪声是立体声，转换为单声道
if len(noise_data.shape) > 1:
    noise_data = noise_data.mean(axis=1)

# 计算噪声频谱模型
frame_size = 1024
overlap = 512
noise_frames = []
for i in range(0, len(noise_data) - frame_size, overlap):
    frame = noise_data[i:i+frame_size]
    windowed_frame = frame * get_window('hamming', frame_size)
    spectrum = fft(windowed_frame)
    noise_frames.append(np.abs(spectrum))
# 计算平均噪声频谱
noise_spectrum = np.mean(noise_frames, axis=0)

# 2. 读取需要降噪的音频信号
rate, data = wav.read('./AudioSource/Source.wav')
if len(data.shape) > 1:
    data = data.mean(axis=1)

# 保存降噪前的频谱，用于后续对比
original_spectrum = []

# 进行降噪处理
output_data = np.zeros(len(data))
window = get_window('hamming', frame_size)
for i in range(0, len(data) - frame_size, overlap):
    # 2. 分帧
    frame = data[i:i+frame_size]
    # 3. 窗函数处理
    windowed_frame = frame * window
    # 4. FFT
    spectrum = fft(windowed_frame)
    # 保存原始频谱
    original_spectrum.append(np.abs(spectrum))

    # 5. 频谱减法
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    subtracted_magnitude = magnitude - noise_spectrum
    # 6. 处理负值和伪影
    subtracted_magnitude = np.maximum(subtracted_magnitude, 0.0)

    # 7. IFFT
    reconstructed_spectrum = subtracted_magnitude * np.exp(1j * phase)
    reconstructed_frame = np.real(ifft(reconstructed_spectrum))
    # 8. Overlap-Add 重建信号
    output_data[i:i+frame_size] += reconstructed_frame * window

# 保存降噪后的频谱，用于对比
denoised_spectrum = []
for i in range(0, len(output_data) - frame_size, overlap):
    frame = output_data[i:i+frame_size]
    windowed_frame = frame * window
    spectrum = fft(windowed_frame)
    denoised_spectrum.append(np.abs(spectrum))

# 9. 后处理（可选，这里暂不实现）

# 保存降噪后的音频
wav.write('./AudioSource/Denosed.wav', rate, output_data.astype(np.int16))

# 绘制频谱对比
# 计算平均频谱
original_spectrum_mean = np.mean(original_spectrum, axis=0)
denoised_spectrum_mean = np.mean(denoised_spectrum, axis=0)
freqs = np.linspace(0, rate, frame_size)

plt.figure(figsize=(12,6))
plt.plot(freqs[:frame_size//2], 20*np.log10(original_spectrum_mean[:frame_size//2]), label='Original')
plt.plot(freqs[:frame_size//2], 20*np.log10(denoised_spectrum_mean[:frame_size//2]), label='Processed')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度 (dB)')
plt.title('降噪前后频谱对比')
plt.legend()
plt.show()
