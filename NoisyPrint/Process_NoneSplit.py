import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# 1. 读取音频文件并预处理
def read_audio(filename):
    sample_rate, data = wav.read(filename)
    # 转换为单声道
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return sample_rate, data

# 读取噪声样本
noise_sample_rate, noise_data = read_audio('./AudioSource/Noisy.wav')

# 读取需要降噪的音频
signal_sample_rate, signal_data = read_audio('./AudioSource/Source.wav')

# 确保采样率一致
if noise_sample_rate != signal_sample_rate:
    print("错误：噪声和信号的采样率不一致！")
    exit(1)

# 2. 参数设置
frame_size = 1024        # 帧大小（样本数）
overlap_size = int(frame_size * 0.5)  # 重叠一半
window = np.hamming(frame_size)       # 汉明窗

# 3. 估计噪声功率谱
def estimate_noise_power(noise_data, frame_size, overlap_size, window):
    num_frames = int(np.ceil(len(noise_data) / (frame_size - overlap_size)))
    padding_length = (num_frames * (frame_size - overlap_size) + overlap_size) - len(noise_data)
    noise_data_padded = np.append(noise_data, np.zeros(padding_length))
    
    noise_frames = []
    for i in range(0, len(noise_data_padded) - frame_size + 1, frame_size - overlap_size):
        frame = noise_data_padded[i:i+frame_size] * window
        spectrum = np.fft.fft(frame)
        power_spectrum = (np.abs(spectrum) ** 2) / frame_size
        noise_frames.append(power_spectrum)
    noise_power = np.mean(noise_frames, axis=0)
    return noise_power

noise_power = estimate_noise_power(noise_data, frame_size, overlap_size, window)

# 4. 对信号进行频谱减法
def spectral_subtraction(signal_data, noise_power, frame_size, overlap_size, window):
    num_frames = int(np.ceil(len(signal_data) / (frame_size - overlap_size)))
    padding_length = (num_frames * (frame_size - overlap_size) + overlap_size) - len(signal_data)
    signal_data_padded = np.append(signal_data, np.zeros(padding_length))
    
    output = np.zeros(len(signal_data_padded))
    
    for i in range(0, len(signal_data_padded) - frame_size + 1, frame_size - overlap_size):
        frame = signal_data_padded[i:i+frame_size] * window
        spectrum = np.fft.fft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # 计算功率谱
        power_spectrum = (magnitude ** 2) / frame_size
        
        # 频谱减法
        subtracted_power = power_spectrum - noise_power
        # 防止出现负值
        subtracted_power = np.maximum(subtracted_power, 1e-10)
        
        # 计算新的幅度谱
        subtracted_magnitude = np.sqrt(subtracted_power * frame_size)
        
        # 重建频谱
        new_spectrum = subtracted_magnitude * np.exp(1j * phase)
        # 逆FFT
        new_frame = np.fft.ifft(new_spectrum).real
        # 重叠相加
        output[i:i+frame_size] += new_frame * window
    
    # 移除填充部分
    output = output[:len(signal_data)]
    return output

denoised_data = spectral_subtraction(signal_data, noise_power, frame_size, overlap_size, window)

# 5. 归一化处理
max_val = np.max(np.abs(denoised_data))
if max_val > 0:
    denoised_data = denoised_data / max_val

denoised_data = (denoised_data * 32767).astype(np.int16)

# 6. 保存降噪后的音频
wav.write('./AudioSource/Denoised.wav', signal_sample_rate, denoised_data)

# 7. 可视化频谱（可选）
def plot_spectrum(data, sample_rate, title):
    n = len(data)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    magnitude = np.abs(np.fft.fft(data))
    plt.figure(figsize=(10, 4))
    plt.plot(freq[:n//2], magnitude[:n//2])
    plt.title(title)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅值')
    plt.grid(True)
    plt.show()

# 原始信号频谱
plot_spectrum(signal_data, signal_sample_rate, '原始信号频谱')
# 降噪后信号频谱
plot_spectrum(denoised_data, signal_sample_rate, '降噪后信号频谱')
