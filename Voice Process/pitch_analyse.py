import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter,find_peaks, medfilt
from scipy.signal.windows import hamming
import warnings
warnings.filterwarnings("ignore")  # 忽略FFT的复数警告

def load_audio(file_path):
    """读取音频文件并归一化"""
    fs, signal = wavfile.read(file_path)
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # 取单声道
    return fs, signal.astype(float) / np.max(np.abs(signal))

def pre_emphasis(signal, alpha=0.97):
    """预加重滤波器"""
    return lfilter([1, -alpha], [1], signal)

def compute_cepstrum(frame, n_fft):
    """改进的复倒谱计算（幅度压缩+相位处理）"""
    spectrum = np.fft.fft(frame, n_fft)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10) + 1j * np.unwrap(np.angle(spectrum))
    cepstrum = np.fft.ifft(log_spectrum).real
    
    # 幅度压缩：使用双曲正切函数限制动态范围
    cepstrum = np.tanh(0.5 * cepstrum / np.max(np.abs(cepstrum)))
    return cepstrum

def detect_pitch(cepstrum, fs, min_lag=None, max_lag=None):
    """改进的基音检测算法"""
    # 动态计算合理滞后范围（人类基频80-400Hz）
    min_lag = int(fs / 400) if min_lag is None else min_lag  # 400Hz上限
    max_lag = int(fs / 80) if max_lag is None else max_lag    # 80Hz下限
    
    lags = np.arange(min_lag, max_lag)
    cepstral_segment = cepstrum[lags]
    
    # 峰值检测+显著性验证
    peaks, props = find_peaks(cepstrum[lags], 
                            height=0.3*np.max(cepstrum[lags]),
                            prominence=0.2)
    
    if len(peaks) > 0:
        main_peak = peaks[np.argmax(props['prominences'])]
        return fs / (lags[main_peak] + 1)  # 避免除零
    return np.nan  # 无效帧

def plot_results(time_axis, signal, frames, pitch_contour, fs, cepstra):
    """绘制完整分析结果（4个子图）"""
    plt.figure(figsize=(15, 12))
    
    # 1. 原始波形
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, signal)
    plt.title("Original Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # 2. 复倒谱（第一帧）
    plt.subplot(4, 1, 2)
    cepstrum = cepstra[0]
    plt.plot(np.arange(len(cepstrum)), cepstrum)
    plt.axvline(x=min_lag, color='r', linestyle='--', label=f'Min Lag ({min_lag})')
    plt.axvline(x=max_lag, color='g', linestyle='--', label=f'Max Lag ({max_lag})')
    plt.title("Complex Cepstrum (First Frame)")
    plt.xlabel("Quefrency (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.xlim(0, max_lag*1.5)
    
    # 3. 对数幅度谱（第一帧）
    plt.subplot(4, 1, 3)
    frame = frames[0]
    spectrum = np.abs(np.fft.fft(frame, 1024))[:512]
    freq_axis = np.linspace(0, fs/2, 512)
    plt.plot(freq_axis, 20 * np.log10(spectrum + 1e-10))
    plt.title("Log Magnitude Spectrum (First Frame)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    
    # 4. 基音轨迹（改进版）
    plt.subplot(4, 1, 4)
    pitch_time = np.linspace(0, len(signal)/fs, len(pitch_contour))
    valid_pitch = np.array(pitch_contour)
    valid_mask = ~np.isnan(valid_pitch)
    plt.plot(pitch_time[valid_mask], valid_pitch[valid_mask], 'r-', label="Pitch")
    plt.scatter(pitch_time[valid_mask], valid_pitch[valid_mask], s=10, c='b', label="Detected Points")
    plt.title("Pitch Contour (Improved Cepstrum Method)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(50, 400)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    global min_lag, max_lag  # 用于绘图标注
    
    # 1. 加载音频
    file_path = "1.wav"  # 替换为您的文件路径
    fs, signal = load_audio(file_path)
    print(f"采样率: {fs}Hz, 音频时长: {len(signal)/fs:.2f}s")
    
    # 2. 预处理
    signal = pre_emphasis(signal)
    
    # 3. 分帧参数（根据采样率自适应）
    frame_length = int(0.04 * fs)  # 40ms窗长提高低频分辨率
    frame_step = int(0.01 * fs)     # 10ms帧移
    num_frames = int(np.ceil((len(signal) - frame_length) / frame_step))
    
    # 4. 动态计算滞后范围
    min_lag = int(fs / 400)  # 400Hz上限
    max_lag = int(fs / 80)   # 80Hz下限
    print(f"基频检测范围: {fs/max_lag:.1f}-{fs/min_lag:.1f}Hz")
    
    # 5. 分帧处理
    pitch_contour = []
    frames = []
    cepstra = []
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_length
        frame = signal[start:end]
        
        # 补零对齐
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), 'constant')
        
        # 加窗处理
        windowed_frame = frame * hamming(frame_length)
        frames.append(windowed_frame)
        
        # 计算倒谱
        cepstrum = compute_cepstrum(windowed_frame, n_fft=2048)
        cepstra.append(cepstrum)
        
        # 基音检测
        pitch_freq = detect_pitch(cepstrum, fs)
        pitch_contour.append(pitch_freq)
    
    # 6. 后处理：中值滤波平滑
    pitch_contour = medfilt(np.nan_to_num(pitch_contour), kernel_size=5)
    
    # 7. 绘制结果
    time_axis = np.arange(len(signal)) / fs
    plot_results(time_axis, signal, frames, pitch_contour, fs, cepstra)

if __name__ == "__main__":
    main()