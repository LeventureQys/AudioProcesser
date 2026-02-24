import numpy as np
import scipy.signal as signal

def highpass_fir_filter(cutoff_freq,sample_rate,num_taps = 101):
    '''
    设计一个高通滤波器 
        参数:
        cutoff_freq (float): 截止频率 (Hz)
        sample_rate (float): 采样率 (Hz)
        num_taps (int): 滤波器的阶数（系数数量）

    返回:
        h (ndarray): FIR 滤波器的系数
    '''
    # 归一化截止频率 (Nyquist 频率为 sample_rate / 2)
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist_freq

    # 使用 firwin 设计高通滤波器
    h = signal.firwin(num_taps, normalized_cutoff, pass_zero=False, window='hamming')
    return h

