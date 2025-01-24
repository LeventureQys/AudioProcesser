import numpy as np
import scipy.signal as signal
def apply_fir_filter(data,filter_coefficients):
    '''
    应用 FIR 滤波器到输入信号。
    参数:
        data (ndarray): 输入信号
        filter_coefficients (ndarray): FIR 滤波器系数
    返回:
        filtered_data (ndarray): 滤波后的信号   
    '''
    filtered_data = signal.lfilter(filter_coefficients, 1, data)
    return filtered_data