import os
import wave
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

def downsample_wav(input_path, output_path, target_rate=16000):
    """
    将WAV文件降采样到目标采样率
    :param input_path: 输入WAV文件路径
    :param output_path: 输出WAV文件路径
    :param target_rate: 目标采样率(默认16kHz)
    """
    try:
        # 读取原始音频文件
        sample_rate, data = wavfile.read(input_path)
        
        # 如果已经是目标采样率，直接复制
        if sample_rate == target_rate:
            wavfile.write(output_path, sample_rate, data)
            return
        
        # 计算重采样因子
        gcd = np.gcd(sample_rate, target_rate)
        up = target_rate // gcd
        down = sample_rate // gcd
        
        # 处理多声道音频
        if len(data.shape) > 1:
            resampled_data = np.zeros((int(data.shape[0] * up / down), data.shape[1]), dtype=data.dtype)
            for channel in range(data.shape[1]):
                resampled_data[:, channel] = resample_poly(data[:, channel], up, down)
        else:
            resampled_data = resample_poly(data, up, down)
        
        # 确保数据类型正确
        if data.dtype == np.int16:
            resampled_data = resampled_data.astype(np.int16)
        elif data.dtype == np.float32:
            resampled_data = resampled_data.astype(np.float32)
        
        # 写入输出文件
        wavfile.write(output_path, target_rate, resampled_data)
        
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {str(e)}")

def process_folder(input_folder, output_folder, target_rate=16000):
    """
    处理文件夹内所有WAV文件
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param target_rate: 目标采样率(默认16kHz)
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"正在处理: {filename}")
            downsample_wav(input_path, output_path, target_rate)
    
    print("所有文件处理完成!")

if __name__ == "__main__":
    # 使用示例
    input_folder = "input_wavs"  # 替换为你的输入文件夹路径
    output_folder = "output_16k"  # 替换为你的输出文件夹路径
    
    process_folder(input_folder, output_folder)