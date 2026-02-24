import os
import librosa
import soundfile as sf

def upsample_to_48khz(input_folder, output_folder=None):
    """
    将输入文件夹中的所有WAV文件升采样到48kHz
    
    参数:
        input_folder (str): 包含原始WAV文件的文件夹路径
        output_folder (str): 可选，保存处理后文件的文件夹路径
                            如果为None，则会在原文件夹中创建"upsampled"子文件夹
    """
    # 确保输出文件夹存在
    if output_folder is None:
        output_folder = os.path.join(input_folder, "upsampled")
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # 加载音频文件
                y, sr = librosa.load(input_path, sr=None)  # sr=None保留原始采样率
                
                # 如果原始采样率已经是48kHz，跳过处理
                if sr == 48000:
                    print(f"文件 {filename} 已经是48kHz，跳过处理")
                    continue
                
                # 升采样到48kHz
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=48000)
                
                # 保存处理后的文件
                sf.write(output_path, y_resampled, 48000)
                print(f"成功处理: {filename} (原始采样率: {sr}Hz -> 目标采样率: 48000Hz)")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 用户输入文件夹路径
    input_folder = input("请输入包含WAV文件的文件夹路径: ").strip()
    
    # 验证路径是否存在
    if not os.path.isdir(input_folder):
        print("错误: 指定的文件夹不存在")
    else:
        # 询问用户是否要指定输出文件夹
        output_choice = input("是否要指定输出文件夹? (y/n): ").strip().lower()
        if output_choice == 'y':
            output_folder = input("请输入输出文件夹路径: ").strip()
            upsample_to_48khz(input_folder, output_folder)
        else:
            upsample_to_48khz(input_folder)
        
        print("处理完成!")