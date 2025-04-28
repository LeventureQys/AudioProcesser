import os
import librosa
import soundfile as sf

def upsample_wav(input_folder, output_folder, target_sr=48000):
    """
    将 input_folder 下的所有 WAV 文件升采样到 target_sr（默认 48kHz），并保存到 output_folder
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有 WAV 文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 加载音频文件（自动读取原始采样率）
            y, orig_sr = librosa.load(input_path, sr=None)  # sr=None 保留原始采样率

            # 如果原始采样率已经是目标采样率，则跳过
            if orig_sr == target_sr:
                print(f"跳过 {filename}（已经是 {target_sr}Hz）")
                continue

            # 升采样到目标采样率
            y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

            # 保存升采样后的文件（保持原始位深度和格式）
            sf.write(output_path, y_resampled, target_sr)
            print(f"已处理 {filename}: {orig_sr}Hz -> {target_sr}Hz")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WAV 文件升采样到 48kHz")
    parser.add_argument("input_folder", help="输入文件夹路径（包含 WAV 文件）")
    parser.add_argument("output_folder", help="输出文件夹路径")
    args = parser.parse_args()

    upsample_wav(args.input_folder, args.output_folder)