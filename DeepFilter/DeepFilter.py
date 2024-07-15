import torchaudio as ta
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import os
def check_and_set_sr(audio_path: str, target_sr: int) -> int:
    """Check the sample rate of the audio file and adjust target_sr if needed."""
    if os.path.exists(audio_path):
        print(f"File exists: {audio_path}")
    else:
        print(f"File does not exist: {audio_path}")

    info = ta.info(audio_path,format="wav")
    orig_sr = info.sample_rate
    if orig_sr != target_sr:
        print(f"Original sample rate {orig_sr} does not match target sample rate {target_sr}. Adjusting...")
        return orig_sr
    return target_sr

if __name__ == "__main__":


    # 列出支持的后端
    print("Available audio backends:", ta.list_audio_backends())

    # 设置sox_io后端
    ta.set_audio_backend("soundfile")
    
    # Load default model
    model, df_state, _ = init_df()
    # Download and open some audio file. You use your audio files here
    print(df_state.sr())
    
    audio_path = r"D:/WorkShop/CurrentWork/FIRFilter_Venture/Audio/voice/3.wav"
    
    # Check and adjust sample rate
    sr = check_and_set_sr(audio_path, df_state.sr())
    
    audio, _ = load_audio(audio_path, sr=sr)
    
    # Denoise the audio
    enhanced = enhance(model, df_state, audio)
    
    # Save for listening
    save_audio("enhanced_2.wav", enhanced, df_state.sr())
