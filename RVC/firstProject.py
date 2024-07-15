from pathlib import Path

from dotenv import load_dotenv
from scipy.io import wavfile

from rvc.modules.vc.modules import VC

def main():
    strFileName="tianmei"
    vc = VC()
    vc.get_vc("./assets/"+strFileName+".pth")
    tgt_sr, audio_opt, times, _ = vc.vc_inference(1, Path("../Audio/voice/3.m4a"),index_rate=0.3,rms_mix_rate=0.5)
    wavfile.write("./result/"+strFileName+".mp3", tgt_sr, audio_opt)

if __name__ == "__main__":
    load_dotenv("./.env")
    main()