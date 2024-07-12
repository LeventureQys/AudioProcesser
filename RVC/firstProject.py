from pathlib import Path

from dotenv import load_dotenv
from scipy.io import wavfile

from rvc.modules.vc.modules import VC

def main():
    vc = VC()
    vc.get_vc("./assets/enen.pth")
    tgt_sr, audio_opt, times, _ = vc.vc_inference(1, Path("../Audio/voice/1.wav"))
    wavfile.write("./result/ret1.mp3", tgt_sr, audio_opt)

if __name__ == "__main__":
    load_dotenv("./.env")
    main()