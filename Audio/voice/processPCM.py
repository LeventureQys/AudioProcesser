import struct
import wave

def pcm_to_wav(pcm_file, wav_file, channels, sample_rate, bits_per_sample):
    with open(pcm_file, 'rb') as pcmf:
        pcm_data = pcmf.read()

    num_samples = len(pcm_data) // (bits_per_sample // 8)
    
    wav_header = b'RIFF' + \
                 struct.pack('<L', 36 + len(pcm_data)) + \
                 b'WAVE' + \
                 b'fmt ' + \
                 struct.pack('<L', 16) + \
                 struct.pack('<H', 1) + \
                 struct.pack('<H', channels) + \
                 struct.pack('<L', sample_rate) + \
                 struct.pack('<L', sample_rate * channels * (bits_per_sample // 8)) + \
                 struct.pack('<H', channels * (bits_per_sample // 8)) + \
                 struct.pack('<H', bits_per_sample) + \
                 b'data' + \
                 struct.pack('<L', len(pcm_data))

    with open(wav_file, 'wb') as wavf:
        wavf.write(wav_header)
        wavf.write(pcm_data)

# Example usage
pcm_file = 'm-k-output.pcm'
wav_file = 'm-koutput.wav'
channels = 2
sample_rate = 48000
bits_per_sample = 16

pcm_to_wav(pcm_file, wav_file, channels, sample_rate, bits_per_sample)
