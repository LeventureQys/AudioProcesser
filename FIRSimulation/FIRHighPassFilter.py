import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, firwin
# High Pass FIR filter design
def highpass_fir_filter(f_c, Fs, N):
    # Ensure the number of coefficients is odd
    if N % 2 == 0:
        N += 1
    
    # Normalize the cutoff frequency with respect to the Nyquist frequency
    nyquist = 0.5 * Fs
    normal_cutoff = f_c / nyquist
    
    # Use firwin to create a high pass FIR filter
    b = firwin(N, normal_cutoff, pass_zero=False, window='hamming')
    
    return b
# Input parameters
f_c = 10000    # Cutoff frequency in Hz
Fs = 96000    # Sampling rate in Hz
N = 101       # Filter order (higher values mean sharper cutoff)

# Get filter coefficients
b = highpass_fir_filter(f_c, Fs, N)

# Frequency response
w, h = freqz(b, worN=8000)
frequencies = w * Fs / (2 * np.pi)
response = 20 * np.log10(np.abs(h))

# Plot frequency response
plt.figure(figsize=(10, 6))
plt.plot(frequencies, response, label='High Pass FIR Filter Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Frequency Response of High Pass FIR Filter')
plt.grid()
plt.legend()
plt.xlim(0, 96000)
plt.ylim(-80, 5)
plt.show()
