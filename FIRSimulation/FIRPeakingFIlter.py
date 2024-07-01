import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, firwin2

def peaking_fir_filter(f0, Fs, G, Q, N):
    # Normalize the center frequency with respect to the Nyquist frequency
    nyquist = 0.5 * Fs
    f0_normalized = f0 / nyquist
    
    # Calculate bandwidth in normalized frequency
    bandwidth = f0 / Q
    bandwidth_normalized = bandwidth / nyquist
    
    # Calculate the gain in linear scale
    A = 10**(G / 20)
    
    # Frequency points for the desired response
    freqs = [0, f0_normalized - bandwidth_normalized / 2, f0_normalized, f0_normalized + bandwidth_normalized / 2, 1]
    
    # Desired gain at the frequency points
    gains = [1, 1, A, 1, 1]
    
    # Design the FIR filter using firwin2
    h = firwin2(N, freqs, gains, window='hamming')
    
    return h

# Input parameters
f0 = 200     # Center frequency in Hz
Fs = 1000    # Sampling rate in Hz
G = 6         # Gain in dB
Q = 1         # Quality factor
N = 101       # Filter order (ensure it's odd)

# Get filter coefficients
h = peaking_fir_filter(f0, Fs, G, Q, N)

# Frequency response
w, H = freqz(h, worN=8000)
frequencies = w * Fs / (2 * np.pi)
response = 20 * np.log10(np.abs(H))

# Plot frequency response
plt.figure(figsize=(10, 6))
plt.plot(frequencies, response, label='Peaking Filter Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Frequency Response of Peaking Filter')
plt.grid()
plt.legend()
plt.xlim(0, 1000)
plt.ylim(-10, 10)
plt.show()
