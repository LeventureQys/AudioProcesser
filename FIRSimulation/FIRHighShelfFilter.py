import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

def highshelf_filter(f0, Fs, G, Q):
    A = 10**(G / 40)
    omega_0 = 2 * np.pi * f0 / Fs
    alpha = np.sin(omega_0) / (2 * Q)
    
    b0 = A * ((A + 1) - (A - 1) * np.cos(omega_0) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega_0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(omega_0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(omega_0) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(omega_0))
    a2 = (A + 1) + (A - 1) * np.cos(omega_0) - 2 * np.sqrt(A) * alpha
    
    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1, a1 / a0, a2 / a0]
    
    return b, a

# Input parameters
f0 = 10000      # Cutoff frequency in Hz
Fs = 96000     # Sampling rate in Hz
G = 6          # Gain in dB
Q = 0.707      # Quality factor

# Get filter coefficients
b, a = highshelf_filter(f0, Fs, G, Q)

# Frequency response
w, h = freqz(b, a, worN=8000)
frequencies = w * Fs / (2 * np.pi)
response = 20 * np.log10(np.abs(h))

# Plot frequency response
plt.figure(figsize=(10, 6))
plt.plot(frequencies, response, label='HighShelf Filter Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Frequency Response of HighShelf Filter')
plt.grid()
plt.legend()
plt.xlim(0, 96000)
plt.ylim(-15, 15)
plt.show()