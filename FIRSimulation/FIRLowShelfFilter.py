import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# LowShelf Filter Parameters
def lowshelf_filter(fc, fs, gain, Q):
    A = 10**(gain / 40)
    omega = 2 * np.pi * fc / fs
    alpha = np.sin(omega) / (2 * Q)
    
    b0 = A * ((A + 1) - (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega))
    b2 = A * ((A + 1) - (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(omega))
    a2 = (A + 1) + (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha
    
    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1, a1 / a0, a2 / a0]
    
    return b, a

# Frequency Response Calculation and Plot
def plot_frequency_response(fc, fs, gain, Q):
    b, a = lowshelf_filter(fc, fs, gain, Q)
    w, h = freqz(b, a, worN=8000)
    
    plt.figure(figsize=(10, 6))
    plt.plot((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h)), 'b')
    plt.title("Frequency Response of the LowShelf Filter")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain [dB]')
    plt.ylim(-24, 24)
    plt.grid()
    plt.show()

# Input parameters
fc = 40000  # Center frequency
fs = 96000  # Sampling rate
gain = 6  # Gain in dB
Q = 0.707  # Q value

# Plot the frequency response
plot_frequency_response(fc, fs, gain, Q)
