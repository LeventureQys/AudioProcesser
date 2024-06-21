import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, firwin

def design_fir_lowpass(cutoff_freq, sample_rate, num_taps, beta):
    nyquist_rate = sample_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist_rate
    # Design the FIR filter using the Kaiser window
    fir_coeff = firwin(num_taps, normalized_cutoff, window=('kaiser', beta))
    return fir_coeff

def plot_frequency_response(fir_coeff, sample_rate):
    w, h = freqz(fir_coeff, worN=8000)
    freqs = w * sample_rate / (2 * np.pi)
    response = 20 * np.log10(np.abs(h))
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, response, 'b')
    plt.title('Frequency Response of the FIR LowPass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.grid()
    plt.xlim(0, 96000)
    plt.ylim(-100, 5)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.show()

# Input parameters
cutoff_freq = 15000  # Cutoff frequency in Hz
sample_rate = 192000  # Sampling rate in Hz
num_taps = 201  # Number of filter coefficients (taps)
beta = 8.6  # Beta parameter for Kaiser window (trade-off between main lobe width and side-lobe levels)

# Design the FIR LowPass filter
fir_coeff = design_fir_lowpass(cutoff_freq, sample_rate, num_taps, beta)

# Plot the frequency response
plot_frequency_response(fir_coeff, sample_rate)
