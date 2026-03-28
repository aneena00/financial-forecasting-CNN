import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

def create_spectrograms(data_array, window_len=100): # Increased window
    spectrograms = []
    for i in range(len(data_array) - window_len):
        segment = data_array[i : i + window_len]
        
        # nperseg: determines frequency resolution (must be <= window_len)
        # nfft: adds padding for a smoother look
        f, t, Zxx = stft(segment, fs=1.0, nperseg=32, noverlap=24, nfft=64) 
        
        spectrograms.append(np.abs(Zxx))
    
    return np.array(spectrograms)

def plot_spectrogram(S, title="High-Resolution Spectrogram"):
    plt.figure(figsize=(8, 5))
    # 'shading=gouraud' makes the colors blend smoothly instead of blocks
    plt.pcolormesh(S, shading='gouraud', cmap='magma') 
    plt.title(title)
    plt.ylabel('Frequency (Scale)')
    plt.xlabel('Time (Within Window)')
    plt.colorbar(label='Magnitude')
    plt.savefig("figure3_spectrogram.png") # Auto-save the better version
    plt.show()
