import numpy as np
import matplotlib.pyplot as plt
import os
# Force Python to find your Graphviz installation
# This is the standard folder for Windows 10/11
graphviz_path = r'C:\Program Files\Graphviz\bin' 
os.environ["PATH"] += os.pathsep + graphviz_path
from scipy.fft import fft, fftfreq
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error

# Import our custom modules
from data_prep import get_aligned_data
from signal_proc import create_spectrograms, plot_spectrogram
from train_predict import run_experiment
from model_cnn import build_cnn_model

# 1. Prepare Data (Task 1)
print("Step 1: Fetching and Normalizing Data...")
df, scaler = get_aligned_data()
target_ticker = "RELIANCE.NS"
prices = df[target_ticker].values

# --- FIGURE 1: TIME SERIES PLOT ---
plt.figure(figsize=(10, 4))
plt.plot(prices, color='blue', linewidth=1)
plt.title(f"Figure 1: Time Series Plot ({target_ticker})")
plt.xlabel("Days")
plt.ylabel("Normalized Price")
plt.grid(True, alpha=0.3)
plt.savefig("figure1_time_series.png")
print("Saved Figure 1.")
plt.show()

# --- FIGURE 2: FREQUENCY SPECTRUM (Global FFT) ---
N = len(prices)
yf = fft(prices)
xf = fftfreq(N, 1)[:N//2]
plt.figure(figsize=(10, 4))
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), color='red')
plt.title("Figure 2: Global Frequency Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig("figure2_frequency_spectrum.png")
print("Saved Figure 2.")
plt.show()

# 2. Signal Processing (Task 2)
print("Step 2: Generating High-Resolution Spectrograms...")
WINDOW_SIZE = 100
X_specs = create_spectrograms(prices, window_len=WINDOW_SIZE)
y = prices[WINDOW_SIZE:] 

# --- FIGURE 3: SPECTROGRAM ---
# This will use the improved 'magma' shading from our signal_proc update
plot_spectrogram(X_specs[0], title=f"Figure 3: STFT Spectrogram ({target_ticker})")
print("Saved Figure 3.")

# 3. Model Development (Task 3)
print("Step 3: Training CNN Model...")
actual, pred = run_experiment(X_specs, y)

# --- FIGURE 4: CNN ARCHITECTURE ---
print("Step 4: Generating Architecture Diagram...")
# Rebuild a temp model to plot it
model_temp = build_cnn_model(X_specs.shape[1:] + (1,))

try:
    # This requires Graphviz installed on Windows
    plot_model(model_temp, to_file='figure4_cnn_architecture.png', 
               show_shapes=True, show_layer_names=True, rankdir='TB')
    print("Saved Figure 4 (Image).")
except Exception as e:
    print(f"Could not generate Figure 4 image (Graphviz missing). Saving text summary instead.")
    with open('figure4_cnn_architecture.txt', 'w', encoding='utf-8') as f:
        model_temp.summary(print_fn=lambda x: f.write(x + '\n'))

# --- FIGURE 5: PREDICTION ANALYSIS (Task 4) ---
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Price', color='blue', alpha=0.6)
plt.plot(pred, label='Predicted Price', color='red', linestyle='--')
plt.title(f"Figure 5: Final Prediction Analysis ({target_ticker})")
plt.xlabel("Days (Test Set)")
plt.ylabel("Normalized Price")
plt.legend()
plt.savefig("final_prediction.png")
print("Saved Figure 5.")
plt.show()

mse = mean_squared_error(actual, pred)
print(f"\n--- FINAL REPORT ---")
print(f"Target Stock: {target_ticker}")
print(f"Mean Squared Error: {mse:.6f}")
print("Assignment Pipeline Complete.")
