import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import tkinter as tk
from tkinter import filedialog

# Function to load ADC data from text file with multiple rows
def load_adc_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        adc_data = [list(map(int, line.strip().split())) for line in lines]
    return np.array(adc_data)

# Function to perform FFT on each row of ADC data after autocorrelation and applying Hamming window
def perform_fft_with_acf(adc_data_row, sampling_frequency):
    # Compute the autocorrelation function (ACF)
    acf = correlate(adc_data_row, adc_data_row, mode='full')
    acf = acf[len(acf) // 2:]  # Take the second half

    # Apply Hamming window to the ACF
    hamming_window = np.hamming(len(acf))
    windowed_acf = acf * hamming_window

    # Perform FFT on the windowed ACF
    num_samples = len(windowed_acf)
    fft_values = np.fft.fft(windowed_acf)
    frequency_axis = np.fft.fftfreq(num_samples, 1 / sampling_frequency)
    positive_freq_indices = np.where(frequency_axis > 0)
    frequency_axis_positive = frequency_axis[positive_freq_indices]
    amplitude_spectrum = np.abs(fft_values[positive_freq_indices]) / num_samples

    # Filter for calculations within the 25 to 50 kHz range
    frequency_range_min = 25 * 1000
    frequency_range_max = 50 * 1000
    within_range_indices = np.where((frequency_axis_positive >= frequency_range_min) & (frequency_axis_positive <= frequency_range_max))
    frequency_axis_range = frequency_axis_positive[within_range_indices]
    amplitude_spectrum_range = amplitude_spectrum[within_range_indices]

    # Normalize amplitude values to the range [0, 1000]
    amplitude_spectrum_range = (amplitude_spectrum_range - amplitude_spectrum_range.min()) / (amplitude_spectrum_range.max() - amplitude_spectrum_range.min()) * 1000
    
    return frequency_axis_range, amplitude_spectrum_range

# Feature extraction functions
def calculate_spectral_centroid(frequency_axis, amplitude_spectrum):
    return np.sum(frequency_axis * amplitude_spectrum) / np.sum(amplitude_spectrum)

def calculate_doppler_shift(center_frequency, transmitted_frequency):
    return center_frequency - transmitted_frequency

# Function to select files using a UI
def select_file(title):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Text Files", "*.txt")])
    return file_path

# File selection UI for object movement cases
file_path_towards = select_file("Select Object Moving Towards FFT File")
file_path_away = select_file("Select Object Moving Away FFT File")
file_path_stationary = select_file("Select Object Stationary FFT File")
file_path_perpendicular = select_file("Select Object Moving Perpendicular FFT File")

# Parameters
sampling_frequency = 1953125  # Hz
transmitted_frequency = 40000  # Hz

# Load ADC data for each case
adc_data_towards = load_adc_data(file_path_towards)
adc_data_away = load_adc_data(file_path_away)
adc_data_stationary = load_adc_data(file_path_stationary)
adc_data_perpendicular = load_adc_data(file_path_perpendicular)

# Number of rows to compare (assuming equal rows in each file)
num_rows = min(adc_data_towards.shape[0], adc_data_away.shape[0], adc_data_stationary.shape[0], adc_data_perpendicular.shape[0])

# Plotting row-wise comparison one row at a time
for row in range(num_rows):
    plt.figure(figsize=(12, 8))

    # Process each case and plot within the desired range
    for adc_data, color, label in zip(
        [adc_data_towards, adc_data_away, adc_data_stationary, adc_data_perpendicular],
        ['blue', 'red', 'green', 'purple'],
        ['Object Moving Towards', 'Object Moving Away', 'Object Stationary', 'Object Moving Perpendicular']
    ):
        frequency_axis, amplitude_spectrum = perform_fft_with_acf(adc_data[row], sampling_frequency)
        centroid = calculate_spectral_centroid(frequency_axis, amplitude_spectrum)
        doppler_shift = calculate_doppler_shift(centroid, transmitted_frequency)
        peak_index = np.argmax(amplitude_spectrum)
        peak_frequency = frequency_axis[peak_index]

        # Plot within 35-45 kHz for visualization
        plot_indices = np.where((frequency_axis >= 35 * 1000) & (frequency_axis <= 45 * 1000))
        plt.plot(frequency_axis[plot_indices] / 1000, amplitude_spectrum[plot_indices], color=color, label=f'{label} (Δf = {doppler_shift:.2f} Hz)')
        
        # Mark the peak frequency and center frequency within the plot range
        if 35 * 1000 <= peak_frequency <= 45 * 1000:
            plt.scatter([peak_frequency / 1000], [amplitude_spectrum[peak_index]], color=color, marker='o', label=f'Peak {label}: {peak_frequency / 1000:.2f} kHz')
        if 35 * 1000 <= centroid <= 45 * 1000:
            plt.axvline(centroid / 1000, color=color, linestyle='--', label=f'Center {label}: {centroid / 1000:.2f} kHz')

    # Configure plot
    plt.title(f'Row {row + 1} Frequency Spectrum Comparison')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Amplitude (0-1000 normalized)')
    plt.legend(fontsize='x-large')
    plt.grid(True)
    plt.xlim([35, 45])

    # Show plot
    plt.show()
