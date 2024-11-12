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
    frequency_axis = np.fft.fftfreq(num_samples, 1/sampling_frequency)
    positive_freq_indices = np.where(frequency_axis > 0)
    frequency_axis_positive = frequency_axis[positive_freq_indices]
    amplitude_spectrum = np.abs(fft_values[positive_freq_indices]) / num_samples

    # Filter to only include frequencies in the specified range (35 kHz to 45 kHz)
    frequency_range_min = 35 * 1000  # 35 kHz in Hz
    frequency_range_max = 45 * 1000  # 45 kHz in Hz
    within_range_indices = np.where((frequency_axis_positive >= frequency_range_min) & (frequency_axis_positive <= frequency_range_max))
    frequency_axis_range = frequency_axis_positive[within_range_indices]
    amplitude_spectrum_range = amplitude_spectrum[within_range_indices]

    # Normalize amplitude values to the range [0, 1000]
    amplitude_spectrum_range = (amplitude_spectrum_range - amplitude_spectrum_range.min()) / (amplitude_spectrum_range.max() - amplitude_spectrum_range.min()) * 1000
    
    return frequency_axis_range, amplitude_spectrum_range

# Feature extraction functions

# Spectral Centroid (Center Frequency)
def calculate_spectral_centroid(frequency_axis, amplitude_spectrum):
    return np.sum(frequency_axis * amplitude_spectrum) / np.sum(amplitude_spectrum)

# Doppler Shift (using center frequency)
def calculate_doppler_shift(center_frequency, transmitted_frequency):
    return center_frequency - transmitted_frequency

# Function to select files using a user interface
def select_file(prompt):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("Text Files", "*.txt")])
    return file_path

# Parameters
sampling_frequency = 1953125  # Hz
transmitted_frequency = 40000  # Hz

# Selecting files for each case
file_path_towards = select_file("Select ADC Data File for Towards Case")
file_path_away = select_file("Select ADC Data File for Away Case")
file_path_stationary = select_file("Select ADC Data File for Stationary Case")
file_path_perpendicular = select_file("Select ADC Data File for Perpendicular Case")

# Load ADC data for each case
adc_data_towards = load_adc_data(file_path_towards)
adc_data_away = load_adc_data(file_path_away)
adc_data_stationary = load_adc_data(file_path_stationary)
adc_data_perpendicular = load_adc_data(file_path_perpendicular)

# Number of rows to compare
num_rows = min(adc_data_towards.shape[0], adc_data_away.shape[0], adc_data_stationary.shape[0], adc_data_perpendicular.shape[0])

# Plotting row-wise comparison one row at a time
for row in range(num_rows):
    plt.figure(figsize=(10, 6))
    
    # Get FFT results for the specified row for each case
    frequency_axis_towards, amplitude_spectrum_towards = perform_fft_with_acf(adc_data_towards[row], sampling_frequency)
    frequency_axis_away, amplitude_spectrum_away = perform_fft_with_acf(adc_data_away[row], sampling_frequency)
    frequency_axis_stationary, amplitude_spectrum_stationary = perform_fft_with_acf(adc_data_stationary[row], sampling_frequency)
    frequency_axis_perpendicular, amplitude_spectrum_perpendicular = perform_fft_with_acf(adc_data_perpendicular[row], sampling_frequency)
    
    # Calculate features for each case
    centroid_towards = calculate_spectral_centroid(frequency_axis_towards, amplitude_spectrum_towards)
    centroid_away = calculate_spectral_centroid(frequency_axis_away, amplitude_spectrum_away)
    centroid_stationary = calculate_spectral_centroid(frequency_axis_stationary, amplitude_spectrum_stationary)
    centroid_perpendicular = calculate_spectral_centroid(frequency_axis_perpendicular, amplitude_spectrum_perpendicular)
    peak_index_towards = np.argmax(amplitude_spectrum_towards)
    peak_index_away = np.argmax(amplitude_spectrum_away)
    peak_index_stationary = np.argmax(amplitude_spectrum_stationary)
    peak_index_perpendicular = np.argmax(amplitude_spectrum_perpendicular)
    peak_frequency_towards = frequency_axis_towards[peak_index_towards]
    peak_frequency_away = frequency_axis_away[peak_index_away]
    peak_frequency_stationary = frequency_axis_stationary[peak_index_stationary]
    peak_frequency_perpendicular = frequency_axis_perpendicular[peak_index_perpendicular]
    doppler_shift_towards = calculate_doppler_shift(centroid_towards, transmitted_frequency)
    doppler_shift_away = calculate_doppler_shift(centroid_away, transmitted_frequency)
    doppler_shift_stationary = calculate_doppler_shift(centroid_stationary, transmitted_frequency)
    doppler_shift_perpendicular = calculate_doppler_shift(centroid_perpendicular, transmitted_frequency)

    # Plot the spectra for each case on the same plot
    plt.plot(frequency_axis_towards / 1000, amplitude_spectrum_towards, label=f'Towards (Δf = {doppler_shift_towards:.2f} Hz)', color='blue')
    plt.plot(frequency_axis_away / 1000, amplitude_spectrum_away, label=f'Away (Δf = {doppler_shift_away:.2f} Hz)', color='red')
    plt.plot(frequency_axis_stationary / 1000, amplitude_spectrum_stationary, label=f'Stationary (Δf = {doppler_shift_stationary:.2f} Hz)', color='green')
    plt.plot(frequency_axis_perpendicular / 1000, amplitude_spectrum_perpendicular, label=f'Perpendicular (Δf = {doppler_shift_perpendicular:.2f} Hz)', color='purple')
    
    # Mark the peak frequencies
    plt.scatter([peak_frequency_towards / 1000], [amplitude_spectrum_towards[peak_index_towards]], color='blue', marker='o', label=f'Peak Towards: {peak_frequency_towards / 1000:.2f} kHz')
    plt.scatter([peak_frequency_away / 1000], [amplitude_spectrum_away[peak_index_away]], color='red', marker='o', label=f'Peak Away: {peak_frequency_away / 1000:.2f} kHz')
    plt.scatter([peak_frequency_stationary / 1000], [amplitude_spectrum_stationary[peak_index_stationary]], color='green', marker='o', label=f'Peak Stationary: {peak_frequency_stationary / 1000:.2f} kHz')
    plt.scatter([peak_frequency_perpendicular / 1000], [amplitude_spectrum_perpendicular[peak_index_perpendicular]], color='purple', marker='o', label=f'Peak Perpendicular: {peak_frequency_perpendicular / 1000:.2f} kHz')
    
    # Mark the center frequencies (centroid)
    plt.axvline(centroid_towards / 1000, color='cyan', linestyle='--', label=f'Center Towards: {centroid_towards / 1000:.2f} kHz')
    plt.axvline(centroid_away / 1000, color='orange', linestyle='--', label=f'Center Away: {centroid_away / 1000:.2f} kHz')
    plt.axvline(centroid_stationary / 1000, color='lime', linestyle='--', label=f'Center Stationary: {centroid_stationary / 1000:.2f} kHz')
    plt.axvline(centroid_perpendicular / 1000, color='purple', linestyle='--', label=f'Center Perpendicular: {centroid_perpendicular / 1000:.2f} kHz')
    
    plt.title(f'Row {row + 1} Frequency Spectrum Comparison')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Amplitude (0-1000 normalized)')
    
    # Increase legend font size
    plt.legend(fontsize='x-large')
    
    plt.grid(True)
    plt.xlim([35, 45])

    # Show the plot
    plt.show()
