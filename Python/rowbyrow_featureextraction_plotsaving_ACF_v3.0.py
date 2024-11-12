import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate
import pandas as pd

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

    # Filter to include frequencies from 25 kHz to 50 kHz for feature extraction
    frequency_range_min = 25 * 1000  # 25 kHz in Hz
    frequency_range_max = 50 * 1000  # 50 kHz in Hz
    within_range_indices = np.where((frequency_axis_positive >= frequency_range_min) & (frequency_axis_positive <= frequency_range_max))
    frequency_axis_range = frequency_axis_positive[within_range_indices]
    amplitude_spectrum_range = amplitude_spectrum[within_range_indices]

    # Normalize amplitude values to the range [0, 1000]
    amplitude_spectrum_range = (amplitude_spectrum_range - amplitude_spectrum_range.min()) / (amplitude_spectrum_range.max() - amplitude_spectrum_range.min()) * 1000
    
    return frequency_axis_range, amplitude_spectrum_range

# Feature extraction functions
def calculate_spectral_centroid(frequency_axis, amplitude_spectrum):
    return np.sum(frequency_axis * amplitude_spectrum) / np.sum(amplitude_spectrum)

def calculate_spectral_bandwidth(frequency_axis, amplitude_spectrum, centroid):
    return np.sqrt(np.sum(((frequency_axis - centroid) ** 2) * amplitude_spectrum) / np.sum(amplitude_spectrum))

def calculate_spectral_entropy(amplitude_spectrum):
    normalized_spectrum = amplitude_spectrum / np.sum(amplitude_spectrum)  # Normalize the spectrum
    return -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-12))  # Calculate entropy

def calculate_doppler_shift(center_frequency, transmitted_frequency):
    return center_frequency - transmitted_frequency

def calculate_energy_distribution(amplitude_spectrum):
    return np.sum(np.square(amplitude_spectrum))

# Directory containing ADC data files
directory_path = "C:/My_Docs/GERMANY/ALAM/Frankfurt/Studies/4_Sem/My_Thesis/Readings/Headers_separated_ADC/Object/Towards/Test"  # Replace with actual directory path

# Parameters
sampling_frequency = 1953125  # Hz
transmitted_frequency = 40000  # Hz (base frequency of your ultrasonic sensor)

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".txt") and 'adc' in filename:
        file_path = os.path.join(directory_path, filename)
        
        # Load ADC data
        adc_data = load_adc_data(file_path)

        # Create a directory to save plots
        plot_directory = os.path.join(directory_path, filename.replace(".txt", ""))
        os.makedirs(plot_directory, exist_ok=True)

        # Lists to store features for saving into Excel
        features_list = []

        # Plotting row-wise and saving plots
        for row in range(adc_data.shape[0]):
            # Get FFT results for the specified row
            frequency_axis, amplitude_spectrum = perform_fft_with_acf(adc_data[row], sampling_frequency)

            # Calculate features
            centroid = calculate_spectral_centroid(frequency_axis, amplitude_spectrum)
            bandwidth = calculate_spectral_bandwidth(frequency_axis, amplitude_spectrum, centroid)
            entropy = calculate_spectral_entropy(amplitude_spectrum)
            energy = calculate_energy_distribution(amplitude_spectrum)
            doppler_shift = calculate_doppler_shift(centroid, transmitted_frequency)

            # Find peak frequency
            peak_index = np.argmax(amplitude_spectrum)
            peak_frequency = frequency_axis[peak_index]

            # Convert units and format the values
            centroid_khz = centroid / 1000  # Convert Hz to kHz
            bandwidth_khz = bandwidth / 1000  # Convert Hz to kHz
            peak_frequency_khz = peak_frequency / 1000  # Convert Hz to kHz
            entropy_formatted = round(entropy, 2)
            energy_formatted = round(energy, 2)
            doppler_shift_formatted = round(doppler_shift, 2)

            # Store features for this row
            features_list.append([row + 1, round(centroid_khz, 2), round(bandwidth_khz, 2), round(peak_frequency_khz, 2), entropy_formatted, doppler_shift_formatted, energy_formatted])

            # Plot frequency spectra with features (only for 35-45 kHz)
            plt.figure(figsize=(10, 6))
            plt.plot(frequency_axis / 1000, amplitude_spectrum, label=f'Row {row + 1}')
            plt.axvline(centroid / 1000, color='cyan', linestyle='--', label=f'Center Frequency: {centroid_khz:.2f} kHz')
            plt.scatter([peak_frequency_khz], [amplitude_spectrum[peak_index]], color='red', marker='o', label=f'Peak Frequency: {peak_frequency_khz:.2f} kHz')
            plt.title(f'Row {row + 1} Frequency Spectrum')
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Amplitude (0-1000 normalized)')
            plt.legend()
            plt.grid(True)
            plt.xlim([35, 45])  # Plot only the range 35-45 kHz
            
            # Save the plot
            plot_filename = os.path.join(plot_directory, f'Row_{row + 1}.png')
            plt.savefig(plot_filename)
            plt.close()

        # Save features to Excel file
        excel_filename = os.path.join(plot_directory, 'features.xlsx')
        df = pd.DataFrame(features_list, columns=['Row', 'Centroid (kHz)', 'Bandwidth (kHz)', 'Peak Frequency (kHz)', 'Entropy (unitless)', 'Doppler Shift (Hz)', 'Energy (unitless)'])
        df.to_excel(excel_filename, index=False)

        print(f"Plots and features have been saved for {filename} in the directory: {plot_directory}")
