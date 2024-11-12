import numpy as np
import os
from scipy.signal import correlate

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

    # Filter to only include frequencies in the specified range
    frequency_range_min = 35 * 1000  # 25 kHz in Hz
    frequency_range_max = 45 * 1000  # 50 kHz in Hz
    within_range_indices = np.where((frequency_axis_positive >= frequency_range_min) & (frequency_axis_positive <= frequency_range_max))
    frequency_axis_range = frequency_axis_positive[within_range_indices]
    amplitude_spectrum_range = amplitude_spectrum[within_range_indices]
    
    # Normalize the amplitude values to the range [0, 1000]
    min_value = np.min(amplitude_spectrum_range)
    max_value = np.max(amplitude_spectrum_range)
    amplitude_spectrum_range = (amplitude_spectrum_range - min_value) / (max_value - min_value) * 1000
    
    return frequency_axis_range, amplitude_spectrum_range

# Function to save the FFT data to a new file in the specified format without decimals


def save_fft_data(file_path, frequency_axis, fft_data):
    new_file_path = file_path.replace('adc', 'fft')
    with open(new_file_path, 'w') as f:
        # Write the frequency components in the first row, rounded to the nearest whole number
        f.write('\t'.join(f'{int(round(freq))}' for freq in frequency_axis) + '\n')
        # Write the amplitude data row by row, rounded to the nearest whole number
        for row in fft_data:
            f.write('\t'.join(f'{int(round(amp))}' for amp in row) + '\n')

# Directory containing the ADC data files
directory_path = "C:/My_Docs/GERMANY/ALAM/Frankfurt/Studies/4_Sem/My_Thesis/Readings/Headers_separated_ADC/Object/Towards/New"  # Replace with your directory

# Sampling frequency
sampling_frequency = 1953125  # Hz

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith("_extracted.txt") and 'adc' in filename:
        file_path = os.path.join(directory_path, filename)
        
        # Load ADC data
        adc_data = load_adc_data(file_path)
        
        # Process each row of the ADC data and perform FFT
        fft_data = []
        frequency_axis = None
        for row in adc_data:
            frequency_axis_range, amplitude_spectrum = perform_fft_with_acf(row, sampling_frequency)
            fft_data.append(amplitude_spectrum)
            if frequency_axis is None:
                frequency_axis = frequency_axis_range
        
        # Save the FFT data to a new file
        save_fft_data(file_path, frequency_axis, fft_data)

print("FFT data files generated successfully with normalized values from 0 to 1000.")
