import tkinter as tk
from tkinter import filedialog
import os

def collect_fft_data(output_file="master_fft_data.txt"):
    # Open the master file for appending the data
    with open(output_file, 'w') as master_file:
        # Ask the user to select multiple FFT files
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_paths = filedialog.askopenfilenames(title="Select FFT Files", filetypes=[("Text Files", "*.txt")])

        # Variable to store whether frequency bins have been added
        frequency_bins_written = False

        # Iterate through all selected files
        for file_path in file_paths:
            # Only process files that have 'fft' in the filename
            if file_path.endswith(".txt") and 'fft' in os.path.basename(file_path):
                with open(file_path, 'r') as fft_file:
                    lines = fft_file.readlines()

                    # Write frequency bins from the first file, if not already done
                    if not frequency_bins_written:
                        master_file.write(lines[0].strip() + "\n")  # Write the first row (frequency bins)
                        frequency_bins_written = True
                    
                    # Append the remaining data (without frequency bins) to the master file
                    for line in lines[1:]:
                        master_file.write(line.strip() + "\n")  # Write each row without empty lines
                    
    print(f"All selected FFT data has been saved in: {output_file}")

# Specify the location to save the centralized FFT data file
output_file = "C:/My_Docs/GERMANY/ALAM/Frankfurt/Studies/4_Sem/My_Thesis/Readings/Headers_separated_ADC/Human_1ms/Away/fft_HWAML__extracted.txt"

# Call the function
collect_fft_data(output_file=output_file)
