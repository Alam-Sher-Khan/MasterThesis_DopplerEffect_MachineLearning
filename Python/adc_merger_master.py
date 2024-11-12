import tkinter as tk
from tkinter import filedialog
import os

# Function to collect ADC data from multiple files and save to a master file
def collect_adc_data():
    # Ask the user to select multiple ADC files
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select ADC Files", filetypes=[("Text Files", "*.txt")])

    if not file_paths:
        print("No files selected.")
        return

    # Ask the user to select a location to save the master file
    save_location = filedialog.asksaveasfilename(defaultextension=".txt", title="Save Master ADC File", filetypes=[("Text Files", "*.txt")])

    if not save_location:
        print("No save location selected.")
        return

    # Open the master file for appending the data
    with open(save_location, 'w') as master_file:
        # Iterate through all selected files
        for file_path in file_paths:
            # Only process files that have 'adc' in the filename
            if file_path.endswith(".txt") and 'adc' in os.path.basename(file_path):
                with open(file_path, 'r') as adc_file:
                    lines = adc_file.readlines()

                    # Append all data from the current ADC file to the master file
                    for line in lines:
                        master_file.write(line.strip() + "\n")  # Write each row without empty lines

    print(f"All selected ADC data has been saved in: {save_location}")

# Call the function to execute the process
collect_adc_data()
