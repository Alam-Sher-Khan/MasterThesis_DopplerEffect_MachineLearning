import os
import pandas as pd

# Define paths
input_path = ("C:/My_Docs/GERMANY/ALAM/Frankfurt/Studies/4_Sem/"
               "My_Thesis/Readings/Headers_separated_ADC/Object/Perpendicular/New")
output_path = ("C:/My_Docs/GERMANY/ALAM/Frankfurt/Studies/4_Sem/"
               "My_Thesis/Readings/Headers_separated_ADC/Object/Perpendicular/New/extracted")

# Iterate over files in the specified directory
for entry in os.scandir(input_path):
    if entry.is_file():  # Ensure that we're processing files only
        print(f"Processing {entry.name}")
        
        # Read the file
        with open(entry.path, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            values = line.split()[16:]  # Adjust the slice as needed
            data.append(values)

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)

        # Prepare the output file name
        # Get the base file name without extension
        base_name = os.path.splitext(entry.name)[0]
        output_file_name = f"{base_name}_extracted.txt"
        output_file_path = os.path.join(output_path, output_file_name)

        # Save DataFrame to .txt file without commas
        df.to_csv(output_file_path, index=False, header=False, sep=' ')

print("Done!")
