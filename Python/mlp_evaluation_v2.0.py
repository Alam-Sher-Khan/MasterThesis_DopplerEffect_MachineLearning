from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
import pandas as pd
import numpy as np
import joblib
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os

# Function to load FFT data while skipping the top row (frequency bins)
def load_fft_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip the first line
    data = []
    for line in lines:
        values = line.split()
        numValues = [int(x) for x in values]
        data.append(numValues)
    return np.array(data)

# File selection interface for loading datasets and pretrained model
def select_file(prompt):
    Tk().withdraw()  # Close the root window
    print(prompt)
    file_path = askopenfilename(title=prompt)  # Ask user to select file with a title prompt
    if file_path:
        print(f"Selected file: {file_path}")
    else:
        print("No file selected.")
    return file_path

# Directory selection interface for saving the output file
def select_save_directory():
    Tk().withdraw()  # Close the root window
    print("Please select the directory where you want to save the results.")
    directory_path = askdirectory(title="Select the directory to save results")
    if directory_path:
        print(f"Selected directory: {directory_path}")
    else:
        print("No directory selected.")
    return directory_path

# Load pretrained model


print("Please select the pretrained MLP model file (trained on stationary dataset).")
model_filename = select_file("Select the pretrained MLP model file (trained on stationary dataset)")
mlp = joblib.load(model_filename)



# Load moving dataset for testing
print("Please select the FFT data file for moving dataset (human).")
file_path_human = select_file("Select the FFT data file for moving dataset (human)")
print("Please select the FFT data file for moving dataset (object).")
file_path_object = select_file("Select the FFT data file for moving dataset (object)")

# Load FFT data
fft_data_object = load_fft_data(file_path_object)
fft_data_human = load_fft_data(file_path_human)

# Combine moving dataset data and labels for testing
X_test_moving = np.vstack([fft_data_object, fft_data_human])
y_test_moving = np.array([1] * len(fft_data_object) + [2] * len(fft_data_human))  # 1 for Object, 2 for Human

# Predict on the moving dataset using the pretrained model
pred_moving = mlp.predict(X_test_moving)

# Calculate performance metrics
cf_matrix_moving = confusion_matrix(y_test_moving, pred_moving)
accuracy_moving = accuracy_score(y_test_moving, pred_moving)
f1_moving = f1_score(y_test_moving, pred_moving, average='weighted')
precision_moving = precision_score(y_test_moving, pred_moving, average='weighted')
recall_moving = recall_score(y_test_moving, pred_moving, average='weighted')

# Print results for the moving dataset
print("\nPerformance on the moving dataset (human vs. object):")
print("Confusion matrix:")
print(cf_matrix_moving)
print(f"Accuracy: {accuracy_moving:.5f}")
print(f"F1 score: {f1_moving:.5f}")
print(f"Precision: {precision_moving:.5f}")
print(f"Recall: {recall_moving:.5f}")

# Generate and print the classification report
report_moving = classification_report(y_test_moving, pred_moving, target_names=['Object', 'Human'], digits=5)
print("\nClassification Report for Moving Dataset:")
print(report_moving)

# Choose directory to save the results
save_directory = select_save_directory()

if save_directory:
    output_filename = os.path.join(save_directory, 'MLP_performance_moving_dataset.xlsx')
    print(f"Saving results to: {output_filename}")

    # Save the confusion matrix, classification report, and overall metrics in Excel
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        # Save confusion matrix
        cf_df = pd.DataFrame(cf_matrix_moving, index=['Actual Object', 'Actual Human'], columns=['Predicted Object', 'Predicted Human'])
        cf_df.to_excel(writer, sheet_name='Confusion Matrix')

        # Save classification report
        report_df = pd.DataFrame(classification_report(y_test_moving, pred_moving, target_names=['Object', 'Human'], output_dict=True)).T
        report_df.to_excel(writer, sheet_name='Classification Report')

        # Save overall performance metrics (accuracy, precision, recall, f1 score)
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy_moving, precision_moving, recall_moving, f1_moving]
        })
        metrics_df.to_excel(writer, sheet_name='Overall Metrics', index=False)

    print(f"Results have been saved to {output_filename}")
else:
    print("No directory selected for saving. Results were not saved.")
