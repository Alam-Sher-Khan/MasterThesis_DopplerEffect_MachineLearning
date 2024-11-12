import tkinter as tk
from tkinter import filedialog
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
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

# Function to select files using a user interface
def select_file(title):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Text Files", "*.txt")])
    return file_path

# Select FFT data files for human and object
file_path_object = select_file("Select FFT Data File for Object")
file_path_human = select_file("Select FFT Data File for Human")

# Load FFT data
fft_data_object = load_fft_data(file_path_object)
fft_data_human = load_fft_data(file_path_human)

# Combine data and labels
X = np.vstack([fft_data_object, fft_data_human])
y = np.array([1] * len(fft_data_object) + [2] * len(fft_data_human))  # 1 for Object, 2 for Human

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# Save the trained model to a file
model_filename = filedialog.asksaveasfilename(defaultextension=".pkl", title="Save Trained Model", filetypes=[("Pickle Files", "*.pkl")])
if model_filename:
    joblib.dump(mlp, model_filename)
    print(f"Model has been saved to {model_filename}")

# Make predictions on test set
pred = mlp.predict(X_test)

# Calculate confusion matrix and
cf_matrix = confusion_matrix(y_test, pred)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='weighted')
precision = precision_score(y_test, pred, average='weighted')
recall = recall_score(y_test, pred, average='weighted')

# Extract TP, TN, FP, FN from the confusion matrix
TP = cf_matrix[1, 1]
TN = cf_matrix[0, 0]
FP = cf_matrix[0, 1]
FN = cf_matrix[1, 0]

# Print metrics
print("Confusion matrix:")
print(cf_matrix)
print(f"Accuracy: {accuracy:.5f}")  # Display the exact accuracy up to 5 decimal places
print(f"F1 score: {f1:.5f}")  # Display the exact F1 score up to 5 decimal places
print(f"Precision: {precision:.5f}")  # Display the precision up to 5 decimal places
print(f"Recall: {recall:.5f}")  # Display the recall up to 5 decimal places
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# Generate and print the classification report with custom formatting
report = classification_report(y_test, pred, target_names=['Object', 'Human'], digits=5)
print("\nClassification Report:")
print(report)

# Plot the confusion matrix with TP, TN, FP, FN annotations
plt.figure(figsize=(8, 6))
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Object', 'Human'], yticklabels=['Object', 'Human'], annot_kws={"size": 18})
plt.title('Confusion Matrix',fontsize=18)
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.xticks(fontsize=14)                     # Increase font size of x-axis ticks
plt.yticks(fontsize=14)                     # Increase font size of y-axis ticks

# Add a legend to display TP, TN, FP, FN values
handles = [
    plt.Line2D([0], [0], color='white', marker='o', markersize=0, label=f'TP = {TP}'),
    plt.Line2D([0], [0], color='white', marker='o', markersize=0, label=f'TN = {TN}'),
    plt.Line2D([0], [0], color='white', marker='o', markersize=0, label=f'FP = {FP}'),
    plt.Line2D([0], [0], color='white', marker='o', markersize=0, label=f'FN = {FN}')
]
plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, borderaxespad=0.)

plt.tight_layout()
plt.show()

# Create an Excel file to save all displayed metrics and the classification report
output_filename = filedialog.asksaveasfilename(defaultextension=".xlsx", title="Save Metrics and Confusion Matrix", filetypes=[("Excel Files", "*.xlsx")])
if output_filename:
    # Save metrics and confusion matrix to Excel
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        # Save confusion matrix
        cf_matrix_df = pd.DataFrame(cf_matrix, index=['Object', 'Human'], columns=['Predicted Object', 'Predicted Human'])
        cf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'TP', 'TN', 'FP', 'FN'],
            'Value': [accuracy, f1, precision, recall, TP, TN, FP, FN]
        })
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
        # Save classification report
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row_data = ' '.join(line.split()).split(' ')
            report_data.append(row_data)
        
        report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        report_df.to_excel(writer, sheet_name='Classification Report', index=False)
    
    print(f"Metrics and confusion matrix have been saved to {output_filename}")
