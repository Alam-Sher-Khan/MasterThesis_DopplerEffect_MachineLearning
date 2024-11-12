import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Function to load FFT data while skipping the top row (frequency bins)
def load_fft_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip the first line with frequency bins
    data = []
    for line in lines:
        values = line.split()
        numValues = [int(x) for x in values]  # Convert values to integers
        data.append(numValues)
    return np.array(data)

# Function to select files using a user interface
def select_file(title):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Text Files", "*.txt")])
    return file_path

# Load FFT data files for human and object
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

# Reshape data for CNN (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Add the third dimension for features
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# One-hot encode the labels for CNN
y_train = to_categorical(y_train - 1, num_classes=2)
y_test = to_categorical(y_test - 1, num_classes=2)

# Create the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))  # Prevent overfitting
model.add(Dense(2, activation='softmax'))  # 2 classes: Object and Human

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained CNN model
model_filename = filedialog.asksaveasfilename(defaultextension=".h5", title="Save Trained Model", filetypes=[("H5 Files", "*.h5")])
if model_filename:
    model.save(model_filename)
    print(f"Model has been saved to {model_filename}")

# Make predictions on the test set
pred = model.predict(X_test)
y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate confusion matrix and other metrics
cf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

# Calculate TP, TN, FP, and FN
TP = cf_matrix[1, 1]
TN = cf_matrix[0, 0]
FP = cf_matrix[0, 1]
FN = cf_matrix[1, 0]

# Print metrics
print("Confusion matrix:")
print(cf_matrix)
print(f"Accuracy: {accuracy:.5f}")
print(f"F1 score: {f1:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

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

# Generate and print the classification report
report = classification_report(y_true, y_pred, target_names=['Object', 'Human'], digits=5)
print("\nClassification Report:")
print(report)

# Save the confusion matrix, classification report, and overall metrics in an Excel file
output_filename = filedialog.asksaveasfilename(defaultextension=".xlsx", title="Save Metrics and Confusion Matrix", filetypes=[("Excel Files", "*.xlsx")])
if output_filename:
    # Save the confusion matrix, classification report, and metrics in Excel
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        # Save confusion matrix
        cf_matrix_df = pd.DataFrame(cf_matrix, index=['Actual Object', 'Actual Human'], columns=['Predicted Object', 'Predicted Human'])
        cf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')

        # Save classification report
        report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=['Object', 'Human'], output_dict=True)).T
        report_df.to_excel(writer, sheet_name='Classification Report')

        # Save overall performance metrics (accuracy, precision, recall, f1 score, TP, TN, FP, FN)
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'True Positives (TP)', 'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)'],
            'Value': [accuracy, precision, recall, f1, TP, TN, FP, FN]
        })
        metrics_df.to_excel(writer, sheet_name='Overall Metrics', index=False)

    print(f"Results have been saved to {output_filename}")
else:
    print("No directory selected for saving. Results were not saved.")
