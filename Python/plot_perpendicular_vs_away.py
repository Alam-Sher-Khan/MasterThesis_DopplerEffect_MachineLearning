import matplotlib.pyplot as plt
import numpy as np

# Function to generate comparison plots for accuracy, precision, recall, and F1 score
def plot_comparison(metrics_scenario1, metrics_scenario2, labels, scenario1_name, scenario2_name):
    # Set up the plot with a larger figure size
    x = np.arange(len(labels))  # The label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(14, 10))  # Increase figure size

    # Plotting bars for Scenario 1 and Scenario 2
    bars1 = ax.bar(x - width/2, metrics_scenario1, width, label=scenario1_name, color='blue')
    bars2 = ax.bar(x + width/2, metrics_scenario2, width, label=scenario2_name, color='green')

    # Add labels, title, and grid with increased font sizes
    ax.set_ylabel('Scores', fontsize=18)
    ax.set_title('Performance Improvement', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='both', labelsize=18)

    # Adjust legend position and font size
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=18)

    # Add grid lines behind the bars
    ax.grid(True, which='both', axis='y', zorder=0)

    # Set the z-order of individual bars higher than grid lines so that grid stays behind
    for bar in bars1:
        bar.set_zorder(2)
    for bar in bars2:
        bar.set_zorder(2)

    # Adding data labels on top of the bars with increased font size
    def add_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom', zorder=3, fontsize=18)

    add_labels(bars1)
    add_labels(bars2)

    # Adjust layout to give space for the legend
    plt.tight_layout(pad=3.0)
    
    # Show the plot
    plt.show()

# Define the labels for the metrics
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Define the metrics for Scenario 1 (Stationary model tested with Moving Towards dataset)
metrics_scenario1 = [0.55, 0.55, 0.55, 0.54]  # Example values for accuracy, precision, recall, f1 score

# Define the metrics for Scenario 2 (Moving Towards model tested with Moving Towards dataset)
metrics_scenario2 = [0.99, 0.99, 0.99, 0.99]  # Example values for accuracy, precision, recall, f1 score

# Scenario names for plotting
scenario1_name = 'MLP Model Trained on Perpendicular Dataset(Non-Doppler) for classifying Moving Away Dataset(Doppler)'
scenario2_name = 'MLP Model Trained on Moving Away Dataset(Doppler) for classifying Moving Away Dataset'

# Call the function to generate comparison plots
plot_comparison(metrics_scenario1, metrics_scenario2, labels, scenario1_name, scenario2_name)
