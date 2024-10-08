import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Define the JSON files (assuming they're in the current folder)
json_files = [
    "raw_clean_flicker_data.json",
    "deepcad_clean_flicker_data.json",
    "deepin_clean_flicker_data.json",
    "pseudo_clean_flicker_data.json",
    "200ms_clean_flicker_data.json"
]

# Updated sequence for the methods
methods = ["Raw", "DeepCAD", "Deep Interpolation", "Pseudo", "200ms"]

# Initialize lists to store data for plotting
area_data = []
duration_data = []

# Initialize a dictionary to store percentage of flickers with duration < 200ms
duration_less_than_200ms = {"Raw": 0, "DeepCAD": 0, "Deep Interpolation": 0}

# Load data from JSON files
for i, file_name in enumerate(json_files):
    with open(file_name, 'r') as f:
        data = json.load(f)
        areas = [entry['Area'] * 0.16 for entry in data]  # Convert pixel^2 to µm^2 (0.4 µm per pixel)
        durations = [entry['Duration (ms)'] for entry in data]

        # Count the percentage of flickers with duration < 200ms for specific methods
        if methods[i] in duration_less_than_200ms:
            duration_less_than_200ms[methods[i]] = sum(d < 200 for d in durations)

        # Extend the lists with the corresponding method names
        area_data.extend([(area, methods[i]) for area in areas])
        duration_data.extend([(duration, methods[i]) for duration in durations])

# Convert data to pandas DataFrames for easier plotting
area_df = pd.DataFrame(area_data, columns=['Area', 'Denoising Method'])
duration_df = pd.DataFrame(duration_data, columns=['Duration (ms)', 'Denoising Method'])

# Filter the area data to only include areas < 50 µm²
filtered_area_df = area_df[area_df['Area'] < 50]
filtered_duration_df = duration_df[duration_df['Duration (ms)'] < 3000]

# Function to perform t-tests between pairs of methods
def perform_t_tests(df, value_col):
    """ Perform t-tests between specific pairs of methods on the given column. """
    selected_pairs = [
        ('Raw', 'DeepCAD'),
        ('Raw', 'Deep Interpolation'),
        ('Raw', 'Pseudo'),
        ('Pseudo', '200ms')
    ]

    t_test_results = []

    # Perform t-test for each selected pair
    for method1, method2 in selected_pairs:
        data1 = df[df['Denoising Method'] == method1][value_col]
        data2 = df[df['Denoising Method'] == method2][value_col]
        t_stat, p_value = ttest_ind(data1, data2, equal_var=False)  # Welch's t-test for unequal variance
        t_test_results.append((method1, method2, t_stat, p_value))

    return pd.DataFrame(t_test_results, columns=['Method1', 'Method2', 't-statistic', 'p-value'])

# Perform t-tests for both Area and Duration
t_test_area_results = perform_t_tests(filtered_area_df, 'Area')
t_test_duration_results = perform_t_tests(filtered_duration_df, 'Duration (ms)')

# Function to add t-test results to the violin plot
def add_significance(df_ttest, order, ax):
    """ Annotate the violin plot with t-test significance results. """
    """ Annotate the violin plot with t-test significance results, with more separation. """
    y_offset = 0.05  # Starting y-offset to move lines further up each time
    for i, row in df_ttest.iterrows():
        method1 = row['Method1']
        method2 = row['Method2']
        p_value = row['p-value']

        significance = ''
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'

        if significance:
            # Ensure correct x-axis placement based on method order
            x1, x2 = order.index(method1), order.index(method2)

            # Get the maximum y-value from the current plot
            y_max = ax.get_ylim()[1]  # This gives the max value on the y-axis

            # Adjust y position and height for the lines and text
            # Increase y-position incrementally for each line to avoid overlap
            y, h = y_max * (0.95 + y_offset), y_max * 0.05

            # Plot the horizontal line and the significance marker
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
            ax.text((x1 + x2) * 0.5, y + h, significance, ha='center', va='bottom', color='black')
# Function to plot violin plots with percentiles and mean as dashed lines
def plot_violin(data_df, value_col, title, ylabel, t_test_results):
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10, 6))

    order = ['Raw', 'Deep Interpolation', 'DeepCAD', 'Pseudo', '200ms']

    # Define color palette for the violin outlines
    outline_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot the violin plots
    ax = sns.violinplot(x='Denoising Method', y=value_col, data=data_df, cut=0, inner="quart", order=order, fill=False, linewidth=2.5,
                        palette=outline_colors)

    # Add t-test significance to the plot
    add_significance(t_test_results, order, ax)

    # Adding title and labels
    plt.title(title)
    plt.ylabel(ylabel)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Plot for Area (normalized to µm²) - Only areas less than 50 µm²
plot_violin(filtered_area_df, 'Area', 'Calcium Flicker Areas < 50 µm²', 'Area (µm²)', t_test_area_results)

# Plot for Duration (if needed for future use)
plot_violin(filtered_duration_df, 'Duration (ms)', 'Calcium Flicker Durations < 3 s', 'Duration (ms)', t_test_duration_results)
