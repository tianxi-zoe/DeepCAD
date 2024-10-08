import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Function to process data (normalize intensity for each method to 0-255)
def process_data(data):
    intensities = [entry["Intensity"] for entry in data]  # Get all intensity values for the method
    max_intensity = max(intensities)  # Find the maximum intensity for the method

    cleaned_data = []
    for entry in data:
        Duration = entry["Duration (ms)"]  # Duration is already in milliseconds
        normalized_intensity = (entry["Intensity"] / max_intensity) * 255  # Normalize intensity to 0-255 for each method
        cleaned_data.append({"Duration": Duration, "Intensity": normalized_intensity})
    return cleaned_data


# Define the paths to the JSON files for each method
method_files = {
    "Raw": 'raw_clean_flicker_data_new.json',
    "DeepInterpolation": 'deepin_clean_flicker_data_new.json',
    "DeepCAD": 'deepcad_clean_flicker_data_new.json',
    "Pseudo": 'pseudo_clean_flicker_data_new.json'
}

# Load and process data for each method
all_data = []  # Store data for plotting

# Normalize and process data for each method individually
for method, file_path in method_files.items():
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            processed_data = process_data(data)  # Normalize intensities for this method
            for entry in processed_data:
                entry["Method"] = method  # Add method label for plotting
            all_data.extend(processed_data)
    else:
        print(f"File not found: {file_path}")

# Convert the combined data to a pandas DataFrame
df = pd.DataFrame(all_data)
filtered_Duration_df = df[df['Duration'] < 3000]

# Create scatter plots for each method separately
methods = filtered_Duration_df["Method"].unique()

# Plot Duration vs Intensity for each method
for method in methods:
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Intensity", y="Duration", data=filtered_Duration_df[filtered_Duration_df["Method"] == method], s=50)

    # Add labels and title
    plt.title(f'Duration vs Intensity ({method})')
    plt.ylabel('Duration')
    plt.xlabel('Normalized Intensity (0-255)')
    plt.grid(True)
    plt.xlim(0, 255)  # Intensity is normalized to 0-255 for each method
    plt.ylim(0, filtered_Duration_df['Duration'].max() + 10)  # Adjust y-axis to start from 0 and extend slightly beyond the max duration

    # Show the plot
    plt.tight_layout()
    plt.show()
