import pandas as pd
import os

# Read data from the CSV file
# Get the directory of the script file
script_dir = os.path.dirname(__file__)

# Path to the main.py directory (one level up from script_dir)
main_dir = os.path.dirname(script_dir)
# Read the selected image names from the text file
with open(os.path.join(main_dir, 'output','selected_image_names.txt'), 'r') as file:
    selected_images = file.read().splitlines()

# Extract the root numbers and corresponding Image names
selected_roots = {}
for image_name in selected_images:
    parts = image_name.split('_')
    # Construct the image type (everything before the root number and extension)
    image_type = '_'.join(parts[:-1])
    # Extract the root number (the number before '.png')
    root_number = 'root_' + parts[-1].split('.')[0]
    # If the image type is not already in the dictionary, add it with an empty list
    if image_type not in selected_roots:
        selected_roots[image_type] = []
    # Append the root number to the list for this image type
    selected_roots[image_type].append(root_number)

# Read the CSV file into a DataFrame
df = pd.read_csv(os.path.join(main_dir, 'output','periderm_length_pixels.csv'))

# Create a new DataFrame to store the quality controlled data
df_qc = df.copy()

# Iterate over the DataFrame and nullify the cells that are not in the selected images
for index, row in df_qc.iterrows():
    image_type = row['Image']
    # Check all root columns for this image type
    for col in df_qc.columns[1:]:  # Skip the 'Image' column
        root_col = col.split('_')[1]  # Get the root number from the column name
        # If the root number is not in the selected_roots for this image type, set its value to NaN
        if image_type in selected_roots and f'root_{root_col}' not in selected_roots[image_type]:
            df_qc.at[index, col] = pd.NA

# Save the processed DataFrame to a new CSV file
output_csv_file_path = os.path.join(main_dir, 'output','periderm_length_after_QC_pixels.csv')
df_qc.to_csv(output_csv_file_path, index=False)

# The path to the new CSV file for download
output_csv_file_path

