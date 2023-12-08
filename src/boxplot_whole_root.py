import pandas as pd
import os
import matplotlib.pyplot as plt

# Read data from the CSV file
# Get the directory of the script file
script_dir = os.path.dirname(__file__)

# Path to the main.py directory (one level up from script_dir)
main_dir = os.path.dirname(script_dir)

# Path to the CSV file
csv_file_path = os.path.join(main_dir, 'output','whole_root_length.csv')

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Set the Image column as the index
df.set_index('Image', inplace=True)

# Transpose the dataframe
df_transposed = df.transpose()

# Plot the boxplots
plt.figure(figsize=(10, 6))
df_transposed.boxplot(grid=False)

plt.title('Whole Root Length for Each Image')
plt.ylabel('Value')
plt.xlabel('Image')
plt.xticks(rotation=45)

# Save the figure
plt.tight_layout()
output_path = os.path.join(main_dir, 'output', 'whole_root_length_boxplot.png')
plt.savefig(output_path)

plt.show()
