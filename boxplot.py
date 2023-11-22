import pandas as pd
import matplotlib.pyplot as plt

# Read data from the CSV file
df = pd.read_csv('/home/lzhang/Desktop/FY_test/periderm_length.csv')

# Set the Image column as the index
df.set_index('Image', inplace=True)

# Transpose the dataframe
df_transposed = df.transpose()

# Plot the boxplots
plt.figure(figsize=(10, 6))
df_transposed.boxplot(grid=False)

plt.title('Boxplots for Each Image')
plt.ylabel('Value')
plt.xlabel('Image')
plt.xticks(rotation=45)

# Save the figure
plt.tight_layout()
plt.savefig('boxplots.png')
plt.show()
