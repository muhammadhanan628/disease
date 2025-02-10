import os
import pandas as pd

# Define the path to the dataset
data_path = os.path.join('data', 'training.csv')

# Load the dataset
data = pd.read_csv(data_path)

# Display the first few rows of the data
print(data.head())

