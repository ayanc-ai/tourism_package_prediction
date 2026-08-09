import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define the path to the dataset
DATA_PATH = "tourism_project/data/tourism.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Remove unnecessary columns (e.g., CustomerID)
# Based on the problem description, CustomerID is a unique identifier and not a feature.
df = df.drop(columns=['CustomerID'])

# Define features (X) and target (y)
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

# Split data into training and testing sets
# Using a common test size of 0.2 (20%) and a random state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split datasets locally as CSV files
X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index=False)
y_train.to_csv('ytrain.csv', index=False, header=True)
y_test.to_csv('ytest.csv', index=False, header=True)

print("Dataset loaded, unnecessary columns removed, and data split into training and testing sets.")
print("Splits saved as Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv in the current directory.")
