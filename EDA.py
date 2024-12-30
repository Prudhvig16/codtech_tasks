import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset from seaborn
df = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(df.head())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Distribution of each feature
plt.figure(figsize=(10, 6))
df.hist()
plt.suptitle('Histograms of Features')
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(df, hue='species')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title('Boxplot of Features to Detect Outliers')
plt.show()
