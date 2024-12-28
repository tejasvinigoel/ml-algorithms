# Import the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns

# Load the Titanic Dataset
titanic_df = sns.load_dataset('titanic')

# Ensure that 'age' and 'fare' columns do not have null values
titanic_df = titanic_df.dropna(subset=['age', 'fare'])

# Compute Z-scores for 'age' and 'fare' using sample standard deviation (ddof=1)
mean_age = np.mean(titanic_df['age'])
std_dev_age = np.std(titanic_df['age'], ddof=1)
Z_scores_age = (titanic_df['age'] - mean_age) / std_dev_age
outliers_age = titanic_df['age'][np.abs(Z_scores_age) > 2.5]

# TODO: Calculate the mean and sample standard deviation for the 'fare' column, then compute the Z-scores.
meanfare = np.mean(titanic_df['fare'])
stddevfare = np.std(titanic_df['fare'], ddof=1)
zscoresfare = (titanic_df['fare'] - meanfare) / stddevfare
outliersfare = titanic_df['fare'][np.abs(zscoresfare) > 2.5]

# Print the outliers
print("Outliers in 'Age' using Z-score: \n", outliers_age)
# TODO: Print the outliers for 'fare' using Z-scores.
print("Outliers in 'Fare' using Z-score: \n", outliersfare)