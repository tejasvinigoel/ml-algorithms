#values from 0 to 1->normalise
#mean 0,sd 1->standardise

# Import necessary libraries
import seaborn as sns
import pandas as pd

# Load the Titanic Dataset
titanic_df = sns.load_dataset('titanic')

# Normalize 'age'
titanic_df['age'] = (titanic_df['age'] - titanic_df['age'].min()) / (titanic_df['age'].max() - titanic_df['age'].min())

# Display the normalized ages
print(titanic_df['age'])

# Standardize 'fare'
titanic_df['fare'] = (titanic_df['fare'] - titanic_df['fare'].mean()) / titanic_df['fare'].std()

# Display the standardized fares
print(titanic_df['fare'])