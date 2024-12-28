# Import necessary libraries
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the Titanic Dataset
titanic_df = sns.load_dataset('titanic')

# Create a StandardScaler object
scaler = StandardScaler()

# Incorrectly using fit_transform on the 'fare' column with NaN values included
#titanic_df['stand_fare'] = scaler.fit_transform(titanic_df[['fare']])

#fit scaler on data without nan
scaler.fit(titanic_df[['fare']].dropna())

#transform everything
titanic_df['stand_fare'] = scaler.transform(titanic_df[['fare']])

# Display standardized fare values
print(titanic_df[['fare', 'stand_fare']])