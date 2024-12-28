# Import the necessary libraries
import numpy as np
import seaborn as sns

# TODO: Load the Titanic dataset and store it in a variable named 'titanic_df'
titanic_df = sns.load_dataset('titanic')

# TODO: Drop any rows with missing values in the 'age' column
titanic_df = titanic_df.dropna(subset=['age'])

# TODO: Calculate the first and third quartile of the 'age' column and store them in variables 'Q1_age' and 'Q3_age'
Q1_age = titanic_df['age'].quantile(0.25)
Q3_age = titanic_df['age'].quantile(0.75)

# TODO: Calculate the Interquartile Range (IQR) for the 'age' column and store it in a variable 'IQR_age'
IQR = Q3_age - Q1_age

# TODO: Using IQR, identify any age values that are outliers and store them in a variable called 'outliers_age'
outliers_age = titanic_df['age'][
    (titanic_df['age'] < (Q1_age - 1.5 * IQR))| 
    (titanic_df['age'] > (Q3_age + 1.5 * IQR))
]

# TODO: Output the outliers found in the 'age' column
print('outliers in fare using IQE\n', outliers_age)