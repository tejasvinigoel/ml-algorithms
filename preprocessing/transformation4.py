# Import necessary libraries
import seaborn as sns
import pandas as pd

# TODO: Load the Titanic dataset and assign it to a variable named 'titanic_df'
titanic_df = sns.load_dataset('titanic')
# TODO: Perform one-hot encoding on the 'class' column to create binary columns for each class
#using just titanic_df would have given class_first, class_second etc..
#now its just first,second,third
encoded_features = pd.get_dummies(titanic_df['class'], columns=['class'])
# TODO: Join the new binary columns to 'titanic_df'
titanic_df = pd.concat([titanic_df, encoded_features], axis=1)
# TODO: Display the first 5 rows of the modified dataframe
print(titanic_df.head())