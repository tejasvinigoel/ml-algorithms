# Import necessary libraries
import seaborn as sns
import pandas as pd

# Load the Titanic dataset
titanic_df = sns.load_dataset("titanic")

# Create a new binary encoded feature 'embarked_southampton'
# Set 1 if 'embark_town' is 'Southampton', else 0
embark_town_col = pd.DataFrame([1 if i == 'Southampton' else 0 for i in titanic_df['embark_town']], columns=["embarked_southampton"])

# Join to the main dataframe with aligned indices
titanic_df = titanic_df.reset_index(drop=True)
embark_town_col = embark_town_col.reset_index(drop=True)
titanic_df = pd.concat([titanic_df, embark_town_col], axis=1)

# Print the first 5 rows of the dataframe
print(titanic_df.head())