from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

titanic_df = sns.load_dataset('titanic')

# Select 'age' column and drop NaN values
age = titanic_df[['age']].dropna()

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Use the scaler
titanic_df['norm_age'] = pd.DataFrame(scaler.fit_transform(age), columns=age.columns, index=age.index)

# Display normalized age values
print(titanic_df['norm_age'])

from sklearn.preprocessing import StandardScaler

# Select 'fare' column and drop NaN values
fare = titanic_df[['fare']].dropna()

# Create a StandardScaler object
scaler = StandardScaler()

# Use the scaler
titanic_df['stand_fare'] = pd.DataFrame(scaler.fit_transform(fare), columns=fare.columns, index=fare.index)

# Display standardized fare values
print(titanic_df['stand_fare'])