from sklearn.preprocessing import MinMaxScaler

#numerical features transformation
# Converting dataframe into an array for the scaler.... between 0 to 1... custom range also possible
titanic_df[['age', 'fare']] = MinMaxScaler().fit_transform(titanic_df[['age', 'fare']])


print('After numerical feature scaling:\n', titanic_df[['age','fare']].head())

#categorical features transformation
# Transform the categorical features with One-Hot Encoding... 0->absence of feature .. 1->present
titanic_df = pd.get_dummies(titanic_df, columns=['sex', 'embarked']) #true false values

print('After one-hot encoding of categorical features:\n', titanic_df.head())

# for 0,1 values in one hot encoding
sex_dummies = pd.get_dummies(titanic_df['sex'], dtype=int)

