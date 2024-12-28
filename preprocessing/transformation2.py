import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define example DataFrame
example_data = pd.DataFrame({
    'age': [23, 45, 56, 78, 21],
    'income': [5000, 7000, 11000, 8000, 7600],
    'embarked': ['S', 'C', 'Q', 'S', 'S']
})

# Initialize scalers
income_scaler = MinMaxScaler()  # Default feature range (0, 1)
age_scaler = MinMaxScaler(feature_range=(0, 10))  # Custom feature range (0, 10)

# Scale 'income' and 'age'
example_data['income'] = income_scaler.fit_transform(example_data[['income']])  # Convert column to 2D
example_data['age'] = age_scaler.fit_transform(example_data[['age']])  # Convert column to 2D

# Display the transformed DataFrame
print('After MinMaxScaler Transformation:\n', example_data)
