# Importing necessary libraries
from sklearn.model_selection import train_test_split
import seaborn as sns

# Loading the Titanic dataset
titanic_df = sns.load_dataset('titanic')

# Splitting the full dataset into the training and testing datasets
train_data, test_data = train_test_split(titanic_df, test_size=0.2, random_state=42)

# Printing out the shapes of the datasets
print(f"Train data shape: {train_data.shape}") # Expected Output: (712, 15)
print(f"Test data shape: {test_data.shape}") # Expected Output: (179, 15)


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression() # Initialize a Logistic Regression model

# We separate the target variable ("survived") from the rest of the training data
x_train = train_data.drop("survived", axis=1)
y_train = train_data["survived"]

# Training the Logistic Regression model
logreg.fit(x_train, y_train)

# Importing necessary libraries
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Separating the independent (x_test) and dependent (y_test) variables from the testing dataset
x_test = test_data.drop("survived", axis=1)
y_test = test_data["survived"]

# Using the model to make predictions on the testing dataset
predictions = logreg.predict(x_test)

# Displaying metrics
print("Classification Report:")
print(classification_report(y_test, predictions)) 

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Accuracy Score:")
print(accuracy_score(y_test, predictions))