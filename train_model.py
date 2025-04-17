import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Make sure the file is in the correct directory.")
    exit()

# Calculate the IQR for the 'Amount' column
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1

# Define the upper and lower bounds for outlier detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers from the DataFrame
df_no_outliers = df[(df['Amount'] >= lower_bound) & (df['Amount'] <= upper_bound)]

# Separate features and target variable
X = df_no_outliers.drop('Class', axis=1)
y = df_no_outliers['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Fit the model to the training data
rf_classifier.fit(X_train, y_train)

# Save the model to disk
filename = 'rf_model.pkl'
pickle.dump(rf_classifier, open(filename, 'wb'))

print("Model saved to disk!")
