import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the model
try:
    filename = 'rf_model.pkl'
    rf_classifier = pickle.load(open(filename, 'rb'))
except FileNotFoundError:
    st.error("Model file 'rf_model.pkl' not found. Please train and save the model first.")
    st.stop()

# Load the dataset
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    st.error("Error: 'creditcard.csv' not found. Make sure the file is in the correct directory.")
    st.stop()

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

# Anomaly Threshold Slider
st.sidebar.header("Adjust Anomaly Detection Sensitivity")
anomaly_threshold = st.sidebar.slider(
    "Threshold",
    min_value=0.01,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Adjust the threshold to control the sensitivity of anomaly detection. Lower values detect more anomalies."
)

# Get anomaly probabilities for the test set
anomaly_probabilities = rf_classifier.predict_proba(X_test)[:, 1]

# Predict anomalies based on the adjusted threshold
y_pred_adjusted = (anomaly_probabilities > anomaly_threshold).astype(int)

# Add predictions to the test data
test_data = X_test.copy()
test_data['Class'] = y_test
test_data['Prediction'] = y_pred_adjusted
test_data['Amount'] = df['Amount']

# Key Metrics
total_transactions = len(df)
average_transaction_amount = df['Amount'].mean()
number_of_anomalies = sum(y_pred_adjusted)

# Visualization of key metrics
st.subheader('Key Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", total_transactions)
col2.metric("Average Transaction Amount", f"${average_transaction_amount:,.2f}")
col3.metric("Anomalies Detected", number_of_anomalies)

# Visualizations using Plotly
st.subheader('Transaction Amount Distribution')
amount_fig = px.histogram(df, x='Amount', nbins=50, title='Distribution of Transaction Amounts')
st.plotly_chart(amount_fig, use_container_width=True)

st.subheader('Class Distribution')
class_counts = df['Class'].value_counts()
class_fig = px.bar(x=class_counts.index, y=class_counts.values,
                   labels={'x': 'Class', 'y': 'Number of Transactions'},
                   title='Distribution of Transaction Classes')
st.plotly_chart(class_fig, use_container_width=True)

# Real-Time Transaction Feed
st.subheader('Real-Time Transaction Feed')
st.write("Displaying the most recent transactions, highlighting any anomalies.")

# Highlight anomalies in the transaction feed
def highlight_anomalies(row):
    if row['Prediction'] == 1:
        return ['background-color: pink'] * len(row)
    else:
        return [''] * len(row)

# Display recent transactions with predictions
recent_transactions = test_data.tail(10).style.apply(highlight_anomalies, axis=1)
st.dataframe(recent_transactions)

st.subheader('Model Performance')
st.write("Classification Report:")
report = classification_report(y_test, y_pred_adjusted, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.write("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
st.dataframe(pd.DataFrame(conf_matrix, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))
