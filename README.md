# Financial Transaction Anomaly Detection Project

## Overview

This project aims to detect fraudulent transactions from a dataset of credit card transactions using machine learning techniques. It includes data preprocessing, exploratory data analysis (EDA), model building, and an interactive dashboard for real-time monitoring.

## Project Structure

The project consists of the following key components:

1.  `creditcard.csv`: The dataset containing credit card transactions.
2.  `train_model.py`: A Python script to train the machine learning model and save it to disk.
3.  `streamlit_app.py`: A Python script that creates the Streamlit dashboard to visualize key metrics, recent transactions, and model performance.
4.  `rf_model.pkl`: The pickled (saved) machine learning model file.
5.  `README.md`: This file, providing an overview and instructions for the project.

## Workflow

1.  **Data Loading and Inspection**:
    -   Load the credit card transactions dataset.
    -   Check for missing values, data types, and class imbalance.

2.  **Exploratory Data Analysis (EDA)**:
    -   Visualize distributions of `Time` and `Amount`.
    -   Confirm the severe class imbalance (very few frauds).
    -   Examine feature correlations (most features are uncorrelated due to PCA).

    *Example Plots*:

    -   Distribution of Transaction Time:
    -   Distribution of Transaction Amount:
    -   Correlation Matrix:
3.  **Data Preprocessing**:
    -   Scale `Time` and `Amount` features.
    -   Remove outliers from the `Amount` column using the IQR method.

4.  **Model Building and Training**:

    *   Use Random Forest Classifier with class weighting to handle imbalance.
    *   Save the trained model to disk using pickle (`rf_model.pkl`).
5.  **Dashboard Implementation**:

    *   Create a Streamlit app (`streamlit_app.py`) to visualize key metrics.

    *   Provide a real-time transaction feed.

    *   Visualize model performance using a classification report and confusion matrix.

    *   Add a user-adjustable anomaly threshold slider for real-time sensitivity control.
6.  **Optimization**:

    *   Avoid training the model on every dashboard launch by loading a pre-trained model (using `pickle`). This significantly speeds up the dashboard.

## Instructions for Running the Project

1.  **Clone the Repository**:
    ```
    git clone [repository URL]
    cd [repository directory]
    ```

2.  **Install Dependencies**:
    ```
    pip install pandas scikit-learn streamlit plotly
    ```

3.  **Train the Model**:
    ```
    python train_model.py
    ```
    This will train the Random Forest model and save it to `rf_model.pkl`.

4.  **Run the Streamlit App**:
    ```
    streamlit run streamlit_app.py
    ```
    This command will open the dashboard in your web browser.

5.  **Interact with the Dashboard**:
    -   Use the slider in the sidebar to adjust the anomaly detection threshold.
    -   Observe the key metrics, transaction feed, and model performance in real-time.

## Key Points to Remember

*   The `creditcard.csv` dataset should be placed in the same directory as the Python scripts.
*   The `train_model.py` script should be run first to generate the `rf_model.pkl` file.
*   The Streamlit dashboard reads the pre-trained model, which speeds up the dashboard.
*   Adjust the anomaly threshold to control the sensitivity of the anomaly detection.

## Additional Notes

*   Consider deploying the Streamlit app to platforms like Streamlit Cloud for wider accessibility.
*   Further improvements could involve feature engineering, hyperparameter tuning, and exploring other machine-learning models.
*   Implement anomaly detection systems to proactively identify and resolve inconsistencies in financial data.
*   Design impactful dashboards & visualizations that provide real-time financial intelligence.
*   Automate financial reporting pipelines—eliminating inefficiencies and manual work.

## By Vaibhav Sharma
