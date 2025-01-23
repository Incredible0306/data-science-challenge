import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_path = '../data/monthly_report_expanded.csv'
data = pd.read_csv(data_path)

# Data overview
print("Data Overview:")
print(data.head())
print("\nData Information:")
print(data.info())

# Data cleaning and preparation
def preprocess_data(data):
    # Handle missing values
    data.fillna(method='ffill', inplace=True)

    # Feature engineering (example: converting timestamps to datetime and extracting month/year)
    if 'activity_date' in data.columns:
        data['activity_date'] = pd.to_datetime(data['activity_date'])
        data['activity_month'] = data['activity_date'].dt.month
        data['activity_year'] = data['activity_date'].dt.year

    return data

data = preprocess_data(data)

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    print("\nSummary Statistics:")
    print(data.describe())

    print("\nCorrelation Matrix:")
    correlation_matrix = data.corr()
    print(correlation_matrix)

    # Visualization: Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

perform_eda(data)

# Predictive Modeling: Lead Scoring
def build_lead_scoring_model(data):
    # Assuming 'conversion_rate' is the target variable and rest are features
    if 'conversion_rate' in data.columns:
        X = data.drop(['conversion_rate'], axis=1)
        y = data['conversion_rate']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Lead Scoring Model Accuracy: {accuracy}")

        # Feature importance analysis
        feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        print("\nFeature Importances:")
        print(feature_importances)

        return model
    else:
        print("Target variable 'conversion_rate' not found.")
        return None

lead_scoring_model = build_lead_scoring_model(data)

# Time Series Analysis: Seasonal Patterns
def analyze_seasonality(data):
    if 'activity_date' in data.columns:
        data['activity_date'] = pd.to_datetime(data['activity_date'])
        data.set_index('activity_date', inplace=True)

        # Analyzing monthly sales trends
        monthly_data = data.resample('M').sum()
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_data.index, monthly_data['conversion_rate'], marker='o')
        plt.title('Monthly Conversion Rate Trends')
        plt.xlabel('Month')
        plt.ylabel('Conversion Rate')
        plt.grid()
        plt.show()

analyze_seasonality(data)

# Prescriptive Analytics: Recommendations
def recommend_activities(data):
    # Basic recommendations based on average activity levels
    if 'activity_type' in data.columns and 'conversion_rate' in data.columns:
        avg_conversion_rate = data.groupby('activity_type')['conversion_rate'].mean()
        print("\nRecommended Activity Targets:")
        print(avg_conversion_rate)
    else:
        print("Required columns 'activity_type' or 'conversion_rate' not found.")

recommend_activities(data)
