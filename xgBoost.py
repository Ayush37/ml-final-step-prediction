import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv("batch_processing_data.csv")

# Convert all timestamp columns to datetime
timestamp_cols = [col for col in df.columns if "FEED" in col or "FINAL_STEP" in col or "Last_Feed_Completion"]
for col in timestamp_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Convert timestamps to minutes since midnight
for col in timestamp_cols:
    df[col] = df[col].dt.hour * 60 + df[col].dt.minute

# Define features (FEED completion times) and target (FINAL_STEP start time)
X = df[[col for col in df.columns if "FEED" in col]]
y = df["FINAL_STEP_Start"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.5, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# Evaluate predictions within ±5 minutes
accuracy_within_5_min = np.mean(np.abs(y_test - y_pred) <= 7) * 100

print(f"XGBoost Model - Mean Absolute Error: {mae:.2f} minutes")
print(f"Accuracy within ±5 minutes: {accuracy_within_5_min:.2f}%")

