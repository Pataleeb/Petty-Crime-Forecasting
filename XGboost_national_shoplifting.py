import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv('combined_Shoplifting.csv')

# Extract only "United States Total" row
df_us = df[df["series"] == "United States Total"].drop(columns=["series"])

# Transpose so that months become the index
df_us = df_us.T

# Convert index to datetime
df_us.index = pd.to_datetime(df_us.index, format='%b-%y', errors='coerce')

# Rename the only column
df_us.columns = ["Shoplifting_Cases"]

# Sort by date
df_us = df_us.sort_index()

# Set frequency to Monthly Start ('MS')
df_us = df_us.asfreq('MS')

# Log Transformation (to stabilize variance)
df_us["Shoplifting_Cases_Log"] = np.log1p(df_us["Shoplifting_Cases"])

# First Differencing (to remove trend)
df_us["Shoplifting_Cases_Diff"] = df_us["Shoplifting_Cases_Log"].diff()

# Feature Engineering: Creating Lag Features
def create_features(data, lags=[1, 2, 3, 6, 12, 18, 24]):
    df = data.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["Shoplifting_Cases_Diff"].shift(lag)
    df["month"] = df.index.month  # Capture seasonality
    df["year"] = df.index.year    # Capture long-term trends
    return df

# Apply feature engineering
df_us = create_features(df_us)

# Drop NaNs after differencing
df_us = df_us.dropna()

# Train-Test Split
train = df_us.loc[:"2022-12-01"]
test = df_us.loc["2023-01-01":"2023-12-01"]

# Define Features and Target
X_train, y_train = train.drop(columns=["Shoplifting_Cases_Diff", "Shoplifting_Cases_Log", "Shoplifting_Cases"]), train["Shoplifting_Cases_Diff"]
X_test, y_test = test.drop(columns=["Shoplifting_Cases_Diff", "Shoplifting_Cases_Log", "Shoplifting_Cases"]), test["Shoplifting_Cases_Diff"]

# Train XGBoost Model
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=1,
    colsample_bytree=0.9,
    subsample=0.8,
    gamma=0,
    objective='reg:squarederror',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Make Predictions (Differenced Scale)
y_pred_diff = xgb_model.predict(X_test)

# Reverse Differencing (to get original scale)
last_known_value = df_us.loc["2022-12-01", "Shoplifting_Cases_Log"]
y_pred_log = last_known_value + np.cumsum(y_pred_diff)

# Convert back from Log Scale
y_pred_final = np.expm1(y_pred_log)

# Create Forecast DataFrame
forecast_df = pd.DataFrame({"Forecasted Cases": y_pred_final}, index=X_test.index)

# Print Data Info
print(f"Test Set Shape: {test.shape}")
print(test.head())
print("Predicted Differenced Values:", y_pred_diff[:5])
print("Final Predictions:", y_pred_final[:5])


if len(test["Shoplifting_Cases"]) != len(y_pred_final):
    print(f"Mismatch in lengths! y_test: {len(test['Shoplifting_Cases'])}, y_pred: {len(y_pred_final)}")
else:

    rmse = np.sqrt(mean_squared_error(test["Shoplifting_Cases"], y_pred_final))


    mae = mean_absolute_error(test["Shoplifting_Cases"], y_pred_final)


    nonzero_mask = test["Shoplifting_Cases"] != 0  # Ensure no division by zero
    mape = np.mean(np.abs((test["Shoplifting_Cases"][nonzero_mask] - y_pred_final[nonzero_mask]) / test["Shoplifting_Cases"][nonzero_mask])) * 100

    # Print All Three Error Metrics
    print("\nXGBoost Model Error Metrics for 2023:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}% (ignoring zero actual values)")

# Plot Actual vs Predicted Values
plt.figure(figsize=(12, 5))
plt.plot(test.index, test["Shoplifting_Cases"], label="Actual Shoplifting Cases (2023)", color="black", marker="o")
plt.plot(forecast_df.index, forecast_df["Forecasted Cases"], label="XGBoost Predicted Cases (2023)", color="blue", linestyle="dashed", marker="s")
plt.title("XGBoost Actual vs Predicted Shoplifting Cases (2023)")
plt.xlabel("Date")
plt.ylabel("Number of Cases")
plt.legend()
plt.grid()
plt.show(block=True)
