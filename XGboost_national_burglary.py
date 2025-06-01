import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv('combined_Burglary.csv')

# Extract only "United States Total" row
df_us = df[df["series"] == "United States Total"].drop(columns=["series"])

# Transpose so that months become the index
df_us = df_us.T
df_us.columns = ["burglary_Cases"]

# Ensure burglary cases are numeric
df_us["burglary_Cases"] = pd.to_numeric(df_us["burglary_Cases"], errors='coerce')

# Fix Date Parsing
df_us.index = df_us.index.astype(str).str.strip()
df_us.index = pd.to_datetime(df_us.index, format='%b-%y', errors='coerce')

# Print before filtering
print("\nBefore filtering NaT values, DataFrame shape:", df_us.shape)

# Remove NaT Values
df_us = df_us[~df_us.index.isna()]


# Ensure Monthly Frequency
df_us = df_us.asfreq('MS')


# Log Transformation
df_us["Log_burglary"] = np.log1p(df_us["burglary_Cases"])

# First Differencing
df_us["Log_Diff1"] = df_us["Log_burglary"].diff()

# Feature Engineering: Lag Features
def create_features(data, lags=[1, 2, 3, 6, 9, 12, 15, 18, 21, 24]):
    df = data.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["Log_Diff1"].shift(lag)
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df

# Apply feature engineering
df_us = create_features(df_us)

# Convert Lag Features to Float
df_us = df_us.astype(float)

# Drop NaNs after differencing
df_us = df_us.dropna()


# Train-Test Split
train = df_us.loc[:"2022-12-01"]
test = df_us.loc["2023-01-01":"2023-12-01"]

# Define Features and Target
X_train, y_train = train.drop(columns=["Log_Diff1", "Log_burglary", "burglary_Cases"]), train["Log_Diff1"]
X_test, y_test = test.drop(columns=["Log_Diff1", "Log_burglary", "burglary_Cases"]), test["Log_Diff1"]

# Train XGBoost Model
xgb_model = XGBRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight=1,
    colsample_bytree=0.8,
    subsample=0.8,
    gamma=0.1,
    objective='reg:squarederror',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Make Predictions
y_pred_diff = xgb_model.predict(X_test)

# **Find the Last Available Date Dynamically**
last_valid_date = df_us.index.max()
print(f"\nLast available date in dataset: {last_valid_date}")

# **Check if last_valid_date is still NaT**
if pd.isna(last_valid_date):
    print("\n ERROR: Last valid date is NaT. Cannot proceed with forecasting.")
    exit()

# Reverse Differencing
last_known_value = df_us.loc[last_valid_date, "Log_burglary"]
y_pred_log = last_known_value + np.cumsum(y_pred_diff)

# Convert Back from Log Scale
y_pred_final = np.expm1(y_pred_log)

# Create Forecast DataFrame
forecast_df = pd.DataFrame({"Forecasted Cases": y_pred_final}, index=X_test.index)

# Error Metrics
if len(test["burglary_Cases"]) != len(y_pred_final):
    print(f"Mismatch in lengths! y_test: {len(test['burglary_Cases'])}, y_pred: {len(y_pred_final)}")
else:
    rmse = np.sqrt(mean_squared_error(test["burglary_Cases"], y_pred_final))
    mae = mean_absolute_error(test["burglary_Cases"], y_pred_final)
    nonzero_mask = test["burglary_Cases"] != 0
    mape = np.mean(np.abs((test["burglary_Cases"][nonzero_mask] - y_pred_final[nonzero_mask]) / test["burglary_Cases"][nonzero_mask])) * 100

    print("\n Optimized XGBoost Model Error Metrics for 2023:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

# Plot Actual vs Predicted Values
plt.figure(figsize=(12, 5))
plt.plot(test.index, test["burglary_Cases"], label="Actual Burglary Cases (2023)", color="black", marker="o")
plt.plot(forecast_df.index, forecast_df["Forecasted Cases"], label="XGBoost Predicted Cases (2023)", color="blue", linestyle="dashed", marker="s")
plt.title("Optimized XGBoost Actual vs Predicted Burglary Cases (2023)")
plt.xlabel("Date")
plt.ylabel("Number of Cases")
plt.legend()
plt.grid()
plt.show()
