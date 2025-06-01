import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv('combined_Pocket_Picking.csv')

# Extract only "United States" row
df_us = df[df["series"] == "United States"].drop(columns=["series"])

# Transpose so that months become the index
df_us = df_us.T

# Convert index to datetime
df_us.index = pd.to_datetime(df_us.index, format='%y-%b', errors='coerce')

# Rename the only column
df_us.columns = ["pp_Cases"]

# Sort by date
df_us = df_us.sort_index()

# Set frequency to Monthly Start ('MS')
df_us = df_us.asfreq('MS')

# Log Transformation (to stabilize variance)
df_us["Log_pp"] = np.log1p(df_us["pp_Cases"])

# First Differencing (to remove trend)
df_us["Log_Diff1"] = df_us["Log_pp"].diff()

# Feature Engineering: Creating More Lag Features
def create_features(data, lags=[1, 2, 3, 6, 9, 12, 15, 18, 21, 24]):
    df = data.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["Log_Diff1"].shift(lag)
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df

# Apply feature engineering
df_us = create_features(df_us)

# Drop NaNs after differencing
df_us = df_us.dropna()

# Train-Test Split
train = df_us.loc[:"2022-12-01"]
test = df_us.loc["2023-01-01":"2023-12-01"]

# Define Features and Target
X_train, y_train = train.drop(columns=["Log_Diff1", "Log_pp", "pp_Cases"]), train["Log_Diff1"]
X_test, y_test = test.drop(columns=["Log_Diff1", "Log_pp", "pp_Cases"]), test["Log_Diff1"]

# Train Optimized XGBoost Model
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

# Make Predictions (Differenced Scale)
y_pred_diff = xgb_model.predict(X_test)

# Reverse Differencing (to get original scale)
last_known_value = df_us.loc["2022-12-01", "Log_pp"]
y_pred_log = last_known_value + np.cumsum(y_pred_diff)

# Convert back from Log Scale
y_pred_final = np.expm1(y_pred_log)

# Create Forecast DataFrame
forecast_df = pd.DataFrame({"Forecasted Cases": y_pred_final}, index=X_test.index)


importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": xgb_model.feature_importances_})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
print("\nFeature Importance (Top 10):")
print(importance_df.head(10))

# Error Metrics Calculation
if len(test["pp_Cases"]) != len(y_pred_final):
    print(f"Mismatch in lengths! y_test: {len(test['pp_Cases'])}, y_pred: {len(y_pred_final)}")
else:
    rmse = np.sqrt(mean_squared_error(test["pp_Cases"], y_pred_final))
    mae = mean_absolute_error(test["pp_Cases"], y_pred_final)
    nonzero_mask = test["pp_Cases"] != 0
    mape = np.mean(np.abs((test["pp_Cases"][nonzero_mask] - y_pred_final[nonzero_mask]) / test["pp_Cases"][nonzero_mask])) * 100

    print("\nOptimized XGBoost Model Error Metrics for 2023:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}% (ignoring zero actual values)")


plt.figure(figsize=(12, 5))
plt.plot(test.index, test["pp_Cases"], label="Actual Pocket Picking Cases (2023)", color="black", marker="o")
plt.plot(forecast_df.index, forecast_df["Forecasted Cases"], label="XGBoost Predicted Cases (2023)", color="blue", linestyle="dashed", marker="s")
plt.title("Optimized XGBoost Actual vs Predicted Pocket Picking Cases (2023)")
plt.xlabel("Date")
plt.ylabel("Number of Cases")
plt.legend()
plt.grid()
plt.show(block=True)
