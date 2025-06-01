import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore

# Load dataset
df = pd.read_csv('combined_Shoplifting.csv')

# Extract only "United States Total" row
df_us = df[df["series"] == "United States Total"].drop(columns=["series"])
df_us = df_us.T

# Convert index to datetime (from format like "Feb-15", "Mar-15")
df_us.index = pd.to_datetime(df_us.index, format='%b-%y', errors='coerce')

# Rename the only column
df_us.columns = ["Shoplifting_Cases"]

# Sort by date
df_us = df_us.sort_index()

# Print first few rows for validation
print("\nExtracted U.S. Data (United States Total):")
print(df_us.head())

# Plot the original time series
plt.figure(figsize=(12, 5))
plt.plot(df_us, label="United States Shoplifting Cases", color="blue")
plt.title("United States Total Shoplifting Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Cases")
plt.legend()
plt.grid()
plt.show()

# ADF Test for Stationarity
def check_stationarity(series, desc="Original Series"):
    result = adfuller(series.dropna())
    print(f"\n ADF Test for {desc}:")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value:.4f}')
    if result[1] < 0.05:
        print("Stationary (No Differencing Needed).")
    else:
        print("Non-Stationary (Differencing Needed).")


check_stationarity(df_us["Shoplifting_Cases"])
###The data is not stationary
df_us["Log_Shoplifting"] = np.log(df_us["Shoplifting_Cases"] + 1)

# Apply First-Order Differencing
df_us["Log_Diff1"] = df_us["Log_Shoplifting"].diff()

# Check Stationarity After First Differencing
check_stationarity(df_us["Log_Diff1"], "Log + First Differencing")

# Plot Transformed Data
plt.figure(figsize=(12, 5))
plt.plot(df_us["Log_Diff1"], label="Log + First Differencing", color="red")
plt.title("Transformed Series: Log + First Differencing")
plt.xlabel("Date")
plt.ylabel("Differenced Log Cases")
plt.legend()
plt.grid()
plt.show()

###SARIMA model

df_us.index = pd.to_datetime(df_us.index)

# Set frequency to Monthly Start ('MS')
df_us = df_us.asfreq('MS')

# Get the last available date in the dataset
last_valid_date = df_us.last_valid_index()
print(f"\nLast available date in dataset: {last_valid_date}")

# Use the last available log-transformed value instead of hardcoded date
last_log_value = df_us.loc[last_valid_date, "Log_Shoplifting"]
print(f"Last log-transformed value used: {last_log_value}")

###SARIMA
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure test data is available (2023 actual values)
train = df_us.loc[:"2022-12-01", "Log_Diff1"]  # Train only up to Dec 2022
test = df_us.loc["2023-01-01":"2023-12-01", "Shoplifting_Cases"]  # Actual data for 2023

# Define SARIMA Order
sarima_order = (2, 1, 2)
seasonal_order = (0, 0, 0, 12)

# Fit SARIMA Model (Train only on data before 2023)
model = sm.tsa.statespace.SARIMAX(
    train,
    order=sarima_order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()

# Forecast 12 months (2023) in log-differenced scale
forecast_diff = model.forecast(steps=12)

# Reverse Differencing (Cumulative Sum)
last_log_value = df_us.loc["2022-12-01", "Log_Shoplifting"]  # Last known log-transformed value before 2023
forecast_log = last_log_value + forecast_diff.cumsum()

# Reverse Log Transformation (Convert back to Original Scale)
forecast_actual = np.exp(forecast_log) - 1

# Create Forecast DataFrame
forecast_dates = pd.date_range(start="2023-01-01", periods=12, freq="MS")
forecast_df = pd.DataFrame({"Forecasted Shoplifting Cases": forecast_actual}, index=forecast_dates)

# Print Forecast for 2023
print("\nSARIMA Forecast for U.S. Shoplifting (2023):")
print(forecast_df)

# Plot Actual vs Predicted Values for 2023
plt.figure(figsize=(12, 5))

# Plot Actual Values (2023)
plt.plot(test.index, test, label="Actual Shoplifting Cases (2023)", color="black", marker="o")

# Plot Predicted Values (2023)
plt.plot(forecast_df.index, forecast_df["Forecasted Shoplifting Cases"],
         label="SARIMA Predicted Cases (2023)", color="red", linestyle="dashed", marker="s")


plt.title("Actual vs Predicted Shoplifting Cases (2023)")
plt.xlabel("Date")
plt.ylabel("Number of Cases")
plt.legend()
plt.grid()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ensure test and forecast lengths match
if len(test) != len(forecast_actual):
    print("Warning: Test and forecast lengths do not match! Check data.")
else:
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test, forecast_actual))

    # Calculate MAE
    mae = mean_absolute_error(test, forecast_actual)

    # Calculate MAPE
    nonzero_mask = test != 0  # Mask for nonzero actual values
    mape = np.mean(np.abs((test[nonzero_mask] - forecast_actual[nonzero_mask]) / test[nonzero_mask])) * 100

    # Print Results
    print("\n Model Error Metrics for 2023:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}% (ignoring zero actual values)")

###Plot for holiday months
# Define holiday months
holiday_months = [1, 7, 10, 11, 12]  # January, July, October, November, December

# Filter test data (actual values) for holiday months
test_holiday = test[test.index.month.isin(holiday_months)]

# Filter forecasted data (predicted values) for holiday months
forecast_holiday = forecast_df[forecast_df.index.month.isin(holiday_months)]

# Ensure same length
if len(test_holiday) != len(forecast_holiday):
    print("Mismatch in test and forecast holiday months! Check data.")
else:
    # Plot Holiday Months Actual vs Predicted
    plt.figure(figsize=(10, 5))

    # Plot Actual Holiday Values
    plt.plot(test_holiday.index, test_holiday, label="Actual Shoplifting Cases (Holiday Months)",
             color="black", marker="o")

    # Plot Predicted Holiday Values
    plt.plot(forecast_holiday.index, forecast_holiday["Forecasted Shoplifting Cases"],
             label="SARIMA Predicted Cases (Holiday Months)", color="red", linestyle="dashed", marker="s")


    plt.title("Actual vs Predicted Shoplifting Cases (Holiday Months: Jan, Jul, Oct, Nov, Dec)")
    plt.xlabel("Date")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.grid()
    plt.show()
###Paramter tuning for SARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Plot ACF & PACF for Log-Differenced Series
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(df_us["Log_Diff1"].dropna(), ax=axes[0])  # Autocorrelation Function
axes[0].set_title("ACF (Autocorrelation Function)")

plot_pacf(df_us["Log_Diff1"].dropna(), ax=axes[1])  # Partial Autocorrelation Function
axes[1].set_title("PACF (Partial Autocorrelation Function)")

plt.show()
####Grid search

###Anomaly detection using confidence bands
###Get confidence bands for 2023 forecast

forecast_2023 = model.get_forecast(steps=12)
conf_int_2023 = forecast_2023.conf_int(alpha=0.10)  # 95% CI

forecast_diff = forecast_2023.predicted_mean

last_log_value = df_us.loc["2022-12-01", "Log_Shoplifting"]
forecast_log = last_log_value + forecast_diff.cumsum()

forecast_actual = np.exp(forecast_log) - 1

conf_int_2023["upper"] = last_log_value + conf_int_2023.iloc[:, 1].cumsum()
conf_int_2023["lower"] = last_log_value + conf_int_2023.iloc[:, 0].cumsum()

conf_int_2023["upper"] = np.exp(conf_int_2023["upper"]) - 1
conf_int_2023["lower"] = np.exp(conf_int_2023["lower"]) - 1

growth_cap=1.5
conf_int_2023["upper"]=pd.Series(conf_int_2023["upper"]).pct_change().clip(upper=growth_cap)
#forecast_dates = pd.date_range(start="2023-01-01", periods=12, freq="MS")
forecast_df = pd.DataFrame({
    "Forecasted Cases": forecast_actual,
    "Lower Bound": conf_int_2023["lower"],
    "Upper Bound": conf_int_2023["upper"]
}, index=forecast_dates)


test_df = test.to_frame().rename(columns={"Shoplifting_Cases": "Actual Cases"})
test_df = test_df.join(forecast_df)


#test_df["Anomaly"] = (test_df["Actual Cases"] > test_df["Upper Bound"]).astype(int)
test_df["Anomaly"] = (test_df["Actual Cases"] > test_df["Upper Bound"] * 0.95).astype(int)

print("\nPredictions in Original Scale:")
print(test_df[["Actual Cases", "Forecasted Cases", "Upper Bound"]].head())

holiday_months = [1, 7, 10, 11, 12]
test_df_holiday = test_df[test_df.index.month.isin(holiday_months)]

# Plot anomalies
plt.figure(figsize=(12, 6))

# Actual vs Predicted
plt.plot(test_df_holiday.index, test_df_holiday["Actual Cases"], label="Actual Cases (Holiday Months)", color="black", marker="o")
plt.plot(test_df_holiday.index, test_df_holiday["Forecasted Cases"], label="Predicted Cases (Holiday Months)", color="red", linestyle="dashed", marker="s")

# Confidence Bands
plt.fill_between(test_df_holiday.index, test_df_holiday["Lower Bound"], test_df_holiday["Upper Bound"],
                 color='blue', alpha=0.2, label="95% Confidence Interval")


anomaly_points = test_df_holiday[test_df_holiday["Anomaly"] == 1]
plt.scatter(anomaly_points.index, anomaly_points["Actual Cases"], color='red', label="Anomalies (High Crime)", zorder=3)

plt.title("Anomaly Detection: Holiday Month Shoplifting Cases (2023)")
plt.xlabel("Date")
plt.ylabel("Number of Cases")
plt.legend()
plt.grid()
plt.show()

###Seveerity detection with z_scores

test_df["Residual"] = test_df["Actual Cases"] - test_df["Forecasted Cases"]

test_df["Z-Score"] = zscore(test_df["Residual"])

test_df_holiday = test_df[test_df.index.month.isin([1, 7, 10, 11, 12])]

anomalies = test_df_holiday[test_df_holiday["Anomaly"] == 1][["Actual Cases", "Forecasted Cases", "Residual", "Z-Score"]]
print("\n Holiday Month Anomalies with Z-Scores:")
print(anomalies)

plt.figure(figsize=(10, 5))
plt.scatter(anomalies.index, anomalies["Z-Score"], color="red", marker="o", s=100, label="Anomaly Severity (Z-Score)")
plt.axhline(y=2, color="black", linestyle="dashed", label="Z = 2 Threshold")
plt.axhline(y=3, color="black", linestyle="dotted", label="Z = 3 Threshold")

plt.title("Anomaly Severity (Z-Scores) for Holiday Months (2023)")
plt.xlabel("Date")
plt.ylabel("Z-Score")
plt.legend()
plt.grid()
plt.show()




