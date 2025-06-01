import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load all state data
folder_path = "/Users/patalee/Desktop/GT_Analytics/Spring_2025/ISYE_6740/project/ISYE6740_Crime-initial_data_cleaning/data/Burglary"
file_list = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

crime_data = {}
for file in file_list:
    state = file.replace("Burglary Reported by Population_", "").replace(".csv", "")
    try:
        df = pd.read_csv(os.path.join(folder_path, file))
        df = df.iloc[[0]]  # Keep only the first row (state-level data)
        df["State"] = state
        crime_data[state] = df
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Combine all state data
df_all_states = pd.concat(crime_data.values(), ignore_index=True)

# Clean and reshape data
df_cleaned = df_all_states.dropna(axis=1, how="any")
df_long = df_cleaned.melt(id_vars=["State"], var_name="Date", value_name="Crime_Count")
df_long["Date"] = pd.to_datetime(df_long["Date"], format="%m-%Y", errors="coerce")
df_long = df_long.sort_values(["State", "Date"]).reset_index(drop=True)
df_long["Crime_Count"] = pd.to_numeric(df_long["Crime_Count"], errors="coerce")
df_long = df_long.dropna(subset=["Crime_Count"])

# Aggregate national crime count per month
df_national = df_long.groupby("Date")["Crime_Count"].sum().reset_index()
df_national.set_index("Date", inplace=True)
df_national.index.freq = "MS"
# Define holiday months (January, July, October, November, December)
holiday_months = [1, 7, 10, 11, 12]


df_holiday = df_national[df_national.index.month.isin(holiday_months)].copy()

###  Stationarity
adf_test = adfuller(df_holiday["Crime_Count"])
if adf_test[1] > 0.05:
    df_holiday["Crime_Count"] = df_holiday["Crime_Count"].diff().dropna()
    df_holiday.dropna(inplace=True)
    print("Applied differencing to make data stationary.")
else:
    print("Data is already stationary.")

# SARIMA Modeling for 2023 and 2024 Holiday Months**
sarima_order = (2, 1, 2)
seasonal_order = (2, 1, 1, 12)

# Train SARIMA model
sarima_model = sm.tsa.statespace.SARIMAX(df_holiday["Crime_Count"],
                                         order=sarima_order,
                                         seasonal_order=seasonal_order,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False).fit(disp=False)

#Forecast 2023 Holiday Months & Compare with Actual Data
forecast_dates_2023 = pd.date_range(start="2023-01-01", periods=12, freq='MS')
forecast_2023 = sarima_model.forecast(steps=12)
forecast_2023.index = forecast_dates_2023

if adf_test[1] > 0.05:
    last_observed_value = df_holiday["Crime_Count"].iloc[-1]  # Get last actual value before differencing
    forecast_2023 = forecast_2023.cumsum() + last_observed_value
# Filter only holiday months
forecast_2023 = forecast_2023[forecast_2023.index.month.isin(holiday_months)]
actual_2023 = df_holiday[df_holiday.index.year == 2023]

# Merge forecasts and actuals
comparison_2023 = actual_2023.merge(forecast_2023.rename("Forecast"), left_index=True, right_index=True)
comparison_2023.dropna(inplace=True)


mae = mean_absolute_error(comparison_2023["Crime_Count"], comparison_2023["Forecast"])
mse = mean_squared_error(comparison_2023["Crime_Count"], comparison_2023["Forecast"])
rmse = np.sqrt(mse)


print("\n Model Performance on 2023 Holiday Months:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


plt.figure(figsize=(12, 6))

# Plot actual vs forecasted values for 2023
plt.plot(comparison_2023.index, comparison_2023["Crime_Count"], marker='o', linestyle='-', label="Actual 2023")
plt.plot(comparison_2023.index, comparison_2023["Forecast"], marker='s', linestyle='--', label="Forecast 2023")


plt.xlabel("Date")
plt.ylabel("Pocket Picking Incidents")
plt.title("Comparison of Actual vs Forecasted Pocket Picking (2023 & 2024 Holiday Months)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
