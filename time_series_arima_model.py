# We will use ARIMA model to forecast the electricity demand in the UK. 
# ARIMA is a time series forecasting model that uses past data to predict future values.  
# The model has three main parameters: p, d, and q.
# p: The number of lag observations included in the model (lag order).
# d: The number of times that the raw observations are differenced (degree of differencing).
# q: The size of the moving average window (order of moving average).
# We will use the auto_arima function from the pmdarima library to automatically find the best parameters for the ARIMA model.
# We will then fit the ARIMA model to the data and forecast the electricity demand for the next 30 days.
# Finally, we will plot the forecasted demand along with the observed data to visualize the results.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings("ignore")

# Load dataset and combine all years data into one dataset
year = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
df = pd.read_csv('data/UK_Load_demand_data/demanddata_2011.csv', parse_dates=['SETTLEMENT_DATE'], index_col='SETTLEMENT_DATE')

#Use only 2021 to 2025 data to reduce the size of the dataset for ARIMA model
#df = pd.read_csv('data/UK_Load_demand_data/demanddata_2024.csv', parse_dates=['SETTLEMENT_DATE'], index_col='SETTLEMENT_DATE')
#year = [2025]
for i in range(len(year)):
    data_path = f'data/UK_Load_demand_data/demanddata_{year[i]}.csv'    
    df_year = pd.read_csv(data_path, parse_dates=['SETTLEMENT_DATE'], index_col='SETTLEMENT_DATE') 
    df = pd.concat([df, df_year], ignore_index=False)

# Save the combined dataset
df.to_csv('data/UK_Load_demand_data/demanddata_2011_2025.csv')

#Exploratory data analysis
# Check the first few rows of the data
print(df.head())

# Check the last few rows of the data
print(df.tail())

# Check the data types of the columns
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Check the summary statistics of the data
print(df.describe())   

# Select the demand column for analysis
data = df['ND']  # ND is the column for demand

"""
# Resample to average values for hourly, daily, weekly and monthly data
daily_data = data.resample('D').mean()
weekly_data = data.resample('W').mean() 
monthly_data = data.resample('M').mean()    

# Plot the energy demand data
plt.figure(figsize=(12, 6))
plt.plot(daily_data, label="Daily Energy Demand")
plt.title("UK Electricity Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
plt.show()

# Plot the energy demand data
plt.figure(figsize=(12, 6))
plt.plot(weekly_data, label="Weekly Energy Demand")
plt.title("UK Electricity Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
plt.show()
"""
# Plot the energy demand data
plt.figure(figsize=(12, 6))
plt.plot(monthly_data, label="Monthly Energy Demand")
plt.title("UK Electricity Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
plt.show()

# Perform Augmented Dickey-Fuller (ADF)test to check for stationarity
def adf_test(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is NOT stationary, differencing is required.")

# Perform ADF test
adf_test(data)

# Auto ARIMA to find best (p, d, q)
auto_model = auto_arima(data, seasonal=True, m=7, trace=True, suppress_warnings=True)
print(auto_model.summary())

# Extract best parameters
p, d, q = auto_model.order

# Fit ARIMA model
model = ARIMA(data, order=(p, d, q), seasonal_order=(p, d, q, 7))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Forecast next 30 days demand
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Plot Forecast
plt.figure(figsize=(12, 6))
plt.plot(data, label="Observed")
plt.plot(pd.date_range(data.index[-1], periods=forecast_steps+1, freq='D')[1:], forecast, label="Forecast", color='red')
plt.title("UK Electricity Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
plt.show()


