# We will use ARIMA model to forecast the electricity load demand in the UK. 
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from pmdarima.arima import auto_arima, StepwiseContext
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error
from pandas.plotting import autocorrelation_plot
from math import sqrt

import warnings
warnings.filterwarnings("ignore")

# Load dataset and combine all years data into one dataset
year = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
df = pd.read_csv('data/UK_Load_demand_data/demanddata_2011.csv', parse_dates=['SETTLEMENT_DATE'], index_col='SETTLEMENT_DATE')
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
df['ND'] = df['ND'].astype('float32')
data = df['ND']  # ND is the column for demand
print(data.head())

# Resample to average values for weekly data to reduce the size of the dataset
data = data.resample('W').mean()
daily_data = data.copy()

# Plot the electricity load demand data
plt.figure(figsize=(12, 6))
plt.plot(data, label="Monthly Electricity Demand")
plt.title("UK Electricity Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
plt.show()

# Plot the autocorrelation
autocorrelation_plot(data)
plt.show()

# Visualise acf and pacf
plot_acf(data)
plot_pacf(data)
plt.show()

# Visualise the seasonal decomposition 
seasonal_p = 52
decomposition=seasonal_decompose(data, model='additive', period=seasonal_p)
decomposition.plot()
plt.show()

# Perform Augmented Dickey-Fuller (ADF)test to check for stationarity
def adf_test(series):
	is_stationary = False
	result = adfuller(series.dropna())
	print(f'ADF Statistic: {result[0]}')
	print(f'p-value: {result[1]}')
	if result[1] <= 0.05:
		is_stationary = True
		print("The series is stationary.")		
	else:		
		print("The series is NOT stationary, differencing is required.")
	return is_stationary 

is_stationary = adf_test(data)

# Perform differencing if data is Not stationary
max_d = 0
if not is_stationary:
	data_diff = data.diff().dropna()
	plot_acf(data_diff, lags=50)
	plot_pacf(data_diff, lags=50)
	plt.show()	
	seasonal_data_diff = data_diff.diff(seasonal_p).dropna()
	plot_acf(seasonal_data_diff, lags=seasonal_p)
	plot_pacf(seasonal_data_diff, lags=seasonal_p)
	plt.show()
	max_d = max_d + 1
	is_stationary = adf_test(data_diff)	

# Split the dataset into train and test set
X = data.values
size = int(len(X) * 0.8)
X_train, X_test = X[0:size], X[size:len(X)]

"""
# Evaluate arima model to determine the order
#auto_model = auto_arima(data,start_p=1,start_q=1, d=max_d, test='adf', m=seasonal_p,D=max_d, seasonal_test='ocsb', stepwise=True, seasonal=True,trace=True)

# Summary of best ARIMA model
print(auto_model.summary())
arima_order = auto_model.order
seasonal_order = auto_model.seasonal_order
"""
#r2 = 0.82
#arima_order = (0,1,1)
#seasonal_order = (0,1,1,seasonal_p)
# r2 = 0.84
# arima_order = (1,1,1)
#seasonal_order = (1,1,1,seasonal_p)	

arima_order = (2,1,2)
seasonal_order = (1,1,1,seasonal_p)

# Fit ARIMA model
model = SARIMAX(X_train, order= arima_order, seasonal_order=seasonal_order) 
model_fit = model.fit()

# Line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

# Density plot of residuals
residuals.plot(kind='kde')
plt.show()

# Summary stats of residuals
print(residuals.describe())

# Print model summary
print(model_fit.summary())

# Forecast solar generation using the test data
forecast_steps = len(X) - size 
forecast = model_fit.forecast(steps=forecast_steps)
print(forecast)

# Plot the results with specified colors
plt.figure(figsize=(14,7))
plt.plot(data.iloc[:size].index, X_train, label='Train', color='#203147')
plt.plot(data.iloc[size:].index, X_test, label='Test', color='#01ef63')
plt.plot(data.iloc[size:].index, forecast, label='Forecast', color='orange')
plt.title("Electricity Load Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Electricity Load Demand (MW)")
plt.legend()
plt.show()

# Evaluate forecasts
rmse = sqrt(mean_squared_error(X_test, forecast))
print('Test RMSE: %.3f' % rmse)

r2 = r2_score(X_test, forecast)
mse = mean_squared_error(X_test, forecast)
rmse = np.sqrt(mse)
rmse = float("{:.4f}".format(rmse))         
mae = mean_absolute_error(X_test, forecast)

print(f'R2: {r2:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}')

