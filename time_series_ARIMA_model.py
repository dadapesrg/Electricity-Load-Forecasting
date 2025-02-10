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
import pmdarima as pm
from pmdarima.arima import auto_arima, StepwiseContext
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
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

# Plot the energy demand data
# Resample to average values for daily data
daily_data = data.resample('D').mean()

# Visualise acf and pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.figure(figsize=(12, 6))
plt.plot(daily_data)
plot_acf(daily_data, lags=40)
plot_pacf(daily_data, lags=40)
plt.show()

# Plot the energy demand data
plt.figure(figsize=(12, 6))
plt.plot(daily_data, label="Daily Energy Demand")
plt.title("UK Electricity Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
#plt.savefig('plots/uk_electricity_demand_daily.png')
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
    
def evaluate_models(dataset, max_p=None, max_d=None, max_q=None, 
					seasonal_p=None, is_stationary=False, stepwise=False):

	# Use StepwiseContext
	if is_stationary:
		seasonal_p = 0				
	with StepwiseContext():
		best_model = auto_arima(
			dataset,
			start_p=0, max_p=max_p,   # AR terms
			start_q=0, max_q=max_q,   # MA terms
			start_d=0, max_d=max_d,    # Auto-detect differencing
			# simple_differencing=True,  # Use simple differencing
			seasonal=False,       # Seasonal ARIMA
       		m=seasonal_p,           # Seasonal period, energy demand has weekly seasonality
			stepwise=stepwise,        # Stepwise search 
			suppress_warnings=True,
			error_action="ignore",
			cache_size=1,
			trace=True
   		)
		return best_model
		
# walk-forward validation
def fit_and_evaluate_arima_model(X_train, X_test, arima_order):
	history = [x for x in X_train]
	predictions = list()
	for t in range(len(X_test)):
		model = ARIMA(history, order= arima_order) 
		model_fit = model.fit()
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = X_test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
	return model_fit, predictions

# Perform ADF test
is_stationary = adf_test(daily_data)

#Evaluate arima model to determine the order
best_model = evaluate_models(daily_data, max_p=10, max_d=5, max_q=10, 
							 seasonal_p=7, is_stationary=is_stationary, stepwise=True)

# Summary of best ARIMA model
print(best_model.summary())

#Split the dataset into train
X = daily_data.values
size = int(len(X) * 0.90)
X_train, X_test = X[0:size], X[size:len(X)]

# Extract best parameters
# Print the parameters of the best model
print("The order of best model is:", best_model.order)
model_fit, predictions = fit_and_evaluate_arima_model(X_train, X_test,  best_model.order)

# evaluate forecasts
rmse = sqrt(mean_squared_error(X_test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot forecasts against actual outcomes
plt.figure(figsize=(12, 6))
plt.plot(X_test, label="Observed", linestyle='--', color='blue')
#plt.plot(pd.date_range(daily_data.index[-1], periods=len(X_test)+1, freq='D')[1:], predictions)
plt.plot(predictions, label="Forecast", color='red')
plt.title("UK Electricity Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
plt.show()

# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())

# Print model summary
print(model_fit.summary())

# Forecast next 30 days demand
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Plot Forecast
plt.figure(figsize=(12, 6))
plt.plot(daily_data[4700:], label="Observed", linestyle='--', color='blue')
plt.plot(pd.date_range(daily_data.index[-1], periods=forecast_steps+1, freq='D')[1:], forecast, label="Forecast", color='red')
plt.title("UK Electricity Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
#plt.savefig('plots/uk_electricity_demand_forecast.png')
plt.show()
