# We will use LSTM and CNN model to forecast the electricity load demand in the UK. 
# LSTM is a recurrent neural network that uses past data to predict future values.  
# CNN is a convolutional neural network that uses past data to predict future values.  
# The model has two main parameters: time_step and batch.
# time_step: The number of time steps to use as input to the model (lag order).
# batch: The batch size to use when training the model.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
from tensorflow import keras

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
# Resample to average values for hourly, daily, weekly and monthly data
daily_data = data.resample('D').mean()

# Plot the energy demand data
plt.figure(figsize=(12, 6))
plt.plot(daily_data, label="Daily Energy Demand")
plt.title("UK Electricity Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
#plt.savefig('plots/uk_electricity_demand_daily.png')
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

# Split into train & test sets (80% train, 20% test)
train_size = int(len(data_scaled) * 0.85)
train, test = data_scaled[:train_size], data_scaled[train_size:]

def create_sequences(dataset, time_step=30):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step])
        Y.append(dataset[i + time_step])
    return np.array(X), np.array(Y)

time_step = 30  # Use past 30 days to predict next day
X_train, Y_train = create_sequences(train, time_step)
X_test, Y_test = create_sequences(test, time_step)

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

epochs = 50
batch = 16

# Train the model
lstm_history= lstm_model.fit(X_train, Y_train, batch_size = batch, epochs = epochs, verbose=2)

# Predict
lstm_y_pred = lstm_model.predict(X_test)

# Inverse transform the predictions and actual values to original scales
lstm_pred = scaler.inverse_transform(lstm_y_pred)  # Convert back to original scale

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(Y_test):], scaler.inverse_transform(Y_test.reshape(-1, 1)), label="Actual Demand")
plt.plot(data.index[-len(lstm_pred):], lstm_pred, label="LSTM Forecast", linestyle='--')
plt.title("UK Electricity Demand Forecast (LSTM)")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
plt.show()

#Define CNN model
# Define the learning rate and optimizer
lr = 0.0003

# Define the convolusion neural network (CNN) model for Time Series Forecasting
adam = keras.optimizers.Adam(lr)
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(1))
model_cnn.compile(loss='mse', optimizer=adam)

# Print model summary
model_cnn.summary()

# Train the model
cnn_history = model_cnn.fit(X_train, Y_train, epochs=epochs, batch_size=batch, verbose=2)

# Make predictions on the test set
cnn_y_pred = model_cnn.predict(X_test)

# Inverse transform the predictions and actual values to original scales
cnn_pred = scaler.inverse_transform(cnn_y_pred)  # Convert back to original scale

# Evaluate the models
from sklearn.metrics import r2_score
R2_Score_lstm = round(r2_score(lstm_y_pred, Y_test) * 100, 2)
print("R2 Score for LSTM : ", R2_Score_lstm,"%")
rmse_lstm = np.sqrt(np.mean(lstm_y_pred - Y_test) ** 2)
print("Root Mean Squared Error for LSTM:", rmse_lstm)

R2_Score_cnn = round(r2_score(cnn_y_pred, Y_test) * 100, 2)
print("R2 Score for LSTM : ", R2_Score_cnn,"%")
rmse_lstm = np.sqrt(np.mean(cnn_y_pred - Y_test) ** 2)
print("Root Mean Squared Error for CNN:", rmse_lstm)

# Visualize the predictions
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True,figsize=(22,12))
fig.suptitle('UK Electricity Demand Forecast')
fig.supxlabel('Time')
fig.supylabel('Demand (MW)')
ax1 = axes[0]
ax2 = axes[1]

ax1.plot(data.index[-len(Y_test):], scaler.inverse_transform(Y_test.reshape(-1, 1)), label="Actual Demand")
ax1.plot(data.index[-len(lstm_pred):], lstm_pred, label="LSTM Forecast", linestyle='--')

ax2.plot(data.index[-len(Y_test):], scaler.inverse_transform(Y_test.reshape(-1, 1)), label="Actual Demand")
ax2.plot(data.index[-len(cnn_pred):], cnn_pred, label="CNN Forecast", linestyle='--')

ax1.legend(loc='best')
ax2.legend(loc='best')   
ax1.set_title('LSTM Predictions')
ax2.set_title('CNN Predictions')
   
# Plotting the train loss for both models
plt.figure(figsize=(10, 6))
plt.plot(lstm_history.history['loss'], label='LSTM Train loss')
plt.plot(cnn_history.history['loss'], label='CNN Train loss')
plt.legend(loc='best')
plt.title('Comparison of Train Loss CNN and LSTM model')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()
