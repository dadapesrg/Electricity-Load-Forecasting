#import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
import keras.optimizers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Load the stock data
df1 = pd.read_csv('data/Continuous_dataset.csv') 

#Use the specified features of the continous dataset
features = ['datetime','nat_demand', 'T2M_toc','QV2M_toc',	'TQL_toc',	'W2M_toc', 'T2M_san', 'QV2M_san',\
'TQL_san',	'W2M_san',	'T2M_dav',	'QV2M_dav',	'TQL_dav',	'W2M_dav']

#data = df1[features].values
df = df1[features]
print(df.head)

# Set 'Date' as the index
df.set_index('datetime', inplace=True)

print("Sample Data:\n", df.head())

# Define the split point
split_fraction = 0.8
split_index = int(len(df) * split_fraction)

# Split the data
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print("Training Data:\n", train_df.tail())
print("\nTesting Data:\n", test_df.head())

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data
train_scaled = scaler.fit_transform(train_df)

# Apply the same transformation to the testing data
test_scaled = scaler.transform(test_df)

# Convert back to DataFrame for easier manipulation
train_scaled = pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index)
test_scaled = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)

#Create model sequence
def create_sequences(data, target_col, timesteps=5):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data.iloc[i:i+timesteps].values)
        y.append(data.iloc[i+timesteps][target_col])
    return np.array(X), np.array(y)

# Create sequences from the scaled training data
timesteps = 5
target_column = 'nat_demand'  # Choose the target variable to predict

X_train, y_train = create_sequences(train_scaled, target_column, timesteps)
X_test, y_test = create_sequences(test_scaled, target_column, timesteps)

print("Training Data Shape (X_train):", X_train.shape)
print("Training Labels Shape (y_train):", y_train.shape)

#number of epoch and batch size
epochs = 200
batch = 24

model_lstm = Sequential()
# Add LSTM layers with Dropout regularization
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dropout(0.2))
# Add Dense layer
model_lstm.add(Dense(units=25))
model_lstm.add(Dense(units=1))  # Output layer, predicting the 'close' price

# Compile the model

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
lstm_history = model_lstm.fit(X_train, y_train, batch_size = batch, epochs=epochs, verbose=2) 

# Print model summary
model_lstm.summary()

# Make predictions on the test set
lstm_y_pred = model_lstm.predict(X_test)

# Since we scaled all features but are predicting only one, we'll
# need to inverse transform the predictions using the appropriate feature column.
def invert_transform(data, shape, column_index, scaler):
    dummy_array = np.zeros((len(data), shape))
    dummy_array[:, column_index] = data.flatten()
    return scaler.inverse_transform(dummy_array)[:, column_index]

# Create a DataFrame to hold predictions and actual values
lstm_test_pred_df = pd.DataFrame({
    'Actual': invert_transform(y_test, X_train.shape[2], 0, scaler), # Inverse scale the nat_demand
    'Predicted': invert_transform(lstm_y_pred, X_train.shape[2], 0, scaler) #inverse sclae the predictions
})

lr = 0.0003
adam = keras.optimizers.Adam(lr)

#develope convolusion CNN model for Time Series Forecasting
adam = keras.optimizers.Adam(lr)
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(1))
model_cnn.compile(loss='mse', optimizer=adam)

model_cnn.summary()

cnn_history = model_cnn.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=2)

# Make predictions on the test set
cnn_y_pred = model_cnn.predict(X_test)

# Create a DataFrame to hold predictions and actual values
cnn_test_pred_df = pd.DataFrame({
    'Actual': invert_transform(y_test, X_train.shape[2], 0, scaler), # scaler.inverse_transform(y_test),  # Inverse scale the nat_demand
    'Predicted': invert_transform(cnn_y_pred, X_train.shape[2], 0, scaler) #inversed_predictions #scaler.inverse_transform(np.concatenate([y_pred, np.zeros_like(y_pred)], axis=1))[:, 0]
})

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(lstm_test_pred_df['Actual'], label='Actual Demand')
plt.plot(lstm_test_pred_df['Predicted'], label='Predicted Demand', linestyle='--')
plt.title('Energy Demand Prediction using LSTM Model - Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Energy demand)')
plt.legend()
#plt.show() 
# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(cnn_test_pred_df['Actual'], label='Actual Demand')
plt.plot(cnn_test_pred_df['Predicted'], label='Predicted Demand', linestyle='--')
plt.title('Energy Demand Prediction using CNN model - Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Energy demand)')
plt.legend()
#plt.show()

plt.figure(figsize=(10, 6))
plt.plot(lstm_history.history['loss'], label='Train loss')
plt.plot(cnn_history.history['loss'], label='Train loss')
plt.legend(loc='best')
plt.title('Comparison of Train Loss CNN and LSTM model')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()

# Evaluate the models
from sklearn.metrics import r2_score
R2_Score_dtr_lstm = round(r2_score(lstm_y_pred, y_test) * 100, 2)
print("R2 Score for LSTM : ", R2_Score_dtr_lstm,"%")

R2_Score_dtr_cnn = round(r2_score(cnn_y_pred, y_test) * 100, 2)
print("R2 Score for CNN : ", R2_Score_dtr_cnn,"%")

rmse_lstm = np.sqrt(np.mean(lstm_y_pred - y_test) ** 2)
print("Root Mean Squared Error for LSTM:", rmse_lstm)
rmse_cnn = np.sqrt(np.mean(cnn_y_pred - y_test) ** 2)
print("Root Mean Squared Error for CNN:", rmse_cnn)