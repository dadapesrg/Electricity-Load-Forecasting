#import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt


# Load the stock data
df1 = pd.read_csv('data/Continuous_dataset.csv') 

# Print the shape of the dataset
print(df1.info())
# Print the first 5 rows
print(df1.head())

# Print the last 5 rows
print(df1.tail())

# Use the specified features of the continous dataset
features = ['datetime','nat_demand', 'T2M_toc','QV2M_toc',	'TQL_toc',	'W2M_toc', 'T2M_san', 'QV2M_san',\
'TQL_san',	'W2M_san',	'T2M_dav',	'QV2M_dav',	'TQL_dav',	'W2M_dav']

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

# Save the test data for later use
test_df.to_csv('data/test_data.csv')
test_df.to_json('data/test_data.json')

# Convert back to DataFrame for easier manipulation
train_scaled = pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index)
test_scaled = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)

# Create sequences from the scaled training data
timesteps = 5
target_column = 'nat_demand'  # This is the target variable to predict

X_train = train_scaled #.drop(columns=[target_column])
y_train = train_scaled[target_column]   

X_test = test_scaled #.drop(columns=[target_column])
y_test = test_scaled[target_column] 

print("Training Data Shape (X_train):", X_train.shape)
print("Training Labels Shape (y_train):", y_train.shape)

sgdr = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000)
gb = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=5) 
rf = RandomForestRegressor(n_estimators=40, random_state=42)
lr = LinearRegression()

model = lr
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model    
mse = mean_squared_error(y_test, y_pred)
R2_Score = round(r2_score(y_pred, y_test) * 100, 2)
print(f'MSE and R2 Score for Linear Regression Model: {mse:.2f}, {R2_Score:.2f}')   

#Function for inverse transform the predicted data
def invert_transform(data, shape, column_index, scaler):
    dummy_array = np.zeros((len(data), shape))    
    dummy_array[:, column_index] = data    
    return scaler.inverse_transform(dummy_array)[:, column_index]

# Create a DataFrame to hold predictions and actual values
test_pred_df = pd.DataFrame({
    'Actual': invert_transform(y_test, test_scaled.shape[1], 0, scaler), # scaler.inverse_transform(y_test),  # Inverse scale the nat_demand
    'Predicted': invert_transform(y_pred, test_scaled.shape[1], 0, scaler) #inversed_predictions #scaler.inverse_transform(np.concatenate([y_pred, np.zeros_like(y_pred)], axis=1))[:, 0]
})

# Plot the results
plt.figure(figsize=(14,5))
plt.plot(test_pred_df['Actual'], color='blue', label='Actual Electricity Load Demand')
plt.plot(test_pred_df['Predicted'], color='red', label='Predicted Electricity Load Demand') #plt.plot(y_pred, color='red', label='Predicted Stock Price')
plt.title('Electricity Load Demand Prediction')
plt.xlabel('Time')
plt.ylabel('Electricity Load Demand')
plt.legend()
plt.show()

# Evaluate the model on the testing data
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")

# Save the models to a file
import pickle
with open('results/lr_model.pkl', 'wb') as f:
    pickle.dump(model, f)