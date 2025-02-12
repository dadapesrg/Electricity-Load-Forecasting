#import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error
from automodelselector.machine_learning_model_selector import MLRegressionModelSelector

# AutoModelSelector

# Load the stock data
df = pd.read_csv('data/continuous_dataset.csv') 

# Print the shape of the dataset
print(df.info())
# Print the first 5 rows
print(df.head())

# Print the last 5 rows
print(df.tail())

# Use the specified features of the continous dataset
features = ['datetime','nat_demand', 'T2M_toc','QV2M_toc',	'TQL_toc',	'W2M_toc', 'T2M_san', 'QV2M_san',\
'TQL_san',	'W2M_san',	'T2M_dav',	'QV2M_dav',	'TQL_dav',	'W2M_dav']

data_df = df[features]
print(data_df.head)

# Set 'Date' as the index
data_df.set_index('datetime', inplace=True)

print("Sample Data:\n", data_df.head())


from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Scale the data
scaled_data = scaler.fit_transform(data_df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=data_df.columns, index=df.index)
print(df_scaled.head())
target_col = "nat_demand"

# Create lagged features
def create_lagged_features(df, target_col, n_lags=3):
    df_lagged = df.copy()
    for col in df.columns:
        for lag in range(1, n_lags + 1):
            df_lagged[f"{col}_lag{lag}"] = df_lagged[col].shift(lag)
    df_lagged.dropna(inplace=True)  # Drop rows with NaN due to shifting
    return df_lagged

# Create lagged features with 3 time steps
df_lagged = create_lagged_features(data_df, target_col, 5)
#df_lagged = create_lagged_features(df_scaled, target_col, 5)
print(df_lagged.head())

# Split into features (X) and target (y)
X = df_lagged.drop(columns=target_col)  # All features except the target
y = df_lagged[target_col]  # Target variable
print(X.head())
print(y.head())

# Define the model parameters for each regression model
model_params = {
     'ridge': {'alpha': 1.0},
     'decision_tree': {'max_depth': 15},
     'random_forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
     'linear_regression': {},     
     'gradient_boosting': {'n_estimators': 100, 'max_depth': 15, 'learning_rate': 0.1},
     'xgboost': {'n_estimators': 100, 'max_depth': 15, 'learning_rate': 0.1},
     "ada_boost": {'n_estimators': 100, 'learning_rate': 0.1},
     'svr': {'C': 1.0, 'kernel': 'rbf'},
     'knn': {'n_neighbors': 5}      
}

# Initialize the MLRegressionModelHandler
model_selector = MLRegressionModelSelector(X, y, model_params=model_params, random_state=0, scale_target= False, shuffle=False)
model_selector.train_and_evaluate_all()
best_model, best_mse, best_r2 = model_selector.get_best_model()
print(f"Best Model: {best_model}, MSE: {best_mse:.4f}, R²: {best_r2:.4f}")
model_selector.plot_predictions()
model_selector.plot_predictions_vs_time(time_column= -1)
print(df_lagged)

model_selector.plot_model_performance()

