#import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error

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

def invert_transform(data, shape, column_index, scaler):
    dummy_array = np.zeros((len(data), shape))    
    dummy_array[:, column_index] = data    
    return scaler.inverse_transform(dummy_array)[:, column_index]

# Build and train the random forest model
def build_train_random_forest_model(X_train, y_train, n_estimators=100, min_samples_split=2, max_depth=15):
    from sklearn.ensemble import RandomForestRegressor    
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)
    rf.fit(X_train, y_train)
    return rf

# Build and train the decision tree model
def build_train_decision_tree_model(X_train_ml, y_train_ml, max_depth=15):
    from sklearn.tree import DecisionTreeRegressor
    dt = DecisionTreeRegressor(max_depth=max_depth)
    dt.fit(X_train_ml, y_train_ml)
    return dt

# Build and train the gradient boosting model   
def build_train_gradient_boosting_model(X_train_ml, y_train_ml, rando_state=42):
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(random_state=rando_state)
    gb.fit(X_train_ml, y_train_ml)
    return gb

# Build and train the xgboost model
def build_train_xgboost_model(X_train, y_train, n_estimators=100, max_depth=7, learning_rate=0.1):
    import xgboost as xgb
    params = {'objective': 'reg:squarederror', 'max_depth': max_depth, 'learning_rate': learning_rate, 'n_estimators': n_estimators}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

# Build and train the AdaBoost model
def build_train_ada_boost_model(X_train, y_train, n_estimators=200, learning_rate=0.1):
    from sklearn.ensemble import AdaBoostRegressor
    ada = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    ada.fit(X_train, y_train)
    return ada

# Build and train the linear regression model
def build_train_linear_regression_model(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr
  
table_column_index = 3  # Index of the 'close' price column as the target

# Train the models
models = {
    'RF': build_train_random_forest_model(X_train, y_train), 
    "GB": build_train_gradient_boosting_model(X_train, y_train), 
    'XGB': build_train_xgboost_model(X_train, y_train), 
    'DT': build_train_decision_tree_model(X_train, y_train),
    'ADA': build_train_ada_boost_model(X_train, y_train),
    'LR': build_train_linear_regression_model(X_train, y_train) # Linear Regression
}

# Evaluate the models
rmse_scores = dict()
predictions = dict()
results = {}
for name, model in models.items():
        train_r2 = model.score(X_train, y_train)
        y_pred = model.predict(X_test)                
        predictions[name] = y_pred
        # calculating evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        rmse = float("{:.4f}".format(rmse))
        rmse_scores[name] = rmse           
        mae = mean_absolute_error(y_test, y_pred)
       
        # store the metrics    
        results[name] = {"Training R²": train_r2, " Testing R²": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}     

# Print the results
print("Summary of Models' Metrics")
results_df = pd.DataFrame(results)  # convert the results to a DataFrame for better readability
results_df_transposed = results_df.T # Transpose the DataFrame
print(results_df_transposed)

# Visualize the model metrics
def visualize_model_metrics(df, title="Model Evalation", xlabel="X-axis", ylabel="Y-axis"):
    df.plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Visualize the model metrics
visualize_model_metrics(results_df, title="Models' Metrics Comparison", xlabel="Evaluation Metrics", ylabel="Value")

# Visualize the RMSE scores
def add_plot_labels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

plt.bar(list(rmse_scores.keys()), list(rmse_scores.values()), color ='red')
add_plot_labels(list(rmse_scores.keys()), list(rmse_scores.values()))
plt.xlabel('') 
plt.ylabel('RMSE') 
plt.title('Models') 
plt.show()

# Visualize the predictions
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True,figsize=(22,12))
fig.suptitle('Electricity Load Demand Predictions')
fig.supxlabel('Time')
fig.supylabel('Electricity Load Demand')
ax1, ax2 = axes[0]
ax3, ax4 = axes[1]
ax5, ax6 = axes[2]

# Invert the transformation 
y_test = invert_transform(y_test, len(df.columns), table_column_index, scaler)
def add_plot(x,y):
    for i in range(len(x)):
        y[i] = invert_transform(y[i], len(df.columns), table_column_index, scaler)
        if x[i] == 'RF':
            ax1.plot(y[i], label=x[i])
        elif x[i] == 'XGB':
            ax2.plot(y[i], label=x[i])               
        elif x[i] == 'GB':
            ax3.plot(y[i], label=x[i])
        elif x[i] == 'DT':
            ax4.plot(y[i], label=x[i])
        elif x[i] == 'ADA':
            ax5.plot(y[i], label=x[i])
        else:
            ax6.plot(y[i], label=x[i])

    ax1.plot(y_test, label='Actual Electricity Load Demand')    
    ax2.plot(y_test, label='Actual Electricity Load Demand')   
    ax3.plot(y_test, label='Actual Electricity Load Demand')      
    ax4.plot(y_test, label='Actual Electricity Load Demand')
    ax5.plot(y_test, label='Actual Electricity Load Demand')
    ax6.plot(y_test, label='Actual Electricity Load Demand')
    
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    ax4.legend(loc='best')  
    ax5.legend(loc='best')
    ax6.legend(loc='best')

    ax1.set_title('Random Forest Predictions')
    ax2.set_title('Extreme Gradient Boosting Predictions')
    ax3.set_title('Gradient Boosting Predictions')
    ax4.set_title('Decision Tree Predictions')  
    ax5.set_title('Ada Boost Predictions')  
    ax6.set_title('Linear Regression Predictions')
   
add_plot(list(predictions.keys()), list(predictions.values()))  

plt.tight_layout()
plt.show()
