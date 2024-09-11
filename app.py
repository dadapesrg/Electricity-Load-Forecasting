import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
import pickle
#load the lstm model
with open('results/lstm_model.pkl', 'rb') as f:
    lstm_model = pickle.load(f)

#load the cnn model
with open('results/cnn_model.pkl', 'rb') as f:
    cnn_model = pickle.load(f)

#the features used for training the model
features = ['nat_demand', 'T2M_toc','QV2M_toc',	'TQL_toc',	'W2M_toc', 'T2M_san', 'QV2M_san',\
'TQL_san',	'W2M_san',	'T2M_dav',	'QV2M_dav',	'TQL_dav',	'W2M_dav']

# Initialize the scaler
scaler = MinMaxScaler() #feature_range=(0, 1))

 # function to reshape data for LSTM [samples, timesteps, features] prediction as used during the training
def create_sequences(data, target_col, timesteps=5):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data.iloc[i:i+timesteps].values)
        y.append(data.iloc[i+timesteps][target_col])
    return np.array(X), np.array(y)

# function to inverse transform the data
def invert_transform(data, shape, column_index, scaler):
    dummy_array = np.zeros((len(data), shape))
    dummy_array[:, column_index] = data.flatten()
    return scaler.inverse_transform(dummy_array)[:, column_index]

# Prediction function
def make_prediction(input_data):
        
    #load the scaler used for training the model
    with open("results/scaler.pkl", "rb") as infile:
        scaler = pickle.load(infile)
    
    # Scale the data since scaler was used for the training)
    input_data_scaled = scaler.transform(input_data)
    test_data_scaled = pd.DataFrame(input_data_scaled, columns=input_data.columns, index=input_data.index)
   
    # Reshape data for LSTM [samples, timesteps, features] as used during the training 
    timesteps = 5
    target_column = 'nat_demand'  # Choose the target variable to predict

    # Create sequences from the scaled input data
    input_pred_data, y_test = create_sequences(test_data_scaled, target_column, timesteps)
          
    # Make prediction
    lstm_pred = lstm_model.predict(input_pred_data)

    #inverse transform the prediction because of scaling applied during the training
    lstm_prediction = invert_transform(lstm_pred, input_pred_data.shape[2], 0, scaler) #inverse sclae the predictions
       
    return lstm_prediction

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract input data (expecting JSON format)  
    input_data = pd.DataFrame(data, columns=features) #, index="datetime")
   
    # Perform the prediction
    try:
        prediction = make_prediction(input_data)
        
        response = {
            'status': 'success',
            'prediction': prediction.tolist()  # Convert numpy array to list for JSON
        }
    except Exception as e:
        response = {
            'status': 'error',
            'message': str(e)
        }   
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8484)