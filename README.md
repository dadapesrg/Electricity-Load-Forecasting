# Electricity Load Forecasting Project
This electricity load forecasting project aims at developing machine learning models using different algorithms such as LSTM, CNN, classical ML algorithms, and time series forecasting algorithms. The main objective is to develop, train and test different AI models for forecasting electricity load demand.
The following models have been added to the project:\
    a. CNN: Convolution Neural Network model \
    b. LSTM: Long Short-Term Memory model \
    c. GB: Gradient Boosting model \
    d. RF: Random Forest model \
    e. XGB: Extreme Gradient Boost model with hyperparameters tuning \
    f. DT: Decision Tree model \   
    g. RF: Random Forest model \
    h. ADA: Ada Boost model \
    i. LR: Linear Regression model \
    j. Time series ARIMA model using UK electricity load demand.
   
The trained models are deployed to the cloud using Google and AWS Cloud infrastructures. 
The project code directory consists of the following:
1. automodelselector: Package/module developed to automatically select machine learning model for a given dataset. 
2. data: Store the data for training the models. The data used for the project was obtained from https://www.kaggle.com/ and UK electricity load demand was obtained from https://www.neso.energy/data-portal
3. results: Store the models after the training and validation
4. app.py: Flask APP for serving the models in deployment. It uses the stored AI models to serve the request and make predictions.
5. electricity_load_demand_forecasting.py: The module for preprocessing the data, training and testing the models using the datasets.
6. machine_learning_model_comparison.py: Module for comparing different classical machine learning algorithms using auto model selector package/module.
7. Dockerfile: For building the image for the Docker container.
8. requirements.txt: Contains the libraries neccessary to build the image for the container
9. plots: For the generated models' comaprison plots 
10. time_series_ARIMA_model.py: Module for time series model developed for UK electricity load demand using arima.
11. time_series_arima_lstm_cnn_model.py: Module for time series model developed for UK electricity load demand using arima, lstm, cnn. 


