# Electricity Load Forecasting Project
This electricity load forecasting project aims at developing machine learning models using different algorithms such as LSTM, CNN, and classical ML algorithms. The main objective is to develop, train and test different AI models for forecasting electricity load demand.
The following models have been added to the project:\
    a. Convolution Neural Network (CNN) model \
    b. Long Short-Term Memory (LSTM) model \     
    c. Gradient Boosting model \ 
    d. Random Forest model \
    e. Extreme Gradient Boost model with hyperparameters tuning \ 
    f. Gradient Boosting model \
    g. Random Forest model \
    h. Ada Boost model \
    i. Linear Regression model \

The trained models are deployed to the cloud using Google and AWS Cloud infrastructures. 
The project code directory consists of the following:
1. data: Store the data for training the models. The data used for the project was obtained from https://www.kaggle.com/
2. results: Store the models after the training and validation
3. app.py: Flask APP for serving the models in deployment. It uses the stored AI models to serve the request and make predictions.
4. electricity_load_demand_forecasting.py: The module for preprocessing the data, training and testing the models using the datasets.
5. machine_learning_model_comparison.py: Module for comparing different classical machine learning algorithms.
6. Dockerfile: For building the image for the Docker container.
7. requirements.txt: Contains the libraries neccessary to build the image for the container
8. plots: For the generated models' comaprison plots 


