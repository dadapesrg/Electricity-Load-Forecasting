# Electricity-Load-Forecasting
This electricity load forecasting project aims at developing machine learning models using different algorithms such as LSTM, CNN, RNN, etc. The main objective is to develop, train and test different AI models for forecasting electricity load demand.
The trained models are deployed to the cloud using Google and AWS Cloud infrastructures. 

The project code directory consists of the following components:
1. Data: Store the data for training the models
2. Results: Store the models after the training and validation
3. app.py: FlaskAP for serving the models in deployment. It uses the stored AI models to serve the request and make predictions.
4. electricity_load_demand_forecasting.py: This is the function that where the models are trained using the datasets.
5. Dockerfile: For building the image for the Docker container.
6. requirements.txt: Contain the libraries neccessary to build the image for the container


