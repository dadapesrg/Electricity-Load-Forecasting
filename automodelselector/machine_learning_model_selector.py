# Load the required libraries
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define the RegressionHandler class
class MLRegressionModelSelector:
    def __init__(self, X, y, test_size=0.2, random_state=None, model_params=None, shuffle=False, scale_target=False):
        """
        Initialize the MLRegressionModelSelector with data and split it into training and testing sets.
        
        :param X: Features (numpy array or pandas DataFrame)
        :param y: Target values (numpy array or pandas Series)
        :param test_size: Proportion of the dataset to include in the test split
        :param random_state: Seed for random number generation
        :param model_params: Dictionary of parameters for each model (optional)
        :param shuffle: Whether to shuffle the data before splitting
        :param scale_target: Whether to scale the target variable (y) using StandardScaler
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )

        # Convert y_train and y_test to numpy arrays if they are pandas Series
        if hasattr(self.y_train, 'to_numpy'):
            self.y_train = self.y_train.to_numpy()
            self.y_test = self.y_test.to_numpy()
        
        # Scaling for the target variable (if enabled)
        self.scale_target = scale_target
        self.y_scaler = StandardScaler() if scale_target else None
        if self.scale_target:
            self.y_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
                
        # Default regression models with optional parameters
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(),            
            'decision_tree': DecisionTreeRegressor(),
            'random_forest': RandomForestRegressor(),
            'gradient_boosting': GradientBoostingRegressor(),            
            'xgboost': xgb.XGBRegressor(),
            'ada_boost': AdaBoostRegressor(),
            'svr': SVR(),
            'knn': KNeighborsRegressor()
        }
        
        # Update models with user-provided parameters
        if model_params:
            self._update_model_params(model_params)
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_mse = float('inf')  # Initialize with a large value for MSE
        self.best_r2 = -float('inf')  # Initialize with a small value for R²

    def _update_model_params(self, model_params):
        """
        Update the models with user-provided parameters.
        
        :param model_params: Dictionary of parameters for each model
        """
        for model_name, params in model_params.items():
            if model_name in self.models:
                self.models[model_name].set_params(**params)
            else:
                raise ValueError(f"Model '{model_name}' not found in available models.")

    def train_model(self, model_name):
        """
        Train a specific regression model by name.
        
        :param model_name: Name of the model to train (e.g., 'linear_regression', 'random_forest')
        :return: Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in available models.")
        
        model = self.models[model_name]
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('regressor', model)
        ])
        
        pipeline.fit(self.X_train, self.y_train)
        return pipeline

    def evaluate_model(self, model):
        """
        Evaluate a trained regression model on the test set.
        
        :param model: Trained model to evaluate
        :return: Mean Squared Error (MSE) and R² score
        """
        y_pred = model.predict(self.X_test)
        if self.scale_target:
            y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test = self.y_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        else:
            y_test = self.y_test
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def train_and_evaluate_all(self):
        """
        Train and evaluate all available regression models, and keep track of the best one.
        """
        for model_name in self.models:
            print(f"Training and evaluating {model_name}...")
            model = self.train_model(model_name)
            mse, r2 = self.evaluate_model(model)
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
            print("-" * 60)
            
            if mse < self.best_mse:
                self.best_mse = mse
                self.best_r2 = r2
                self.best_model = model

    def get_best_model(self):
        """
        Get the best regression model based on Mean Squared Error (MSE).
        
        :return: Best model, its MSE, and R² score
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        return self.best_model, self.best_mse, self.best_r2

    def predict(self, X=None):
        """
        Make predictions using the best model.
        
        :param X: Input features for prediction (optional, uses test data if None)
        :return: Predicted values (inverse transformed if target was scaled)
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        if X is None:
            X = self.X_test
        y_pred = self.best_model.predict(X)
        if self.scale_target:
            y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred   

    def plot_predictions(self):
        """
        Plot the true vs predicted values for the test data using the best model.
        Inverse transforms the values if the target was scaled.
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        
        y_pred = self.predict(self.X_test)
        
        if self.scale_target:
            y_test = self.y_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        else:
            y_test = self.y_test
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"True vs Predicted Values (Best Model: {self.best_model.named_steps['regressor'].__class__.__name__})")
        plt.show()

    def plot_predictions_vs_time(self, time_column):
        """
        Plot the true and predicted values against time for the test data.
        
        :param time_column: Name or index of the column in X representing time
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        
        y_pred = self.predict(self.X_test)
        
        if self.scale_target:
            y_test = self.y_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        else:
            y_test = self.y_test
        
        # Extract time values from the test data
        if isinstance(time_column, str):
            time_values = self.X_test[time_column]
        else:
            time_values = self.X_test.iloc[:, time_column]
        
        time = pd.date_range(self.X_test.index[-1], periods=len(y_test), freq='D')
        plt.figure(figsize=(12, 6))
        plt.plot(time, y_test, label="True Values", color='blue', alpha=0.7)
        plt.plot(time, y_pred, label="Predicted Values", color='orange', alpha=0.7)
        plt.xlabel("Time")
        plt.ylabel("Target Value")
        plt.title(f"True and Predicted Values Over Time (Best Model: {self.best_model.named_steps['regressor'].__class__.__name__})")
        plt.legend()
        plt.show()

    def save_model(self, filename):
        """
        Save the best model to a file using joblib.
        
        :param filename: Name of the file to save the model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        joblib.dump(self.best_model, filename)  # Save the model to a file
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """
        Load a model from a file using joblib.
        
        :param filename: Name of the file containing the saved model
        """
        self.best_model = joblib.load(filename)
        print(f"Model loaded from {filename}")  

    def plot_model_performance(self):
        """
        Plot the performance of all models based on Mean Squared Error (MSE).
        """
        mse_scores = {}
        for model_name in self.models:
            model = self.train_model(model_name)
            mse, _ = self.evaluate_model(model)
            mse_scores[model_name] = mse

        plt.figure(figsize=(10, 6))
        plt.bar(mse_scores.keys(), mse_scores.values(), color='skyblue')
        plt.xlabel("Model")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("Model Performance Comparison")
        plt.savefig('plots/model_performance_comparison.png')
        plt.show()    
   
