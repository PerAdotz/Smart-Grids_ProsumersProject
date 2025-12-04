import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os

class PvModel:
    """
    Handles the machine learning workflow for predicting PhotoVoltaic (PV) power generation.
    Uses an XGBoost Regressor model.
    """

    def __init__(self):
        """
        Initializes the PvModel with empty lists for data splits and no trained model.
        """
        self.X_train = [] # Training features
        self.y_train = [] # Training target (Generation_kW)
        self.X_test = [] # Testing features
        self.y_test = [] # Testing target (Generation_kW)
        self.model = None # Placeholder for the trained XGBoost model

    def split(self, dataset, train_ratio):
        """
        Splits the dataset into training and testing sets based on a time-series approach.
        Features used: 'PV_Capacity_kW', 'Latitude', 'Longitude', 'Month', 'Day', 'Hour_sin', 'Hour_cos'.
        Target: 'Generation_kW'.

        Args:
            dataset (pd.DataFrame): The input DataFrame containing PV historical data.
            train_ratio (float): The proportion of the data to use for training.

        Returns:
            None: Updates the internal self.X_train, self.y_train, self.X_test, self.y_test attributes.
        """
        # Define the columns that will be used as features
        X_cols = ['PV_Capacity_kW', 'Latitude', 'Longitude', 'Month', 'Day', 'Hour_sin', 'Hour_cos']
        X = dataset[X_cols]

        # Define the target variable
        y = dataset['Generation_kW']

        # Calculate the index position where the temporal split should occur
        split_point = int(len(X) * train_ratio)

        # Split the data sequentially
        self.X_train = X.iloc[:split_point]
        self.y_train = y.iloc[:split_point]
        self.X_test = X.iloc[split_point:]
        self.y_test = y.iloc[split_point:]
        print(f"Data split complete. Training size: {len(self.X_train)}, Testing size: {len(self.X_test)}")

    def train(self):
        """
        Initializes and trains an XGBoost Regressor model using the training data.

        Returns:
            None: Updates the internal self.model attribute with the trained regressor.
        """
        # Initialize the XGBoost Regressor model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror', # Standard objective for regression tasks
            n_estimators=1000, # Number of boosting rounds (trees)
            learning_rate=0.05, # Step size shrinkage to prevent overfitting
            n_jobs=-1, # Use all available CPU cores for parallel training
            random_state=42 # Seed for reproducibility
        )

        # Train the model on the split training data
        print("\nTraining XGBoost Regressor...")
        self.model.fit(self.X_train, self.y_train)
        print("Training Complete.")

    def test(self):
        """
        Evaluates the trained model's performance on the test set and visualizes the results.

        Returns:
            None: Prints evaluation metrics (MAE, RMSE, R²) and displays a plot.
        """
        if self.model is None:
            print("ERROR: Model is not trained. Please run the train method first.")
            return
        
        # Make predictions on the unseen test set
        y_pred = self.model.predict(self.X_test)

        # Evaluate the predictions
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae:.3f} kW")
        print(f"Root Mean Square Error (RMSE): {rmse:.3f} kW")
        print(f"R-squared (R²): {r2:.4f}")

        # Plot a comparison for visual validation
        plt.figure(figsize=(14, 6))
        # Plot the actual values
        plt.plot(self.y_test.index, self.y_test, label='Actual PV Generation', alpha=0.7)
        # Plot the predicted values
        plt.plot(self.y_test.index, y_pred, label='Predicted PV Generation', linestyle='--', color='red')
        plt.title('Actual vs. Predicted PV Output (Test Set)')
        plt.xlabel('Timestamp')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, filepath):
        """
        Saves the trained XGBoost model to a file using joblib for persistent storage.

        Args:
            filepath (str): The complete path and filename for saving the model.

        Returns:
            None: Prints a success or error message.
        """
        if self.model is None:
            print("ERROR: Model is not trained. Cannot save.")
            return

        try:
            # Save the model object
            joblib.dump(self.model, filepath)
            print(f"Model successfully saved to: {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to save model: {e}")
    
    def load_model(self, filepath):
        """
        Loads a trained XGBoost model from a file using joblib.

        Args:
            filepath (str): The complete path and filename of the model file.

        Returns:
            None: Updates the internal self.model attribute. Prints an error message.
        """
        try:
            # Load the model object
            self.model = joblib.load(filepath)
        except FileNotFoundError:
            print(f"ERROR: Model file not found at: {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")

    def predict_single_point(self, prosumer_data, date, hour):
        """
        Prepares a single input vector for prediction based on prosumer attributes and time and returns the predicted PV generation.

        Args:
            prosumer_data (dict): A dictionary containing required prosumer attributes: 
                                'pv_capacity', 'latitude', 'longitude'.
            date (pd.Timestamp): The date for prediction.
            hour (int): The hour for prediction.

        Returns:
            float: Predicted PV generation in kW. Returns 0.0 if the model is not loaded.
        """
        if self.model is None:
            print("ERROR: Model is not loaded or trained. Cannot predict.")
            return 0.0
        
        # Prepare temporal features
        month = date.month
        day = date.day

        # Apply sine/cosine transformation for cyclical hour feature
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Create the feature vector
        input_data = pd.DataFrame({
            'PV_Capacity_kW': [prosumer_data['pv_capacity']],
            'Latitude': [prosumer_data['latitude']],
            'Longitude': [prosumer_data['longitude']],
            'Month': [month],
            'Day': [day],
            'Hour_sin': [hour_sin],
            'Hour_cos': [hour_cos]
        })
        
        # Make prediction
        prediction = self.model.predict(input_data)
        return prediction[0]

if __name__ == "__main__":
    training = True

    TRAIN_RATIO = 0.80

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    imput_data_path = os.path.join(BASE_DIR, CURRENT_DIR, "pv_historical_dataset.csv")
    output_model_path = os.path.join(BASE_DIR, CURRENT_DIR, "pv_predictor_xgb.joblib")

    # Initialize model
    model = PvModel()

    # Train
    if training:
        # Load dataset
        dataset = pd.read_csv(imput_data_path)

        # Split the dataset
        model.split(dataset, TRAIN_RATIO)

        # Train model
        model.train()

        # Test model
        model.test()

        # Save model
        model.save_model(output_model_path)
    
    # Predict
    else:
        # Load model
        model.load_model(output_model_path)

        # Predict for a test prosumer at noon on the 15th of June
        prosumer = {
            'id': 'P_Demo',
            'pv_capacity': 4.0,
            'latitude': 45.0725,
            'longitude': 7.6875
        }
        date = pd.to_datetime('2022-06-15')
        hour = 12

        generation = model.predict_single_point(prosumer, date, hour)
        print(f"Prediction for {prosumer['id']} at {date} {hour}: {generation:.3f} kW")
