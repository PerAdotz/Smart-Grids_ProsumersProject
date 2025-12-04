import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

class PvModel:

    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.model = []

    def split(self, dataset, train_ratio):
        X_cols = ['PV_Capacity_kW', 'Latitude', 'Longitude', 'Month', 'Day', 'Hour_sin', 'Hour_cos']
        X = dataset[X_cols]
        y = dataset['Generation_kW']

        # Calculate the index position where the split should occur
        split_point = int(len(X) * train_ratio)

        # Split temporaly
        self.X_train = X.iloc[:split_point]
        self.y_train = y.iloc[:split_point]
        self.X_test = X.iloc[split_point:]
        self.y_test = y.iloc[split_point:]

    def train(self):
        # Initialize the model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=1000, 
            learning_rate=0.05, 
            n_jobs=-1, # Use all CPU cores
            random_state=42
        )

        # Train the model
        print("\nTraining XGBoost Regressor...")
        self.model.fit(self.X_train, self.y_train)
        print("Training Complete.")

    def test(self):
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
        # Plot the actual values (y_test) and the predicted values (y_pred)
        plt.plot(self.y_test.index, self.y_test, label='Actual PV Generation', alpha=0.7)
        plt.plot(self.y_test.index, y_pred, label='Predicted PV Generation', linestyle='--', color='red')
        plt.title('Actual vs. Predicted PV Output (Test Set)')
        plt.xlabel('Timestamp')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, filepath):
        """
        Saves the trained XGBoost model to a file using joblib.
        """
        if self.model is None:
            print("ERROR: Model is not trained. Cannot save.")
            return

        try:
            joblib.dump(self.model, filepath)
            print(f"Model successfully saved to: {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to save model: {e}")
    
    def load_model(self, filepath):
        """
        Loads a trained XGBoost model from a file using joblib.
        """
        try:
            self.model = joblib.load(filepath)
        except FileNotFoundError:
            print(f"ERROR: Model file not found at: {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")

    def predict_single_point(self, prosumer_data, date, hour):
        """
        Prepares a single input vector for prediction based on prosumer attributes and time.

        Args:
            prosumer_data (object): An object containing prosumer attributes 
                                    (pv_capacity, latitude, longitude).
            date (datetime): The date for prediction.
            hour (int): The hour for prediction.

        Returns:
            float: Predicted PV generation in kW.
        """
        
        # Prepare time features
        month = date.month
        day = date.day
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
    training = False

    TRAIN_RATIO = 0.80

    input_filename = 'PvForecast/pv_historical_dataset.csv'
    output_filename = 'PvForecast/pv_predictor_xgb.joblib'

    # Initialize model
    model = PvModel()

    # Train
    if training:
        # Load dataset
        dataset = pd.read_csv(input_filename)

        # Split the dataset
        model.split(dataset, TRAIN_RATIO)

        # Train model
        model.train()

        # Test model
        model.test()

        # Save model
        model.save_model(output_filename)
    
    # Predict
    else:
        # Load model
        model.load_model(output_filename)

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
