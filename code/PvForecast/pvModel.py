import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform
import glob
import seaborn as sns

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

    def tune(self):
        """
        Performs hyperparameter tuning for the XGBoost Regressor using Randomized Search 
        with Time Series Cross-Validation (TSCV).

        The best model and parameters found are stored internally.

        Returns:
            best_model (xgb.XGBRegressor): The XGBoost model object trained with the optimal hyperparameters.
            best_params (dict): A dictionary containing the best hyperparameters found by the search.
        """
        # Define the Parameter Search Space
        param_distributions = {
            # 1. Conservative Model Control (to reduce over-prediction and improve generalization)
            'learning_rate': uniform(0.01, 0.1),    # Step size shrinkage (0.01 to 0.11)
            'max_depth': randint(3, 5),             # Maximum depth of a tree (3 to 6)
            'gamma': uniform(0.01, 0.5),            # Minimum loss reduction required for a split
            'lambda': uniform(0.1, 2.0),            # L2 regularization term on weights
            'alpha': uniform(0.0, 1.0),             # L1 regularization term on weights

            # 2. Variance Reduction (subsampling)
            'subsample': uniform(0.6, 0.4),         # Fraction of samples used for training each tree (0.6 to 1.0)
            'colsample_bytree': uniform(0.6, 0.4),  # Fraction of features used for training each tree (0.6 to 1.0)

            # 3. Boosting Rounds
            'n_estimators': randint(1000, 2000)     # Number of boosting rounds/trees (1000 to 1999)
        }

        # Initialize the base model
        # Its parameters will be set by the search
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )

        # Use TimeSeriesSplit for Cross-Validation (TSCV) to avoid data leakage
        # The data is split chronologically, respecting the time dependency
        tscv = TimeSeriesSplit(n_splits=5)

        # Initialize the Randomized Search
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_distributions,
            n_iter=5,             # Number of different parameter combinations to try (50-100 is often enough)
            scoring='neg_mean_squared_error', # Standard for regression: we want to maximize the negative MSE
            cv=tscv,
            verbose=2,             # Higher verbosity prints more details during the run
            random_state=42,
            n_jobs=-1              # Use all cores for parallelizing the search
        )

        # Fit the search to the training data
        print("Starting Randomized Search with Time Series Cross-Validation...")
        random_search.fit(self.X_train, self.y_train)

        # Get the best parameters and the best model
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        # Print results
        print("\n--- Tuning Results ---")
        print(f"Best Parameters: {best_params}")
        print(f"Best Negative MSE: {best_score}")
        print(f"Equivalent Best RMSE: {(-best_score)**0.5:.4f}")
        """
        --- Tuning Results ---
        Best Parameters: {'alpha': np.float64(0.8661761457749352), 'colsample_bytree': np.float64(0.8404460046972835), 'gamma': np.float64(0.36403628889802275), 'lambda': np.float64(0.1411689885916049), 'learning_rate': np.float64(0.10699098521619943), 'max_depth': 4, 'n_estimators': 1413, 'subsample': np.float64(0.6849356442713105)}
        Best Negative MSE: -0.0961074152548122
        Equivalent Best RMSE: 0.3100
        """

    def train(self):
        """
        Initializes and trains an XGBoost Regressor model using the training data.

        Returns:
            None: Updates the internal self.model attribute with the trained regressor.
        """
        # Initialize the XGBoost Regressor model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror', # Standard objective for regression tasks
            colsample_bytree=0.840,
            n_estimators=1413, # Number of boosting rounds (trees)
            gamma=0.364,
            alpha=0.866,
            reg_lambda=0.141,
            max_depth=4,
            subsample=0.685,
            learning_rate=0.107, # Step size shrinkage to prevent overfitting
            n_jobs=-1, # Use all available CPU cores for parallel training
            random_state=42, # Seed for reproducibility
            tree_method='hist'
        )

        # Train the model on the split training data
        print("\nTraining XGBoost Regressor...")
        self.model.fit(self.X_train, self.y_train)
        print("Training Complete.")
        """
        Mean Absolute Error (MAE): 0.179 kW
        Root Mean Square Error (RMSE): 0.350 kW
        R-squared (R²): 0.8095
        """

    def get_test_predictions(self):
        """
        Calculates the actual test values and corresponding predictions.
        
        Raises:
            ValueError: If the model is not trained or test data is missing.

        Returns:
            tuple: (pd.Series y_test, np.array y_pred)
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please run the train method first.")
        if self.X_test.empty or self.y_test.empty:
            raise ValueError("Test data is not available. Please run the split method first.")
            
        # Make predictions on the unseen test set
        y_pred = self.model.predict(self.X_test)
        return self.y_test, y_pred

    def test(self):
        """
        Evaluates the trained model's performance on the test set.

        Returns:
            None: Prints evaluation metrics (MAE, RMSE, R²).
        """
        try:
            y_test, y_pred = self.get_test_predictions()
        except ValueError as e:
            print(f"ERROR: {e}")
            return
        
        # Evaluate the predictions
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae:.3f} kW")
        print(f"Root Mean Square Error (RMSE): {rmse:.3f} kW")
        print(f"R-squared (R²): {r2:.4f}")

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
    
    # --- Plotting Methods ---

    def plot_importance_scores(self):
        """
        Plots the feature importance scores of the trained XGBoost model.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(self.model, ax=ax, importance_type='gain', title='Feature Importance (Gain)')
        for text in ax.texts:
            val = float(text.get_text())
            text.set_text(f'{val:.2f}')
        plt.title("PV Generation Forecasting Feature Importance")
        plt.tight_layout()
        plt.show()

    def plot_seasonal_days(self, dates_to_plot):
        """
        Plots the daily mean and standard deviation of actual vs predicted PV generation
        for specified dates to illustrate seasonal variations. 

        Args:
            dates_to_plot (list): List of date strings (YYYY-MM-DD) to plot.
        """
        try:
            y_test, y_pred = self.get_test_predictions()
        except ValueError as e:
            print(f"Plotting Error: {e}")
            return
        
        y_test.index = pd.to_datetime(y_test.index)
        
        # Preparation of a unique DataFrame for Seaborn
        df_plot = pd.DataFrame({
            'Hour': y_test.index.hour,
            'Date': y_test.index.strftime('%Y-%m-%d'),
            'Actual': y_test.values,
            'Predicted': y_pred
        })
        
        for i, date_str in enumerate(dates_to_plot):
            plt.figure(figsize=(12, 6))
            day_df = df_plot[df_plot['Date'] == date_str]
            
            # Real mean and standard deviation
            sns.lineplot(data=day_df, x='Hour', y='Actual', errorbar='sd', 
                        label='Actual', color='tab:blue', linewidth=2)
            
            # Predicted mean and standard deviation
            sns.lineplot(data=day_df, x='Hour', y='Predicted', errorbar='sd', 
                        label='Predicted', color='tab:orange', linestyle='--')

            plt.title(f'Daily mean and standard deviation of PV generation - {date_str}')
            plt.ylabel('Power (kW)')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    def plot_scatter_error(self):
        """
        Plots the predicted PV generation against the actual PV generation, 
        colored by the magnitude of the absolute prediction error.
        """
        try:
            y_test, y_pred = self.get_test_predictions()
        except ValueError as e:
            print(f"Plotting Error: {e}")
            return

        actual = y_test.values
        pred = y_pred
        
        # Calculate the absolute error
        errors = np.abs(pred - actual)

        # Plot the predicted and actual prices colored with the error
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(
            actual,
            pred,
            c=errors,
            cmap="coolwarm",
            s=20,
            alpha=0.8
        )
        plt.colorbar(sc, label="Absolute Prediction Error (kW)")

        # Plot the ideal prediction line (Actual == Predicted)
        min_val = min(actual.min(), pred.min())
        max_val = max(actual.max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1)

        plt.xlabel("Actual PV Generation (kW)")
        plt.ylabel("Predicted PV Generation (kW)")
        plt.title("Actual vs Predicted PV Generation colored by error magnitude (Test set)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_error_vs_actual(self):
        """
        Plots the prediction residual error (Predicted - Actual) against the actual PV generation.
        This helps identify if the model's error is correlated with the magnitude of the target variable.
        """
        try:
            y_test, y_pred = self.get_test_predictions()
        except ValueError as e:
            print(f"Plotting Error: {e}")
            return

        actual = y_test.values
        pred = y_pred

        # Calculate residual error (Predicted - Actual)
        errors = pred - actual

        # Plot the residual error
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, errors, s=15, alpha=0.7)
        
        # Add a zero line for reference
        plt.axhline(0, color="black", linewidth=1, linestyle='--')

        plt.xlabel("Actual PV Generation (kW)")
        plt.ylabel("Prediction Error (kW)")
        plt.title("Error vs Actual Generation (Test Set)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    tuning = False # Set to True to tune the model to find the best hyperparameters
    training = False # Set to False to skip training and only predict

    TRAIN_RATIO = 0.80

    # Get the directory of the current file (PvForecast)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Get the directory of the 'code' folder (one level up)
    BASE_DIR = os.path.dirname(CURRENT_DIR)

    # Define the paths of the input dataset and the output model
    DATA_DIR = "Data_PvProduction"
    input_data_path = os.path.join(BASE_DIR, DATA_DIR, "pv_historical_dataset_part_*.csv.gz")
    output_model_path = os.path.join(BASE_DIR, CURRENT_DIR, "pv_predictor_xgb.joblib")

    # Initialize the model
    model = PvModel()

    if tuning or training:
        # Load and concatenate all dataset parts
        print("Loading dataset...")
        files = sorted(glob.glob(input_data_path))
        dataset = pd.concat(
            (pd.read_csv(f, index_col=0, parse_dates=True) for f in files),
            ignore_index=False
        )

        # Split the dataset
        print("Splitting dataset...")
        model.split(dataset, TRAIN_RATIO)

    # Tune
    if tuning:
        # Tune the model
        model.tune()

    # Train
    elif training:
        # Train the model 
        model.train()

        # Test the model and print metrics
        model.test()

        # Plot
        print("Displaying plots...")
        model.plot_importance_scores()
        model.plot_seasonal_days(['2023-06-21', '2023-12-21'])
        model.plot_scatter_error()
        model.plot_error_vs_actual()

        # Save the model
        model.save_model(output_model_path)
    
    # Predict
    else:
        # Load the model
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