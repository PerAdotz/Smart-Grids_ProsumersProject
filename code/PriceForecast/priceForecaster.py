import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import os
import joblib
import json

class PriceForecaster:
    """
    Manages the end-to-end workflow for electricity market price forecasting 
    using LightGBM, including data loading, cleaning, feature engineering, 
    training a multi-output model, and prediction utilities.
    """

    def __init__(self, file_paths, lookback=24):
        """
        Initializes the forecaster, loads the raw data from Excel files, and cleans it.

        Args:
            file_paths (dict): Dictionary mapping year/key to the Excel file path.
            lookback (int, optional): The number of historical hours to use as lag features. Defaults to 24.
        """
        self.lookback = lookback
        self.target_columns = []
        self.feature_columns = []
        self.df_all = self.load_and_clean_data(file_paths)
        self.multi_output_model = None
        self.spike_aware_model = None

    def load_and_clean_data(self, file_paths):
        """
        Loads, standardizes, handles missing values, and concatenates
        all yearly price data into a single DataFrame (self.df_all).

        Args:
            file_paths (dict): Dictionary of file paths.

        Returns:
            pd.DataFrame: The cleaned and concatenated DataFrame.
        """
        dataframes_list = []

        # 1. File Extraction and Column Standardization
        for name, path in file_paths.items():
            try:
                # Read the file
                df = pd.read_excel(path, sheet_name='Prezzi-Prices')
                
                # Standardize column names by stripping whitespace
                df.columns = [col.strip() for col in df.columns]
                
                # Standardize Date and Hour column names
                if 'Data/Date\n(YYYYMMDD)' in df.columns:
                    df.rename(columns={'Data/Date\n(YYYYMMDD)': 'Date'}, inplace=True)
                if 'Ora\n/Hour' in df.columns:
                    df.rename(columns={'Ora\n/Hour': 'Hour'}, inplace=True)
                
                dataframes_list.append(df)

            except FileNotFoundError:
                print(f"Error: File not found at path: {path}")
            except Exception as e:
                print(f"Error processing file {name}: {e}")

        df_all = pd.concat(dataframes_list, ignore_index=True)
        print(f"Initial concatenated data shape: {df_all.shape}")

        # 2. Datetime Indexing and Hour Cleaning
        print("\nApplying datetime transformations...")
        df_all['date_only'] = pd.to_datetime(df_all['Date'].astype(str), format='%Y%m%d')

        # Adjust hour 24 to hour 0 of the next day
        mask_24 = df_all['Hour'] == 24
        if mask_24.any():
            df_all.loc[mask_24, 'date_only'] += pd.Timedelta(days=1)
            df_all.loc[mask_24, 'Hour'] = 0

        df_all['datetime'] = df_all['date_only'] + pd.to_timedelta(df_all['Hour'], unit='h')
        df_all.set_index('datetime', inplace=True)
        df_all.drop(columns=['date_only'], inplace=True)
        df_all = df_all.sort_index()

        # 3. Cleaning: Missing Values + Duplicates
        print("\nCleaning missing values and duplicates...")
        initial_missing = df_all.isnull().sum().sum()
        if initial_missing > 0:
            # Impute missing values using forward-fill then backward-fill
            df_all.fillna(method='ffill', inplace=True)
            df_all.fillna(method='bfill', inplace=True)
            print(f"Imputed {initial_missing} missing values.")
        else:
            print("No missing values detected.")

        duplicates = df_all.duplicated(subset=['Date', 'Hour']).sum()
        if duplicates > 0:
            df_all.drop_duplicates(subset=['Date', 'Hour'], keep='first', inplace=True)
            print(f"Removed {duplicates} duplicate rows based on Date/Hour.")
        else:
            print("No duplicates found.")

        # Identify target columns (all numeric columns excluding Date/Hour/Index)
        time_cols = ["Date", "Hour"]
        self.target_columns = [
            c for c in df_all.columns if c not in time_cols and pd.api.types.is_numeric_dtype(df_all[c])
        ]
        
        print(f"\nData cleaning complete. Final shape: {df_all.shape}")
        print(f"Target series identified: {self.target_columns}")
        
        return df_all.reset_index(drop=True) # Remove datetime index temporarily for feature engineering

    def add_time_features(self, df):
        """
        Addd time-based features (DayOfWeek, Month, cyclical Hour).

        Args:
            df (pd.DataFrame): DataFrame containing 'Date' and 'Hour' columns.

        Returns:
            pd.DataFrame: DataFrame with added time features.
        """
        df = df.copy()

        # Ensure 'Date' is datetime object for feature extraction
        dt = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
        hour = df["Hour"].astype(int)

        df["DayOfWeek"] = dt.dt.dayofweek
        df["Month"] = dt.dt.month
        df["Day"] = dt.dt.day
        df["Hour"] = hour # Keep integer hour feature

        # Cyclical encoding of the hour
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

        return df

    def build_general_dataset(self, lookback=None):
        """
        Builds the feature matrix (X) and target matrix (y) for the multi-output 
        forecasting model using lagged price values.

        Args:
            lookback (int, optional): Number of past hours to use as lag features. 
                                    Defaults to self.lookback.

        Returns:
            tuple: (X, y, feature_columns)
                - X (pd.DataFrame): Feature matrix.
                - y (pd.DataFrame): Target matrix (current prices for all zones).
                - feature_columns (list): List of column names used in X.
        """
        lb = lookback if lookback is not None else self.lookback
        df = self.df_all.copy()
        df = self.add_time_features(df)
        
        targets = self.target_columns

        # Create lag features (t-1 to t-lookback) for every target series
        # for col in targets:
        #     for i in range(1, lb + 1):
        #         df[f"{col}_t-{i}"] = df[col].shift(i)
        lag_data = {
                f"{col}_t-{i}": df[col].shift(i)
                for col in targets
                for i in range(1, lb + 1)
            }

        df = pd.concat([df, pd.DataFrame(lag_data)], axis=1)
        df = df.copy()  # defragmentation

        lag_features = [c for c in df.columns if "_t-" in c]
        time_features = ["DayOfWeek", "Month", "Day", "Hour", "Hour_sin", "Hour_cos"]

        feature_columns = lag_features + time_features
        self.feature_columns = feature_columns # Store feature columns for later prediction

        # Drop initial rows containing NaN values due to lagging
        data = df[feature_columns + targets].dropna()

        X = data[feature_columns]
        y = data[targets]

        return X, y, feature_columns

    def train_multi_output_model(self, lookback=None, train_ratio=0.8):
        """
        Trains the LightGBM Multi-Output Regressor on the historical data.

        Args:
            lookback (int, optional): Number of past hours to use as lag features. Defaults to self.lookback.
            train_ratio (float, optional): Proportion of data to use for training. Defaults to 0.8.

        Returns:
            tuple: (model, global_rmse, global_r2)
        """
        X, y, _ = self.build_general_dataset(lookback)
        
        n = len(X)
        n_train = int(n * train_ratio)
        
        # Split data chronologically
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

        base_lgbm = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model = MultiOutputRegressor(base_lgbm)

        print("\nTraining LightGBM model (multi-output)...")
        model.fit(X_train, y_train)
        self.multi_output_model = model

        # Evaluate performance on test set
        y_pred = model.predict(X_test)
        y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=self.target_columns)

        print("\n=== LightGBM forecast metrics per zone ===")
        for col in self.target_columns:
            rmse = sqrt(mean_squared_error(y_test[col], y_pred_df[col]))
            r2 = r2_score(y_test[col], y_pred_df[col])
            print(f"{col}: RMSE={rmse:.3f}, R2={r2:.3f}")

        overall_rmse = sqrt(mean_squared_error(y_test.values, y_pred_df.values))
        overall_r2 = r2_score(y_test.values, y_pred_df.values)

        print("\n=== Overall LightGBM model performance ===")
        print(f"Global RMSE: {overall_rmse:.3f}")
        print(f"Global R2:   {overall_r2:.3f}")
        """
        === LightGBM forecast metrics per zone ===
        PUN: RMSE=7.751, R2=0.941
        AUST: RMSE=7.938, R2=0.937
        BSP: RMSE=7.938, R2=0.937
        CALA: RMSE=9.326, R2=0.923
        CNOR: RMSE=8.298, R2=0.934
        COAC: RMSE=13.336, R2=0.887
        CORS: RMSE=54.108, R2=0.984
        CSUD: RMSE=8.533, R2=0.933
        FRAN: RMSE=7.938, R2=0.937
        GREC: RMSE=9.424, R2=0.925
        MALT: RMSE=10.579, R2=0.909
        MONT: RMSE=8.596, R2=0.932
        NORD: RMSE=7.938, R2=0.937
        SARD: RMSE=13.257, R2=0.888
        SICI: RMSE=10.547, R2=0.909
        SLOV: RMSE=7.938, R2=0.937
        SUD: RMSE=9.424, R2=0.925
        SVIZ: RMSE=7.938, R2=0.937
        XAUS: RMSE=7.938, R2=0.937
        XFRA: RMSE=7.938, R2=0.937
        XGRE: RMSE=9.424, R2=0.925

        === Overall LightGBM model performance ===
        Global RMSE: 14.861
        Global R2:   0.929
        """

        return overall_rmse, overall_r2

    def prepare_latest_features(self, df_hist, lookback=None):
        """
        Creates the feature vector (X) for the hour immediately 
        following the last timestamp in df_hist.

        Args:
            df_hist (pd.DataFrame): The truncated historical DataFrame containing data 
                                    up to the time point preceding the prediction.
            lookback (int): The number of hours to use for lag features. Defaults to self.lookback.

        Returns:
            tuple: (latest, feature_columns)
                - latest (pd.DataFrame): A DataFrame containing a single row, 
                                        which is the feature vector ready for prediction (X for t+1).
                - feature_columns (list of str): The list of column names in the feature vector.
        """
        lb = lookback if lookback is not None else self.lookback
        df = df_hist.copy()
        df = self.add_time_features(df)
        
        targets = self.target_columns

        # Apply lagging features
        # for col in targets:
        #     for i in range(1, lb + 1):
        #         df[f"{col}_t-{i}"] = df[col].shift(i)
        lag_data = {
            f"{col}_t-{i}": df[col].shift(i)
            for col in targets
            for i in range(1, lb + 1)
        }

        df = pd.concat([df, pd.DataFrame(lag_data)], axis=1)
        df = df.copy()  # defragmentazione

        lag_features = [c for c in df.columns if "_t-" in c]
        time_features = ["DayOfWeek", "Month", "Day", "Hour", "Hour_sin", "Hour_cos"]
        feature_columns = lag_features + time_features
        
        # The latest row contains the features for the NEXT hour prediction (t+1)
        latest = df[feature_columns].iloc[-1:]

        if latest.isna().any().any():
            raise ValueError("Not enough history to build latest features (check lookback setting).")

        return latest, feature_columns

    def predict_next_hour(self, date_int, hour_int, lookback=None):
        """
        Predicts the prices for the hour immediately following (date_int, hour_int) 
        using the trained multi-output model.

        Args:
            date_int (int): Date of the last known price, in YYYYMMDD format.
            hour_int (int): Hour of the last known price (0-23).
            lookback (int, optional): Same lookback used during training. Defaults to self.lookback.

        Returns:
            dict: {zone_name: predicted_price_for_next_hour}
        """
        if self.multi_output_model is None:
            raise ValueError("Multi-output model must be trained first.")
            
        lb = lookback if lookback is not None else self.lookback

        # Work on a copy, sorted by time
        df_sub = self.df_all.copy().sort_values(["Date", "Hour"])

        # 1. Truncate history up to the target time point
        mask = (df_sub["Date"] < date_int) | (
            (df_sub["Date"] == date_int) & (df_sub["Hour"] <= hour_int)
        )
        df_hist = df_sub[mask]

        if df_hist.shape[0] < lb + 5:
            raise ValueError("Not enough history before this hour to build features.")

        # 2. Build latest feature vector (X for t+1)
        latest_features, _ = self.prepare_latest_features(df_hist, lb)

        # 3. Predict next hour for all zones
        pred_values = self.multi_output_model.predict(latest_features)[0]

        # 4. Return as a dictionary
        return dict(zip(self.target_columns, pred_values))

    def predict_future_horizon(self, start_date_int, start_hour_int, horizon=6, lookback=None):
        """
        Predicts prices for a sequence of future hours (horizon) recursively.
        Each prediction uses the previous prediction as input data.

        Args:
            start_date_int (int): Date YYYYMMDD from which to start forecasting (last known data point).
            start_hour_int (int): Hour from which to start forecasting.
            horizon (int, optional): Number of future hours to predict. Defaults to 6.
            lookback (int, optional): Same lookback used during training. Defaults to self.lookback.

        Returns:
            pd.DataFrame: Predicted prices for all zones, indexed by step: t+1, t+2, ...
        """
        if self.multi_output_model is None:
            raise ValueError("Multi-output model must be trained first.")
            
        lb = lookback if lookback is not None else self.lookback
        
        # Initialize working history
        df_work = self.df_all.copy().sort_values(["Date", "Hour"])

        # Keep only history up to the start point
        mask = (df_work["Date"] < start_date_int) | (
            (df_work["Date"] == start_date_int) & (df_work["Hour"] <= start_hour_int)
        )
        df_hist = df_work[mask].copy()

        preds = []

        # Recursive prediction loop
        for step in range(1, horizon + 1):
            
            # 1. Build features (X for t+step) from the extended history
            latest_features, _ = self.prepare_latest_features(df_hist, lb)

            # 2. Predict next hour
            pred_values = self.multi_output_model.predict(latest_features)[0]
            pred_dict = dict(zip(self.target_columns, pred_values))
            preds.append(pred_dict)

            # 3. Create the time index for the new predicted point (t+step)
            # Find the last time point in df_hist
            last_date_int = df_hist["Date"].iloc[-1]
            last_hour_int = df_hist["Hour"].iloc[-1]
            
            # Convert date to string
            last_date_str = str(last_date_int)
            
            # Combine the date and hour information
            dt_combined_str = f"{last_date_str} {last_hour_int:02d}"
            
            # Convert to pandas datetime object representing the last known point
            dt = pd.to_datetime(dt_combined_str, format="%Y%m%d %H")
            
            # Advance one hour
            new_dt = dt + pd.Timedelta(hours=1)
            new_date_int = int(new_dt.strftime("%Y%m%d"))
            new_hour_int = new_dt.hour

            # 4. Create a new row (synthetic data point) and append to history
            new_row = {"Date": new_date_int, "Hour": new_hour_int}
            new_row.update(pred_dict)
            df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)

        return pd.DataFrame(preds, index=[f"t+{i}" for i in range(1, horizon + 1)])

    def save_model(self, filepath):
        """
        Saves the trained MultiOutputRegressor model to a file using joblib 
        for persistent storage.

        Args:
            filepath (str): The complete path and filename for saving the model.

        Returns:
            None: Prints a success or error message.
        """
        if self.multi_output_model is None:
            print("ERROR: Multi-output model is not trained. Cannot save.")
            return

        try:
            # Save the model object
            joblib.dump(self.multi_output_model, filepath)
            print(f"Model successfully saved to: {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to save model: {e}")
    
    def load_model(self, filepath):
        """
        Loads a trained MultiOutputRegressor model from a file using joblib.

        Args:
            filepath (str): The complete path and filename of the model file.

        Returns:
            None: Updates the internal self.multi_output_model attribute. Prints an error message.
        """
        try:
            # Load the model object
            self.multi_output_model = joblib.load(filepath)
            print(f"Model successfully loaded from: {filepath}")
        except FileNotFoundError:
            print(f"ERROR: Model file not found at: {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")

    def get_test_predictions(self, lookback=None):
        """
        Extracts the actual and predicted values for the test set of the 
        multi-output model (last 20% of the historical data).

        Args:
            lookback (int, optional): The lookback window to use for feature creation. 
                                    Defaults to self.lookback.

        Returns:
            tuple: (y_test, y_pred_df)
                - y_test (pd.DataFrame): The actual target values for the test set.
                - y_pred_df (pd.DataFrame): The predicted values for the test set, 
                                            with columns corresponding to price zones.
        """
        if self.multi_output_model is None: 
            raise ValueError("General model not trained.")
        
        X, y, _ = self.build_general_dataset(lookback)
        model = self.multi_output_model
        targets = self.target_columns

        n = len(X)
        n_train = int(n * 0.8)
        X_test = X.iloc[n_train:]
        y_test = y.iloc[n_train:]

        y_pred = model.predict(X_test)
        y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=targets)
        
        return y_test, y_pred_df

    # --- Plotting Methods ---

    def plot_feature_importance(self, zone, top_n=20):
        """
        Plots the feature importance for a specific zone's model.
        
        Args:
            zone (str): The target zone name to inspect.
            top_n (int): Number of top features to display.
        """
        if self.multi_output_model is None:
            raise ValueError("Model must be trained before plotting importance.")
        
        if zone not in self.target_columns:
            raise ValueError(f"Zone '{zone}' not found in targets.")

        # Identify the index of the zone in the multi-output wrapper
        zone_index = self.target_columns.index(zone)
        
        # Extract the specific LightGBM model for that zone
        target_model = self.multi_output_model.estimators_[zone_index]
        
        # Get importance scores
        importances = target_model.feature_importances_
        indices = np.argsort(importances)[-top_n:]  # Get indices of top N features

        # 4. Plot
        plt.figure(figsize=(10, 8))
        plt.title(f"Price Forecasting Feature Importance for {zone}")
        plt.barh(range(len(indices)), importances[indices], color='steelblue', align='center')
        plt.yticks(range(len(indices)), [self.feature_columns[i] for i in indices])
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()

    def plot_zone_timeseries(self, zone, lookback=None, last_n=200):
        """
        Plots actual vs predicted prices over the test set time index for a specific zone 
        using the multi-output model.

        Args:
            zone (str): The price zone column name to plot.
            lookback (int, optional): The lookback window used for feature creation. Defaults to self.lookback.
            last_n (int): Number of last samples to plot.
        """
        try:
            y_test, y_pred_df = self.get_test_predictions(lookback)
        except ValueError as e:
            print(f"Plotting Error: {e}")
            return

        if zone not in self.target_columns:
            raise ValueError(f"Zone '{zone}' not found in target columns.")
        
        actual = y_test[zone].tail(last_n)
        predicted = y_pred_df[zone].tail(last_n)

        # Plot the actual values
        plt.figure(figsize=(12, 6))
        plt.plot(actual.values, label='Actual Price', color='tab:blue', linewidth=2)
        
        # Plot the predicted values
        plt.plot(predicted.values, label='Predicted Price', linestyle='--', color='tab:orange')

        # Format the dates on the x-axis
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        plt.xlabel("Test samples (time order)")
        plt.ylabel("Price (€)")
        plt.title(f"{zone} - Actual vs Predicted Price (Test Set, last {last_n} points)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_rmse_per_zone(self, lookback=None):
        """
        Plots the Root Mean Square Error (RMSE) for each zone, calculated over the test set
        using the general multi-output model.

        Args:
            lookback (int, optional): The lookback window used for feature creation. Defaults to self.lookback.
        """
        try:
            y_test, y_pred_df = self.get_test_predictions(lookback)
        except ValueError as e:
            print(f"Plotting Error: {e}")
            return

        rmse_vals = {}
        for col in self.target_columns:
            rmse_vals[col] = sqrt(mean_squared_error(y_test[col], y_pred_df[col]))

        rmse_series = pd.Series(rmse_vals).sort_values()

        plt.figure(figsize=(10, 6))
        rmse_series.plot(kind="bar")
        plt.ylabel("RMSE")
        plt.title("RMSE per zone (Test Set)")
        plt.tight_layout()
        plt.show()
    
    def plot_zone_scatter_error(self, zone, lookback=None):
        """
        Plots the predicted prices against the actual prices for a specific zone, 
        colored by the magnitude of the absolute prediction error.

        Args:
            zone (str): The price zone column name to plot.
            lookback (int, optional): The lookback window used for feature creation. Defaults to self.lookback.
        """
        try:
            y_test, y_pred_df = self.get_test_predictions(lookback)
        except ValueError as e:
            print(f"Plotting Error: {e}")
            return

        if zone not in self.target_columns:
            raise ValueError(f"Zone '{zone}' not found in target columns.")

        actual = y_test[zone]
        pred = y_pred_df[zone]
        
        # Calculate the absolute error
        errors = (pred - actual).abs()

        # Plot the predicted and actual prices colored with the error
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(
            actual.values,
            pred.values,
            c=errors.values,
            cmap="coolwarm",
            s=20,
            alpha=0.8
        )
        plt.colorbar(sc, label="Absolute Prediction Error (€)")

        # Plot the ideal prediction line (Actual == Predicted)
        min_val = min(actual.min(), pred.min())
        max_val = max(actual.max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1)

        plt.xlabel("Actual Price (€)")
        plt.ylabel("Predicted Price (€)")
        plt.title(f"{zone} - Actual vs Predicted Price colored by error (Test Set)")
        plt.tight_layout()
        plt.show()

    def plot_zone_error_vs_price(self, zone, lookback=None):
        """
        Plots the prediction residual error (Predicted - Actual) against the actual price 
        for a specific zone.

        Args:
            zone (str): The price zone column name to plot.
            lookback (int, optional): The lookback window used for feature creation. Defaults to self.lookback.
        """
        try:
            y_test, y_pred_df = self.get_test_predictions(lookback)
        except ValueError as e:
            print(f"Plotting Error: {e}")
            return

        if zone not in self.target_columns:
            raise ValueError(f"Zone '{zone}' not found in target columns.")

        actual = y_test[zone]
        pred = y_pred_df[zone]

        # Calculate residual error (Predicted - Actual)
        errors = pred - actual

        # Plot the residual error
        plt.figure(figsize=(10, 6))
        plt.scatter(actual.values, errors.values, s=15, alpha=0.7)
        
        # Add a zero line for reference
        plt.axhline(0, color="black", linewidth=1, linestyle='--')

        plt.xlabel("Actual Price (€)")
        plt.ylabel("Prediction Error (€)")
        plt.title(f"{zone} - Error vs Actual price (Test Set)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Get the directory of the current file (PriceForecast)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Get the directory of the 'code' folder (one level up)
    BASE_DIR = os.path.dirname(CURRENT_DIR)

    # Define the directory containing the datasets
    DATA_DIR = 'Data_ElectricityMarketPrices'
        
    file_paths = {
        'df_2021': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2021.xlsx'),
        'df_2022': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2022.xlsx'),
        'df_2023': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2023.xlsx'),
        'df_2024': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2024.xlsx'),
        'df_2025': os.path.join(BASE_DIR, DATA_DIR, 'Anno 2025_10.xlsx'),
    }

    # Define the path to save the model
    output_model_path = os.path.join(BASE_DIR, CURRENT_DIR, "price_forecaster_multi.joblib")

    # Configuration Parameters
    CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
    with open(CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    PRICE_TYPE = config["price_forecaster"]["price_type"]
    
    # Initialize and load data
    lookback = 24
    forecaster = PriceForecaster(file_paths, lookback=lookback)
    
    # Train Multi-Output Model
    g_rmse, g_r2 = forecaster.train_multi_output_model()
    
    print("\n---------------------------------------------------------")
    print(f"Global metrics: RMSE={g_rmse:.3f}, R2={g_r2:.3f}")
    print("---------------------------------------------------------")

    # Predict NEXT-HOUR (using the latest timestamp in the dataset)
    df_all_raw = forecaster.df_all # The raw clean data
    latest_date = int(df_all_raw["Date"].iloc[-1])
    latest_hour = int(df_all_raw["Hour"].iloc[-1])
    print(f"Latest point in the dataset: {latest_date} {latest_hour}")
    # 20251031 0

    latest_prices_dict = forecaster.predict_next_hour(
        date_int=latest_date,
        hour_int=latest_hour
    )

    print(f"\nPredicted NEXT-hour prices for all zones at latest timestamp {latest_date} hour {latest_hour}:")
    print(latest_prices_dict)
    
    # Predict FUTURE HORIZON (next 6 hours)
    future_preds_df = forecaster.predict_future_horizon(
        start_date_int=latest_date,
        start_hour_int=latest_hour,
        horizon=6
    )
    print("\nPredicted 6 future hours recursively:")
    print(future_preds_df)

    # Plot the RMSE by zone
    print("\nPlotting the RMSE zone...")
    forecaster.plot_rmse_per_zone()
    
    # Plot for PRICE_TYPE zone
    print(f"\nDisplaying plots for {PRICE_TYPE} zone...")
    forecaster.plot_zone_timeseries(zone=PRICE_TYPE)
    forecaster.plot_zone_scatter_error(zone=PRICE_TYPE)
    forecaster.plot_zone_error_vs_price(zone=PRICE_TYPE)

    # Save the model
    forecaster.save_model(output_model_path)