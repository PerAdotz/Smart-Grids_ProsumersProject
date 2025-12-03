import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


# =====File Extraction=====

df_2021 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2021.xlsx', sheet_name='Prezzi-Prices')
df_2022 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2022.xlsx', sheet_name='Prezzi-Prices')
df_2023 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2023.xlsx', sheet_name='Prezzi-Prices')
df_2024 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2024.xlsx', sheet_name='Prezzi-Prices')
df_2025 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2025_10.xlsx', sheet_name='Prezzi-Prices')
# after exploring the excel files , I discovered that all of the columns name were similar except for "PUN " in df_2025 :



dataframes_list = [ df_2021, df_2022, df_2023, df_2024, df_2025 ]
df_names = ['df_2021', 'df_2022', 'df_2023', 'df_2024', 'df_2025']

for i, df in enumerate(dataframes_list):
    # Strip whitespace from all column names
    df.columns = [col.strip() for col in df.columns]

    # Assign the modified DataFrame back to its original variable name
    if df_names[i] == 'df_2021':
        df_2021 = df
    elif df_names[i] == 'df_2022':
        df_2022 = df
    elif df_names[i] == 'df_2023':
        df_2023 = df
    elif df_names[i] == 'df_2024':
        df_2024 = df
    elif df_names[i] == 'df_2025':
        df_2025 = df


# Verification step: 
print("Column names after standardization:")
for name, df in zip(df_names, dataframes_list):
    print(f"\n{name} columns: {list(df.columns)}")


# === Creating datetime index and standardizing column names ===

for i, df in enumerate(dataframes_list):
    print(f"\nProcessing {df_names[i]}...")

    # Standardize column names
    if 'Data/Date\n(YYYYMMDD)' in df.columns:
        df.rename(columns={'Data/Date\n(YYYYMMDD)': 'Date'}, inplace=True)
    if 'Ora\n/Hour' in df.columns:
        df.rename(columns={'Ora\n/Hour': 'Hour'}, inplace=True)

    # Create temp copy to avoid chained assignment issues
    df_temp = df.copy()

    # Convert Date to datetime
    df_temp['date_only'] = pd.to_datetime(df_temp['Date'].astype(str), format='%Y%m%d')

    # Handle Hour == 24
    mask_24 = df_temp['Hour'] == 24
    if mask_24.any():
        df_temp.loc[mask_24, 'date_only'] += pd.Timedelta(days=1)
        df_temp.loc[mask_24, 'Hour'] = 0

    # Build datetime index
    df_temp['datetime'] = df_temp['date_only'] + pd.to_timedelta(df_temp['Hour'], unit='h')
    df_temp.set_index('datetime', inplace=True)

    # Drop unused columns
    df_temp.drop(columns=['Date', 'Hour', 'date_only'], inplace=True)

    # Assign back
    if df_names[i] == 'df_2021':
        df_2021 = df_temp
    elif df_names[i] == 'df_2022':
        df_2022 = df_temp
    elif df_names[i] == 'df_2023':
        df_2023 = df_temp
    elif df_names[i] == 'df_2024':
        df_2024 = df_temp
    elif df_names[i] == 'df_2025':
        df_2025 = df_temp

print("\nDatetime index creation completed for all DataFrames.")


# === Cleaning DataFrames: Missing Values + Duplicates ===

for i, df in enumerate(dataframes_list):
    print(f"\nProcessing {df_names[i]}...")

    # ----- Missing value handling -----
    initial_missing = df.isnull().sum().sum()
    print(f"Initial missing values: {initial_missing}")

    if initial_missing > 0:
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        final_missing = df.isnull().sum().sum()
        print(f"Missing values after ffill/bfill: {final_missing}")

        if final_missing > 0:
            print(f"Warning: {final_missing} missing values remain in {df_names[i]}.")
    else:
        print("No missing values detected; skipping imputation.")

    # ----- Duplicate handling -----
    print("Checking for and removing duplicate rows...")
    initial_rows = df.shape[0]
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        removed = initial_rows - df.shape[0]
        print(f"{df_names[i]}: Removed {removed} duplicate rows.")
    else:
        print(f"{df_names[i]}: No duplicates found.")

    # Assign cleaned DF back to its variable name
    if df_names[i] == 'df_2021':
        df_2021 = df
    elif df_names[i] == 'df_2022':
        df_2022 = df
    elif df_names[i] == 'df_2023':
        df_2023 = df
    elif df_names[i] == 'df_2024':
        df_2024 = df
    elif df_names[i] == 'df_2025':
        df_2025 = df

print("\n=== Data Cleaning Complete for All DataFrames ===")

print(df_2021.head())

'''
####
#The individual yearly DataFrames (df_2021, df_2022, df_2023, df_2024, df_2025) are prepared for the price prediction. 
#They have undergone column name standardization, creation of a proper DatetimeIndex, verification of no duplicate rows, absence of missing values, 
#and all relevant columns, including the 'PUN' column, are of the float64 data type.
####
'''
'''
# ===combining all dataframes into one single dataframe===
# this step could be skipped

df_combined = pd.concat([df_2021, df_2022, df_2023, df_2024, df_2025])

print("Verification of df_combined shape:")
print(df_combined.shape)

print("First 5 rows of df_combined:")
print(df_combined.head())

'''

def build_forecasting_dataset(df, target_columns, lookback=5):
    """
    Build supervised dataset:
    X = past values + time features
    y = current values for each price column.
    """
    df = df.copy()

    # ---- time features from Date & Hour ----
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df["Hour"] = df["Hour"].astype(int)

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # cyclical encoding for Hour (captures 24h cycle)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # ---- lag features for each price series ----
    for col in target_columns:
        for i in range(1, lookback + 1):
            df[f"{col}_t-{i}"] = df[col].shift(i)

    lag_features = [c for c in df.columns if "_t-" in c]
    time_features = ["DayOfWeek", "Month", "Day", "Hour", "Hour_sin", "Hour_cos"]

    feature_columns = lag_features + time_features

    data = df[feature_columns + target_columns].dropna()

    X = data[feature_columns]
    y = data[target_columns]

    return X, y, feature_columns


def run_price_forecast(df, lookback=5):
    """
    Train RandomForest to forecast all price columns.
    Returns model and metrics.
    """
    df = df.copy()
    df = df.sort_values(["Date", "Hour"])

    time_columns = ["Date", "Hour"]
    target_columns = [c for c in df.columns if c not in time_columns]

    print("Price series used as targets:")
    print(target_columns)

    X, y, feature_columns = build_forecasting_dataset(df, target_columns, lookback)

    print(f"\nUsing past {lookback} hours as features")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Rows after lagging: {len(X)}")

    n = len(X)
    n_train = int(n * 0.8)

    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # ---- evaluation ----
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=target_columns)

    print("\n=== Forecast Metrics per zone ===")
    for col in target_columns:
        rmse = sqrt(mean_squared_error(y_test[col], y_pred_df[col]))
        r2 = r2_score(y_test[col], y_pred_df[col])
        print(f"{col}: RMSE={rmse:.3f}, R2={r2:.3f}")

    # overall performance (this is what you asked for)
    overall_rmse = sqrt(mean_squared_error(y_test.values, y_pred_df.values))
    overall_r2 = r2_score(y_test.values, y_pred_df.values)

    print("\n=== Overall model performance ===")
    print(f"Global RMSE: {overall_rmse:.3f}")
    print(f"Global R2:   {overall_r2:.3f}")

    return model, target_columns, feature_columns, (overall_rmse, overall_r2)


def prepare_latest_features(df, target_columns, lookback=5):
    """
    Build the latest feature row (most recent history) for prediction.
    """
    df = df.copy()
    df = df.sort_values(["Date", "Hour"])

    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df["Hour"] = df["Hour"].astype(int)

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    for col in target_columns:
        for i in range(1, lookback + 1):
            df[f"{col}_t-{i}"] = df[col].shift(i)

    lag_features = [c for c in df.columns if "_t-" in c]
    time_features = ["DayOfWeek", "Month", "Day", "Hour", "Hour_sin", "Hour_cos"]
    feature_columns = lag_features + time_features

    latest = df[feature_columns].iloc[-1:]

    if latest.isna().any().any():
        raise ValueError("Not enough history for prediction.")

    return latest, feature_columns


def predict_next(df, model, target_columns, lookback=5):
    """
    Forecast next-hour prices for all target columns.
    """
    latest_features, _unused = prepare_latest_features(df, target_columns, lookback)
    pred = model.predict(latest_features)[0]
    return pd.Series(pred, index=target_columns)

if __name__ == "__main__":
    df_for_model = df_2021   

    lookback = 12

    model, targets, features, global_metrics = run_price_forecast(
        df_for_model,
        lookback=lookback
    )

    print("\nGlobal metrics tuple (RMSE, R2):", global_metrics)

    print("\nNext-hour prediction:")
    print(predict_next(df_for_model, model, targets, lookback))
