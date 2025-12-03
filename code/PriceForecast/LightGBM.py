import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import math
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# =====File Extraction=====

df_2021 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2021.xlsx', sheet_name='Prezzi-Prices')
df_2022 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2022.xlsx', sheet_name='Prezzi-Prices')
df_2023 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2023.xlsx', sheet_name='Prezzi-Prices')
df_2024 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2024.xlsx', sheet_name='Prezzi-Prices')
df_2025 = pd.read_excel('code/Data_ElectricityMarketPrices/Anno 2025_10.xlsx', sheet_name='Prezzi-Prices')
# after exploring the excel files , I discovered that all of the columns name were similar except for "PUN " in df_2025 :

dataframes_list = [df_2021, df_2022, df_2023, df_2024, df_2025]
df_names = ['df_2021', 'df_2022', 'df_2023', 'df_2024', 'df_2025']

# strip whitespace in column names
for i, df in enumerate(dataframes_list):
    df.columns = [col.strip() for col in df.columns]
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

print("Column names after standardization:")
for name, df in zip(df_names, dataframes_list):
    print(f"\n{name} columns: {list(df.columns)}")

# === Creating datetime index and standardizing Date/Hour names ===

for i, df in enumerate(dataframes_list):
    print(f"\nProcessing {df_names[i]}...")

    if 'Data/Date\n(YYYYMMDD)' in df.columns:
        df.rename(columns={'Data/Date\n(YYYYMMDD)': 'Date'}, inplace=True)
    if 'Ora\n/Hour' in df.columns:
        df.rename(columns={'Ora\n/Hour': 'Hour'}, inplace=True)

    # NOTE: this creates a datetime index on a temp copy,
    # but we keep Date and Hour columns in the main dfs so the models can use them.
    df_temp = df.copy()
    df_temp['date_only'] = pd.to_datetime(df_temp['Date'].astype(str), format='%Y%m%d')

    mask_24 = df_temp['Hour'] == 24
    if mask_24.any():
        df_temp.loc[mask_24, 'date_only'] += pd.Timedelta(days=1)
        df_temp.loc[mask_24, 'Hour'] = 0

    df_temp['datetime'] = df_temp['date_only'] + pd.to_timedelta(df_temp['Hour'], unit='h')
    df_temp.set_index('datetime', inplace=True)
    df_temp.drop(columns=['date_only'], inplace=True)

    # we *do not* drop Date/Hour here, so later models can use them
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

print("\nDatetime index creation step finished (Date & Hour preserved for modeling).")

# === Cleaning DataFrames: Missing Values + Duplicates ===

for i, df in enumerate(dataframes_list):
    print(f"\nProcessing {df_names[i]}...")

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

    print("Checking for and removing duplicate rows...")
    initial_rows = df.shape[0]
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        removed = initial_rows - df.shape[0]
        print(f"{df_names[i]}: Removed {removed} duplicate rows.")
    else:
        print(f"{df_names[i]}: No duplicates found.")

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

# ----------------------------------------------------------------------
# GENERAL MULTI-OUTPUT LIGHTGBM (all zones)
# ----------------------------------------------------------------------

def build_forecasting_dataset(df, target_columns, lookback=24):
    """
    Build X, y for forecasting:
    - X: lagged prices + Date/Hour features
    - y: current prices for each zone
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
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

    data = df[feature_columns + target_columns].dropna()

    X = data[feature_columns]
    y = data[target_columns]

    return X, y, feature_columns


def run_lgbm_forecast(df, lookback=24):
    """
    Train LightGBM on your dataframe (all zones).
    """
    df = df.copy()
    df = df.sort_values(["Date", "Hour"])

    time_cols = ["Date", "Hour"]
    target_columns = [c for c in df.columns if c not in time_cols]

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

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=target_columns)

    print("\n=== LightGBM forecast metrics per zone ===")
    for col in target_columns:
        rmse = sqrt(mean_squared_error(y_test[col], y_pred_df[col]))
        r2 = r2_score(y_test[col], y_pred_df[col])
        print(f"{col}: RMSE={rmse:.3f}, R2={r2:.3f}")

    overall_rmse = sqrt(mean_squared_error(y_test.values, y_pred_df.values))
    overall_r2 = r2_score(y_test.values, y_pred_df.values)

    print("\n=== Overall LightGBM model performance ===")
    print(f"Global RMSE: {overall_rmse:.3f}")
    print(f"Global R2:   {overall_r2:.3f}")

    return model, target_columns, feature_columns, (overall_rmse, overall_r2)


def prepare_latest_features(df, target_columns, lookback=24):
    df = df.copy()
    df = df.sort_values(["Date", "Hour"])

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
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
        raise ValueError("Not enough history to build latest features.")

    return latest, feature_columns


def predict_next_lgbm(df, model, target_columns, lookback=24):
    latest_features, _ = prepare_latest_features(df, target_columns, lookback)
    pred = model.predict(latest_features)[0]
    return pd.Series(pred, index=target_columns)


def get_lgbm_test_predictions(df, model, target_columns, lookback=24):
    df = df.copy()
    df = df.sort_values(["Date", "Hour"])

    X, y, _ = build_forecasting_dataset(df, target_columns, lookback)

    n = len(X)
    n_train = int(n * 0.8)
    X_test = X.iloc[n_train:]
    y_test = y.iloc[n_train:]

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=target_columns)

    return y_test, y_pred_df


def plot_zone_timeseries(df, model, target_columns, zone, lookback=24):
    y_test, y_pred_df = get_lgbm_test_predictions(df, model, target_columns, lookback)

    if zone not in target_columns:
        raise ValueError(f"Zone '{zone}' not found in target columns.")

    plt.figure()
    plt.plot(y_test.index, y_test[zone].values, label="Actual")
    plt.plot(y_test.index, y_pred_df[zone].values, label="Predicted")
    plt.xlabel("Test samples (time order)")
    plt.ylabel("Price")
    plt.title(f"{zone} - Actual vs Predicted (test set)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_zone_scatter_error(df, model, target_columns, zone, lookback=24):
    y_test, y_pred_df = get_lgbm_test_predictions(df, model, target_columns, lookback)

    if zone not in target_columns:
        raise ValueError(f"Zone '{zone}' not in target columns.")

    actual = y_test[zone]
    pred = y_pred_df[zone]
    errors = (pred - actual).abs()

    plt.figure()
    sc = plt.scatter(
        actual.values,
        pred.values,
        c=errors.values,
        cmap="coolwarm",
        s=20,
        alpha=0.8
    )
    plt.colorbar(sc, label="Absolute error")

    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1)

    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title(f"{zone} – Actual vs Predicted (colored by error)")
    plt.tight_layout()
    plt.show()


def plot_rmse_per_zone(df, model, target_columns, lookback=24):
    y_test, y_pred_df = get_lgbm_test_predictions(df, model, target_columns, lookback)

    rmse_vals = {}
    for col in target_columns:
        rmse_vals[col] = sqrt(mean_squared_error(y_test[col], y_pred_df[col]))

    rmse_series = pd.Series(rmse_vals).sort_values()

    plt.figure()
    rmse_series.plot(kind="bar")
    plt.ylabel("RMSE")
    plt.title("RMSE per zone (test set)")
    plt.tight_layout()
    plt.show()


def plot_zone_error_vs_price(df, model, target_columns, zone, lookback=24):
    y_test, y_pred_df = get_lgbm_test_predictions(df, model, target_columns, lookback)

    if zone not in target_columns:
        raise ValueError(f"Zone '{zone}' not in target columns.")

    actual = y_test[zone]
    pred = y_pred_df[zone]
    errors = pred - actual

    plt.figure()
    plt.scatter(actual.values, errors.values, s=15, alpha=0.7)
    plt.axhline(0, color="black", linewidth=1)

    plt.xlabel("Actual price")
    plt.ylabel("Prediction error (Predicted - Actual)")
    plt.title(f"{zone} – Error vs Actual price")
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# SPIKE-AWARE MODEL FOR A SINGLE ZONE (e.g. PUN)
# ----------------------------------------------------------------------

def add_time_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    hour = df["Hour"].astype(int)

    df["DayOfWeek"] = dt.dt.dayofweek
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = hour

    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    return df


def build_spike_dataset(df, zone="PUN",
                        lookback=24,
                        weekly_lag=168,
                        roll_windows=(24, 168)):
    df = df.copy()
    df = df.sort_values(["Date", "Hour"])
    df = add_time_features(df)

    target = df[zone]

    for i in range(1, lookback + 1):
        df[f"{zone}_t-{i}"] = target.shift(i)

    df[f"{zone}_t-{weekly_lag}"] = target.shift(weekly_lag)

    for w in roll_windows:
        df[f"{zone}_roll_mean_{w}"] = target.rolling(w).mean()
        df[f"{zone}_roll_std_{w}"] = target.rolling(w).std()

    df[f"{zone}_diff_1"] = target.diff(1)
    df[f"{zone}_diff_24"] = target.diff(24)

    feature_cols = [
        c for c in df.columns
        if c.startswith(zone + "_t-")
        or c.startswith(zone + "_roll_")
        or c.startswith(zone + "_diff_")
    ] + ["DayOfWeek", "Month", "Day", "Hour", "Hour_sin", "Hour_cos"]

    data = df[feature_cols + [zone]].dropna()

    X = data[feature_cols]
    y = data[zone]

    return X, y, feature_cols


def run_spike_aware_lgbm(df, zone="PUN",
                         lookback=24,
                         weekly_lag=168,
                         roll_windows=(24, 168),
                         spike_threshold=350):

    X, y, feature_cols = build_spike_dataset(
        df, zone=zone,
        lookback=lookback,
        weekly_lag=weekly_lag,
        roll_windows=roll_windows
    )

    n = len(X)
    n_train = int(n * 0.8)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    w_train = np.ones(len(y_train))
    spike_mask = y_train > spike_threshold
    w_train[spike_mask] = 3.0

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print(f"\nTraining spike-aware model for zone {zone}...")
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    overall_rmse = sqrt(mean_squared_error(y_test, y_pred))
    overall_r2 = r2_score(y_test, y_pred)

    normal_mask = y_test <= spike_threshold
    spike_mask_test = y_test > spike_threshold

    normal_rmse = sqrt(mean_squared_error(y_test[normal_mask], y_pred[normal_mask])) \
        if normal_mask.sum() > 0 else np.nan
    spike_rmse = sqrt(mean_squared_error(y_test[spike_mask_test], y_pred[spike_mask_test])) \
        if spike_mask_test.sum() > 0 else np.nan

    print("\n=== Spike-aware model performance ===")
    print(f"Zone: {zone}")
    print(f"Overall RMSE: {overall_rmse:.3f}, R2: {overall_r2:.3f}")
    print(f"Normal  RMSE (<= {spike_threshold}): {normal_rmse:.3f}")
    print(f"Spike   RMSE (>  {spike_threshold}): {spike_rmse:.3f}")
    print(f"Spikes in test set: {spike_mask_test.sum()}")

    return model, (overall_rmse, overall_r2, normal_rmse, spike_rmse), (X_test, y_test, y_pred)


def predict_next_spike_aware(df, model, zone="PUN",
                             lookback=24,
                             weekly_lag=168,
                             roll_windows=(24, 168)):

    X, y, feature_cols = build_spike_dataset(
        df, zone=zone,
        lookback=lookback,
        weekly_lag=weekly_lag,
        roll_windows=roll_windows
    )

    latest_features = X.iloc[[-1]]
    pred = model.predict(latest_features)[0]
    return pred

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":

    df_for_model = df_2021   # choose year

    lookback = 24

    # 1) General multi-output model (all zones)
    model, targets, features, global_metrics = run_lgbm_forecast(
        df_for_model,
        lookback=lookback
    )

    print("\nGlobal metrics (RMSE, R2):", global_metrics)

    print("\nNext-hour prediction (all zones):")
    print(predict_next_lgbm(df_for_model, model, targets, lookback))

    # Plots for PUN (multi-output model)
    plot_zone_timeseries(df_for_model, model, targets, zone="PUN", lookback=lookback)
    plot_zone_scatter_error(df_for_model, model, targets, zone="PUN", lookback=lookback)
    plot_zone_error_vs_price(df_for_model, model, targets, zone="PUN", lookback=lookback)
    plot_rmse_per_zone(df_for_model, model, targets, lookback=lookback)

    # 2) Spike-aware model for PUN
    zone = "PUN"
    model_spike, metrics_spike, (X_test_s, y_test_s, y_pred_s) = run_spike_aware_lgbm(
        df_for_model,
        zone=zone,
        lookback=lookback,
        weekly_lag=168,
        roll_windows=(24, 168),
        spike_threshold=350
    )

    print("\nSpike-aware metrics (overall RMSE, R2, normal RMSE, spike RMSE):")
    print(metrics_spike)

    next_price_spike = predict_next_spike_aware(
        df_for_model,
        model_spike,
        zone=zone,
        lookback=lookback,
        weekly_lag=168,
        roll_windows=(24, 168)
    )

    print(f"\nNext-hour predicted price for {zone} (spike-aware): {next_price_spike:.2f}")
