import pandas as pd
from pvlib.iotools import get_pvgis_hourly
import numpy as np
import os
import sys
import json

# Get the directory of the current file (PvForecast)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the directory of the 'code' folder (one level up)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Add the 'code' directory to the Python search path
# This allows importing generate_community from the parent directory structure
sys.path.append(BASE_DIR)

from Community.community import generate_community

def fetch_pvgis_generation(prosumer, date_start, date_end):
    """
    Calls the PVGIS API (via pvlib) to fetch a time series of hourly PV generation 
    for a single prosumer's system configuration and location.

    Args:
        prosumer (Prosumer): The Prosumer object containing latitude, longitude, 
                            pv_capacity, and losses.
        date_start (int): The starting year for the data fetching.
        date_end (int): The ending year for the data fetching.

    Returns:
        pd.DataFrame: A DataFrame of hourly PV generation data, including feature 
                    columns for model training, or an empty DataFrame if the API call fails.
    """
    try:
        # Call the PVGIS API to retrieve hourly data
        data, metadata = get_pvgis_hourly(
            latitude=prosumer.latitude, # Prosumer latitude
            longitude=prosumer.longitude, # Prosumer longitude
            peakpower=prosumer.pv_capacity, # Nominal power of PV system in kW
            pvcalculation=True, # Request PV production estimate
            loss=prosumer.losses, # Sum of PV system losses in percent
            optimalangles=True, # Request calculation for optimal tilt and azimuth
            raddatabase='PVGIS-SARAH3', # European radiation database
            start=date_start, # First year of the radiation time series
            end=date_end, # Last year of the radiation time series
        )
        
        # Select the generation column and rename it
        df = data[['P']].rename(columns={'P': 'Generation_kW'})
        
        # Convert generation from Watts (W) to Kilowatts (kW)
        df['Generation_kW'] = df['Generation_kW'] / 1000.0

        # Add system metadata as static feature columns for the machine learning model
        df['PV_Capacity_kW'] = prosumer.pv_capacity
        df['Latitude'] = prosumer.latitude
        df['Longitude'] = prosumer.longitude

        # Extract time-based features from the index
        df['Hour'] = df.index.hour
        df["DayOfWeek"] = df.index.dayofweek
        df["Month"] = df.index.month
        df["Day"] = df.index.day

        # Create sine and cosine transformations for the cyclical 'Hour' feature
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

        return df
    
    except Exception as e:
        print(f"Error fetching data for {prosumer.id}: {e}")
        # Return an empty DataFrame on failure
        return pd.DataFrame(index=pd.to_datetime([], utc=True))

if __name__ == "__main__":
    # --- Configuration Parameters ---
    CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
    with open(CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    MULTIPLIER = 5
    NUM_PROSUMERS = config["community"]["num_prosumers"] * MULTIPLIER

    NEIGHBOURHOOD_POOL = config["community"]["neighbourhoods_pool"]
    NUM_NEIGHBOURHOODS = len(NEIGHBOURHOOD_POOL)

    PV_NUMBER_RANGE = tuple(config["community"]["pv_number_range"])
    PV_CAPACITY = config["community"]["pv_capacity"]
    BATTERY_RANGE = config["community"]["battery_capacity_range"]
    LOSSES = config["community"]["pv_losses"]

    # Fetching data from 2021-2023 (exclusive end year)
    DATE_START = 2021
    DATE_END = 2023

    # Define the output path for the collected dataset
    ROWS_PER_FILE = 2_000_000
    OUTPUT_PREFIX = "pv_historical_dataset_part"

    # --- Data Collection Process ---
    
    # 1. Generate the initial community structure (Prosumer objects and Neighbourhoods setup)
    prosumers, neighbourhoods = generate_community(NUM_PROSUMERS, NUM_NEIGHBOURHOODS, NEIGHBOURHOOD_POOL, PV_NUMBER_RANGE, PV_CAPACITY, BATTERY_RANGE, LOSSES)

    all_dataframes = [] # List to hold the full feature set for each prosumer

    # 2. Iterate through all generated prosumers and fetch historical data
    for i, prosumer in enumerate(prosumers):
        print(f"Processing Prosumer {i + 1}/{len(prosumers)}")
        df_features = fetch_pvgis_generation(prosumer, DATE_START, DATE_END)
        if not df_features.empty:
            all_dataframes.append(df_features)

    # 3. Combine data into a single dataset
    # Concatenate all individual feature sets vertically (stacking the time series)
    dataset = pd.concat(all_dataframes, axis=0)

    print("\n--- Data Collection Summary ---")
    print("Total rows:", len(dataset))
    print("Features:", list(dataset.columns))
    print("------------------------------")

    # 4. Save the dataset in chunks to manage large file sizes
    num_chunks = int(np.ceil(len(dataset) / ROWS_PER_FILE))
    for i in range(num_chunks):
        start = i * ROWS_PER_FILE
        end = (i + 1) * ROWS_PER_FILE
        chunk = dataset.iloc[start:end]

        chunk_path = os.path.join(
            BASE_DIR,
            CURRENT_DIR,
            f"{OUTPUT_PREFIX}_{i+1:02d}.csv.gz"
        )

        chunk.to_csv(chunk_path, index=True, compression="gzip")
        print(f"Saved {chunk_path} ({len(chunk)} rows)")