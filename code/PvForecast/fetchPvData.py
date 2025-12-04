import pandas as pd
from pvlib.iotools import get_pvgis_hourly
import numpy as np
import os
import sys

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
        data, _ = get_pvgis_hourly(
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
    NUM_PROSUMERS = 100
    NUM_NEIGHBOURHOODS = 10

    # Geographical bounding boxes for 10 simulated neighbourhoods
    NEIGHBOURHOOD_POOL = {
        'Centro': [45.065, 45.080, 7.675, 7.690],
        'San_Salvario': [45.050, 45.065, 7.675, 7.695],
        'Crocetta': [45.060, 45.075, 7.645, 7.660],
        'Aurora': [45.088, 45.100, 7.675, 7.695],
        'Vanchiglia': [45.068, 45.083, 7.690, 7.710],
        'Lingotto': [45.015, 45.035, 7.640, 7.665],
        'Santa_Rita': [45.040, 45.055, 7.630, 7.655],
        'San_Donato': [45.080, 45.095, 7.640, 7.658],
        'Cit_Turin': [45.070, 45.085, 7.660, 7.675],
        'Barriera_di_Milano': [45.100, 45.115, 7.665, 7.685],
    }

    PV_NUMBER_RANGE = (0, 20) # Range for number of PV panels
    PV_CAPACITY = 0.25 # Single PV panel capacity in kW
    BATTERY_RANGE = [0, 5, 10] # Available battery capacities in kW
    LOSSES = 14 # Total system losses in percent

    # Fetching data from 2021-2023 (exclusive end year)
    DATE_START = 2021
    DATE_END = 2023

    # Define the output path for the collected dataset
    output_data_path = os.path.join(BASE_DIR, CURRENT_DIR, "pv_historical_dataset.csv")

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

    # 4. Save the dataset
    dataset.to_csv(output_data_path, index=True)

    print("\n--- Data Collection Summary ---")
    print(f"Dataset saved to: {output_data_path}")
    print("Total rows:", len(dataset))
    print("Features:", list(dataset.columns))
    print("------------------------------")