import pvlib
import pandas as pd
from pvlib.iotools import get_pvgis_hourly
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Get the directory of the current file (PvForecast)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the directory of the 'code' folder (one level up)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Add the 'code' directory to the Python search path
sys.path.append(BASE_DIR)

from Community.community import generate_community

def fetch_pvgis_generation(prosumer, date_start, date_end):
    """
    Calls PVGIS API for a single prosumer and returns a DataFrame
    of hourly PV generation (P) in Watts.
    """
    try:
        data, metadata = get_pvgis_hourly(
            latitude=prosumer.latitude, # In decimal degrees, between -90 and 90, north is positive (ISO 19115)
            longitude=prosumer.longitude, # In decimal degrees, between -180 and 180, east is positive (ISO 19115)
            peakpower=prosumer.pv_capacity, # Nominal power of PV system in kW
            pvcalculation=True, # Estimate hourly PV production 
            loss=prosumer.losses, # Sum of PV system losses in percent
            optimalangles=True, # Optimize the surface_tilt and surface_azimuth
            raddatabase='PVGIS-SARAH3', # Name of the radiation database: "PVGIS-SARAH" for Europe
            start=date_start, # First year of the radiation time series
            end=date_end, # Last year of the radiation time series
        )
        
        # Select key feature
        df = data[['P']].rename(columns={'P': 'Generation_kW'})
        
        # Convert W to kW
        df['Generation_kW'] = df['Generation_kW'] / 1000.0
        
        # Add system metadata as columns
        df['PV_Capacity_kW'] = prosumer.pv_capacity
        df['Latitude'] = prosumer.latitude
        df['Longitude'] = prosumer.longitude

        # Add time features
        df['Hour'] = df.index.hour
        df["DayOfWeek"] = df.index.dayofweek
        df["Month"] = df.index.month
        df["Day"] = df.index.day

        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

        return df
    
    except Exception as e:
        print(f"Error fetching data for {prosumer.id}: {e}")
        # Return an empty DataFrame column for failed prosumers to maintain structure
        return pd.DataFrame(index=pd.to_datetime([], utc=True))

if __name__ == "__main__":
    NUM_PROSUMERS = 100
    NUM_NEIGHBOURHOODS = 10

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

    PV_NUMBER_RANGE = (0, 20) # Number of PV pannels
    PV_CAPACITY = 0.25 # PV pannel capacity in kW
    BATTERY_RANGE = [0, 5, 10] # Battery capacity in kW
    LOSSES = 14 # Sum of PV system losses in percent

    DATE_START = 2021
    DATE_END = 2023

    output_data_path = os.path.join(BASE_DIR, CURRENT_DIR, "pv_historical_dataset.csv")

    # Generate a community for which retreiving data
    prosumers, neighbourhoods = generate_community(NUM_PROSUMERS, NUM_NEIGHBOURHOODS, NEIGHBOURHOOD_POOL, PV_NUMBER_RANGE, PV_CAPACITY, BATTERY_RANGE, LOSSES)

    all_dataframes = [] # List to hold the full feature set for each neighborhood

    for i, prosumer in enumerate(prosumers):
        print(f"Processing Prosumer {i + 1}/{len(prosumers)}")
        df_features = fetch_pvgis_generation(prosumer, DATE_START, DATE_END)
        if not df_features.empty:
            all_dataframes.append(df_features)

    # Concatenate all individual feature sets vertically (stacking the time series)
    dataset = pd.concat(all_dataframes, axis=0)

    # Save the DataFrame to a CSV file
    dataset.to_csv(output_data_path, index=True)

    print("Total rows:", len(dataset))
    print("Features:", list(dataset.columns))