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
import matplotlib.pyplot as plt




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
