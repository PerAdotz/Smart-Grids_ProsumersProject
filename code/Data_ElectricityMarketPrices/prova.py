import pandas as pd

import os
print('working path:' , os.getcwd())

df = pd.read_excel("code/Data_ElectricityMarketPrices/Anno 2021.xlsx",sheet_name="Prezzi-Prices")
print(df.head())