import pandas as pd
import numpy as np
import os, pickle
import warnings
import datetime
warnings.filterwarnings('ignore')

path_to_data = r"./data/raw/oil and other controls"

# Get the list of all csv files in path_to_data and all subfolders
csv_files = []
for root, dirs, files in os.walk(path_to_data):
    for f in files:
        if f.endswith('.csv'):
            csv_files.append(os.path.join(root, f))

# Loop through the files, limiting columns and appending airline tickers to a df
for file in csv_files:
    if file == csv_files[0]:
        df_main = pd.read_csv(file)
    else:
        df_temp = pd.read_csv(file)
        df_main = pd.concat([df_main, df_temp], ignore_index=True)

# Drop duplicates rows
df_main = df_main.drop_duplicates()
df_main = df_main.dropna()

# Convert 'Time' column to datetime format
df_main['Time'] = pd.to_datetime(df_main['Time'], format='%Y-%m-%d %H:%M')
df_main['%Chg'] = df_main['%Chg'].str.replace('%', '').astype(float)

df_main.sort_values(by=['Time'], inplace=True)
df_main.index = df_main['Time']
df_main.drop(columns=['Time'], inplace=True)

df_main['Last'].rolling(window=10).std()

# Finance variables
df_main['High-Low']   =  df_main['High'] - df_main['Low']
df_main['High-Low%']  = (df_main['High'] / df_main['Open']) - 1
df_main['Last-Open']  =  df_main['Open'] - df_main['Last']
df_main['Last-Open%'] = (df_main['Open'] / df_main['Last']) - 1

# 10-period rolling variance
df_main['Roll_SD_Last_10']   = df_main['Last'].rolling(window=10).std()
df_main['Roll_SDSD_Last_10'] = df_main['Roll_SD_Last_10'].rolling(window=10).std()
df_main['Roll_SD_Volume_10']   = df_main['Volume'].rolling(window=10).std()
df_main['Roll_SDSD_Volume_10'] = df_main['Roll_SD_Volume_10'].rolling(window=10).std()

# Moving averages
# 4-period moving average of 'Last'
df_main['MA4_Last']  = df_main['Last'].transform(lambda x: x.rolling(window=4).mean())
df_main['MA12_Last'] = df_main['Last'].transform(lambda x: x.rolling(window=12).mean())

# Forward fill missing values
df_wide = df_main.ffill()
df_wide = df_wide[df_wide.index >= datetime.datetime(2018, 1, 1, 0, 0)]

# add the prefix "Oil_" to all columns
df_wide = df_wide.add_prefix('Oil_')

# Save the processed DataFrame to a pickle file
output_path = r"./data/processed/oil_data.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(df_wide, f)