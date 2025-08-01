import pandas as pd
import numpy as np
import os, pickle
import warnings, datetime
from sklearn.metrics import r2_score as r2 
warnings.filterwarnings('ignore')

path_to_data = r"./data/raw/stock prices"

# Get the list of all csv files in path_to_data and all subfolders
csv_files = []
for root, dirs, files in os.walk(path_to_data):
    for f in files:
        if f.endswith('.csv'):
            csv_files.append(os.path.join(root, f))

# Loop through the files, limiting columns and appending airline tickers to a df
for file in csv_files:
    ticker = file.split('_')[0].split('\\')[-1].upper()  # Extract ticker from filename

    if file == csv_files[0]:
        df = pd.read_csv(file)
        df['ticker'] = ticker
    else:
        df_temp = pd.read_csv(file)
        df_temp['ticker'] = ticker
        df = pd.concat([df, df_temp], ignore_index=True)

# Drop duplicates rows
df = df.drop_duplicates()
df = df.dropna()

# Convert 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M')
df['ti']=df['Time'].dt.time
df.sort_values(by=['ticker', 'Time'], inplace=True)

# More efficient approach for lagged variables
for day_lag in range(1, 6):
    df[f'Volume_Day_lag{day_lag:02d}'] = None
    
    for time_val in df['ti'].unique():
        mask = df['ti'] == time_val
        subset = df[mask].copy()
        subset = subset.sort_values(['ticker', 'Time'])
        
        # Calculate lag for each ticker separately
        lagged_values = subset.groupby('ticker')['Volume'].shift(day_lag)
        df.loc[mask, f'Volume_Day_lag{day_lag:02d}'] = lagged_values.values

df['Volume_Day_lagma5'] = df[['Volume_Day_lag01', 'Volume_Day_lag02', 'Volume_Day_lag03', 'Volume_Day_lag04', 'Volume_Day_lag05']].mean(axis=1)

df = df.dropna()
df.sort_values(by=['Time', 'ticker'], inplace=True)
df.index = df[['Time','ticker']]
df['date'] = df['Time']
df = df[df['date'] >= datetime.datetime(2018, 1, 2, 9, 45)]
df = df[df['ticker'].isin(['AAL', 'ALGT', 'ALK', 'DAL', 'JBLU', 'LUV', 'UAL'])]

y_cols = ['Volume']
x_cols = ['Volume_Day_lagma5']

y = df[y_cols]
x = df[x_cols]

# Train/test splitting
split_val  = round(0.8 * len(y))
split_test = round(0.9 * len(y))

y_train = y[:split_val]
y_val   = y[split_val:split_test]
y_test  = y[split_test:]

x_train = x[:split_val]
x_val   = x[split_val:split_test]
x_test  = x[split_test:]

x_eval = np.concatenate((x_val, x_test), axis=0)
y_eval = y[split_val:]

print(f"Baseline: {r2(y_eval, x_eval)}")

predictions = x
pickle.dump(predictions, open(f'./output/models/baseline predictions.pkl', 'wb'))