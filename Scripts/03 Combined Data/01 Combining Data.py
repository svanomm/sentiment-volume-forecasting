import pandas as pd
import numpy as np
import pickle, warnings, datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
def downcast(df, verbose = True):
    start_memory = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        dtype_name = df[col].dtype.name
        if dtype_name == "object" or 'date' in dtype_name:
            pass
        elif dtype_name == "bool":
            df[col] = df[col].astype("int8")
        elif dtype_name.startswith("int") or (df[col].round() == df[col]).all():
            df[col] = pd.to_numeric(df[col], downcast = "integer")
        else:
            df[col] = pd.to_numeric(df[col], downcast = "float")
    end_memory = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print("{:.1f}% compressed".format(100 * (start_memory - end_memory) / start_memory))
    return df

with open('./Data/Processed/GDELT_Clean_finance.pkl', 'rb') as f:
    gdelt = pickle.load(f)
gdelt.index.names = [None,None]

with open('./Data/Processed/stock_data_long.pkl', 'rb') as f:
    stocks = pickle.load(f)

# Merge the datasets by row index
df = pd.merge(stocks, gdelt, on=['date', 'ticker'], how='inner')

print(f'Number of colums with lag in name: {len([i for i in stocks.columns if 'lag' in i])}')

# Ensure no missing values
df = df.ffill().bfill()
# Sort columns alphabetically
df = df.reindex(sorted(df.columns), axis=1)
# Move 'date' and 'ticker' to the front
df = df[['date', 'ticker'] + [col for col in df.columns if col not in ['date', 'ticker']]]
# Downcast numeric columns to reduce memory usage
df = downcast(df, verbose = True)

# Write to pickle
with open('./Data/Processed/merged_data_finance.pkl', 'wb') as f:
    pickle.dump(df, f)

print(df.head())
print(df['ticker'].value_counts())
