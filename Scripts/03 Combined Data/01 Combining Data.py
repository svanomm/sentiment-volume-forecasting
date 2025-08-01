import pandas as pd
import numpy as np
import pickle, warnings, datetime

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

print("Starting data combination process...")

print("Loading GDELT data...")
with open('./Data/Processed/GDELT_Clean_finance.pkl', 'rb') as f:
    gdelt = pickle.load(f)
gdelt.index.names = [None,None]
print(f"GDELT data loaded: {gdelt.shape[0]:,} rows, {gdelt.shape[1]} columns")

print("Loading stock data...")
with open('./Data/Processed/stock_data_long.pkl', 'rb') as f:
    stocks = pickle.load(f)
print(f"Stock data loaded: {stocks.shape[0]:,} rows, {stocks.shape[1]} columns")

print("Merging datasets...")
# Merge the datasets by row index
df = pd.merge(stocks, gdelt, on=['date', 'ticker'], how='inner')
print(f"Merged data: {df.shape[0]:,} rows, {df.shape[1]} columns")

print(f'Number of colums with lag in name: {len([i for i in stocks.columns if 'lag' in i])}')

print("Handling missing values...")
# Ensure no missing values
df = df.ffill().bfill()
print(f"Missing values handled. Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

print("Sorting columns alphabetically...")
# Sort columns alphabetically
df = df.reindex(sorted(df.columns), axis=1)

print("Reordering columns (date and ticker first)...")
# Move 'date' and 'ticker' to the front
df = df[['date', 'ticker'] + [col for col in df.columns if col not in ['date', 'ticker']]]

print("Downcasting numeric columns to reduce memory usage...")
# Downcast numeric columns to reduce memory usage
df = downcast(df, verbose = True)

print("Saving merged data to pickle file...")
# Write to pickle
with open('./Data/Processed/merged_data_finance.pkl', 'wb') as f:
    pickle.dump(df, f)
print("Data saved successfully!")

print("\n=== FINAL RESULTS ===")
print(df.head())
print(f"\nTicker distribution:")
print(df['ticker'].value_counts())
print(f"\nFinal dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print("Data combination process completed!")
