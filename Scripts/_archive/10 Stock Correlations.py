# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf

# Set plot styling
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

tickers = ['AAL','ALGT','ALK','DAL','ITA','IYT','JBLU','JETS','LUV','UAL','BNO']
tickers_sub = ['AAL','ALGT','ALK','DAL','JBLU','LUV','UAL']

# Load the stock price data
data_path = Path("./Data/Processed/stock_data_long.pkl")
with open(data_path, 'rb') as f:
    df = pickle.load(f)

df['ti']=df['date'].dt.time
df.sort_values(by=['ticker', 'date'], inplace=True)

df[f'Volume_Day_lag01'] = None

for time_val in df['ti'].unique():
    mask = df['ti'] == time_val
    subset = df[mask].copy()
    subset = subset.sort_values(['ticker', 'date'])
    
    # Calculate lag for each ticker separately
    lagged_values = subset.groupby('ticker')['Volume'].shift(1)
    df.loc[mask, f'Volume_Day_lag01'] = lagged_values.values

df[f'Volume_Day_lag01'] = df[f'Volume_Day_lag01'].astype(float)

controls = df[['date', 'BNO_Volume', 'ITA_Volume', 'IYT_Volume', 'JETS_Volume'
               , 'BNO_Volume_lag01', 'ITA_Volume_lag01', 'IYT_Volume_lag01', 'JETS_Volume_lag01']].drop_duplicates().reset_index(drop=True)
controls.rename(columns={
    'BNO_Volume': 'BNO',
    'ITA_Volume': 'ITA',
    'IYT_Volume': 'IYT',
    'JETS_Volume': 'JETS',
    'BNO_Volume_lag01': 'BNO_lag01',
    'ITA_Volume_lag01': 'ITA_lag01',
    'IYT_Volume_lag01': 'IYT_lag01',
    'JETS_Volume_lag01': 'JETS_lag01'
}, inplace=True)

volumes = df[['date', 'ticker', 'Volume', 'Volume_lag01', 'Volume_Day_lag01', 'Change_Volume']].pivot(index='date', columns='ticker', values=['Volume', 'Volume_lag01', 'Volume_Day_lag01', 'Change_Volume'])
# Flatten the MultiIndex columns
volumes.columns = ['_'.join(col).strip() for col in volumes.columns.values]
volumes.columns = [col.replace('Volume_', '') for col in volumes.columns.values]

#volumes = pd.merge(volumes, controls, on='date', how='left').reset_index(drop=True)
volumes = volumes.dropna()

# Compute correlation matrix of daily returns
correlation_matrix = volumes.corr()

# Visualize the correlation matrix as a heatmap
plot_data = correlation_matrix.filter([f'{i}' for i in tickers_sub], axis=0).copy()
plot_data=plot_data[[f'lag01_{i}' for i in tickers_sub]]
plt.figure(figsize=(14, 12))
sns.heatmap(plot_data, annot=True, cmap='vlag', vmin=-0, vmax=0.8, fmt='.3f',linewidths=0.5,square=True,annot_kws={'size': 14})
plt.savefig('./Output/Correlation Matrices/Lagged_Volume.png', dpi=600)

# Visualize the correlation matrix as a heatmap
plot_data = correlation_matrix.filter([f'{i}' for i in tickers_sub], axis=0).copy()
plot_data=plot_data[[f'Day_lag01_{i}' for i in tickers_sub]]
plt.figure(figsize=(14, 12))
sns.heatmap(plot_data, annot=True, cmap='vlag', vmin=-0, vmax=0.8, fmt='.3f',linewidths=0.5,square=True,annot_kws={'size': 14})
plt.savefig('./Output/Correlation Matrices/Lagged_Volume_Day.png', dpi=600)

# Plot autocorrelation for AAL stock for the first 50 lags
plt.figure(figsize=(14, 8))

# Create a clean autocorrelation plot
plot_acf(volumes['AAL'].dropna(), 
         lags=260, 
         alpha=0.05, 
         title=f'Autocorrelation of AAL Stock Price Change (First 50 Lags)',
         zero=False,
         auto_ylims=True)

# Enhance plot appearance
plt.xlabel('Lag', fontsize=14)
plt.ylabel('Autocorrelation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create a matrix of autocorrelation plots for all tickers
fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=True, sharey=True)
axes = axes.flatten()

for i, ticker in enumerate(tickers_sub):
    # Plot autocorrelation for each ticker
    plot_acf(volumes[f'Change_{ticker}'].dropna(), 
             lags=26, 
             alpha=0.05, 
             title=f'{ticker}',
             zero=False,
             auto_ylims=False,
             ax=axes[i])
    
    # Customize each subplot
    axes[i].set_xlabel('Lag (15-Minute Periods)', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_title(f'{ticker}', fontsize=14)

# Remove unused subplots
for j in range(len(tickers_sub), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('./Output/Correlation Matrices/Autocorrelations.png', dpi=600)
