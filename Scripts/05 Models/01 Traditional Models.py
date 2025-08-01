import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score as r2
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
# Open the pickle file
df = pickle.load(open('./data/processed/merged_data_finance.pkl', 'rb'))
print(f"Data loaded successfully. Shape: {df.shape}")

print("Preparing feature sets...")
sentiment_cols = [i for i in df.columns if any(x in i for x in ['Article Count', 'Tone', 'llm'])] + [i for i in df.columns if i.startswith('c') or i.startswith('v')]
time_cols = ['hour_of_day_10','hour_of_day_11','hour_of_day_12','hour_of_day_13','hour_of_day_14','hour_of_day_15','hour_of_day_9','is_close','is_open','month_of_year_1','month_of_year_10','month_of_year_11','month_of_year_12','month_of_year_2','month_of_year_3','month_of_year_4','month_of_year_5','month_of_year_6','month_of_year_7','month_of_year_8','month_of_year_9','day_of_week_0','day_of_week_1','day_of_week_2','day_of_week_3','day_of_week_4']
self_finance_vars = [i for i in df.columns if 'lag' in i and i not in sentiment_cols and all(x not in i for x in ['BNO','JETS','IYT','ITA'])]
oil_vars          = [i for i in df.columns if 'lag' in i and 'BNO' in i]
etf_finance_vars  = [i for i in df.columns if 'lag' in i and i not in sentiment_cols and any(x in i for x in ['JETS','IYT','ITA'])]
finance_vars = self_finance_vars + oil_vars + etf_finance_vars

# Define the different sets of features to try
feature_sets = {
    'time_only': time_cols,
    'sentiment_only': sentiment_cols,
    'self_finance_only': self_finance_vars,
    'finance_only': finance_vars,
    'finance_time': finance_vars + time_cols,
    'all': sentiment_cols + finance_vars + time_cols 
}

print("Feature set summary:")
for f in feature_sets:
    print(f"{f}: {len(feature_sets[f])} features")

y_cols = ['Volume']
y = df[y_cols]

print("\nSetting up train/validation/test splits...")
# Train/test splitting
split_val  = round(0.8 * len(y))
split_test = round(0.9 * len(y))

y_train = y[:split_val]
y_val   = y[split_val:split_test]
y_test  = y[split_test:]
print(f"Train size: {len(y_train)}, Validation size: {len(y_val)}, Test size: {len(y_test)}")

print(f"\nTraining models on {len(feature_sets)} feature sets...")
for i, feature_set in enumerate(feature_sets, 1):
    print(f'\n[{i}/{len(feature_sets)}] Processing feature set: {feature_set}')

    print(f"  Preparing features ({len(feature_sets[feature_set])} features)...")
    x_cols = feature_sets[feature_set]
    x = df[x_cols]
    x_train = x[:split_val]
    x_val   = x[split_val:split_test]
    x_test  = x[split_test:]

    print("  Normalizing features...")
    # Normalize the features to [0,1]
    sc2 = MinMaxScaler(feature_range=(0, 1))

    x_train = sc2.fit_transform(x_train)
    x_val   = sc2.transform(x_val)
    x_test  = sc2.transform(x_test)

    print("  Training OLS model...")
    ols = LinearRegression()
    ols.fit(x_train, y_train)
    pickle.dump(ols, open(f'./output/models/ols/ols_{feature_set}.pkl', 'wb'))
    print(f"  OLS saved. Val R²: {ols.score(x_val, y_val):.4f}, Test R²: {ols.score(x_test, y_test):.4f}")

    print("  Training LASSO model...")
    lasso = Lasso(
        alpha=1,
        selection='random',
    )

    lasso.fit(x_train, y_train)
    pickle.dump(lasso, open(f'./output/models/lasso/lasso_{feature_set}.pkl', 'wb'))
    print(f"  LASSO saved. Val R²: {lasso.score(x_val, y_val):.4f}, Test R²: {lasso.score(x_test, y_test):.4f}")

print("\nTraining baseline model with single feature...")
x_cols = ['Volume_Day_lagma5']
x = df[x_cols]
x_train = x[:split_val]
x_val   = x[split_val:split_test]
x_test  = x[split_test:]

print("Normalizing baseline features...")
# Normalize the features to [0,1]
sc2 = MinMaxScaler(feature_range=(0, 1))

x_train = sc2.fit_transform(x_train)
x_val   = sc2.transform(x_val)
x_test  = sc2.transform(x_test)

print("Training baseline OLS model...")
ols = LinearRegression()
ols.fit(x_train, y_train)

x_eval = np.concatenate((x_val, x_test), axis=0)
y_eval = y[split_val:]

print(f"Baseline OLS (combined val+test): {ols.score(x_eval, y_eval):.4f}")
print("\nScript completed successfully!")
