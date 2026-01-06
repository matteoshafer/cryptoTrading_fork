import pandas as pd

df = pd.read_csv('BTC_trading_signals_2025-01-01_2025-05-07.csv')
print('Bull count stats:')
print(df['bull_count'].describe())
print(f'\nMax bull_count: {df["bull_count"].max()}')
print(f'Days with bull_count >= 3: {(df["bull_count"] >= 3).sum()}')
print(f'Days with bull_count >= 1: {(df["bull_count"] >= 1).sum()}')
print(f'\nPredicted return stats:')
print(df['predicted_return'].describe())
print(f'\nDays with predicted_return > 0.002: {(df["predicted_return"] > 0.002).sum()}')
print(f'\nSample of days with highest bull_count:')
print(df.nlargest(10, 'bull_count')[['time', 'P_t', 'bull_count', 'predicted_return', 'ensemble_buy']].to_string())
