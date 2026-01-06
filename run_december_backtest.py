"""Run backtest for December 2024 and show detailed trade results"""
import sys
from main import run_trading_strategy, backtest_strategy
import pandas as pd

print("="*60)
print("DECEMBER 2024 BACKTEST - BTC")
print("="*60)

# Run strategy - load data from October to allow model training
# Models need historical data to train, but we'll analyze December trades
result_df = run_trading_strategy(
    coin='BTC',
    start_date='2024-10-01',  # Start earlier to allow model training (needs 50+ points)
    end_date='2024-12-31',
    verbose=True
)

if result_df.empty:
    print("\nERROR: No results generated")
    sys.exit(1)

# Filter to December only for backtest analysis
print("\nFiltering to December 2024 only...")
if 'time' in result_df.columns:
    result_df['date'] = pd.to_datetime(result_df['time'])
else:
    result_df['date'] = pd.to_datetime(result_df.index)

dec_mask = (result_df['date'] >= '2024-12-01') & (result_df['date'] <= '2024-12-31')
result_df_dec = result_df[dec_mask].copy()

if result_df_dec.empty:
    print("ERROR: No December data found")
    sys.exit(1)

print(f"December data points: {len(result_df_dec)} days")

# Run backtest on December data
print("\n" + "="*60)
print("RUNNING BACKTEST (December 2024)")
print("="*60)
backtest_results = backtest_strategy(result_df_dec, initial_capital=10000.0)

# Print backtest summary
print("\n" + "="*60)
print("BACKTEST SUMMARY")
print("="*60)
for key, value in backtest_results.items():
    if key != 'trades':
        if isinstance(value, float):
            if 'return' in key.lower() or 'rate' in key.lower():
                print(f"{key}: {value*100:.2f}%")
            elif 'capital' in key.lower():
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

# Show detailed trades
print("\n" + "="*60)
print("DETAILED TRADES (December 2024)")
print("="*60)
trades = backtest_results.get('trades', [])
if trades:
    print(f"\nTotal trades executed: {len(trades)}")
    print("\nTrade Log:")
    for i, trade in enumerate(trades, 1):
        print(f"\nTrade {i}:")
        print(f"  Date: {trade['date']}")
        print(f"  Action: {trade['action']}")
        print(f"  Price: ${trade['price']:,.2f}")
        print(f"  Shares: {trade['shares']:.6f}")
        if 'return_pct' in trade:
            print(f"  Return: {trade['return_pct']*100:.2f}%")
            print(f"  Profit/Loss: ${trade['shares'] * trade['price'] - (trade['shares'] * trade['price'] / (1 + trade['return_pct'])):.2f}")
else:
    print("\nNo trades executed during December 2024.")
    print("\nReason: No ensemble buy/sell signals were generated in December.")
    
    # Show signal statistics for December
    print("\nDecember Signal Statistics:")
    print(f"  Total ensemble buy signals: {result_df_dec['ensemble_buy'].sum()}")
    print(f"  Total ensemble sell signals: {result_df_dec['ensemble_sell'].sum()}")
    
    # Show December predictions
    print("\nDecember Daily Predictions:")
    sample_cols = ['date', 'P_t', 'bull_count', 'predicted_return', 'SMA_20', 'ensemble_buy', 'ensemble_sell']
    available_cols = [col for col in sample_cols if col in result_df_dec.columns]
    if 'date' not in available_cols and 'time' in result_df_dec.columns:
        available_cols.insert(0, 'time')
    print(result_df_dec[available_cols].to_string())

print("="*60)

# Save results
output_file = 'BTC_december_trading_signals.csv'
result_df_dec.to_csv(output_file)
print(f"\nDecember signals saved to: {output_file}")
