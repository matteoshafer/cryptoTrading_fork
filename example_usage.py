"""
Example usage of the Trading Strategy System
"""

from main import run_trading_strategy, backtest_strategy
import pandas as pd

# Example 1: Run strategy for default coin (from CONSTANTS.py)
print("Example 1: Running strategy for default coin...")
result_df = run_trading_strategy(verbose=True)

if not result_df.empty:
    # Display some results
    print("\nFirst few signals:")
    print(result_df[['P_t', 'bull_count', 'predicted_return', 'ensemble_buy', 'ensemble_sell']].head(20))
    
    # Run backtest
    print("\nRunning backtest...")
    backtest_results = backtest_strategy(result_df, initial_capital=10000.0)
    print(f"Total Return: {backtest_results['total_return']*100:.2f}%")
    print(f"Win Rate: {backtest_results['win_rate']*100:.2f}%")
    print(f"Total Trades: {backtest_results['total_trades']}")

# Example 2: Run strategy for specific coin
print("\n" + "="*60)
print("Example 2: Running strategy for BTC...")
result_df_btc = run_trading_strategy(coin='BTC', verbose=True)

# Example 3: Run strategy for specific date range
print("\n" + "="*60)
print("Example 3: Running strategy for date range...")
result_df_range = run_trading_strategy(
    coin='ETH',
    start_date='2024-01-01',
    end_date='2024-06-01',
    verbose=True
)

