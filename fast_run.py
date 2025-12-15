"""Fast run - skip slow model loading"""
import pandas as pd
import sys

# Skip sentiment model loading
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("="*60)
print("FAST TRADING STRATEGY TEST")
print("="*60)

try:
    from main import load_data
    from model_manager import ModelManager
    from ensemble_model import EnsembleModel
    from main import backtest_strategy
    
    print("\n1. Loading data (last 150 rows only)...")
    data = load_data('ETH')
    data_subset = data.iloc[-150:].copy()  # Use only last 150 rows
    print(f"   Loaded {len(data_subset)} rows")
    
    # Skip sentiment model
    print("\n2. Initializing models (skipping sentiment model)...")
    model_manager = ModelManager()
    model_manager.sentiment_model = None  # Skip slow loading
    
    training_cols = ['volume', 'SMA_20', 'SMA_50']
    training_cols = [col for col in training_cols if col in data_subset.columns]
    print(f"   Using {len(training_cols)} training columns")
    
    model_manager.initialize_models(data_subset, training_cols)
    print("   Models ready")
    
    print("\n3. Generating signals...")
    signals_df = model_manager.generate_signals(data_subset, training_cols)
    print(f"   Generated {len(signals_df)} signals")
    
    # Count buy signals
    buy_cols = [col for col in signals_df.columns if col.endswith('_buy')]
    print("\n   Individual model buy signals:")
    for col in sorted(buy_cols):
        count = signals_df[col].sum()
        print(f"     {col}: {count}")
    
    print("\n4. Running ensemble...")
    ensemble = EnsembleModel()
    result_df = ensemble.generate_signals(signals_df)
    
    buy_count = result_df['ensemble_buy'].sum()
    sell_count = result_df['ensemble_sell'].sum()
    print(f"   Ensemble buy: {buy_count}, sell: {sell_count}")
    
    print("\n5. Backtesting...")
    backtest = backtest_strategy(result_df, initial_capital=10000.0)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Initial Capital: ${backtest.get('initial_capital', 0):,.2f}")
    print(f"Final Capital: ${backtest.get('final_capital', 0):,.2f}")
    print(f"Total Return: {backtest.get('total_return', 0)*100:.2f}%")
    print(f"Total Trades: {backtest.get('total_trades', 0)}")
    print(f"Win Rate: {backtest.get('win_rate', 0)*100:.2f}%")
    print(f"Avg Return/Trade: {backtest.get('avg_return_per_trade', 0)*100:.2f}%")
    print("="*60)
    
    # Show last few signals
    print("\nLast 5 signals:")
    cols = ['P_t', 'bull_count', 'predicted_return', 'ensemble_buy', 'ensemble_sell']
    cols = [c for c in cols if c in result_df.columns]
    print(result_df[cols].tail().to_string())
    
    print("\nComplete!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


