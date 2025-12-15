"""Quick test with smaller dataset"""
import pandas as pd
from main import load_data
from model_manager import ModelManager
from ensemble_model import EnsembleModel
from main import backtest_strategy

print("="*60)
print("QUICK TEST - Trading Strategy System")
print("="*60)

# Load data
print("\n1. Loading data...")
data = load_data('ETH')
print(f"   ✓ Loaded {len(data)} rows")

# Use last 200 rows for faster testing
print("\n2. Using last 200 rows for quick test...")
data_subset = data.iloc[-200:].copy()
print(f"   ✓ Using {len(data_subset)} rows from {data_subset.index[0]} to {data_subset.index[-1]}")

# Get training columns
training_cols = ['volume', 'SMA_20', 'SMA_50', 'Volume_MA_20', 'OBV', 
                 'BB_Lower', 'BB_Middle', 'BB_Upper', 'avg_sentiment']
training_cols = [col for col in training_cols if col in data_subset.columns]
print(f"   ✓ Using {len(training_cols)} training columns")

# Initialize model manager
print("\n3. Initializing models...")
model_manager = ModelManager()
model_manager.initialize_models(data_subset, training_cols)
print("   ✓ Models initialized")

# Generate signals
print("\n4. Generating signals from individual models...")
signals_df = model_manager.generate_signals(data_subset, training_cols)
print(f"   ✓ Generated signals for {len(signals_df)} time points")

# Show buy signals
buy_cols = [col for col in signals_df.columns if col.endswith('_buy')]
print("\n   Buy signals per model:")
for col in buy_cols:
    count = signals_df[col].sum()
    if count > 0:
        print(f"     {col}: {count}")

# Ensemble
print("\n5. Generating ensemble signals...")
ensemble = EnsembleModel()
result_df = ensemble.generate_signals(signals_df)
print(f"   ✓ Ensemble signals generated")

buy_count = result_df['ensemble_buy'].sum()
sell_count = result_df['ensemble_sell'].sum()
print(f"   ✓ Ensemble buy signals: {buy_count}")
print(f"   ✓ Ensemble sell signals: {sell_count}")

# Backtest
print("\n6. Running backtest...")
backtest_results = backtest_strategy(result_df, initial_capital=10000.0)
print(f"   ✓ Backtest completed")
print(f"\n   Results:")
print(f"     Initial Capital: ${backtest_results.get('initial_capital', 0):,.2f}")
print(f"     Final Capital: ${backtest_results.get('final_capital', 0):,.2f}")
print(f"     Total Return: {backtest_results.get('total_return', 0)*100:.2f}%")
print(f"     Total Trades: {backtest_results.get('total_trades', 0)}")
print(f"     Win Rate: {backtest_results.get('win_rate', 0)*100:.2f}%")
print(f"     Avg Return/Trade: {backtest_results.get('avg_return_per_trade', 0)*100:.2f}%")

# Show sample
print("\n7. Sample results (last 10 rows):")
sample_cols = ['P_t', 'bull_count', 'predicted_return', 'ensemble_buy', 'ensemble_sell', 'position', 'cumulative_return']
available_cols = [c for c in sample_cols if c in result_df.columns]
print(result_df[available_cols].tail(10).to_string())

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)


