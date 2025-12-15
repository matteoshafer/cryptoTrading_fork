"""Test script to run the trading strategy"""
import sys
import traceback

try:
    print("="*60)
    print("TESTING TRADING STRATEGY SYSTEM")
    print("="*60)
    
    print("\n1. Testing imports...")
    from main import run_trading_strategy, backtest_strategy
    from model_manager import ModelManager
    from ensemble_model import EnsembleModel
    print("   ✓ All imports successful")
    
    print("\n2. Testing data loading...")
    from main import load_data
    data = load_data('ETH')
    print(f"   ✓ Data loaded: {len(data)} rows")
    print(f"   ✓ Columns: {len(data.columns)} columns")
    print(f"   ✓ Date range: {data.index[0]} to {data.index[-1]}")
    
    print("\n3. Testing model manager initialization...")
    model_manager = ModelManager()
    print(f"   ✓ ModelManager initialized with {len(model_manager.model_configs)} models")
    
    print("\n4. Testing signal generation (first 100 rows)...")
    # Use a subset for faster testing
    data_subset = data.iloc[:100].copy()
    training_cols = ['volume', 'SMA_20', 'SMA_50', 'Volume_MA_20', 'OBV', 
                     'BB_Lower', 'BB_Middle', 'BB_Upper', 'avg_sentiment']
    training_cols = [col for col in training_cols if col in data_subset.columns]
    
    model_manager.initialize_models(data_subset, training_cols)
    signals_df = model_manager.generate_signals(data_subset, training_cols)
    print(f"   ✓ Signals generated: {len(signals_df)} rows")
    print(f"   ✓ Signal columns: {len([c for c in signals_df.columns if '_buy' in c or '_sell' in c])} signal columns")
    
    # Show buy signals per model
    buy_cols = [col for col in signals_df.columns if col.endswith('_buy')]
    print("\n   Buy signals per model:")
    for col in buy_cols:
        count = signals_df[col].sum()
        print(f"     {col}: {count}")
    
    print("\n5. Testing ensemble model...")
    ensemble = EnsembleModel()
    result_df = ensemble.generate_signals(signals_df)
    print(f"   ✓ Ensemble signals generated: {len(result_df)} rows")
    
    buy_count = result_df['ensemble_buy'].sum()
    sell_count = result_df['ensemble_sell'].sum()
    print(f"   ✓ Ensemble buy signals: {buy_count}")
    print(f"   ✓ Ensemble sell signals: {sell_count}")
    
    print("\n6. Testing backtest...")
    backtest_results = backtest_strategy(result_df, initial_capital=10000.0)
    print(f"   ✓ Backtest completed")
    print(f"   ✓ Total return: {backtest_results.get('total_return', 0)*100:.2f}%")
    print(f"   ✓ Total trades: {backtest_results.get('total_trades', 0)}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    
    # Show sample of results
    print("\nSample of results (last 5 rows):")
    print(result_df[['P_t', 'bull_count', 'predicted_return', 'ensemble_buy', 'ensemble_sell']].tail())
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

