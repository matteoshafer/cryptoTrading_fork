"""Run trading strategy with full error handling"""
import sys
import traceback

try:
    print("Starting trading strategy...")
    sys.stdout.flush()
    
    from main import run_trading_strategy, backtest_strategy
    
    print("Running strategy for ETH...")
    sys.stdout.flush()
    
    # Run with smaller date range for faster execution
    result_df = run_trading_strategy(
        coin='ETH',
        start_date='2024-01-01',  # Limit to recent data
        verbose=True
    )
    
    if result_df.empty:
        print("ERROR: No results generated")
        sys.exit(1)
    
    print(f"\n✓ Strategy completed! Generated {len(result_df)} rows of signals")
    
    # Show summary
    buy_count = result_df['ensemble_buy'].sum()
    sell_count = result_df['ensemble_sell'].sum()
    print(f"\nEnsemble Signals:")
    print(f"  Buy signals: {buy_count}")
    print(f"  Sell signals: {sell_count}")
    
    # Run backtest
    print("\nRunning backtest...")
    backtest_results = backtest_strategy(result_df, initial_capital=10000.0)
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
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
    print("="*50)
    
    # Save results
    output_file = 'ETH_trading_signals.csv'
    result_df.to_csv(output_file)
    print(f"\nResults saved to {output_file}")
    
except Exception as e:
    print(f"\nERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)


