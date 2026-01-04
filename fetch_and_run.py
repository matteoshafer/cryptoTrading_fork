"""Fetch new cryptocurrency data and run the trading strategy"""
import sys
from datetime import datetime, timedelta
import pandas as pd
from functions import prices, fullDataPath, myFillNa
from main import run_trading_strategy, backtest_strategy

def fetch_and_update_data(coin='BTC', start_date=None, end_date=None):
    """
    Fetch new cryptocurrency data from Coinbase API and update the data file.
    
    Args:
        coin: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        start_date: Start date (datetime or string 'YYYY-MM-DD'). If None, fetches last 365 days
        end_date: End date (datetime or string 'YYYY-MM-DD'). If None, uses current date
    """
    print(f"Fetching data for {coin}...")
    
    # Convert dates to datetime if strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Fetch data using the prices function
    data, fetched_coin = prices(
        product_id=coin,
        start=start_date,
        end=end_date
    )
    
    if data is None:
        print(f"Failed to fetch data for {coin}")
        return None
    
    print(f"Fetched {len(data)} rows of data")
    print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    
    # Load existing data if it exists
    try:
        existing_data = pd.read_csv(fullDataPath(coin))
        existing_data['time'] = pd.to_datetime(existing_data['time'])
        print(f"Existing data: {len(existing_data)} rows from {existing_data['time'].min()} to {existing_data['time'].max()}")
        
        # Combine and remove duplicates (keep most recent)
        combined = pd.concat([existing_data, data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['time'], keep='last')
        combined = combined.sort_values('time').reset_index(drop=True)
        data = combined
        print(f"Combined data: {len(data)} rows from {data['time'].min()} to {data['time'].max()}")
    except FileNotFoundError:
        print("No existing data file found. Creating new file.")
    
    # Ensure time is datetime and set as index
    data['time'] = pd.to_datetime(data['time'])
    if data.index.name != 'time':
        data.set_index('time', inplace=True)
    
    # Calculate technical indicators if not present
    if 'SMA_20' not in data.columns:
        data['SMA_20'] = data['close'].rolling(window=20).mean()
    if 'SMA_50' not in data.columns:
        data['SMA_50'] = data['close'].rolling(window=50).mean()
    
    # Fill NaN values
    myFillNa(data)
    
    # Reset index to save time as column
    data = data.reset_index()
    
    # Save to file
    output_path = fullDataPath(coin)
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    return data

if __name__ == "__main__":
    import sys
    
    # Parse arguments: coin start_date end_date
    coin = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    start_date = sys.argv[2] if len(sys.argv) > 2 else None
    end_date = sys.argv[3] if len(sys.argv) > 3 else datetime.now()
    
    print("="*60)
    print("FETCHING NEW DATA")
    print("="*60)
    
    # Fetch data
    data = fetch_and_update_data(coin=coin, start_date=start_date, end_date=end_date)
    
    if data is not None:
        print("\n" + "="*60)
        print("RUNNING TRADING STRATEGY")
        print("="*60)
        
        # Run the strategy - use October-December 2025 to allow model training
        # (models need at least 50 data points to train)
        result_df = run_trading_strategy(coin=coin, start_date='2025-10-01', end_date='2025-12-31', verbose=True)
        
        if not result_df.empty:
            # Filter to December 2025 only for analysis
            print("\nFiltering to December 2025 only...")
            if 'time' in result_df.columns:
                result_df['date'] = pd.to_datetime(result_df['time'])
            else:
                result_df['date'] = pd.to_datetime(result_df.index)
            
            dec_mask = (result_df['date'] >= '2025-12-01') & (result_df['date'] <= '2025-12-31')
            result_df_dec = result_df[dec_mask].copy()
            
            if result_df_dec.empty:
                print("No December 2025 data found in results")
            else:
                print(f"December 2025 data points: {len(result_df_dec)} days")
                
                # Run backtest on December data only
                print("\nRunning backtest on December 2025...")
                backtest_results = backtest_strategy(result_df_dec, initial_capital=10000.0)
            
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
            
            # Show trades
            trades = backtest_results.get('trades', [])
            if trades:
                print("\n" + "="*50)
                print("TRADE LOG")
                print("="*50)
                for i, trade in enumerate(trades, 1):
                    print(f"\nTrade {i}:")
                    print(f"  Date: {trade['date']}")
                    print(f"  Action: {trade['action']}")
                    print(f"  Price: ${trade['price']:,.2f}")
                    print(f"  Shares: {trade['shares']:.6f}")
                    if 'return_pct' in trade:
                        profit_loss = trade['shares'] * trade['price'] - (trade['shares'] * trade['price'] / (1 + trade['return_pct']))
                        print(f"  Return: {trade['return_pct']*100:.2f}%")
                        print(f"  Profit/Loss: ${profit_loss:,.2f}")
                print("="*50)
