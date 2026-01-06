"""
Main Trading Strategy System
Orchestrates individual models and ensemble model for trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict
from model_manager import ModelManager
from ensemble_model import EnsembleModel
import functions
from functions import fullDataPath
import CONSTANTS
from CONSTANTS import COIN
import warnings
warnings.filterwarnings('ignore')


def load_data(coin: str = None) -> pd.DataFrame:
    """
    Load cryptocurrency data from the fulldata directory.
    
    Args:
        coin: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        
    Returns:
        DataFrame with price and feature data
    """
    if coin is None:
        coin = COIN
    
    try:
        data = pd.read_csv(fullDataPath(coin))
        
        # Ensure time column is datetime
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])
            if data.index.name != 'time':
                data.set_index('time', inplace=True)
        elif data.index.name != 'time' and not isinstance(data.index, pd.DatetimeIndex):
            # If no time column, try to convert index to datetime
            try:
                data.index = pd.to_datetime(data.index)
            except:
                # If that fails, create a simple range index
                pass
        
        # Ensure close price exists
        if 'close' not in data.columns:
            raise ValueError(f"'close' column not found in data for {coin}")
        
        # Calculate technical indicators if not present
        if 'SMA_20' not in data.columns:
            data['SMA_20'] = data['close'].rolling(window=20).mean()
        
        if 'SMA_50' not in data.columns:
            data['SMA_50'] = data['close'].rolling(window=50).mean()
        
        # Calculate price change
        if 'price_change' not in data.columns:
            data['price_change'] = data['close'].diff().fillna(0.0)
        
        # Fill NaN values
        functions.myFillNa(data)
        
        return data
    
    except FileNotFoundError:
        print(f"Data file not found for {coin}. Please ensure {fullDataPath(coin)} exists.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def get_training_columns() -> list:
    """
    Load training columns from file.
    
    Returns:
        List of column names for model training
    """
    try:
        with open('training_columns.txt', 'r') as f:
            columns = [line.strip() for line in f.readlines() if line.strip()]
        return columns
    except FileNotFoundError:
        # Default columns if file doesn't exist
        return ['volume', 'SMA_20', 'SMA_50', 'Volume_MA_20', 'OBV', 
                'BB_Lower', 'BB_Middle', 'BB_Upper', 'avg_sentiment']


def run_trading_strategy(coin: str = None, 
                        start_date: str = None, 
                        end_date: str = None,
                        verbose: bool = True,
                        buy_min_bull_count: int = 2,  # Require at least 2 models to agree for buy
                        buy_threshold_return: float = 0.0,  # Allow any positive predicted return
                        sell_max_bull_count: int = 0,
                        sell_threshold_return: float = 0.0) -> pd.DataFrame:
    """
    Run the complete trading strategy with all models and ensemble.
    
    Args:
        coin: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        start_date: Start date for analysis (YYYY-MM-DD format)
        end_date: End date for analysis (YYYY-MM-DD format)
        verbose: Whether to print progress information
        buy_min_bull_count: Minimum number of bullish models for buy signal (default: 3)
        buy_threshold_return: Minimum predicted return for buy signal (default: 0.002 = 0.2%)
        sell_max_bull_count: Maximum number of bullish models for sell signal (default: 2)
        sell_threshold_return: Maximum predicted return for sell signal (default: -0.002 = -0.2%)
        
    Returns:
        DataFrame with all signals and trading decisions
    """
    if coin is None:
        coin = COIN
    
    if verbose:
        print(f"Loading data for {coin}...")
    
    # Load data
    data = load_data(coin)
    
    if data.empty:
        print("No data loaded. Exiting.")
        return pd.DataFrame()
    
    # Filter by date range if provided
    if start_date:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data.index <= pd.to_datetime(end_date)]
    
    if len(data) < 50:
        print(f"Not enough data points ({len(data)}). Need at least 50.")
        return pd.DataFrame()
    
    if verbose:
        print(f"Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    
    # Get training columns
    training_cols = get_training_columns()
    # Filter to only columns that exist in data
    training_cols = [col for col in training_cols if col in data.columns]
    
    if verbose:
        print(f"Using {len(training_cols)} training columns: {training_cols[:5]}...")
    
    # Initialize Model Manager
    if verbose:
        print("Initializing Model Manager...")
    model_manager = ModelManager()
    model_manager.initialize_models(data, training_cols)
    
    # Generate signals from individual models
    if verbose:
        print("Generating signals from individual models...")
    signals_df = model_manager.generate_signals(data, training_cols)
    
    if signals_df.empty:
        print("No signals generated. Exiting.")
        return pd.DataFrame()
    
    if verbose:
        print(f"Generated signals for {len(signals_df)} time points")
        print(f"Buy signals per model:")
        buy_cols = [col for col in signals_df.columns if col.endswith('_buy')]
        for col in buy_cols:
            print(f"  {col}: {signals_df[col].sum()}")
    
    # Initialize Ensemble Model
    if verbose:
        print("Initializing Ensemble Model...")
        print(f"  Buy conditions: {buy_min_bull_count}+ models bullish, return > {buy_threshold_return*100:.2f}%")
        print(f"  Sell conditions: {sell_max_bull_count} or fewer models bullish, return < {sell_threshold_return*100:.2f}%")
    ensemble = EnsembleModel(
        buy_threshold_return=buy_threshold_return,
        buy_min_bull_count=buy_min_bull_count,
        sell_threshold_return=sell_threshold_return,
        sell_max_bull_count=sell_max_bull_count,
        time_stop_days=5,             # Held >= 5 days
        time_stop_min_return=0.05,   # cumulative return > 5%
        stop_loss_threshold=0.98      # P_t < EntryPrice * 0.98
    )
    
    # Generate ensemble signals
    if verbose:
        print("Generating ensemble trading signals...")
    result_df = ensemble.generate_signals(signals_df)
    
    # Get trading summary
    if verbose:
        print("\n" + "="*50)
        print("TRADING SUMMARY")
        print("="*50)
        summary = ensemble.get_trading_summary(result_df)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("="*50)
    
    return result_df


def backtest_strategy(result_df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
    """
    Backtest the trading strategy.
    
    Args:
        result_df: DataFrame from run_trading_strategy()
        initial_capital: Starting capital
        
    Returns:
        Dictionary with backtest results
    """
    if result_df.empty:
        return {}
    
    capital = initial_capital
    position = 0  # Number of shares/coins held
    entry_price = 0.0
    
    trades = []
    
    for idx, row in result_df.iterrows():
        price = row['P_t']
        
        # Buy signal
        if row['ensemble_buy'] == 1 and position == 0:
            position = capital / price
            entry_price = price
            capital = 0
            trades.append({
                'date': idx,
                'action': 'BUY',
                'price': price,
                'shares': position
            })
        
        # Sell signal
        elif row['ensemble_sell'] == 1 and position > 0:
            capital = position * price
            return_pct = (price - entry_price) / entry_price
            trades.append({
                'date': idx,
                'action': 'SELL',
                'price': price,
                'shares': position,
                'return_pct': return_pct
            })
            position = 0
            entry_price = 0.0
    
    # Calculate final value
    if position > 0:
        final_price = result_df['P_t'].iloc[-1]
        final_capital = position * final_price
    else:
        final_capital = capital
    
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Calculate statistics
    trade_returns = [t['return_pct'] for t in trades if 'return_pct' in t]
    if trade_returns:
        avg_return = np.mean(trade_returns)
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        max_return = max(trade_returns)
        min_return = min(trade_returns)
    else:
        avg_return = 0.0
        win_rate = 0.0
        max_return = 0.0
        min_return = 0.0
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'total_trades': len([t for t in trades if t['action'] == 'BUY']),
        'avg_return_per_trade': avg_return,
        'win_rate': win_rate,
        'max_return': max_return,
        'min_return': min_return,
        'trades': trades
    }


def main():
    """Main entry point for the trading strategy system.
    
    Usage:
        python main.py [COIN] [START_DATE] [END_DATE]
        
    Examples:
        python main.py                    # Uses default coin (ETH), all data
        python main.py BTC                # Analyze BTC, all data
        python main.py BTC 2024-12-01     # Analyze BTC from Dec 1, 2024 to end
        python main.py BTC 2024-12-01 2024-12-31  # Analyze BTC for December 2024
    """
    import sys
    
    # Parse command line arguments
    coin = sys.argv[1] if len(sys.argv) > 1 else COIN
    start_date = sys.argv[2] if len(sys.argv) > 2 else None
    end_date = sys.argv[3] if len(sys.argv) > 3 else None
    
    print("="*60)
    print("CRYPTOCURRENCY TRADING STRATEGY SYSTEM")
    print("="*60)
    print(f"Analyzing: {coin}")
    if start_date:
        print(f"Start date: {start_date}")
    if end_date:
        print(f"End date: {end_date}")
    print()
    
    # Run trading strategy
    result_df = run_trading_strategy(coin=coin, start_date=start_date, end_date=end_date, verbose=True)
    
    if not result_df.empty:
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
        
        # Show detailed trades if any
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
        
        # Save results
        date_suffix = f"_{start_date}_{end_date}" if (start_date and end_date) else (f"_{start_date}" if start_date else "")
        output_file = f'{coin}_trading_signals{date_suffix}.csv'
        result_df.to_csv(output_file)
        print(f"\nResults saved to {output_file}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()

