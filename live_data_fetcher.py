"""
Live Bitcoin Data Fetcher
Fetches Bitcoin price data from Coinbase API and updates the local dataset.
Supports scheduled fetching (daily by default, configurable for higher frequency).
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import requests
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = 'fetcher_config.json'
DEFAULT_CONFIG = {
    'coin': 'BTC',
    'data_dir': 'fulldata',
    'fetch_interval_hours': 24,  # Default: once per day
    'granularity': 86400,  # Daily candles (seconds)
    'lookback_days': 365,
    'auto_calculate_indicators': True
}


def load_config() -> dict:
    """Load configuration from file or return defaults."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
    return DEFAULT_CONFIG


def save_config(config: dict):
    """Save configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {CONFIG_FILE}")


def fetch_coinbase_data(
    coin: str = 'BTC',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    granularity: int = 86400
) -> Optional[pd.DataFrame]:
    """
    Fetch historical candlestick data from Coinbase API.

    Args:
        coin: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        start_date: Start date for data fetch
        end_date: End date for data fetch
        granularity: Time interval in seconds (86400 = daily)

    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    product_id = f"{coin.upper()}-USD"
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"

    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    all_data = []
    current_start = start_date

    logger.info(f"Fetching {coin} data from {start_date.date()} to {end_date.date()}")

    while current_start < end_date:
        # Coinbase API returns max 300 candles per request
        current_end = min(current_start + timedelta(seconds=granularity * 300), end_date)

        params = {
            'start': current_start.isoformat(),
            'end': current_end.isoformat(),
            'granularity': granularity
        }

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if data:
                    all_data.extend(data)
                    logger.debug(f"Fetched {len(data)} candles from {current_start.date()}")
            elif response.status_code == 429:
                # Rate limited - wait and retry
                logger.warning("Rate limited. Waiting 10 seconds...")
                time.sleep(10)
                continue
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")

        except requests.exceptions.Timeout:
            logger.error("Request timed out")
        except requests.exceptions.ConnectionError:
            logger.error("Connection error - check internet connection")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

        current_start = current_end
        time.sleep(0.5)  # Rate limiting

    if not all_data:
        logger.warning("No data fetched")
        return None

    # Convert to DataFrame
    columns = ['time', 'low', 'high', 'open', 'close', 'volume']
    df = pd.DataFrame(all_data, columns=columns)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)

    # Calculate basic price changes
    df['change'] = df['close'] - df['open']
    df['pct_change'] = (df['change'] / df['open']) * 100

    logger.info(f"Fetched {len(df)} total candles")
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for the dataset.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    close = df['close']
    volume = df['volume']

    # Simple Moving Averages
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()

    # Exponential Moving Averages
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Middle'] = close.rolling(window=20).mean()
    df['BB_STD'] = close.rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_STD'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_STD'] * 2)

    # Volume indicators
    df['Volume_MA_20'] = volume.rolling(window=20).mean()
    df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    # Fill NaN values
    df = df.fillna(0)

    logger.info("Technical indicators calculated")
    return df


def get_data_path(coin: str, data_dir: str = 'fulldata') -> str:
    """Get the path to the data file for a coin."""
    return os.path.join(data_dir, f'{coin}_df.csv')


def load_existing_data(coin: str, data_dir: str = 'fulldata') -> Optional[pd.DataFrame]:
    """Load existing data from CSV file."""
    path = get_data_path(coin, data_dir)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])
            logger.info(f"Loaded {len(df)} existing records from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
    return None


def update_data(
    coin: str = 'BTC',
    data_dir: str = 'fulldata',
    lookback_days: int = 365,
    calculate_indicators: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch new data and merge with existing data.

    Args:
        coin: Cryptocurrency symbol
        data_dir: Directory for data files
        lookback_days: Days of history to maintain
        calculate_indicators: Whether to recalculate technical indicators

    Returns:
        Updated DataFrame or None if update fails
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load existing data
    existing_df = load_existing_data(coin, data_dir)

    # Determine date range for new data
    end_date = datetime.now()
    if existing_df is not None and len(existing_df) > 0:
        last_date = existing_df['time'].max()
        # Fetch from the day after the last record
        start_date = last_date + timedelta(days=1)
        logger.info(f"Last data point: {last_date.date()}")

        # If we already have today's data, just log and return
        if start_date.date() > end_date.date():
            logger.info("Data is already up to date")
            return existing_df
    else:
        # No existing data - fetch full lookback period
        start_date = end_date - timedelta(days=lookback_days)

    # Fetch new data
    new_df = fetch_coinbase_data(coin, start_date, end_date)

    if new_df is None or len(new_df) == 0:
        logger.warning("No new data fetched")
        return existing_df

    # Merge with existing data
    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Remove duplicates, keeping the most recent
        combined_df = combined_df.drop_duplicates(subset=['time'], keep='last')
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
        logger.info(f"Combined data: {len(combined_df)} total records")
    else:
        combined_df = new_df

    # Trim to lookback period
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    combined_df = combined_df[combined_df['time'] >= cutoff_date].reset_index(drop=True)

    # Calculate technical indicators
    if calculate_indicators:
        combined_df = calculate_technical_indicators(combined_df)

    # Save to file
    path = get_data_path(coin, data_dir)
    combined_df.to_csv(path, index=False)
    logger.info(f"Data saved to {path}")

    return combined_df


def run_scheduler(config: dict):
    """
    Run the data fetcher on a schedule.

    Args:
        config: Configuration dictionary
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.error("APScheduler not installed. Install with: pip install apscheduler")
        logger.info("Running single update instead...")
        update_data(
            coin=config['coin'],
            data_dir=config['data_dir'],
            lookback_days=config['lookback_days'],
            calculate_indicators=config['auto_calculate_indicators']
        )
        return

    scheduler = BlockingScheduler()

    # Schedule the job
    interval_hours = config['fetch_interval_hours']
    scheduler.add_job(
        lambda: update_data(
            coin=config['coin'],
            data_dir=config['data_dir'],
            lookback_days=config['lookback_days'],
            calculate_indicators=config['auto_calculate_indicators']
        ),
        trigger=IntervalTrigger(hours=interval_hours),
        id='data_fetcher',
        name=f'Fetch {config["coin"]} data every {interval_hours} hours',
        replace_existing=True
    )

    logger.info(f"Scheduler started. Fetching {config['coin']} every {interval_hours} hours.")
    logger.info("Press Ctrl+C to stop.")

    # Run immediately on start
    update_data(
        coin=config['coin'],
        data_dir=config['data_dir'],
        lookback_days=config['lookback_days'],
        calculate_indicators=config['auto_calculate_indicators']
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


def get_latest_price(coin: str = 'BTC') -> Optional[dict]:
    """
    Get the latest price for a coin (real-time spot price).

    Args:
        coin: Cryptocurrency symbol

    Returns:
        Dictionary with price info or None
    """
    product_id = f"{coin.upper()}-USD"
    url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'coin': coin,
                'price': float(data['price']),
                'bid': float(data['bid']),
                'ask': float(data['ask']),
                'volume': float(data['volume']),
                'time': data['time']
            }
    except Exception as e:
        logger.error(f"Failed to get latest price: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Live Bitcoin Data Fetcher')
    parser.add_argument('--coin', type=str, default='BTC', help='Cryptocurrency symbol')
    parser.add_argument('--update', action='store_true', help='Run a single data update')
    parser.add_argument('--schedule', action='store_true', help='Run on schedule')
    parser.add_argument('--interval', type=float, default=24, help='Fetch interval in hours')
    parser.add_argument('--price', action='store_true', help='Get current price')
    parser.add_argument('--lookback', type=int, default=365, help='Days of history to maintain')

    args = parser.parse_args()

    # Load or create config
    config = load_config()

    # Override config with command line args
    if args.coin:
        config['coin'] = args.coin.upper()
    if args.interval:
        config['fetch_interval_hours'] = args.interval
    if args.lookback:
        config['lookback_days'] = args.lookback

    # Save updated config
    save_config(config)

    if args.price:
        # Just get current price
        price_info = get_latest_price(config['coin'])
        if price_info:
            print(f"\n{price_info['coin']}/USD Current Price:")
            print(f"  Price: ${price_info['price']:,.2f}")
            print(f"  Bid:   ${price_info['bid']:,.2f}")
            print(f"  Ask:   ${price_info['ask']:,.2f}")
            print(f"  24h Volume: {price_info['volume']:,.2f}")
            print(f"  Time: {price_info['time']}")
        return

    if args.schedule:
        # Run on schedule
        run_scheduler(config)
    else:
        # Single update (default)
        logger.info(f"Running single update for {config['coin']}")
        df = update_data(
            coin=config['coin'],
            data_dir=config['data_dir'],
            lookback_days=config['lookback_days'],
            calculate_indicators=config['auto_calculate_indicators']
        )

        if df is not None:
            print(f"\nData Summary for {config['coin']}:")
            print(f"  Total records: {len(df)}")
            print(f"  Date range: {df['time'].min().date()} to {df['time'].max().date()}")
            print(f"  Latest close: ${df['close'].iloc[-1]:,.2f}")
            print(f"  24h change: {df['pct_change'].iloc[-1]:.2f}%")


if __name__ == '__main__':
    main()
