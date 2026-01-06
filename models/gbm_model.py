"""
Custom GBM (Geometric Brownian Motion) Model
Simulates price paths using GBM.

Buy Signal: mean(pred_paths[-1]) > P_t * (1 + 0.01)
Sell Signal: mean(pred_paths[-1]) < P_t * (1 - 0.005)
"""

import pandas as pd
import numpy as np
from typing import Optional


class GBMModel:
    """Geometric Brownian Motion Model for trading signals."""
    
    def __init__(self,
                 buy_threshold_pct: float = 0.01,  # 1% gain
                 sell_threshold_pct: float = -0.005,  # 0.5% drop
                 n_paths: int = 100,
                 min_train_size: int = 20):
        """
        Initialize GBM Model.
        
        Args:
            buy_threshold_pct: Buy threshold as percentage (default: 0.01 = 1%)
            sell_threshold_pct: Sell threshold as percentage (default: -0.005 = -0.5%)
            n_paths: Number of simulation paths
            min_train_size: Minimum training data size
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.n_paths = n_paths
        self.min_train_size = min_train_size
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict price changes using GBM simulation with rolling window.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series of predicted price changes
        """
        if len(data) < self.min_train_size or 'close' not in data.columns:
            return pd.Series(0.0, index=data.index)
        
        try:
            prices = data['close'].values
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, use previous data to estimate parameters
            T = 1  # Forecast horizon (1 day)
            
            for i in range(self.min_train_size, len(data)):
                try:
                    window_prices = prices[:i+1]
                    window_returns = np.diff(window_prices) / window_prices[:-1]
                    
                    if len(window_returns) < 5:
                        continue
                    
                    mu = np.mean(window_returns)  # Drift
                    sigma = np.std(window_returns)  # Volatility
                    current_price = window_prices[-1]
                    
                    # Simulate paths
                    paths = []
                    for _ in range(self.n_paths):
                        z = np.random.normal(0, 1)
                        future_price = current_price * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
                        paths.append(future_price)
                    
                    mean_pred = np.mean(paths)
                    pred_series.iloc[i] = mean_pred - current_price  # Return as price change
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"GBM Simulator error: {e}")
            return pd.Series(0.0, index=data.index)
    
    def generate_signals(self, data: pd.DataFrame) -> tuple:
        """
        Generate buy/sell signals.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (pred_change, buy_signals, sell_signals)
        """
        pred_change = self.predict(data)
        current_prices = data['close']
        
        # Calculate future prices
        future_prices = current_prices + pred_change
        
        # Buy: mean(pred_paths[-1]) > P_t * (1 + 0.01)
        buy_signals = (future_prices > current_prices * (1 + self.buy_threshold_pct)).astype(int)
        
        # Sell: mean(pred_paths[-1]) < P_t * (1 - 0.005)
        sell_signals = (future_prices < current_prices * (1 + self.sell_threshold_pct)).astype(int)
        
        return pred_change, buy_signals, sell_signals
