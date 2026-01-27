"""
TCN (Temporal Convolutional Network) Model
Convolutional network for time series prediction.

Buy Signal: tcn_pred_change > +0.01 * P_t
Sell Signal: tcn_pred_change < -0.005 * P_t
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class TCNModel:
    """TCN Model for trading signals."""
    
    def __init__(self,
                 buy_threshold_pct: float = 0.003,  # 0.3% gain - more active trading
                 sell_threshold_pct: float = -0.002,  # 0.2% drop
                 lookback: int = 20,
                 decay_rate: float = 0.1,  # For exponential weighting
                 min_train_size: int = 30):
        """
        Initialize TCN Model.
        
        Args:
            buy_threshold_pct: Buy threshold as percentage of price (default: 0.01 = 1%)
            sell_threshold_pct: Sell threshold as percentage of price (default: -0.005 = -0.5%)
            lookback: Number of previous periods to consider
            decay_rate: Exponential decay rate for weighting
            min_train_size: Minimum training data size
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.lookback = lookback
        self.decay_rate = decay_rate
        self.min_train_size = min_train_size
    
    def predict(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """
        Predict price changes using TCN with rolling window.
        
        Args:
            data: DataFrame with price and feature data
            training_cols: List of column names to use for training (not used in simplified version)
            
        Returns:
            Series of predicted price changes
        """
        if len(data) < self.min_train_size:
            return pd.Series(0.0, index=data.index)
        
        try:
            prices = data['close'].values
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, use previous data to predict
            for i in range(self.min_train_size, len(data)):
                try:
                    window_prices = prices[:i+1]
                    window_returns = np.diff(window_prices) / window_prices[:-1]
                    
                    if len(window_returns) < self.lookback:
                        continue
                    
                    # Use weighted average of recent returns (simulating causal convolution)
                    weights = np.exp(-np.arange(self.lookback) * self.decay_rate)
                    weights = weights / weights.sum()
                    recent_returns = window_returns[-self.lookback:]
                    weighted_return = np.sum(recent_returns * weights)
                    
                    current_price = window_prices[-1]
                    pred_change = weighted_return * current_price
                    
                    pred_series.iloc[i] = pred_change
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"TCN prediction error: {e}")
            return pd.Series(0.0, index=data.index)
    
    def generate_signals(self, data: pd.DataFrame, training_cols: List[str]) -> tuple:
        """
        Generate buy/sell signals.
        
        Args:
            data: DataFrame with price and feature data
            training_cols: List of column names to use for training
            
        Returns:
            Tuple of (pred_change, buy_signals, sell_signals)
        """
        pred_change = self.predict(data, training_cols)
        current_prices = data['close']
        
        buy_signals = (pred_change > self.buy_threshold_pct * current_prices).astype(int)
        sell_signals = (pred_change < self.sell_threshold_pct * current_prices).astype(int)
        
        return pred_change, buy_signals, sell_signals
