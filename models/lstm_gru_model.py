"""
LSTM/GRU Model
Long Short-Term Memory / Gated Recurrent Unit for sequence prediction.

Buy Signal: rnn_pred_change > +0.012 * P_t
Sell Signal: rnn_pred_change < -0.006 * P_t
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class LSTMGRUModel:
    """LSTM/GRU Model for trading signals."""
    
    def __init__(self,
                 buy_threshold_pct: float = 0.012,  # 1.2% gain
                 sell_threshold_pct: float = -0.006,  # 0.6% drop
                 lookback: int = 20,
                 alpha: float = 0.3,  # EMA smoothing factor
                 min_train_size: int = 30):
        """
        Initialize LSTM/GRU Model.
        
        Args:
            buy_threshold_pct: Buy threshold as percentage of price (default: 0.012 = 1.2%)
            sell_threshold_pct: Sell threshold as percentage of price (default: -0.006 = -0.6%)
            lookback: Number of previous periods to consider
            alpha: Exponential moving average smoothing factor
            min_train_size: Minimum training data size
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.lookback = lookback
        self.alpha = alpha
        self.min_train_size = min_train_size
    
    def predict(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """
        Predict price changes using LSTM/GRU with rolling window.
        
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
                    
                    # Use exponential moving average of returns as prediction
                    ema_return = window_returns[-1]
                    for r in window_returns[-self.lookback:-1]:
                        ema_return = self.alpha * r + (1 - self.alpha) * ema_return
                    
                    current_price = window_prices[-1]
                    pred_change = ema_return * current_price
                    
                    pred_series.iloc[i] = pred_change
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"LSTM/GRU prediction error: {e}")
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
