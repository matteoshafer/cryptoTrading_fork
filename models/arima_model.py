"""
ARIMA/SARIMA Model
Time series forecasting model.

Buy Signal: fcst_mean[t+1] > P_t * (1 + 0.008)
Sell Signal: fcst_mean[t+1] < P_t * (1 - 0.008)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ARIMAModel:
    """ARIMA Model for trading signals."""
    
    def __init__(self,
                 buy_threshold_pct: float = 0.008,  # 0.8% gain
                 sell_threshold_pct: float = -0.008,  # 0.8% drop
                 order: Tuple[int, int, int] = (1, 1, 1),
                 min_train_size: int = 30):
        """
        Initialize ARIMA Model.
        
        Args:
            buy_threshold_pct: Buy threshold as percentage (default: 0.008 = 0.8%)
            sell_threshold_pct: Sell threshold as percentage (default: -0.008 = -0.8%)
            order: ARIMA order (p, d, q)
            min_train_size: Minimum training data size
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.order = order
        self.min_train_size = min_train_size
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict price changes using ARIMA with rolling window.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series of predicted price changes
        """
        if not STATSMODELS_AVAILABLE:
            return pd.Series(0.0, index=data.index)
        
        if len(data) < self.min_train_size or 'close' not in data.columns:
            return pd.Series(0.0, index=data.index)
        
        try:
            prices = data['close'].values
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, train on previous data and predict
            for i in range(self.min_train_size, len(data)):
                try:
                    window_prices = prices[:i+1]
                    # Fit ARIMA model
                    model = ARIMA(window_prices[:-1], order=self.order)
                    fitted_model = model.fit(disp=0)
                    
                    # Forecast next value
                    forecast = fitted_model.forecast(steps=1)
                    current_price = window_prices[-1]
                    pred_change = forecast[0] - current_price
                    
                    pred_series.iloc[i] = pred_change
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"ARIMA prediction error: {e}")
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
        
        # Buy: fcst_mean[t+1] > P_t * (1 + 0.008)
        buy_signals = (future_prices > current_prices * (1 + self.buy_threshold_pct)).astype(int)
        
        # Sell: fcst_mean[t+1] < P_t * (1 - 0.008)
        sell_signals = (future_prices < current_prices * (1 + self.sell_threshold_pct)).astype(int)
        
        return pred_change, buy_signals, sell_signals
