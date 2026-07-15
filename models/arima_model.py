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
except Exception:
    STATSMODELS_AVAILABLE = False


class ARIMAModel:
    """ARIMA Model for trading signals."""
    
    def __init__(self,
                 buy_threshold_pct: float = 0.008,  # 0.8% gain
                 sell_threshold_pct: float = -0.008,  # 0.8% drop
                 order: Tuple[int, int, int] = (1, 1, 1),
                 min_train_size: int = 30,
                 retrain_interval: int = 5):
        """
        Initialize ARIMA Model.

        Args:
            buy_threshold_pct: Buy threshold as percentage (default: 0.008 = 0.8%)
            sell_threshold_pct: Sell threshold as percentage (default: -0.008 = -0.8%)
            order: ARIMA order (p, d, q)
            min_train_size: Minimum training data size
            retrain_interval: Refit the model every N bars; in between, use the
                cached fit's multi-step forecast for the current horizon
                (walk-forward safe: the cached fit only saw data up to its
                refit bar). 1 = retrain every bar.
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.order = order
        self.min_train_size = min_train_size
        self.retrain_interval = max(1, retrain_interval)
    
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

            # Walk-forward with periodic refits. At refit bar f the model is
            # fit on prices[:f+1] (all data known at bar f); at a later bar i
            # (before the next refit) we take the (i - f + 1)-step-ahead
            # forecast so the prediction always targets bar i+1 using only
            # information available at the refit bar.
            # Note: statsmodels' ARIMAResults.fit() takes no `disp` argument —
            # the old `fit(disp=0)` call raised TypeError on every bar and was
            # swallowed by the except, silently zeroing all ARIMA signals.
            fitted_model = None
            last_fit = -1
            for i in range(self.min_train_size, len(data)):
                try:
                    if fitted_model is None or (i - last_fit) >= self.retrain_interval:
                        model = ARIMA(prices[:i + 1], order=self.order)
                        fitted_model = model.fit()
                        last_fit = i

                    steps = i - last_fit + 1
                    forecast = fitted_model.forecast(steps=steps)
                    pred_series.iloc[i] = forecast[-1] - prices[i]
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
