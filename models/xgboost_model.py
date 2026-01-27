"""
XGBoost Model
Gradient boosting model for price prediction.

Buy Signal: pred_change > +0.01 * P_t (>1% predicted gain)
Sell Signal: pred_change < -0.005 * P_t (>0.5% predicted drop)
"""

import pandas as pd
import numpy as np
from typing import List, Optional

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostModel:
    """XGBoost Model for trading signals."""
    
    def __init__(self,
                 buy_threshold_pct: float = 0.003,  # 0.3% gain - more active trading
                 sell_threshold_pct: float = -0.002,  # 0.2% drop
                 n_estimators: int = 50,
                 max_depth: int = 5,
                 min_train_size: int = 50):
        """
        Initialize XGBoost Model.
        
        Args:
            buy_threshold_pct: Buy threshold as percentage of price (default: 0.01 = 1%)
            sell_threshold_pct: Sell threshold as percentage of price (default: -0.005 = -0.5%)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            min_train_size: Minimum training data size
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_train_size = min_train_size
    
    def predict(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """
        Predict price changes using rolling window.
        
        Args:
            data: DataFrame with price and feature data
            training_cols: List of column names to use for training
            
        Returns:
            Series of predicted price changes
        """
        if not XGBOOST_AVAILABLE:
            return pd.Series(0.0, index=data.index)
        
        if len(data) < self.min_train_size:
            return pd.Series(0.0, index=data.index)
        
        try:
            X = data[training_cols].fillna(0)
            y = data['price_change'].fillna(0)
            
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, train on previous data and predict
            for i in range(self.min_train_size, len(data)):
                try:
                    model = XGBRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=42,
                        verbosity=0
                    )
                    model.fit(X.iloc[:i], y.iloc[:i])
                    pred = model.predict(X.iloc[[i]])
                    pred_series.iloc[i] = pred[0]
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"XGBoost prediction error: {e}")
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
