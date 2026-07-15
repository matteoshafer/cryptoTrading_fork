"""
LightGBM Model
Gradient boosting framework for price prediction.

Buy Signal: lgbm_pred_change > +0.01 * P_t
Sell Signal: lgbm_pred_change < -0.01 * P_t
"""

import pandas as pd
import numpy as np
from typing import List, Optional

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    # Broad except: same rationale as xgboost_model.py — a broken native
    # library at import time isn't always an ImportError.
    LIGHTGBM_AVAILABLE = False


class LightGBMModel:
    """LightGBM Model for trading signals."""
    
    def __init__(self,
                 buy_threshold_pct: float = 0.003,  # 0.3% gain - more active trading
                 sell_threshold_pct: float = -0.003,  # 0.3% drop
                 n_estimators: int = 50,
                 max_depth: int = 5,
                 min_train_size: int = 50,
                 retrain_interval: int = 5):
        """
        Initialize LightGBM Model.

        Args:
            buy_threshold_pct: Buy threshold as percentage of price (default: 0.01 = 1%)
            sell_threshold_pct: Sell threshold as percentage of price (default: -0.01 = -1%)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            min_train_size: Minimum training data size
            retrain_interval: Refit the model every N bars and reuse it for the
                bars in between (walk-forward safe: the cached model was trained
                only on data prior to its refit bar). 1 = retrain every bar.
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_train_size = min_train_size
        self.retrain_interval = max(1, retrain_interval)
    
    def predict(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """
        Predict price changes using LightGBM with rolling window.
        
        Args:
            data: DataFrame with price and feature data
            training_cols: List of column names to use for training
            
        Returns:
            Series of predicted price changes
        """
        if not LIGHTGBM_AVAILABLE:
            return pd.Series(0.0, index=data.index)
        
        if len(data) < self.min_train_size:
            return pd.Series(0.0, index=data.index)
        
        try:
            X = data[training_cols].fillna(0)
            # Forward-shifted target: predict the *next* bar's change, not the
            # already-realized change baked into this bar's own SMA/BB features.
            y = data['close'].diff().shift(-1).fillna(0)
            
            pred_series = pd.Series(0.0, index=data.index)

            # Walk-forward: refit every `retrain_interval` bars on data up to
            # that bar, reuse the fitted model for the bars in between.
            model = None
            last_fit = -1
            for i in range(self.min_train_size, len(data)):
                try:
                    if model is None or (i - last_fit) >= self.retrain_interval:
                        model = LGBMRegressor(
                            n_estimators=self.n_estimators,
                            max_depth=self.max_depth,
                            random_state=42,
                            verbose=-1
                        )
                        model.fit(X.iloc[:i], y.iloc[:i])
                        last_fit = i
                    pred = model.predict(X.iloc[[i]])
                    pred_series.iloc[i] = pred[0]
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"LightGBM prediction error: {e}")
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
