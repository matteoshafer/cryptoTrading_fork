"""
SVR (Support Vector Regression) Model
Support vector machine for price prediction.

Buy Signal: svr_pred_change > +0.008 * P_t
Sell Signal: svr_pred_change < -0.008 * P_t
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class SVRModel:
    """SVR Model for trading signals."""
    
    def __init__(self,
                 buy_threshold_pct: float = 0.008,  # 0.8% gain
                 sell_threshold_pct: float = -0.008,  # 0.8% drop
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 min_train_size: int = 50):
        """
        Initialize SVR Model.
        
        Args:
            buy_threshold_pct: Buy threshold as percentage of price (default: 0.008 = 0.8%)
            sell_threshold_pct: Sell threshold as percentage of price (default: -0.008 = -0.8%)
            kernel: SVR kernel type
            C: Regularization parameter
            epsilon: Epsilon parameter
            min_train_size: Minimum training data size
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.min_train_size = min_train_size
    
    def predict(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """
        Predict price changes using SVR with rolling window.
        
        Args:
            data: DataFrame with price and feature data
            training_cols: List of column names to use for training
            
        Returns:
            Series of predicted price changes
        """
        if len(data) < self.min_train_size:
            return pd.Series(0.0, index=data.index)
        
        try:
            X = data[training_cols].fillna(0)
            y = data['price_change'].fillna(0)
            
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, train on previous data and predict
            for i in range(self.min_train_size, len(data)):
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X.iloc[:i])
                    X_scaled_pred = scaler.transform(X.iloc[[i]])
                    
                    model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
                    model.fit(X_scaled, y.iloc[:i])
                    
                    pred = model.predict(X_scaled_pred)
                    pred_series.iloc[i] = pred[0]
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"SVR prediction error: {e}")
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
