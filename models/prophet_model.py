"""
Prophet Model
Facebook Prophet time series forecasting.

Buy Signal: prophet_pred > P_t with CI lower > P_t
Sell Signal: prophet_pred < P_t with CI upper < P_t
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetModel:
    """Prophet Model for trading signals."""
    
    def __init__(self,
                 yearly_seasonality: bool = False,
                 daily_seasonality: bool = False,
                 min_train_size: int = 30):
        """
        Initialize Prophet Model.
        
        Args:
            yearly_seasonality: Enable yearly seasonality
            daily_seasonality: Enable daily seasonality
            min_train_size: Minimum training data size
        """
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.min_train_size = min_train_size
    
    def predict(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Predict price changes using Prophet with rolling window.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (prediction, lower_ci, upper_ci)
        """
        if not PROPHET_AVAILABLE:
            zeros = pd.Series(0.0, index=data.index)
            return zeros, zeros, zeros
        
        if len(data) < self.min_train_size:
            zeros = pd.Series(0.0, index=data.index)
            return zeros, zeros, zeros
        
        try:
            # Prepare data for Prophet
            if 'time' in data.columns:
                time_col = pd.to_datetime(data['time'])
            else:
                time_col = pd.to_datetime(data.index)
            
            pred_series = pd.Series(0.0, index=data.index)
            lower_ci = pd.Series(0.0, index=data.index)
            upper_ci = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, train on previous data and predict
            for i in range(self.min_train_size, len(data)):
                try:
                    prophet_df = pd.DataFrame({
                        'ds': time_col.iloc[:i+1],
                        'y': data['close'].iloc[:i+1]
                    })
                    
                    # Fit model
                    model = Prophet(
                        yearly_seasonality=self.yearly_seasonality,
                        daily_seasonality=self.daily_seasonality
                    )
                    model.fit(prophet_df.iloc[:-1])
                    
                    # Forecast
                    future = model.make_future_dataframe(periods=1)
                    forecast = model.predict(future)
                    
                    # Get last prediction
                    last_pred = forecast.iloc[-1]
                    current_price = data['close'].iloc[i]
                    
                    pred_series.iloc[i] = last_pred['yhat'] - current_price
                    lower_ci.iloc[i] = last_pred['yhat_lower']
                    upper_ci.iloc[i] = last_pred['yhat_upper']
                except:
                    pass
            
            return pred_series, lower_ci, upper_ci
        except Exception as e:
            print(f"Prophet prediction error: {e}")
            zeros = pd.Series(0.0, index=data.index)
            return zeros, zeros, zeros
    
    def generate_signals(self, data: pd.DataFrame) -> tuple:
        """
        Generate buy/sell signals.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (pred_change, buy_signals, sell_signals, lower_ci, upper_ci)
        """
        pred_change, lower_ci, upper_ci = self.predict(data)
        current_prices = data['close']
        
        # Calculate predicted prices
        prophet_pred = current_prices + pred_change
        
        # Buy: prophet_pred > P_t with CI lower > P_t
        buy_signals = ((prophet_pred > current_prices) & (lower_ci > current_prices)).astype(int)
        
        # Sell: prophet_pred < P_t with CI upper < P_t
        sell_signals = ((prophet_pred < current_prices) & (upper_ci < current_prices)).astype(int)
        
        return pred_change, buy_signals, sell_signals, lower_ci, upper_ci
