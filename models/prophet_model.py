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
except Exception:
    PROPHET_AVAILABLE = False


class ProphetModel:
    """Prophet Model for trading signals."""
    
    def __init__(self,
                 yearly_seasonality: bool = False,
                 daily_seasonality: bool = False,
                 min_train_size: int = 30,
                 retrain_interval: int = 10):
        """
        Initialize Prophet Model.

        Args:
            yearly_seasonality: Enable yearly seasonality
            daily_seasonality: Enable daily seasonality
            min_train_size: Minimum training data size
            retrain_interval: Refit Prophet every N bars; in between, use the
                cached fit's multi-step forecast for the current horizon
                (walk-forward safe: the cached fit only saw data up to its
                refit bar). Prophet fits are expensive, so this matters.
        """
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.min_train_size = min_train_size
        self.retrain_interval = max(1, retrain_interval)
    
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
            
            # Walk-forward with periodic refits. At refit bar f the model is
            # fit on all rows known at bar f and asked for the next
            # `retrain_interval` days in one shot; at a later bar i (before
            # the next refit) we read off the forecast row targeting bar i+1,
            # so every prediction uses only information available at the
            # refit bar.
            fitted_model = None
            future_fcst = None
            last_fit = -1
            for i in range(self.min_train_size, len(data)):
                try:
                    if fitted_model is None or (i - last_fit) >= self.retrain_interval:
                        prophet_df = pd.DataFrame({
                            'ds': time_col.iloc[:i + 1].values,
                            'y': data['close'].iloc[:i + 1].values
                        })

                        fitted_model = Prophet(
                            yearly_seasonality=self.yearly_seasonality,
                            daily_seasonality=self.daily_seasonality
                        )
                        fitted_model.fit(prophet_df)

                        future = fitted_model.make_future_dataframe(periods=self.retrain_interval)
                        future_fcst = fitted_model.predict(
                            future.tail(self.retrain_interval)
                        ).reset_index(drop=True)
                        last_fit = i

                    # Row (i - last_fit) of the future-only forecast targets
                    # bar i+1 (step i - last_fit + 1 after the refit bar).
                    fcst_row = future_fcst.iloc[i - last_fit]
                    current_price = data['close'].iloc[i]

                    pred_series.iloc[i] = fcst_row['yhat'] - current_price
                    lower_ci.iloc[i] = fcst_row['yhat_lower']
                    upper_ci.iloc[i] = fcst_row['yhat_upper']
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
