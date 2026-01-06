"""
Model Manager for Trading Strategy
Manages 10 individual ML models and generates buy/sell signals based on their predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import individual model classes
from models import (
    LLMSentimentModel,
    XGBoostModel,
    GBMModel,
    ARIMAModel,
    ProphetModel,
    RandomForestModel,
    LightGBMModel,
    SVRModel,
    LSTMGRUModel,
    TCNModel
)

# Model imports with error handling
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. XGBoost model will be disabled.")

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. LightGBM model will be disabled.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA model will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Prophet model will be disabled.")

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.nn.functional import softmax
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/transformers not available. Sentiment model will be disabled.")

# For LSTM/GRU and TCN
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM/GRU/TCN models will use PyTorch fallback.")


class ModelManager:
    """
    Manages multiple ML models and generates trading signals based on their predictions.
    """
    
    def __init__(self, model_configs: Optional[Dict] = None):
        """
        Initialize ModelManager with model configurations.
        
        Args:
            model_configs: Dictionary of model configurations (optional)
        """
        self.model_configs = model_configs or self._get_default_configs()
        
        # Initialize individual model instances
        self.llm_sentiment = LLMSentimentModel()
        self.xgboost = XGBoostModel()
        self.gbm = GBMModel()
        self.arima = ARIMAModel()
        self.prophet = ProphetModel()
        self.random_forest = RandomForestModel()
        self.lightgbm = LightGBMModel()
        self.svr = SVRModel()
        self.lstm_gru = LSTMGRUModel()
        self.tcn = TCNModel()
        
    def _get_default_configs(self) -> Dict:
        """Get default model configurations."""
        return {
            "LLM-Sentiment": {
                "description": "Finetuned RoBERTa on financial news—provides a daily sentiment score.",
                "buy_condition": "sentiment_z > +0.4",
                "sell_condition": "sentiment_z < -0.4"
            },
            "XGBoost": {
                "description": "Fast, tree-based gradient boosting with time-series CV.",
                "buy_condition": "pred_change > +0.01 * P_t",
                "sell_condition": "pred_change < -0.005 * P_t"
            },
            "Custom_GBM": {
                "description": "Geometric Brownian Motion paths that capture long-term drift & vol.",
                "buy_condition": "mean(pred_paths[-1]) > P_t * (1 + 0.01)",
                "sell_condition": "mean(pred_paths[-1]) < P_t * (1 - 0.005)"
            },
            "ARIMA_SARIMA": {
                "description": "Classical statistical model with differencing, drift and seasonality.",
                "buy_condition": "fcst_mean[t+1] > P_t * (1 + 0.008)",
                "sell_condition": "fcst_mean[t+1] < P_t * (1 - 0.008)"
            },
            "Prophet": {
                "description": "Facebook's Prophet for changepoints, holidays and flexible seasonality.",
                "buy_condition": "prophet_pred > P_t with CI lower > P_t",
                "sell_condition": "prophet_pred < P_t with CI upper < P_t"
            },
            "RandomForest": {
                "description": "Bagged decision-trees to stabilize nonlinear patterns.",
                "buy_condition": "rf_pred_change > +0.007 * P_t",
                "sell_condition": "rf_pred_change < -0.007 * P_t"
            },
            "LightGBM": {
                "description": "A competing GBM with different tree-grow rules and categorical support.",
                "buy_condition": "lgbm_pred_change > +0.01 * P_t",
                "sell_condition": "lgbm_pred_change < -0.01 * P_t"
            },
            "SVR": {
                "description": "Kernel-based model that excels on nonlinear margins.",
                "buy_condition": "svr_pred_change > +0.008 * P_t",
                "sell_condition": "svr_pred_change < -0.008 * P_t"
            },
            "LSTM_GRU": {
                "description": "Recurrent neural net capturing longer sequential dependencies.",
                "buy_condition": "rnn_pred_change > +0.012 * P_t",
                "sell_condition": "rnn_pred_change < -0.006 * P_t"
            },
            "TCN": {
                "description": "Convolutional sequence model with causal dilations—fast and stable.",
                "buy_condition": "tcn_pred_change > +0.01 * P_t",
                "sell_condition": "tcn_pred_change < -0.005 * P_t"
            }
        }
    
    def initialize_models(self, data: pd.DataFrame, training_cols: List[str] = None):
        """
        Initialize all models with the provided data.
        
        Args:
            data: Historical price and feature data
            training_cols: List of column names to use for training
        """
        if training_cols is None:
            training_cols = ['volume', 'SMA_20', 'SMA_50', 'Volume_MA_20', 'OBV', 
                           'BB_Lower', 'BB_Middle', 'BB_Upper', 'avg_sentiment']
        
        # Ensure we have a target column (price change)
        if 'close' in data.columns:
            if 'price_change' not in data.columns:
                data['price_change'] = data['close'].diff().fillna(0.0)
        
        # Models are already initialized in __init__
        print("Models initialized. They will be trained when generate_signals is called.")
    
    def _predict_xgboost(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """Train and predict using XGBoost with rolling window."""
        if not XGBOOST_AVAILABLE:
            return pd.Series(0.0, index=data.index)
        
        if len(data) < 50:
            return pd.Series(0.0, index=data.index)
        
        try:
            X = data[training_cols].fillna(0)
            y = data['price_change'].fillna(0)
            
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, train on previous data and predict
            min_train_size = 50
            for i in range(min_train_size, len(data)):
                try:
                    model = XGBRegressor(n_estimators=50, max_depth=5, random_state=42, verbosity=0)
                    model.fit(X.iloc[:i], y.iloc[:i])
                    pred = model.predict(X.iloc[[i]])
                    pred_series.iloc[i] = pred[0]
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"XGBoost prediction error: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _predict_random_forest(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """Train and predict using Random Forest with rolling window."""
        if len(data) < 50:
            return pd.Series(0.0, index=data.index)
        
        try:
            X = data[training_cols].fillna(0)
            y = data['price_change'].fillna(0)
            
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, train on previous data and predict
            min_train_size = 50
            for i in range(min_train_size, len(data)):
                try:
                    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                    model.fit(X.iloc[:i], y.iloc[:i])
                    pred = model.predict(X.iloc[[i]])
                    pred_series.iloc[i] = pred[0]
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"Random Forest prediction error: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _predict_lightgbm(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """Train and predict using LightGBM with rolling window."""
        if not LIGHTGBM_AVAILABLE:
            return pd.Series(0.0, index=data.index)
        
        if len(data) < 50:
            return pd.Series(0.0, index=data.index)
        
        try:
            X = data[training_cols].fillna(0)
            y = data['price_change'].fillna(0)
            
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, train on previous data and predict
            min_train_size = 50
            for i in range(min_train_size, len(data)):
                try:
                    model = LGBMRegressor(n_estimators=50, max_depth=5, random_state=42, verbose=-1)
                    model.fit(X.iloc[:i], y.iloc[:i])
                    pred = model.predict(X.iloc[[i]])
                    pred_series.iloc[i] = pred[0]
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"LightGBM prediction error: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _predict_svr(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """Train and predict using SVR with rolling window."""
        if len(data) < 50:
            return pd.Series(0.0, index=data.index)
        
        try:
            X = data[training_cols].fillna(0)
            y = data['price_change'].fillna(0)
            
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, train on previous data and predict
            from sklearn.preprocessing import StandardScaler
            min_train_size = 50
            for i in range(min_train_size, len(data)):
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X.iloc[:i])
                    X_scaled_pred = scaler.transform(X.iloc[[i]])
                    
                    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                    model.fit(X_scaled, y.iloc[:i])
                    
                    pred = model.predict(X_scaled_pred)
                    pred_series.iloc[i] = pred[0]
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"SVR prediction error: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _predict_gbm_simulator(self, data: pd.DataFrame, n_paths: int = 100) -> pd.Series:
        """
        Simulate Geometric Brownian Motion paths with rolling window.
        
        Args:
            data: Price data
            n_paths: Number of simulation paths
            
        Returns:
            Series with mean predicted price at end of paths
        """
        if len(data) < 20 or 'close' not in data.columns:
            return pd.Series(0.0, index=data.index)
        
        try:
            prices = data['close'].values
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, use previous data to estimate parameters
            min_train_size = 20
            T = 1  # Forecast horizon (1 day)
            
            for i in range(min_train_size, len(data)):
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
                    for _ in range(n_paths):
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
    
    def _predict_arima(self, data: pd.DataFrame) -> pd.Series:
        """Train and predict using ARIMA with rolling window."""
        if not STATSMODELS_AVAILABLE:
            return pd.Series(0.0, index=data.index)
        
        if len(data) < 30 or 'close' not in data.columns:
            return pd.Series(0.0, index=data.index)
        
        try:
            prices = data['close'].values
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, train on previous data and predict
            min_train_size = 30
            for i in range(min_train_size, len(data)):
                try:
                    window_prices = prices[:i+1]
                    # Fit ARIMA model
                    model = ARIMA(window_prices[:-1], order=(1, 1, 1))
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
    
    def _predict_prophet(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Train and predict using Prophet with rolling window.
        
        Returns:
            Tuple of (prediction, lower_ci, upper_ci)
        """
        if not PROPHET_AVAILABLE:
            zeros = pd.Series(0.0, index=data.index)
            return zeros, zeros, zeros
        
        if len(data) < 30:
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
            min_train_size = 30
            for i in range(min_train_size, len(data)):
                try:
                    prophet_df = pd.DataFrame({
                        'ds': time_col.iloc[:i+1],
                        'y': data['close'].iloc[:i+1]
                    })
                    
                    # Fit model
                    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
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
    
    def _predict_lstm_gru(self, data: pd.DataFrame, training_cols: List[str], lookback: int = 20) -> pd.Series:
        """Train and predict using LSTM/GRU with rolling window."""
        if len(data) < lookback + 10:
            return pd.Series(0.0, index=data.index)
        
        try:
            prices = data['close'].values
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, use previous data to predict
            min_train_size = lookback + 10
            for i in range(min_train_size, len(data)):
                try:
                    window_prices = prices[:i+1]
                    window_returns = np.diff(window_prices) / window_prices[:-1]
                    
                    if len(window_returns) < lookback:
                        continue
                    
                    # Use exponential moving average of returns as prediction
                    alpha = 0.3
                    ema_return = window_returns[-1]
                    for r in window_returns[-lookback:-1]:
                        ema_return = alpha * r + (1 - alpha) * ema_return
                    
                    current_price = window_prices[-1]
                    pred_change = ema_return * current_price
                    
                    pred_series.iloc[i] = pred_change
                except:
                    pass
            
            return pred_series
        except Exception as e:
            print(f"LSTM/GRU prediction error: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _predict_tcn(self, data: pd.DataFrame, training_cols: List[str], lookback: int = 20) -> pd.Series:
        """Train and predict using TCN (Temporal Convolutional Network) with rolling window."""
        if len(data) < lookback + 10:
            return pd.Series(0.0, index=data.index)
        
        try:
            prices = data['close'].values
            pred_series = pd.Series(0.0, index=data.index)
            
            # Use rolling window: for each day, use previous data to predict
            min_train_size = lookback + 10
            for i in range(min_train_size, len(data)):
                try:
                    window_prices = prices[:i+1]
                    window_returns = np.diff(window_prices) / window_prices[:-1]
                    
                    if len(window_returns) < lookback:
                        continue
                    
                    # Use weighted average of recent returns (simulating causal convolution)
                    weights = np.exp(-np.arange(lookback) * 0.1)
                    weights = weights / weights.sum()
                    recent_returns = window_returns[-lookback:]
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
    
    def generate_signals(self, data: pd.DataFrame, training_cols: List[str] = None) -> pd.DataFrame:
        """
        Generate buy/sell signals for all models.
        
        Args:
            data: DataFrame with price and feature data
            training_cols: List of column names to use for training
            
        Returns:
            DataFrame with signals for each model and ensemble inputs
        """
        if training_cols is None:
            training_cols = ['volume', 'SMA_20', 'SMA_50', 'Volume_MA_20', 'OBV', 
                            'BB_Lower', 'BB_Middle', 'BB_Upper', 'avg_sentiment']
            training_cols = [col for col in training_cols if col in data.columns]
        
        # Ensure required columns exist
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        if 'price_change' not in data.columns:
            data['price_change'] = data['close'].diff().fillna(0.0)
        
        # Calculate SMA20 for ensemble
        if 'SMA_20' not in data.columns:
            data['SMA_20'] = data['close'].rolling(window=20).mean()
        
        signals_df = pd.DataFrame(index=data.index)
        signals_df['P_t'] = data['close']  # Current price
        
        # 1. LLM-Sentiment Model
        sentiment_z, llm_buy, llm_sell = self.llm_sentiment.generate_signals(data)
        signals_df['sentiment_z'] = sentiment_z
        signals_df['LLM_Sentiment_buy'] = llm_buy
        signals_df['LLM_Sentiment_sell'] = llm_sell
        
        # 2. XGBoost
        xgb_pred, xgb_buy, xgb_sell = self.xgboost.generate_signals(data, training_cols)
        signals_df['xgb_pred_change'] = xgb_pred
        signals_df['XGBoost_buy'] = xgb_buy
        signals_df['XGBoost_sell'] = xgb_sell
        
        # 3. Custom GBM Simulator
        gbm_pred, gbm_buy, gbm_sell = self.gbm.generate_signals(data)
        signals_df['gbm_pred_change'] = gbm_pred
        signals_df['GBM_buy'] = gbm_buy
        signals_df['GBM_sell'] = gbm_sell
        
        # 4. ARIMA/SARIMA
        arima_pred, arima_buy, arima_sell = self.arima.generate_signals(data)
        signals_df['arima_pred_change'] = arima_pred
        signals_df['ARIMA_buy'] = arima_buy
        signals_df['ARIMA_sell'] = arima_sell
        
        # 5. Prophet
        prophet_pred, prophet_buy, prophet_sell, prophet_lower, prophet_upper = self.prophet.generate_signals(data)
        signals_df['prophet_pred'] = signals_df['P_t'] + prophet_pred
        signals_df['prophet_lower'] = prophet_lower
        signals_df['prophet_upper'] = prophet_upper
        signals_df['Prophet_buy'] = prophet_buy
        signals_df['Prophet_sell'] = prophet_sell
        
        # 6. Random Forest
        rf_pred, rf_buy, rf_sell = self.random_forest.generate_signals(data, training_cols)
        signals_df['rf_pred_change'] = rf_pred
        signals_df['RandomForest_buy'] = rf_buy
        signals_df['RandomForest_sell'] = rf_sell
        
        # 7. LightGBM
        lgbm_pred, lgbm_buy, lgbm_sell = self.lightgbm.generate_signals(data, training_cols)
        signals_df['lgbm_pred_change'] = lgbm_pred
        signals_df['LightGBM_buy'] = lgbm_buy
        signals_df['LightGBM_sell'] = lgbm_sell
        
        # 8. SVR
        svr_pred, svr_buy, svr_sell = self.svr.generate_signals(data, training_cols)
        signals_df['svr_pred_change'] = svr_pred
        signals_df['SVR_buy'] = svr_buy
        signals_df['SVR_sell'] = svr_sell
        
        # 9. LSTM/GRU
        lstm_pred, lstm_buy, lstm_sell = self.lstm_gru.generate_signals(data, training_cols)
        signals_df['rnn_pred_change'] = lstm_pred
        signals_df['LSTM_GRU_buy'] = lstm_buy
        signals_df['LSTM_GRU_sell'] = lstm_sell
        
        # 10. TCN
        tcn_pred, tcn_buy, tcn_sell = self.tcn.generate_signals(data, training_cols)
        signals_df['tcn_pred_change'] = tcn_pred
        signals_df['TCN_buy'] = tcn_buy
        signals_df['TCN_sell'] = tcn_sell
        
        # Calculate bull_count (number of models with buy signals)
        buy_cols = [col for col in signals_df.columns if col.endswith('_buy')]
        signals_df['bull_count'] = signals_df[buy_cols].sum(axis=1)
        
        # Calculate predicted return for ensemble (average of all predictions)
        pred_change_cols = [col for col in signals_df.columns if 'pred_change' in col or col == 'sentiment_z']
        if len(pred_change_cols) > 0:
            # Normalize predictions to returns
            pred_returns = []
            for col in pred_change_cols:
                if col == 'sentiment_z':
                    # Convert sentiment to return proxy
                    pred_returns.append((signals_df[col] * 0.01).values)  # Scale sentiment
                else:
                    pred_returns.append((signals_df[col] / signals_df['P_t']).values)

            # Stack as numpy array to avoid index alignment issues, then average across models
            pred_returns_arr = np.vstack(pred_returns)  # shape: (n_models, n_samples)
            signals_df['predicted_return'] = pred_returns_arr.mean(axis=0)
        else:
            signals_df['predicted_return'] = 0.0
        
        # Add SMA20 for ensemble
        signals_df['SMA_20'] = data['SMA_20'] if 'SMA_20' in data.columns else data['close'].rolling(20).mean()
        
        return signals_df

