"""
Model Manager for Trading Strategy
Manages 10 individual ML models and generates buy/sell signals based on their predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import individual model classes. Each one handles its own optional-dependency
# fallback internally (see models/*.py), so ModelManager doesn't need to
# duplicate those imports or availability checks.
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
                "description": "PyTorch GRU trained walk-forward on return sequences (falls back to an EMA momentum heuristic if torch is not installed).",
                "buy_condition": "rnn_pred_change > +0.012 * P_t",
                "sell_condition": "rnn_pred_change < -0.006 * P_t"
            },
            "TCN": {
                "description": "PyTorch causal dilated conv net trained walk-forward on return sequences (falls back to a weighted-average heuristic if torch is not installed).",
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
        signals_df['prophet_pred_change'] = prophet_pred
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
        
        # Calculate predicted return for ensemble (average of all predictions).
        # 'prophet_pred_change' is included here, so Prophet participates in
        # the average alongside the other nine models.
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
            pred_returns_arr = np.zeros((0, len(signals_df)))
            signals_df['predicted_return'] = 0.0

        # Add SMA20 for ensemble
        signals_df['SMA_20'] = data['SMA_20'] if 'SMA_20' in data.columns else data['close'].rolling(20).mean()

        # Ensemble confidence score (0-100). Pure statistics computed offline
        # from the models' own outputs — no network calls and no LLM inference
        # anywhere in this path.
        self._add_confidence(signals_df, pred_returns_arr)

        return signals_df

    # Rolling window (bars) used for each model's walk-forward hit rate.
    HIT_RATE_WINDOW = 30

    def _add_confidence(self, signals_df: pd.DataFrame, pred_matrix: np.ndarray):
        """
        Attach a quantitative confidence score to every bar.

        The score blends four leak-free, LLM-free components, each mapped to
        [0, 1] and computed relative to *active* models only (a model is
        active if it produced any nonzero prediction this run — models whose
        optional dependency is missing emit all zeros and are excluded):

        1. Directional agreement (35%): fraction of active models whose
           predicted return shares the majority sign, rescaled so a 50/50
           split scores 0 and unanimity scores 1.
        2. Signal-to-noise (25%): |mean| / std of the cross-model predicted
           returns — a strong consensus magnitude relative to model
           disagreement scores high.
        3. Recent reliability (30%): each active model's rolling hit rate over
           the last HIT_RATE_WINDOW resolved calls (did the sign of its
           prediction at bar t match the realized t -> t+1 move?), averaged
           across models. Outcomes are shifted one bar so only already
           realized moves are used — no lookahead.
        4. Prophet interval width (10%, only when Prophet is active): a narrow
           forecast interval relative to price means the statistical forecast
           is tight; a wide one means high uncertainty.

        Adds columns: active_models, confidence (0-100), confidence_bucket
        (LOW < 40 <= MEDIUM < 70 <= HIGH).
        """
        n = len(signals_df)

        active_mask = np.array([np.any(pred_matrix[m] != 0)
                                for m in range(pred_matrix.shape[0])], dtype=bool)
        n_active = int(active_mask.sum())
        signals_df['active_models'] = n_active

        if n_active == 0 or n == 0:
            signals_df['confidence'] = 0.0
            signals_df['confidence_bucket'] = 'LOW'
            return

        preds = pred_matrix[active_mask]  # (n_active, n)

        # 1) Directional agreement among active models
        n_pos = (preds > 0).sum(axis=0)
        n_neg = (preds < 0).sum(axis=0)
        agreement_frac = np.maximum(n_pos, n_neg) / n_active
        c_agree = np.clip((agreement_frac - 0.5) * 2.0, 0.0, 1.0)

        # 2) Signal-to-noise of the cross-model prediction spread
        if n_active >= 2:
            snr = np.abs(preds.mean(axis=0)) / (preds.std(axis=0) + 1e-9)
            c_snr = snr / (1.0 + snr)
        else:
            c_snr = np.full(n, 0.5)  # dispersion undefined with one model

        # 3) Rolling walk-forward hit rate per active model
        price = signals_df['P_t'].values.astype(float)
        next_move = np.full(n, np.nan)
        next_move[:-1] = price[1:] - price[:-1]

        hit_rates = np.full((preds.shape[0], n), np.nan)
        for m in range(preds.shape[0]):
            called = preds[m] != 0
            correct = (np.sign(preds[m]) == np.sign(next_move)).astype(float)
            calls = pd.Series(np.where(called, correct, np.nan))
            # A call at bar t resolves at bar t+1, so shift by one bar before
            # rolling: the rate at bar t only reflects already-settled calls.
            hit_rates[m] = calls.shift(1).rolling(
                self.HIT_RATE_WINDOW, min_periods=5).mean().values

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # all-NaN warm-up columns
            avg_hit_rate = np.nanmean(hit_rates, axis=0)
        # Map: <= 40% hit rate -> 0, >= 65% -> 1; NaN (warm-up) -> neutral 0.5
        c_hit = np.clip((avg_hit_rate - 0.40) / 0.25, 0.0, 1.0)
        c_hit = np.where(np.isnan(avg_hit_rate), 0.5, c_hit)

        # 4) Prophet forecast-interval width (only when Prophet produced CIs)
        c_prophet = None
        if 'prophet_lower' in signals_df.columns and 'prophet_upper' in signals_df.columns:
            lower = signals_df['prophet_lower'].values.astype(float)
            upper = signals_df['prophet_upper'].values.astype(float)
            has_ci = (lower != 0) | (upper != 0)
            if has_ci.any():
                rel_width = np.where(
                    has_ci, (upper - lower) / np.maximum(np.abs(price), 1e-9), np.nan)
                # 0% wide -> 1.0 confidence, >= 10% of price wide -> 0.0
                c_prophet = np.clip(1.0 - rel_width / 0.10, 0.0, 1.0)
                c_prophet = np.where(np.isnan(rel_width), 0.5, c_prophet)

        components = [(0.35, c_agree), (0.25, c_snr), (0.30, c_hit)]
        if c_prophet is not None:
            components.append((0.10, c_prophet))
        total_weight = sum(w for w, _ in components)
        confidence = sum(w * c for w, c in components) / total_weight * 100.0

        signals_df['confidence'] = np.round(confidence, 1)
        signals_df['confidence_bucket'] = pd.cut(
            confidence,
            bins=[-0.1, 40.0, 70.0, 100.1],
            labels=['LOW', 'MEDIUM', 'HIGH']
        ).astype(str)

