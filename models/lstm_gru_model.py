"""
LSTM/GRU Model
Recurrent neural network (PyTorch GRU) for next-bar return prediction,
trained walk-forward with periodic retraining.

If PyTorch is not installed, the model gracefully degrades to a simple
exponential-moving-average-of-returns heuristic (clearly NOT a neural net —
just a momentum baseline), consistent with the optional-dependency pattern
used by the other models.

Buy Signal: rnn_pred_change > +buy_threshold_pct * P_t
Sell Signal: rnn_pred_change < sell_threshold_pct * P_t
"""

import pandas as pd
import numpy as np
from typing import List

from .torch_seq import TORCH_AVAILABLE, walk_forward_torch_predict

if TORCH_AVAILABLE:
    from torch import nn

    class _GRUNet(nn.Module):
        """Single-layer GRU + linear head over the last hidden state."""

        def __init__(self, hidden_size: int = 16):
            super().__init__()
            self.gru = nn.GRU(input_size=1, hidden_size=hidden_size,
                              batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])


class LSTMGRUModel:
    """LSTM/GRU Model for trading signals."""

    def __init__(self,
                 buy_threshold_pct: float = 0.003,  # 0.3% gain - more active trading
                 sell_threshold_pct: float = -0.002,  # 0.2% drop
                 lookback: int = 20,
                 alpha: float = 0.3,  # EMA smoothing factor (heuristic fallback only)
                 min_train_size: int = 30,
                 hidden_size: int = 16,
                 retrain_interval: int = 20,
                 epochs: int = 25):
        """
        Initialize LSTM/GRU Model.

        Args:
            buy_threshold_pct: Buy threshold as percentage of price
            sell_threshold_pct: Sell threshold as percentage of price
            lookback: Number of previous returns fed to the network
            alpha: EMA smoothing factor for the no-torch heuristic fallback
            min_train_size: Minimum training data size
            hidden_size: GRU hidden state size
            retrain_interval: Refit the network every N bars (walk-forward safe)
            epochs: Training epochs per refit
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.lookback = lookback
        self.alpha = alpha
        self.min_train_size = min_train_size
        self.hidden_size = hidden_size
        self.retrain_interval = retrain_interval
        self.epochs = epochs

    def predict(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """
        Predict next-bar price changes walk-forward.

        Args:
            data: DataFrame with price and feature data
            training_cols: Unused (the RNN consumes the raw return sequence)

        Returns:
            Series of predicted price changes
        """
        if len(data) < self.min_train_size:
            return pd.Series(0.0, index=data.index)

        try:
            prices = data['close'].values.astype(float)
            if TORCH_AVAILABLE:
                preds = walk_forward_torch_predict(
                    prices,
                    build_model=lambda: _GRUNet(self.hidden_size),
                    lookback=self.lookback,
                    min_train_size=self.min_train_size,
                    retrain_interval=self.retrain_interval,
                    epochs=self.epochs
                )
                return pd.Series(preds, index=data.index)
            return self._predict_ema_fallback(prices, data.index)
        except Exception as e:
            print(f"LSTM/GRU prediction error: {e}")
            return pd.Series(0.0, index=data.index)

    def _predict_ema_fallback(self, prices: np.ndarray, index) -> pd.Series:
        """
        Heuristic fallback used only when PyTorch is unavailable: an
        exponential moving average of recent returns. This is a momentum
        baseline, not a neural network.
        """
        pred_series = pd.Series(0.0, index=index)

        for i in range(self.min_train_size, len(prices)):
            try:
                window_prices = prices[:i + 1]
                window_returns = np.diff(window_prices) / window_prices[:-1]

                if len(window_returns) < self.lookback:
                    continue

                ema_return = window_returns[-1]
                for r in window_returns[-self.lookback:-1]:
                    ema_return = self.alpha * r + (1 - self.alpha) * ema_return

                pred_series.iloc[i] = ema_return * window_prices[-1]
            except Exception:
                pass

        return pred_series

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
