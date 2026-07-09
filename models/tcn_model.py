"""
TCN (Temporal Convolutional Network) Model
Causal dilated 1-D convolutional network (PyTorch) for next-bar return
prediction, trained walk-forward with periodic retraining.

If PyTorch is not installed, the model gracefully degrades to an
exponentially-weighted average of recent returns (clearly NOT a neural net —
just a smoothing baseline), consistent with the optional-dependency pattern
used by the other models.

Buy Signal: tcn_pred_change > +buy_threshold_pct * P_t
Sell Signal: tcn_pred_change < sell_threshold_pct * P_t
"""

import pandas as pd
import numpy as np
from typing import List

from .torch_seq import TORCH_AVAILABLE, walk_forward_torch_predict

if TORCH_AVAILABLE:
    import torch.nn.functional as F
    from torch import nn

    class _TCNNet(nn.Module):
        """Two causal dilated conv layers (dilations 1 and 2) + linear head."""

        def __init__(self, channels: int = 16, kernel_size: int = 3):
            super().__init__()
            self.conv1 = nn.Conv1d(1, channels, kernel_size, dilation=1)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=2)
            self.head = nn.Linear(channels, 1)
            self.pad1 = (kernel_size - 1) * 1
            self.pad2 = (kernel_size - 1) * 2

        def forward(self, x):
            # x: (batch, seq, 1) -> (batch, channels, seq); left-pad so the
            # convolutions stay causal (no future timesteps leak backwards).
            h = x.transpose(1, 2)
            h = F.relu(self.conv1(F.pad(h, (self.pad1, 0))))
            h = F.relu(self.conv2(F.pad(h, (self.pad2, 0))))
            return self.head(h[:, :, -1])


class TCNModel:
    """TCN Model for trading signals."""

    def __init__(self,
                 buy_threshold_pct: float = 0.003,  # 0.3% gain - more active trading
                 sell_threshold_pct: float = -0.002,  # 0.2% drop
                 lookback: int = 20,
                 decay_rate: float = 0.1,  # weighting for the heuristic fallback only
                 min_train_size: int = 30,
                 channels: int = 16,
                 retrain_interval: int = 20,
                 epochs: int = 25):
        """
        Initialize TCN Model.

        Args:
            buy_threshold_pct: Buy threshold as percentage of price
            sell_threshold_pct: Sell threshold as percentage of price
            lookback: Number of previous returns fed to the network
            decay_rate: Exponential decay for the no-torch heuristic fallback
            min_train_size: Minimum training data size
            channels: Convolutional channel width
            retrain_interval: Refit the network every N bars (walk-forward safe)
            epochs: Training epochs per refit
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.lookback = lookback
        self.decay_rate = decay_rate
        self.min_train_size = min_train_size
        self.channels = channels
        self.retrain_interval = retrain_interval
        self.epochs = epochs

    def predict(self, data: pd.DataFrame, training_cols: List[str]) -> pd.Series:
        """
        Predict next-bar price changes walk-forward.

        Args:
            data: DataFrame with price and feature data
            training_cols: Unused (the TCN consumes the raw return sequence)

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
                    build_model=lambda: _TCNNet(self.channels),
                    lookback=self.lookback,
                    min_train_size=self.min_train_size,
                    retrain_interval=self.retrain_interval,
                    epochs=self.epochs
                )
                return pd.Series(preds, index=data.index)
            return self._predict_weighted_fallback(prices, data.index)
        except Exception as e:
            print(f"TCN prediction error: {e}")
            return pd.Series(0.0, index=data.index)

    def _predict_weighted_fallback(self, prices: np.ndarray, index) -> pd.Series:
        """
        Heuristic fallback used only when PyTorch is unavailable: an
        exponentially-weighted average of recent returns. This is a smoothing
        baseline, not a convolutional network.
        """
        pred_series = pd.Series(0.0, index=index)

        for i in range(self.min_train_size, len(prices)):
            try:
                window_prices = prices[:i + 1]
                window_returns = np.diff(window_prices) / window_prices[:-1]

                if len(window_returns) < self.lookback:
                    continue

                weights = np.exp(-np.arange(self.lookback) * self.decay_rate)
                weights = weights / weights.sum()
                recent_returns = window_returns[-self.lookback:]
                weighted_return = np.sum(recent_returns * weights)

                pred_series.iloc[i] = weighted_return * window_prices[-1]
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
