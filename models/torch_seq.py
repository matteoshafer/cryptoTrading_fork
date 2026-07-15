"""
Shared walk-forward training loop for the small PyTorch sequence models
(LSTM/GRU and TCN).

Runs entirely offline on CPU — no network calls, no external services.
If PyTorch is not installed, TORCH_AVAILABLE is False and callers fall back
to their lightweight heuristic implementations.
"""

import numpy as np

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def walk_forward_torch_predict(prices: np.ndarray,
                               build_model,
                               lookback: int = 20,
                               min_train_size: int = 30,
                               retrain_interval: int = 20,
                               epochs: int = 25,
                               lr: float = 0.005,
                               max_train_windows: int = 500,
                               seed: int = 42) -> np.ndarray:
    """
    Walk-forward next-bar return prediction with periodic retraining.

    The network is refit every `retrain_interval` bars on (return-window,
    next-bar-return) pairs whose targets have already been realized at the
    refit bar, then reused for the bars in between. This preserves the
    no-lookahead property: a prediction at bar j always comes from a model
    trained only on outcomes known strictly before bar j, applied to an
    input window of returns known at bar j.

    Args:
        prices: 1-D array of close prices
        build_model: zero-arg callable returning a fresh nn.Module mapping a
            (batch, lookback, 1) float tensor to a (batch, 1) prediction
        lookback: length of the return window fed to the network
        min_train_size: first bar at which predictions start
        retrain_interval: refit the network every N bars
        epochs: full-batch training epochs per refit
        lr: Adam learning rate
        max_train_windows: cap on training pairs (most recent kept)
        seed: torch manual seed for reproducibility

    Returns:
        Array (len(prices),) of predicted next-bar *price changes* (0.0 for
        bars before enough history exists).
    """
    n = len(prices)
    preds = np.zeros(n)
    start = max(min_train_size, lookback + 7)
    if not TORCH_AVAILABLE or n <= start:
        return preds

    torch.manual_seed(seed)

    returns = np.diff(prices) / np.maximum(np.abs(prices[:-1]), 1e-12)
    # windows[k] = returns[k : k + lookback], i.e. the window *ending at*
    # bar t = k + lookback (the last `lookback` realized returns as of t).
    windows = np.lib.stride_tricks.sliding_window_view(returns, lookback)

    model = None
    scale = 1.0
    last_fit = -1

    for i in range(start, n):
        try:
            if model is None or (i - last_fit) >= retrain_interval:
                # Usable training pairs: window ending at t, target
                # returns[t] (the t -> t+1 move), realized by bar i iff
                # t + 1 <= i.
                t_hi = i - 1
                ks = np.arange(0, t_hi - lookback + 1)
                if len(ks) < 5:
                    continue
                if len(ks) > max_train_windows:
                    ks = ks[-max_train_windows:]
                X = windows[ks]
                y = returns[ks + lookback]
                scale = float(returns[:i].std()) + 1e-9

                X_t = torch.tensor(X / scale, dtype=torch.float32).unsqueeze(-1)
                y_t = torch.tensor(y / scale, dtype=torch.float32).unsqueeze(-1)

                model = build_model()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss_fn = nn.MSELoss()
                model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = loss_fn(model(X_t), y_t)
                    loss.backward()
                    optimizer.step()
                model.eval()
                last_fit = i

            if model is None:
                continue

            # Predict bar i's next move from the window ending at bar i.
            w = windows[i - lookback]
            with torch.no_grad():
                x = torch.tensor(w / scale, dtype=torch.float32).view(1, lookback, 1)
                ret_pred = float(model(x).item()) * scale
            preds[i] = ret_pred * prices[i]
        except Exception:
            pass

    return preds
