"""
Held-out validation for ensemble-level trading parameters.

Usage:
    python validate_ensemble.py [COIN] [TRAIN_FRAC]

Protocol (chronological, leak-free):

1. The per-model signal matrix is generated ONCE over the full dataset.
   Every individual model already predicts walk-forward (its output at bar t
   is produced by a model fit only on data available at bar t), so slicing
   the matrix afterward cannot leak future information into the past.
2. Bars are split chronologically: the first TRAIN_FRAC (default 80%) form
   the tuning slice, the remainder the held-out validation slice.
3. A small grid of ensemble-level parameters (volatility multipliers,
   consensus threshold, entry confirmation) is scored on the tuning slice
   only, ranked by Sharpe ratio (tie-break: total return).
4. The single winning parameter set is then applied, untouched, to the
   held-out slice. That number is the honest out-of-sample estimate; the
   validation slice is never used to select anything.

The pre-round-two baseline (fixed thresholds, unweighted votes, fixed 2%
stop, forced 5-day exit) is evaluated on both slices for comparison.

No network calls, no LLM/API usage — everything runs offline on the local
CSVs.
"""

import itertools
import sys

import numpy as np
import pandas as pd

from main import load_data, get_training_columns, backtest_strategy
from model_manager import ModelManager
from ensemble_model import EnsembleModel


# Replicates the pre-round-two ensemble behavior exactly:
# fixed return thresholds, unweighted majority votes, fixed 2% stop,
# forced exit after 5 days, no entry confirmation.
BASELINE_PARAMS = dict(
    buy_threshold_return=0.002,
    buy_min_bull_count=4,
    sell_threshold_return=-0.002,
    sell_max_bull_count=3,
    time_stop_days=5,
    time_stop_min_return=0.05,
    stop_loss_threshold=0.98,
    max_hold_days=5,
    buy_vol_mult=0.0,
    sell_vol_mult=0.0,
    buy_confirm_bars=1,
    stop_vol_mult=0.0,
    use_weighted_consensus=False,
)

# Parameters shared by every grid candidate (round-two mechanics on).
FIXED_PARAMS = dict(
    buy_threshold_return=0.002,
    sell_threshold_return=-0.002,
    sell_max_bull_count=3,
    time_stop_days=5,
    time_stop_min_return=0.05,
    stop_loss_threshold=0.98,
    stop_vol_mult=1.5,
    max_hold_days=None,
    use_weighted_consensus=True,
)

# Small, coarse grid — deliberately few knobs to limit selection bias.
PARAM_GRID = dict(
    buy_vol_mult=[0.0, 0.15, 0.3],
    sell_vol_mult=[0.0, 0.25, 0.5],
    buy_min_bull_count=[4, 5],
    buy_confirm_bars=[1, 2],
)


def evaluate(signals_slice: pd.DataFrame, params: dict) -> dict:
    """Run the ensemble + fee-aware backtest on one chronological slice."""
    ensemble = EnsembleModel(**params)
    result_df = ensemble.generate_signals(signals_slice)
    bt = backtest_strategy(result_df, initial_capital=10000.0)
    return {
        'total_return': bt.get('total_return', 0.0),
        'buy_hold_return': bt.get('buy_hold_return', 0.0),
        'sharpe_ratio': bt.get('sharpe_ratio', 0.0),
        'max_drawdown': bt.get('max_drawdown', 0.0),
        'total_trades': bt.get('total_trades', 0),
        'win_rate': bt.get('win_rate', 0.0),
    }


def fmt(res: dict) -> str:
    return (f"return {res['total_return']*100:+7.2f}%  "
            f"(B&H {res['buy_hold_return']*100:+7.2f}%)  "
            f"sharpe {res['sharpe_ratio']:+6.2f}  "
            f"maxDD {res['max_drawdown']*100:6.2f}%  "
            f"trades {res['total_trades']:3d}  "
            f"win {res['win_rate']*100:5.1f}%")


def main():
    coin = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    train_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

    print(f"Loading {coin} data...")
    data = load_data(coin)
    if data.empty:
        print("No data. Exiting.")
        return
    training_cols = [c for c in get_training_columns() if c in data.columns]
    print(f"{len(data)} rows, {len(training_cols)} features: {training_cols}")

    print("Generating walk-forward model signals over the full dataset "
          "(done once; slicing afterward cannot leak)...")
    signals_df = ModelManager().generate_signals(data, training_cols)

    split = int(len(signals_df) * train_frac)
    train_sig = signals_df.iloc[:split]
    val_sig = signals_df.iloc[split:]
    print(f"Tuning slice:     {train_sig.index[0]} .. {train_sig.index[-1]} ({len(train_sig)} bars)")
    print(f"Held-out slice:   {val_sig.index[0]} .. {val_sig.index[-1]} ({len(val_sig)} bars)")

    # ---- Grid search on the tuning slice only -----------------------------
    keys = list(PARAM_GRID.keys())
    rows = []
    for values in itertools.product(*(PARAM_GRID[k] for k in keys)):
        combo = dict(zip(keys, values))
        params = {**FIXED_PARAMS, **combo}
        res = evaluate(train_sig, params)
        rows.append((res['sharpe_ratio'], res['total_return'], combo, res))

    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)

    print(f"\nTop 5 of {len(rows)} candidates on the TUNING slice:")
    for sharpe, ret, combo, res in rows[:5]:
        print(f"  {combo}  ->  {fmt(res)}")

    best_combo = rows[0][2]
    best_params = {**FIXED_PARAMS, **best_combo}

    # ---- Honest out-of-sample evaluation -----------------------------------
    print("\n" + "=" * 72)
    print(f"CHOSEN PARAMS (selected on tuning slice only): {best_combo}")
    print("=" * 72)

    for label, sig in (("TUNING slice", train_sig), ("HELD-OUT slice", val_sig)):
        base = evaluate(sig, BASELINE_PARAMS)
        best = evaluate(sig, best_params)
        print(f"\n{label}:")
        print(f"  baseline (round-one) : {fmt(base)}")
        print(f"  tuned    (round-two) : {fmt(best)}")

    print("\nNote: only the HELD-OUT numbers are out-of-sample evidence. If the")
    print("tuned strategy does not beat the baseline there, treat the tuned")
    print("parameters as curve-fit and prefer the defaults' conservatism.")


if __name__ == '__main__':
    main()
