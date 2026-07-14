"""
Ensemble Trading Model
Combines signals from multiple individual models with specific trading rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


class EnsembleModel:
    """
    Ensemble model that combines signals from multiple individual models
    with buy, sell, time stop, and stop-loss rules.
    """
    
    def __init__(self,
                 buy_threshold_return: float = 0.002,
                 buy_min_bull_count: int = 5,
                 sell_threshold_return: float = -0.002,
                 sell_max_bull_count: int = 3,
                 time_stop_days: int = 5,
                 time_stop_min_return: float = 0.05,
                 stop_loss_threshold: float = 0.98,
                 max_hold_days: Optional[int] = None,
                 buy_vol_mult: float = 0.3,
                 sell_vol_mult: float = 0.25,
                 buy_confirm_bars: int = 2,
                 stop_vol_mult: float = 1.5,
                 stop_loss_max_pct: float = 0.12,
                 vol_window: int = 20,
                 use_weighted_consensus: bool = True):
        """
        Initialize Ensemble Model with daily-bar swing-trading rules.

        Defaults require a real plurality of the 10 models to agree (4+) and a
        non-trivial predicted edge before entering — a single bullish model
        out of ten is not a meaningful signal on daily data.

        Regime awareness: the fixed return thresholds act as *floors*; when
        realized volatility is available (the 'realized_vol' column from
        ModelManager, itself walk-forward) the effective entry threshold is
        max(buy_threshold_return, buy_vol_mult * vol) and the effective exit
        threshold is min(sell_threshold_return, -sell_vol_mult * vol), so in
        choppy markets a bigger predicted edge is demanded and small negative
        wobbles don't force an exit (less whipsaw), while in quiet markets the
        original fixed thresholds still apply.

        Args:
            buy_threshold_return: Floor on predicted return for buy (default: 0.2%)
            buy_min_bull_count: Minimum bull count for buy (default: 5). When
                use_weighted_consensus is on this compares against the
                reliability-weighted, active-model-normalized bull count
                ('bull_count_weighted', scaled to the nominal 10-model panel),
                so 5 means "half the active panel, reliability-weighted" —
                held-out validation on both coins never preferred a looser
                setting.
            sell_threshold_return: Floor (least-negative) predicted return for sell (default: -0.2%)
            sell_max_bull_count: Maximum bull count for sell (default: 3)
            time_stop_days: Days to hold before take-profit time stop check (default: 5)
            time_stop_min_return: Minimum cumulative return for time stop (default: 5%)
            stop_loss_threshold: Fallback/floor stop as fraction of entry price
                (default 0.98 = 2% loss). Used verbatim when stop_vol_mult is 0
                or no volatility estimate exists; otherwise it is the tightest
                the volatility-scaled stop is allowed to get.
            max_hold_days: Maximum days to hold a position; None (default)
                disables the forced calendar exit — with ~1% round-trip fees a
                mandatory 5-day churn was a pure cost drag.
            buy_vol_mult: Entry threshold as a multiple of realized daily vol (default: 0.3)
            sell_vol_mult: Exit threshold as a multiple of realized daily vol (default: 0.25)
            buy_confirm_bars: Require the raw buy condition to hold on this many
                consecutive bars before entering (default: 2; 1 = off). Filters
                one-bar noise spikes at the cost of a one-bar-later entry.
            stop_vol_mult: Stop-loss distance as a multiple of realized daily vol
                at entry, clipped to [1 - stop_loss_threshold, stop_loss_max_pct]
                (default: 1.5; 0 = fixed stop_loss_threshold). A fixed 2% stop
                under ~3% daily vol was a coin-flip exit that mostly paid fees.
            stop_loss_max_pct: Widest allowed stop distance (default: 12%)
            vol_window: Rolling window for realized vol if the 'realized_vol'
                column is absent (default: 20)
            use_weighted_consensus: Use 'bull_count_weighted' (reliability-
                weighted votes) when present instead of the raw bull_count
                (default: True)
        """
        self.max_hold_days = max_hold_days
        self.buy_threshold_return = buy_threshold_return
        self.buy_min_bull_count = buy_min_bull_count
        self.sell_threshold_return = sell_threshold_return
        self.sell_max_bull_count = sell_max_bull_count
        self.time_stop_days = time_stop_days
        self.time_stop_min_return = time_stop_min_return
        self.stop_loss_threshold = stop_loss_threshold
        self.buy_vol_mult = buy_vol_mult
        self.sell_vol_mult = sell_vol_mult
        self.buy_confirm_bars = max(1, int(buy_confirm_bars))
        self.stop_vol_mult = stop_vol_mult
        self.stop_loss_max_pct = stop_loss_max_pct
        self.vol_window = vol_window
        self.use_weighted_consensus = use_weighted_consensus

        # Track positions
        self.positions = {}  # {entry_index: {'entry_price': float, 'entry_date': datetime, 'entry_index': int}}
    
    def generate_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble trading signals based on individual model signals.
        
        Args:
            signals_df: DataFrame from ModelManager.generate_signals()
            
        Returns:
            DataFrame with ensemble signals and position tracking
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        result_df = signals_df.copy()
        
        # Initialize signal columns
        result_df['ensemble_signal'] = 0  # 0: hold, 1: buy, -1: sell
        result_df['ensemble_buy'] = 0
        result_df['ensemble_sell'] = 0
        result_df['position'] = 0  # 0: no position, 1: long position
        result_df['entry_price'] = np.nan
        result_df['entry_index'] = -1
        result_df['days_held'] = 0
        result_df['cumulative_return'] = 0.0
        result_df['stop_loss_triggered'] = False
        result_df['time_stop_triggered'] = False
        
        # Ensure required columns exist
        required_cols = ['P_t', 'predicted_return', 'bull_count', 'SMA_20']
        for col in required_cols:
            if col not in result_df.columns:
                if col == 'SMA_20':
                    result_df['SMA_20'] = result_df['P_t'].rolling(window=20).mean()
                else:
                    raise ValueError(f"Required column '{col}' not found in signals_df")
        
        # --- Precompute per-bar arrays (all walk-forward: every value at bar
        # --- t is derived from data known at bar t) -------------------------
        n = len(result_df)
        price_arr = result_df['P_t'].values.astype(float)
        pred_arr = result_df['predicted_return'].values.astype(float)
        sma_arr = result_df['SMA_20'].values.astype(float)

        # Consensus metric: reliability-weighted bull count when available
        if self.use_weighted_consensus and 'bull_count_weighted' in result_df.columns:
            bull_arr = result_df['bull_count_weighted'].values.astype(float)
        else:
            bull_arr = result_df['bull_count'].values.astype(float)

        # Realized daily volatility (rolling std of returns up to bar t)
        if 'realized_vol' in result_df.columns:
            vol_arr = result_df['realized_vol'].values.astype(float)
        else:
            vol_arr = (result_df['P_t'].pct_change()
                       .rolling(self.vol_window, min_periods=10).std().values)

        # Regime-scaled thresholds with the fixed thresholds as floors;
        # where vol is not yet estimable (warm-up), fall back to the floors.
        has_vol = ~np.isnan(vol_arr)
        buy_thr_arr = np.where(
            has_vol,
            np.maximum(self.buy_threshold_return, self.buy_vol_mult * np.nan_to_num(vol_arr)),
            self.buy_threshold_return)
        sell_thr_arr = np.where(
            has_vol,
            np.minimum(self.sell_threshold_return, -self.sell_vol_mult * np.nan_to_num(vol_arr)),
            self.sell_threshold_return)

        # Raw buy condition per bar (position-independent), then require it to
        # hold for buy_confirm_bars consecutive bars (uses only bars <= t).
        sma_ok = np.where(np.isnan(sma_arr), True, price_arr > sma_arr)
        raw_buy_ok = (pred_arr >= buy_thr_arr) & (bull_arr >= self.buy_min_bull_count) & sma_ok
        if self.buy_confirm_bars > 1:
            confirmed_buy = (pd.Series(raw_buy_ok.astype(float))
                             .rolling(self.buy_confirm_bars).min()
                             .fillna(0.0).values >= 1.0)
        else:
            confirmed_buy = raw_buy_ok

        # Track current position
        current_position = None  # None or entry_index

        for idx in range(n):
            row_idx = result_df.index[idx]
            current_price = price_arr[idx]
            predicted_return = pred_arr[idx]
            bull_count = bull_arr[idx]
            sma20 = sma_arr[idx]

            # Check if we have a position
            if current_position is not None:
                entry_info = self.positions[current_position]
                entry_price = entry_info['entry_price']
                entry_index = entry_info['entry_index']
                stop_pct = entry_info['stop_pct']
                days_held = idx - entry_index

                # Calculate cumulative return
                cumulative_return = (current_price - entry_price) / entry_price

                # Update tracking columns
                result_df.loc[row_idx, 'position'] = 1
                result_df.loc[row_idx, 'entry_price'] = entry_price
                result_df.loc[row_idx, 'entry_index'] = entry_index
                result_df.loc[row_idx, 'days_held'] = days_held
                result_df.loc[row_idx, 'cumulative_return'] = cumulative_return

                # Check stop-loss condition (volatility-scaled at entry)
                if current_price < entry_price * (1.0 - stop_pct):
                    result_df.loc[row_idx, 'stop_loss_triggered'] = True
                    result_df.loc[row_idx, 'ensemble_signal'] = -1
                    result_df.loc[row_idx, 'ensemble_sell'] = 1
                    # Close position
                    current_position = None
                    continue

                # Check time stop condition (take profit)
                if days_held >= self.time_stop_days and cumulative_return > self.time_stop_min_return:
                    result_df.loc[row_idx, 'time_stop_triggered'] = True
                    result_df.loc[row_idx, 'ensemble_signal'] = -1
                    result_df.loc[row_idx, 'ensemble_sell'] = 1
                    current_position = None
                    continue

                # Optional forced calendar exit (disabled by default: with
                # ~1% round-trip fees, mandatory churn is a pure cost drag)
                if self.max_hold_days is not None and days_held >= self.max_hold_days:
                    result_df.loc[row_idx, 'time_stop_triggered'] = True
                    result_df.loc[row_idx, 'ensemble_signal'] = -1
                    result_df.loc[row_idx, 'ensemble_sell'] = 1
                    current_position = None
                    continue

                # Check sell conditions (volatility-scaled threshold)
                sell_condition_1 = predicted_return < sell_thr_arr[idx]
                sell_condition_2 = bull_count <= self.sell_max_bull_count
                sell_condition_3 = current_price < sma20 * 0.98 if not pd.isna(sma20) else False  # 2% below SMA

                # Sell on negative momentum or bearish consensus
                if sell_condition_1 or (sell_condition_2 and sell_condition_3):
                    result_df.loc[row_idx, 'ensemble_signal'] = -1
                    result_df.loc[row_idx, 'ensemble_sell'] = 1
                    current_position = None
                    continue

            else:
                # No position - require predicted edge (volatility-scaled),
                # model consensus, and an uptrend filter (all three), held for
                # buy_confirm_bars consecutive bars.
                if confirmed_buy[idx]:
                    # Volatility-scaled stop distance, fixed at entry
                    base_stop = 1.0 - self.stop_loss_threshold
                    if self.stop_vol_mult > 0 and not np.isnan(vol_arr[idx]):
                        stop_pct = min(max(self.stop_vol_mult * vol_arr[idx], base_stop),
                                       self.stop_loss_max_pct)
                    else:
                        stop_pct = base_stop

                    # Open new position
                    entry_index = idx
                    entry_price = current_price
                    entry_date = row_idx if isinstance(row_idx, (pd.Timestamp, datetime)) else datetime.now()

                    self.positions[entry_index] = {
                        'entry_price': entry_price,
                        'entry_date': entry_date,
                        'entry_index': entry_index,
                        'stop_pct': stop_pct
                    }
                    current_position = entry_index
                    
                    result_df.loc[row_idx, 'ensemble_signal'] = 1
                    result_df.loc[row_idx, 'ensemble_buy'] = 1
                    result_df.loc[row_idx, 'position'] = 1
                    result_df.loc[row_idx, 'entry_price'] = entry_price
                    result_df.loc[row_idx, 'entry_index'] = entry_index
                    result_df.loc[row_idx, 'days_held'] = 0
                    result_df.loc[row_idx, 'cumulative_return'] = 0.0
        
        return result_df
    
    def get_trading_summary(self, result_df: pd.DataFrame) -> Dict:
        """
        Generate a summary of trading performance.
        
        Args:
            result_df: DataFrame from generate_signals()
            
        Returns:
            Dictionary with trading statistics
        """
        if result_df.empty:
            return {}
        
        total_trades = result_df['ensemble_buy'].sum()
        total_sells = result_df['ensemble_sell'].sum()
        
        # Calculate returns for closed positions
        closed_positions = result_df[result_df['ensemble_sell'] == 1]
        if len(closed_positions) > 0:
            returns = closed_positions['cumulative_return'].values
            avg_return = returns.mean() if len(returns) > 0 else 0.0
            total_return = returns.sum() if len(returns) > 0 else 0.0
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        else:
            avg_return = 0.0
            total_return = 0.0
            win_rate = 0.0
        
        stop_loss_count = result_df['stop_loss_triggered'].sum()
        time_stop_count = result_df['time_stop_triggered'].sum()
        
        summary = {
            'total_buy_signals': int(total_trades),
            'total_sell_signals': int(total_sells),
            'closed_positions': len(closed_positions),
            'average_return_per_trade': float(avg_return),
            'total_cumulative_return': float(total_return),
            'win_rate': float(win_rate),
            'stop_loss_triggers': int(stop_loss_count),
            'time_stop_triggers': int(time_stop_count)
        }

        # Buy-and-hold benchmark over the same window for context
        if 'P_t' in result_df.columns and len(result_df) > 1 and result_df['P_t'].iloc[0] != 0:
            summary['buy_hold_return'] = float(
                result_df['P_t'].iloc[-1] / result_df['P_t'].iloc[0] - 1.0)

        # Average confidence at entry, if the confidence score is present
        if 'confidence' in result_df.columns:
            buys = result_df[result_df['ensemble_buy'] == 1]
            if len(buys) > 0:
                summary['avg_buy_confidence'] = float(buys['confidence'].mean())

        return summary
    
    def reset_positions(self):
        """Reset all tracked positions."""
        self.positions = {}

