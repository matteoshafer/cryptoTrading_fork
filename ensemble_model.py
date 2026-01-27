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
                 buy_threshold_return: float = 0.0,
                 buy_min_bull_count: int = 1,
                 sell_threshold_return: float = -0.001,
                 sell_max_bull_count: int = 1,
                 time_stop_days: int = 1,
                 time_stop_min_return: float = 0.01,
                 stop_loss_threshold: float = 0.99,
                 max_hold_days: int = 2):
        """
        Initialize Ensemble Model with high-frequency trading rules.

        Args:
            buy_threshold_return: Minimum predicted return for buy signal (default: 0.0%)
            buy_min_bull_count: Minimum number of bullish models for buy (default: 1)
            sell_threshold_return: Maximum predicted return for sell signal (default: -0.1%)
            sell_max_bull_count: Maximum number of bullish models for sell (default: 1)
            time_stop_days: Days to hold before time stop check (default: 1)
            time_stop_min_return: Minimum cumulative return for time stop (default: 1%)
            stop_loss_threshold: Stop loss threshold as fraction of entry price (default: 1% loss)
            max_hold_days: Maximum days to hold a position (default: 2)
        """
        self.max_hold_days = max_hold_days
        self.buy_threshold_return = buy_threshold_return
        self.buy_min_bull_count = buy_min_bull_count
        self.sell_threshold_return = sell_threshold_return
        self.sell_max_bull_count = sell_max_bull_count
        self.time_stop_days = time_stop_days
        self.time_stop_min_return = time_stop_min_return
        self.stop_loss_threshold = stop_loss_threshold
        
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
        
        # Track current position
        current_position = None  # None or entry_index
        
        for idx in range(len(result_df)):
            row_idx = result_df.index[idx]
            current_price = result_df['P_t'].iloc[idx]
            predicted_return = result_df['predicted_return'].iloc[idx]
            bull_count = result_df['bull_count'].iloc[idx]
            sma20 = result_df['SMA_20'].iloc[idx]
            
            # Check if we have a position
            if current_position is not None:
                entry_info = self.positions[current_position]
                entry_price = entry_info['entry_price']
                entry_index = entry_info['entry_index']
                days_held = idx - entry_index
                
                # Calculate cumulative return
                cumulative_return = (current_price - entry_price) / entry_price
                
                # Update tracking columns
                result_df.loc[row_idx, 'position'] = 1
                result_df.loc[row_idx, 'entry_price'] = entry_price
                result_df.loc[row_idx, 'entry_index'] = entry_index
                result_df.loc[row_idx, 'days_held'] = days_held
                result_df.loc[row_idx, 'cumulative_return'] = cumulative_return
                
                # Check stop-loss condition
                if current_price < entry_price * self.stop_loss_threshold:
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

                # Max hold days - force exit to free capital for new trades
                if days_held >= self.max_hold_days:
                    result_df.loc[row_idx, 'time_stop_triggered'] = True
                    result_df.loc[row_idx, 'ensemble_signal'] = -1
                    result_df.loc[row_idx, 'ensemble_sell'] = 1
                    current_position = None
                    continue

                # Check sell conditions
                sell_condition_1 = predicted_return < self.sell_threshold_return
                sell_condition_2 = bull_count <= self.sell_max_bull_count
                sell_condition_3 = current_price < sma20 * 0.98 if not pd.isna(sma20) else False  # 2% below SMA

                # Sell on negative momentum or bearish consensus
                if sell_condition_1 or (sell_condition_2 and sell_condition_3):
                    result_df.loc[row_idx, 'ensemble_signal'] = -1
                    result_df.loc[row_idx, 'ensemble_sell'] = 1
                    current_position = None
                    continue
                
            else:
                # No position - high frequency trading: buy when any model is bullish
                buy_condition_1 = predicted_return >= self.buy_threshold_return
                buy_condition_2 = bull_count >= self.buy_min_bull_count

                # Active trading: buy whenever minimum models are bullish
                # This allows for frequent entry points
                should_buy = buy_condition_2 and (buy_condition_1 or predicted_return > -0.005)
                
                if should_buy:
                    # Open new position
                    entry_index = idx
                    entry_price = current_price
                    entry_date = row_idx if isinstance(row_idx, (pd.Timestamp, datetime)) else datetime.now()
                    
                    self.positions[entry_index] = {
                        'entry_price': entry_price,
                        'entry_date': entry_date,
                        'entry_index': entry_index
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
        
        return summary
    
    def reset_positions(self):
        """Reset all tracked positions."""
        self.positions = {}

