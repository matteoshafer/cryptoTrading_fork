# Trading Strategy System Implementation

## Overview

This implementation provides a comprehensive trading strategy system that combines 10 individual machine learning models with an ensemble approach for cryptocurrency trading decisions.

## Components

### 1. Model Manager (`model_manager.py`)

Manages 10 individual ML models and generates buy/sell signals:

1. **LLM-Sentiment Model** (RoBERTa)
   - Buy: `sentiment_z > +0.4`
   - Sell: `sentiment_z < -0.4`

2. **XGBoost Regressor**
   - Buy: `pred_change > +0.01 * P_t` (>1% predicted gain)
   - Sell: `pred_change < -0.005 * P_t` (>0.5% predicted drop)

3. **Custom GBM Simulator** (Geometric Brownian Motion)
   - Buy: `mean(pred_paths[-1]) > P_t * (1 + 0.01)`
   - Sell: `mean(pred_paths[-1]) < P_t * (1 - 0.005)`

4. **ARIMA/SARIMA**
   - Buy: `fcst_mean[t+1] > P_t * (1 + 0.008)`
   - Sell: `fcst_mean[t+1] < P_t * (1 - 0.008)`

5. **Prophet**
   - Buy: `prophet_pred > P_t` with `CI lower > P_t`
   - Sell: `prophet_pred < P_t` with `CI upper < P_t`

6. **Random Forest Regressor**
   - Buy: `rf_pred_change > +0.007 * P_t`
   - Sell: `rf_pred_change < -0.007 * P_t`

7. **LightGBM (or CatBoost)**
   - Buy: `lgbm_pred_change > +0.01 * P_t`
   - Sell: `lgbm_pred_change < -0.01 * P_t`

8. **Support Vector Regressor (SVR)**
   - Buy: `svr_pred_change > +0.008 * P_t`
   - Sell: `svr_pred_change < -0.008 * P_t`

9. **LSTM/GRU** (PyTorch GRU trained walk-forward on return sequences; if
   torch is not installed it falls back to a simple EMA momentum heuristic —
   a baseline, not a neural net)
   - Buy: `rnn_pred_change > +0.012 * P_t`
   - Sell: `rnn_pred_change < -0.006 * P_t`

10. **Temporal Convolutional Network (TCN)** (PyTorch causal dilated conv net
    trained walk-forward; if torch is not installed it falls back to a
    weighted-average-of-returns heuristic — a baseline, not a neural net)
    - Buy: `tcn_pred_change > +0.01 * P_t`
    - Sell: `tcn_pred_change < -0.005 * P_t`

### 2. Ensemble Model (`ensemble_model.py`)

Combines individual model signals with the following rules:

#### Buy Conditions (ALL must be true):
- `r_{t+1} > 0.002` (predicted return > 0.2%)
- `bull_count >= 4` (at least 4 of 10 models are bullish)
- `P_t > SMA20` (current price above 20-day moving average)

#### Sell Conditions (ANY can be true):
- `r_{t+1} < -0.002` (predicted return < -0.2%)
- `bull_count <= 3` (3 or fewer models are bullish) AND `P_t < SMA20 * 0.98`

#### Time Stop:
- Position held `>= 5 days` AND cumulative return `> 5%`

#### Stop-Loss:
- `P_t < EntryPrice × 0.98` (2% loss from entry price)

### Signal Confidence Score

Every bar gets a quantitative `confidence` score (0-100) and a
`confidence_bucket` (LOW < 40 <= MEDIUM < 70 <= HIGH), computed purely from
statistics on the models' own outputs — **no network calls and no LLM
inference at trading time**. It blends four components, each computed
relative to the models that are actually *active* this run (models whose
optional dependency is missing emit all-zero predictions and are excluded):

| Component | Weight | Meaning |
|---|---|---|
| Directional agreement | 35% | Fraction of active models whose predicted return shares the majority sign (50/50 split → 0, unanimous → 1) |
| Signal-to-noise | 25% | \|mean\| / std of the cross-model predicted returns — strong consensus relative to disagreement scores high |
| Recent reliability | 30% | Each model's rolling hit rate over its last 30 *resolved* directional calls (shifted one bar, so no lookahead), averaged across models |
| Prophet interval width | 10% | Narrow forecast interval relative to price → high; ≥10% of price wide → 0. Dropped (weights renormalized) when Prophet is inactive |

The score appears in the CLI **CURRENT SIGNAL** block, the output CSV
(`confidence`, `confidence_bucket`, `active_models` columns), and the
dashboard banner. `backtest_strategy(size_by_confidence=True)` (the default)
also uses it for position sizing: each buy invests 25% (confidence 0) to
100% (confidence 100) of available cash.

### 3. Main Orchestration (`main.py`)

Coordinates data loading, model execution, and signal generation.

## Usage

### Basic Usage

```python
from main import run_trading_strategy, backtest_strategy

# Run strategy for default coin (from CONSTANTS.py)
result_df = run_trading_strategy(verbose=True)

# Run backtest
backtest_results = backtest_strategy(result_df, initial_capital=10000.0)
print(f"Total Return: {backtest_results['total_return']*100:.2f}%")
```

### Command Line Usage

```bash
python main.py BTC  # Run for Bitcoin
python main.py ETH  # Run for Ethereum
```

### Advanced Usage

```python
# Run for specific coin and date range
result_df = run_trading_strategy(
    coin='BTC',
    start_date='2024-01-01',
    end_date='2024-06-01',
    verbose=True
)
```

## Data Requirements

The system expects data files in `fulldata/{COIN}_df.csv` with the following columns:

**Required:**
- `time`: Timestamp (will be used as index)
- `close`: Closing price

**Recommended:**
- `volume`: Trading volume
- `SMA_20`, `SMA_50`: Moving averages
- `Volume_MA_20`: Volume moving average
- `OBV`: On-Balance Volume
- `BB_Lower`, `BB_Middle`, `BB_Upper`: Bollinger Bands
- `avg_sentiment`: Average sentiment score

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** Some models require additional dependencies:
- XGBoost: `pip install xgboost`
- LightGBM: `pip install lightgbm`
- Prophet: `pip install prophet`
- Statsmodels: `pip install statsmodels`
- PyTorch: `pip install torch` (for the LSTM/GRU and TCN networks; add
  `transformers` for the offline news-sentiment labeling pipeline)

The system will gracefully handle missing dependencies and disable corresponding models.

## Output

The system generates a CSV file `{COIN}_trading_signals.csv` containing:

- Individual model buy/sell signals
- Ensemble buy/sell signals
- Position tracking (entry price, days held, cumulative return)
- Stop-loss and time-stop triggers
- Predicted returns and bull count

## Backtest Results

The backtest function provides:
- Total return percentage (net of fees, default 0.5% per leg)
- Buy-and-hold benchmark return over the same window (same fees)
- Annualized Sharpe ratio (from the daily mark-to-market equity curve)
- Maximum drawdown (peak-to-trough on the equity curve)
- Win rate
- Average return per trade
- Maximum and minimum returns
- Trade history (with the confidence score and cash allocation of each entry)
- The full per-bar `equity_curve` Series

## Notes

- Models are trained incrementally (using all data up to the current point)
- Heavier models retrain periodically instead of every bar (`retrain_interval`
  constructor argument: default 5 bars for XGBoost/LightGBM/Random Forest/
  SVR/ARIMA, 10 for Prophet, 20 for the LSTM/GRU and TCN networks). Between
  refits the cached model — trained only on data before its refit bar — is
  reused, so the walk-forward no-lookahead property is preserved. Set
  `retrain_interval=1` to retrain every bar as before.
- The system requires at least 50 data points for reliable predictions
- Some models (Prophet, ARIMA) require more historical data
- The ensemble model tracks positions and applies risk management rules automatically

## Integration with Existing Codebase

The system integrates with your existing codebase:
- Uses `functions.py` for data loading utilities
- Uses `CONSTANTS.py` for default coin selection
- Reads training columns from `training_columns.txt`
- Compatible with data format from `fulldata/` directory

