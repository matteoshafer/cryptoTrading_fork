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

9. **LSTM/GRU**
   - Buy: `rnn_pred_change > +0.012 * P_t`
   - Sell: `rnn_pred_change < -0.006 * P_t`

10. **Temporal Convolutional Network (TCN)**
    - Buy: `tcn_pred_change > +0.01 * P_t`
    - Sell: `tcn_pred_change < -0.005 * P_t`

### 2. Ensemble Model (`ensemble_model.py`)

Combines individual model signals with the following rules:

#### Buy Conditions (ALL must be true):
- `r_{t+1} > 0.005` (predicted return > 0.5%)
- `bull_count >= 6` (at least 6 models are bullish)
- `P_t > SMA20` (current price above 20-day moving average)

#### Sell Conditions (ANY can be true):
- `r_{t+1} < -0.005` (predicted return < -0.5%)
- `bull_count <= 4` (4 or fewer models are bullish)
- `P_t < SMA20` (current price below 20-day moving average)

#### Time Stop:
- Position held `>= 5 days` AND cumulative return `> 5%`

#### Stop-Loss:
- `P_t < EntryPrice × 0.98` (2% loss from entry price)

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
- PyTorch: `pip install torch transformers`
- TensorFlow: `pip install tensorflow` (optional, for LSTM/GRU/TCN)

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
- Total return percentage
- Win rate
- Average return per trade
- Maximum and minimum returns
- Trade history

## Notes

- Models are trained incrementally (using all data up to the current point)
- The system requires at least 50 data points for reliable predictions
- Some models (Prophet, ARIMA) require more historical data
- The ensemble model tracks positions and applies risk management rules automatically

## Integration with Existing Codebase

The system integrates with your existing codebase:
- Uses `functions.py` for data loading utilities
- Uses `CONSTANTS.py` for default coin selection
- Reads training columns from `training_columns.txt`
- Compatible with data format from `fulldata/` directory

