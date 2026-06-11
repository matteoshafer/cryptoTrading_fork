# Crypto Trading Bot

An ensemble ML trading system for cryptocurrency that generates buy/sell/hold signals and backtests strategy performance across historical data.

## Features

- **10 ML models** combined into an ensemble: XGBoost, LightGBM, Random Forest, ARIMA/SARIMA, Prophet, SVR, LSTM/GRU, TCN, GBM Simulator, and LLM sentiment (RoBERTa)
- **Current signal** — run the script and get an immediate BUY / SELL / HOLD recommendation based on the latest data
- **Backtesting** — simulates trades over historical data with full P&L, win rate, and trade log
- **Interactive dashboard** — Streamlit UI with price chart, portfolio value over time, and signal visualization
- Supports BTC and ETH out of the box

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run from the command line

```bash
python main.py          # Default coin (ETH)
python main.py BTC      # Bitcoin
python main.py BTC 2024-12-01             # From a start date
python main.py BTC 2024-12-01 2025-01-01  # Over a date range
```

Output includes backtest results, a full trade log, and a **CURRENT SIGNAL** block at the end:

```
==================================================
CURRENT SIGNAL
==================================================
  Signal:           🟢 BUY
  Latest Price:     $95,420.00
  Predicted Return: 0.312%
  Bullish Models:   6
  As of:            2025-05-07
==================================================
```

### Run the dashboard

```bash
streamlit run dashboard.py
```

Pick a coin and date range in the sidebar, click **Run Analysis**, and the dashboard shows:
- Color-coded current signal banner (green/red/yellow)
- Portfolio value over time chart
- Price chart with buy/sell markers
- Backtest metrics (total return, win rate, avg return/trade)
- Trade log table
- Per-model signal breakdown

## How It Works

### Signal Generation

Each of the 10 models produces an independent buy/sell signal. The ensemble combines them:

**Buy** when:
- Predicted return ≥ threshold (default 0%)
- At least N models are bullish (default 1)

**Sell** when:
- Predicted return < threshold, OR
- Too few models are bullish, OR
- Price drops 2%+ below SMA-20

**Risk management:**
- Stop-loss: exit if price falls more than 2% from entry
- Time stop: take profit if held 5+ days with >5% cumulative return
- Max hold: force exit after 2 days to free capital

### Backtesting

`backtest_strategy()` simulates $10,000 in starting capital through all historical buy/sell signals and reports:

| Metric | Description |
|---|---|
| Total Return | % gain/loss over the full period |
| Final Capital | Dollar value at end |
| Win Rate | % of trades that were profitable |
| Avg Return/Trade | Mean return across all closed positions |
| Max / Min Return | Best and worst individual trades |

## Data

Historical data lives in `fulldata/`:
- `BTC_df.csv`
- `ETH_df.csv`

Required columns: `time`, `close`  
Optional (improves model accuracy): `volume`, `SMA_20`, `SMA_50`, `OBV`, `BB_Lower`, `BB_Middle`, `BB_Upper`, `avg_sentiment`

To fetch fresh data, run:

```bash
python fetch_and_run.py
```

## Project Structure

```
├── main.py              # CLI entry point — runs strategy, backtest, current signal
├── dashboard.py         # Streamlit dashboard
├── ensemble_model.py    # Combines model signals into buy/sell decisions
├── model_manager.py     # Initializes and runs all 10 ML models
├── functions.py         # Data utilities and feature engineering
├── live_data_fetcher.py # Fetches live price/sentiment data
├── fetch_and_run.py     # Scheduled fetch + strategy run
├── CONSTANTS.py         # Config (default coin, API keys, thresholds)
├── fulldata/            # Historical OHLCV + feature data
└── models/              # Saved model weights
```

## Configuration

Edit `CONSTANTS.py` to change defaults:

```python
COIN = 'ETH'        # Default coin
TRAIN_PCT = 0.8     # Train/test split
```

Ensemble thresholds can be tuned at runtime via CLI args or the dashboard sliders.
