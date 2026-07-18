# Crypto Trading Bot

An ensemble ML trading system for cryptocurrency that generates buy/sell/hold signals and backtests strategy performance across historical data.

## Features

- **10 ML models** combined into an ensemble: XGBoost, LightGBM, Random Forest, ARIMA/SARIMA, Prophet, SVR, LSTM/GRU, TCN, GBM Simulator, and LLM sentiment (RoBERTa)
- **Current signal** — run the script and get an immediate BUY / SELL / HOLD recommendation based on the latest data
- **Confidence score** — every signal carries a 0-100 confidence (LOW/MEDIUM/HIGH) computed offline from model agreement, prediction signal-to-noise, each model's recent hit rate, and Prophet's forecast-interval width. No API or LLM calls at trading time.
- **Backtesting** — simulates trades over historical data with fees, confidence-based position sizing, full P&L, Sharpe ratio, max drawdown, a buy-and-hold benchmark, win rate, and trade log
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
  Bullish Models:   6 of 8 active
  Confidence:       71/100 (HIGH)
  As of:            2025-05-07
==================================================
```

### Run the dashboard

```bash
streamlit run dashboard.py
```

Pick a coin and date range in the sidebar, click **Run Analysis**, and the dashboard shows:
- Color-coded current signal banner (green/red/yellow) with the signal confidence score
- Portfolio value over time chart (fee- and position-sizing-aware equity curve)
- Price chart with buy/sell markers
- Backtest metrics (total return, win rate, avg return/trade, Sharpe ratio, max drawdown, buy & hold comparison)
- Trade log table
- Per-model signal breakdown

## How It Works

### Signal Generation

Each of the 10 models produces an independent buy/sell signal. The ensemble weights every model's vote and predicted return by that model's own rolling walk-forward hit rate (recent resolved calls only — no lookahead), then combines them with volatility-aware thresholds:

**Buy** when (all must hold, on 2 consecutive bars):
- Predicted return ≥ max(0.2%, 0.3 × realized daily volatility)
- Reliability-weighted bull count ≥ 5 (half the active panel)
- Price is above SMA-20

**Sell** when:
- Predicted return < min(-0.2%, -0.25 × realized daily volatility), OR
- Too few models are bullish AND price is 2%+ below SMA-20

**Risk management:**
- Stop-loss: volatility-scaled at entry — exit if price falls more than clip(1.5 × daily vol, 2%, 12%) from entry
- Time stop: take profit if held 5+ days with >5% cumulative return

Ensemble parameters were chosen via `validate_ensemble.py`, which tunes on the first 80% of the data and reports honest out-of-sample results on the untouched final 20%.

### Backtesting

`backtest_strategy()` simulates $10,000 in starting capital through all historical buy/sell signals, charging a 0.5% fee per leg and sizing each entry by the signal's confidence score (25%-100% of available cash; pass `size_by_confidence=False` for all-in trades). It reports:

| Metric | Description |
|---|---|
| Total Return | % gain/loss over the full period, net of fees |
| Buy & Hold Return | Benchmark: buying at the start and holding to the end (same fees) |
| Sharpe Ratio | Annualized risk-adjusted return from the daily equity curve |
| Max Drawdown | Worst peak-to-trough decline of the equity curve |
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

## Telegram Signals

For a hands-off, long-term-hold workflow, `telegram_notify.py` pushes the
current signal to a Telegram chat instead of requiring you to run `main.py`
yourself:

```bash
python telegram_notify.py --setup     # one-time: create a bot via @BotFather,
                                       # paste the token, auto-detects your chat ID
python telegram_notify.py --once      # single check-and-notify pass
python telegram_notify.py --schedule  # run forever: checks every 4h (default),
                                       # sends a daily digest at 9am (default)
```

It fetches fresh data itself (via `live_data_fetcher.py`), then sends:
- **A daily digest** — current signal + confidence for every tracked coin, once
  a day at `--digest-hour`, regardless of whether anything changed.
- **A change alert** — immediately, whenever a coin's signal flips (e.g.
  HOLD → BUY), checked every `--interval-hours`.

Once a day (not on every check — it's slow and doesn't need to be that
fresh) it also runs `update_sentiment.py`, which scrapes recent news via
free Google News RSS, scores it with a local RoBERTa classifier, and merges
a daily-aggregated `avg_sentiment` value into `fulldata/{coin}_df.csv` so
the LLM-Sentiment model has real data to work with. Pass `--no-sentiment`
to skip this. (This replaces `functions.newspaper_sentiment_pipeline()`,
which had two real bugs: it reloaded the sentiment model from scratch per
article, and its merge duplicated every price row once per matching
article instead of aggregating to one value per day.)

Credentials go in `.env` (gitignored); last-seen signal/sentiment state
persists in `telegram_state.json` (also gitignored — runtime state, not
config). No LLM/API calls of any kind beyond the local, offline RoBERTa
classifier — this only talks to the Telegram Bot API, Google News RSS, and
the existing walk-forward signal pipeline.

### Running unattended (macOS)

`--schedule` uses an in-process timer, which is fine on a server but wrong
for a laptop: while your Mac sleeps, the Python process (and its timer)
sleeps too, so a run can come due, be missed, and never fire — nothing
scheduled during that sleep gets caught up. `launchd` doesn't have that
problem: it's sleep-aware and runs missed `StartInterval` jobs promptly on
wake. So on macOS, use `launchd` with `telegram_notify.py --once` instead
of the script's own `--schedule`:

```bash
cp launchd/com.example.cryptotrading-telegram.plist.example \
   ~/Library/LaunchAgents/com.example.cryptotrading-telegram.plist
# then edit that copy: replace /path/to/cryptoTrading_fork with your actual
# path, and rename the Label/filename if you want something other than
# "com.example"

launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.example.cryptotrading-telegram.plist
launchctl print gui/$(id -u)/com.example.cryptotrading-telegram   # check status
launchctl bootout gui/$(id -u)/com.example.cryptotrading-telegram # stop it
```

`RunAtLoad` fires an immediate check on load; `StartInterval` (14400s = 4h)
repeats it, catching up on wake if a run was missed during sleep. Logs go
to `telegram_notify.out.log` / `telegram_notify.err.log`. On an
always-on Linux server, `telegram_notify.py --schedule` (systemd or just a
persistent shell) is fine — the sleep problem only applies to laptops.

**If runs still silently stop happening** (`telegram_state.json`'s dates
fall behind and nothing shows up in the logs, not even an error): the job
is being killed mid-run by the next sleep cycle before it can finish or
flush its output. A model-retrain check takes 1-2 minutes, but on battery
some Macs dark-wake for well under a minute before sleeping again — not
long enough. The template already wraps the command in
`caffeinate -s -i`, which holds the system awake for exactly as long as
that one process runs; if you're seeing this on a plist from before that
fix, add the `caffeinate -s -i` prefix yourself (see the template) and
reload the agent.

To respond to commands typed in the chat (`/help`, `/status`, `/signal
[COIN]`), run `telegram_notify.py --listen` as a *second*, separately
loaded agent — see `launchd/com.example.cryptotrading-telegram-commands.plist.example`.
Unlike the digest job, this one needs `KeepAlive` instead of
`StartInterval`, since it has to be continuously listening rather than
running briefly on a schedule.

## Project Structure

```
├── main.py              # CLI entry point — runs strategy, backtest, current signal
├── dashboard.py         # Streamlit dashboard
├── telegram_notify.py   # Telegram digest + change alerts for long-term holds
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
