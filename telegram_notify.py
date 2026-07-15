"""
Telegram delivery for BTC/ETH trading signals.

Two delivery modes, matching the existing signal generation in main.py /
live_data_fetcher.py:
  - Daily digest: current signal + confidence for every tracked coin, sent
    once per day at a configurable hour, regardless of whether it changed.
  - Change alert: sent immediately whenever a coin's signal flips
    (e.g. HOLD -> BUY), so you're not paged for a signal that's holding
    steady between digests.

No LLM/API calls of any kind here — this only talks to the Telegram Bot API
and reuses the existing walk-forward signal pipeline (main.py, live_data_
fetcher.py). Credentials live in .env (gitignored); state (last signal seen
per coin, last digest date) persists in telegram_state.json (also gitignored
— it's runtime state, not configuration).

Setup:
    python telegram_notify.py --setup

Usage:
    python telegram_notify.py --once                       # single check, right now
    python telegram_notify.py --schedule                    # run forever
    python telegram_notify.py --schedule --interval-hours 4 --digest-hour 9
    python telegram_notify.py --once --coins BTC             # BTC only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime

import requests
from dotenv import load_dotenv

from live_data_fetcher import update_data
from main import run_trading_strategy, get_current_signal

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
STATE_FILE = os.path.join(os.path.dirname(__file__), "telegram_state.json")
API_BASE = "https://api.telegram.org/bot{token}/{method}"

SIGNAL_EMOJI = {"BUY": "\U0001F7E2", "SELL": "\U0001F534", "HOLD": "\U0001F7E1"}


# ── Low-level Telegram helpers ──────────────────────────────────────────────

def _tg_get(token: str, method: str, **params) -> dict:
    url = API_BASE.format(token=token, method=method)
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _autodetect_chat_id(token: str) -> str | None:
    try:
        data = _tg_get(token, "getUpdates", limit=10, timeout=0)
        results = data.get("result", [])
        if results:
            msg = results[-1].get("message") or results[-1].get("channel_post")
            if msg:
                return str(msg["chat"]["id"])
    except Exception:
        pass
    return None


def _setup_telegram() -> tuple[str, str]:
    print()
    print("Telegram Bot First-Run Setup")
    print("=" * 40)
    print()
    print("STEP 1 -- Create a bot with @BotFather")
    print("  1. Open Telegram and search for @BotFather")
    print("  2. Send:  /newbot")
    print("  3. Choose a name and username (must end in 'bot')")
    print("  4. Copy the token BotFather gives you")
    print()

    while True:
        token = input("Paste your bot token here: ").strip()
        if ":" in token and len(token) > 20:
            break
        print("  That doesn't look like a valid token. Try again.")

    bot_name = "the bot"
    try:
        me = _tg_get(token, "getMe")
        bot_name = me["result"].get("username", "unknown")
        print(f"\n  Token valid -- Bot: @{bot_name}")
    except Exception as e:
        print(f"\n  Could not validate token: {e}")

    print()
    print("STEP 2 -- Get your Chat ID")
    print(f"  1. Find your bot @{bot_name} on Telegram and send it any message")
    input("\n  [Press Enter after you've sent a message] ")

    chat_id: str | None = None
    for attempt in range(3):
        chat_id = _autodetect_chat_id(token)
        if chat_id:
            print(f"\n  Chat ID detected: {chat_id}")
            break
        print(f"  No messages found yet ({attempt + 1}/3), retrying...")
        time.sleep(3)

    if not chat_id:
        chat_id = input("\n  Enter your Chat ID manually: ").strip()

    with open(ENV_FILE, "a") as f:
        f.write(f"\nTELEGRAM_BOT_TOKEN={token}\n")
        f.write(f"TELEGRAM_CHAT_ID={chat_id}\n")
    print("\n  Credentials saved to .env\n")
    return token, chat_id


def get_credentials() -> tuple[str, str]:
    load_dotenv(ENV_FILE)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        token, chat_id = _setup_telegram()
    return token, chat_id


def send_message(token: str, chat_id: str, text: str, retries: int = 3) -> bool:
    url = API_BASE.format(token=token, method="sendMessage")
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 429:
                wait = int(r.json().get("parameters", {}).get("retry_after", 5))
                print(f"  [Telegram] Rate limited -- waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [Telegram] Send failed: {e}")
    return False


# ── State ────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Signal computation ──────────────────────────────────────────────────────

def compute_signal(coin: str, lookback_days: int = 400) -> dict:
    """Pull fresh data for `coin` and return the current walk-forward signal."""
    update_data(coin=coin, lookback_days=lookback_days)
    result_df = run_trading_strategy(coin=coin, verbose=False)
    return get_current_signal(result_df)


# ── Message formatting ──────────────────────────────────────────────────────

def format_signal_block(coin: str, sig: dict) -> str:
    emoji = SIGNAL_EMOJI.get(sig.get("signal"), "⚪")
    lines = [f"{emoji} <b>{coin}: {sig.get('signal', 'UNKNOWN')}</b>"]
    if sig.get("price") is not None:
        lines.append(f"Price: ${sig['price']:,.2f}")
    if sig.get("confidence") is not None:
        bucket = sig.get("confidence_bucket") or "?"
        lines.append(f"Confidence: {sig['confidence']:.0f}/100 ({bucket})")
    if sig.get("bull_count") is not None and sig.get("active_models") is not None:
        lines.append(f"Consensus: {int(sig['bull_count'])} of {int(sig['active_models'])} models")
    if sig.get("predicted_return") is not None:
        lines.append(f"Predicted return: {sig['predicted_return'] * 100:+.3f}%")
    if sig.get("timestamp") is not None:
        lines.append(f"As of: {str(sig['timestamp']).split(' ')[0]}")
    return "\n".join(lines)


def format_digest(signals: dict[str, dict]) -> str:
    header = f"\U0001F4CA <b>Daily Signal Digest</b> — {date.today().isoformat()}"
    blocks = [format_signal_block(coin, sig) for coin, sig in signals.items()]
    return header + "\n\n" + "\n\n".join(blocks)


def format_alert(coin: str, old_signal: str, sig: dict) -> str:
    new_signal = sig.get("signal", "UNKNOWN")
    return (f"\U0001F514 <b>{coin} signal changed: {old_signal} → {new_signal}</b>\n\n"
            + format_signal_block(coin, sig))


# ── Core check-and-notify ───────────────────────────────────────────────────

def run_once(coins=("BTC", "ETH"), digest_hour: int = 9, lookback_days: int = 400,
             update_sentiment_daily: bool = True):
    token, chat_id = get_credentials()
    state = load_state()
    today = date.today().isoformat()
    now = datetime.now()

    digest_due = state.get("last_digest_date") != today and now.hour >= digest_hour

    # Refresh avg_sentiment from fresh news once per day, before computing
    # signals, so the LLM-Sentiment model gets today's data on this run
    # rather than a day late. Scraping is slow (dozens of HTTP fetches + a
    # local model pass) and Google News RSS doesn't need re-hitting every
    # few hours, so this deliberately doesn't run on every scheduled check.
    sentiment_due = update_sentiment_daily and state.get("last_sentiment_date") != today
    if sentiment_due:
        try:
            from update_sentiment import update_sentiment
            for coin in coins:
                print(f"Updating sentiment for {coin}...")
                update_sentiment(coin)
            state["last_sentiment_date"] = today
        except Exception as e:
            print(f"  Sentiment update failed (non-fatal, continuing): {e}")

    signals = {}
    for coin in coins:
        print(f"Checking {coin}...")
        try:
            sig = compute_signal(coin, lookback_days=lookback_days)
        except Exception as e:
            print(f"  Failed to compute signal for {coin}: {e}")
            continue
        signals[coin] = sig

        coin_state = state.get(coin, {})
        old_signal = coin_state.get("last_signal")
        new_signal = sig.get("signal")

        if old_signal is not None and new_signal is not None and old_signal != new_signal:
            print(f"  {coin}: {old_signal} -> {new_signal} -- sending alert")
            send_message(token, chat_id, format_alert(coin, old_signal, sig))

        state[coin] = {
            "last_signal": new_signal,
            "last_confidence": sig.get("confidence"),
            "last_checked": now.isoformat(),
        }

    if digest_due and signals:
        print("Sending daily digest...")
        send_message(token, chat_id, format_digest(signals))
        state["last_digest_date"] = today

    save_state(state)
    print("Done.")


def run_scheduler(coins, interval_hours: int, digest_hour: int, lookback_days: int,
                  update_sentiment_daily: bool = True):
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        print("APScheduler not installed. Install with: pip install apscheduler")
        print("Running a single check instead...")
        run_once(coins, digest_hour, lookback_days, update_sentiment_daily)
        return

    scheduler = BlockingScheduler()
    scheduler.add_job(
        lambda: run_once(coins, digest_hour, lookback_days, update_sentiment_daily),
        trigger=IntervalTrigger(hours=interval_hours),
        id="telegram_notify",
        name=f"Check signals every {interval_hours}h, digest at {digest_hour}:00",
        replace_existing=True,
    )
    print(f"Scheduler started: checking every {interval_hours}h, daily digest at {digest_hour}:00.")
    print("Press Ctrl+C to stop.")
    run_once(coins, digest_hour, lookback_days, update_sentiment_daily)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")


def main():
    parser = argparse.ArgumentParser(description="Telegram delivery for BTC/ETH trading signals")
    parser.add_argument("--setup", action="store_true", help="Force (re)run the Telegram setup flow")
    parser.add_argument("--once", action="store_true", help="Run a single check-and-notify pass")
    parser.add_argument("--schedule", action="store_true", help="Run continuously on a schedule")
    parser.add_argument("--interval-hours", type=float, default=4, help="Hours between checks (default: 4)")
    parser.add_argument("--digest-hour", type=int, default=9, help="Local hour (0-23) to send the daily digest (default: 9)")
    parser.add_argument("--lookback-days", type=int, default=400, help="Days of history to fetch/train on (default: 400)")
    parser.add_argument("--coins", nargs="+", default=["BTC", "ETH"], help="Coins to track (default: BTC ETH)")
    parser.add_argument("--no-sentiment", action="store_true", help="Skip the once-daily news sentiment update")
    args = parser.parse_args()

    if args.setup:
        _setup_telegram()
        return

    if args.schedule:
        run_scheduler(tuple(args.coins), args.interval_hours, args.digest_hour, args.lookback_days,
                     update_sentiment_daily=not args.no_sentiment)
    else:
        # --once or no flag: single pass
        run_once(tuple(args.coins), args.digest_hour, args.lookback_days,
                 update_sentiment_daily=not args.no_sentiment)


if __name__ == "__main__":
    main()
