"""
Fetches recent crypto news, scores sentiment, and merges a daily-aggregated
`avg_sentiment` column into fulldata/{coin}_df.csv — the column the
LLM-Sentiment model actually reads (models/llm_sentiment_model.py).

This exists instead of automating functions.newspaper_sentiment_pipeline()
because that pipeline has two real bugs:
  1. get_sentiment() reloaded the RoBERTa tokenizer/model from disk for every
     single article. This loads it once per run and reuses it.
  2. Its merge joined one row per ARTICLE onto one-row-per-day price data —
     a many-to-one merge on date that duplicates every OHLCV row once per
     matching article. This aggregates to one avg_sentiment value per
     calendar day first, then does a one-to-one update keyed by date, so the
     row count of fulldata/{coin}_df.csv never changes.

No LLM/API calls beyond a local, offline RoBERTa sentiment classifier
(downloaded once from Hugging Face and cached) and free Google News RSS.

Usage:
    python update_sentiment.py BTC
    python update_sentiment.py BTC ETH --max-per-query 15
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from functions import fullDataPath
from news_scraper import scrape_news_for_coin

MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


def _clean_text(text: str) -> str:
    """Whitespace-only cleanup — deliberately NOT functions.preprocess(), which
    uppercases the whole string and strips all punctuation. Empirically that
    collapses this RoBERTa model's predictions to near-100% "neutral"
    regardless of actual sentiment (verified: a clearly bearish sentence went
    from 99.8% negative to 99.98% neutral after that transform). The model
    was fine-tuned on ordinary-cased financial text; keep it that way."""
    return re.sub(r"\s+", " ", text).strip()


def _load_sentiment_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def score_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Add a numeric 'score' column in [-100, 100] to each article (one model load, reused)."""
    import torch
    from torch.nn.functional import softmax

    tokenizer, model = _load_sentiment_model()
    id2label = {k: v.lower() for k, v in model.config.id2label.items()}
    sign = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

    scores = []
    for text in df["text"].astype(str):
        tokens = tokenizer(_clean_text(text), padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
        probs = softmax(outputs.logits, dim=-1).tolist()[0]
        score = 100.0 * sum(sign.get(id2label[i], 0.0) * p for i, p in enumerate(probs))
        scores.append(score)

    df = df.copy()
    df["score"] = scores
    return df


def update_sentiment(coin: str, max_per_query: int = 6) -> pd.DataFrame:
    """Scrape recent news, score it, aggregate to daily avg_sentiment, merge into fulldata CSV.

    max_per_query default lowered from an earlier 15 to 6: on a slow/flaky
    network each article fetch can take several seconds even with a working
    timeout, and 4 queries x 15 articles x 2 coins made a scheduled run's
    total runtime unpredictable enough to matter for reliability. 6 still
    gives a reasonable daily sample while keeping worst-case runtime bounded.
    """
    print(f"Scraping news for {coin}...")
    articles = scrape_news_for_coin(coin, max_per_query=max_per_query)

    df = pd.read_csv(fullDataPath(coin))
    original_len = len(df)
    df["time"] = pd.to_datetime(df["time"])
    if "avg_sentiment" not in df.columns:
        df["avg_sentiment"] = np.nan

    if articles.empty:
        print("No articles found; leaving existing avg_sentiment untouched.")
        return df

    print(f"Scoring {len(articles)} articles...")
    scored = score_articles(articles)
    daily = scored.groupby("date")["score"].mean()
    daily.index = pd.to_datetime(daily.index)

    df = df.set_index("time")
    updated = daily.reindex(df.index)
    df["avg_sentiment"] = updated.where(updated.notna(), df["avg_sentiment"])
    df = df.reset_index()

    # Merging must never change how many price rows we have -- that would
    # silently corrupt every other model's training data.
    assert len(df) == original_len, (
        f"Row count changed ({original_len} -> {len(df)}) -- refusing to save, "
        f"this would corrupt fulldata/{coin}_df.csv"
    )

    df.to_csv(fullDataPath(coin), index=False)
    n_updated = updated.notna().sum()
    print(f"Updated avg_sentiment for {n_updated} day(s), saved to {fullDataPath(coin)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Update avg_sentiment from fresh news for one or more coins")
    parser.add_argument("coins", nargs="+", help="Coins to update, e.g. BTC ETH")
    parser.add_argument("--max-per-query", type=int, default=6, help="Articles per search query (default: 6)")
    args = parser.parse_args()

    for coin in args.coins:
        update_sentiment(coin, max_per_query=args.max_per_query)


if __name__ == "__main__":
    main()
