"""
Google News Scraper
Fetches headlines and full article text via Google News RSS (free, no API key).
Designed to feed into the existing sentiment analysis pipeline in functions.py.
"""

from __future__ import annotations

import time
import re
import urllib.parse
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests
import feedparser
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
}

# Sites that reliably block scrapers — fall back to title only
BLOCKED_DOMAINS = {
    'wsj.com', 'ft.com', 'bloomberg.com', 'reuters.com',
    'nytimes.com', 'washingtonpost.com',
}


def _rss_url(query: str) -> str:
    return (
        f"https://news.google.com/rss/search"
        f"?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )


def _parse_date(date_str: str) -> str:
    """Parse RFC-2822 date string to YYYY-MM-DD, fallback to today."""
    try:
        return parsedate_to_datetime(date_str).strftime('%Y-%m-%d')
    except Exception:
        return datetime.now().strftime('%Y-%m-%d')


def _resolve_google_redirect(url: str) -> str:
    """Follow Google News redirect to get the real article URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        return resp.url
    except Exception:
        return url


def _is_blocked(url: str) -> bool:
    for domain in BLOCKED_DOMAINS:
        if domain in url:
            return True
    return False


def fetch_article_text(url: str, timeout: int = 15) -> str | None:
    """
    Fetch and extract main article text from a URL.
    Returns None if the page can't be scraped (caller falls back to title).
    """
    if _is_blocked(url):
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')

        # Strip boilerplate
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'figure']):
            tag.decompose()

        # Prefer <article> body, then fall back to all <p> tags
        container = soup.find('article') or soup
        paragraphs = container.find_all('p')
        text = ' '.join(
            p.get_text(strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 40
        )
        return text if len(text) > 100 else None
    except Exception:
        return None


def fetch_rss_articles(query: str, max_articles: int = 20) -> list[dict]:
    """Fetch article metadata from Google News RSS for a query."""
    # feedparser.parse(url) fetches the URL itself with no timeout at all --
    # if the RSS endpoint accepts the connection but stalls (observed in
    # practice), this can hang forever with no way to recover. Fetching the
    # bytes ourselves with an explicit timeout and handing feedparser
    # already-downloaded content (no network I/O of its own) bounds it.
    try:
        resp = requests.get(_rss_url(query), headers=HEADERS, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"  RSS fetch failed for '{query}': {e}")
        return []
    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            'title': entry.get('title', '').split(' - ')[0].strip(),
            'link': entry.get('link', ''),
            'date': _parse_date(entry.get('published', '')),
            'source': entry.get('source', {}).get('title', '') if isinstance(entry.get('source'), dict) else '',
        })
    return articles


def scrape_news(query: str, max_articles: int = 20, sleep: float = 1.0) -> pd.DataFrame:
    """
    Scrape Google News for a query, fetch full article text where possible.

    Returns DataFrame with columns: title, link, date, source, text
    """
    articles = fetch_rss_articles(query, max_articles)

    for article in articles:
        time.sleep(sleep)
        real_url = _resolve_google_redirect(article['link'])
        article['link'] = real_url
        text = fetch_article_text(real_url)
        article['text'] = text if text else article['title']

    return pd.DataFrame(articles)


def scrape_news_for_coin(
    coin: str,
    queries: list[str] | None = None,
    max_per_query: int = 15,
    sleep: float = 1.0,
) -> pd.DataFrame:
    """
    Scrape Google News across multiple queries for a cryptocurrency.
    Deduplicates by URL and returns a DataFrame ready for sentimentAnalysis().

    Args:
        coin: e.g. 'BTC' or 'ETH'
        queries: list of search queries (defaults to sensible crypto queries)
        max_per_query: articles to fetch per query
        sleep: seconds between article fetches

    Returns:
        DataFrame with columns: title, link, date, source, text
    """
    if queries is None:
        queries = [
            f'{coin} cryptocurrency price prediction',
            f'{coin} crypto market news',
            f'{coin} trading analysis today',
            f'{coin} bullish bearish outlook',
        ]

    all_articles = pd.DataFrame()
    for query in queries:
        print(f"  Fetching: {query}")
        df = scrape_news(query, max_articles=max_per_query, sleep=sleep)
        all_articles = pd.concat([all_articles, df], ignore_index=True)

    # Deduplicate
    all_articles = all_articles.drop_duplicates(subset=['link']).reset_index(drop=True)
    print(f"  Total unique articles: {len(all_articles)}")
    return all_articles


if __name__ == '__main__':
    import sys
    coin = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    print(f"\nScraping Google News for {coin}...")
    df = scrape_news_for_coin(coin)
    print(df[['date', 'source', 'title']].to_string(index=False))
    out = f'news_{coin}.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
