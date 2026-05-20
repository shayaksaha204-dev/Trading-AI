"""
News Fetcher — Real Financial Headlines from RSS & NewsAPI
============================================================
Fetches actual financial news headlines for sentiment analysis.

Two backends:
  1. RSS feeds (free, no key): Yahoo Finance, Reuters, MarketWatch
  2. NewsAPI (free tier: 100 req/day, requires key)

Headlines are cached per ticker to avoid excessive requests.
"""

import time
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg, DATA_DIR


# ── Ticker → Search Term Mapping ─────────────────────────────────────────────
TICKER_TO_COMPANY = {
    # US Stocks
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google Alphabet",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "META": "Meta Facebook",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "V": "Visa",
    "WMT": "Walmart",
    "JNJ": "Johnson Johnson",
    "PG": "Procter Gamble",
    "XOM": "Exxon Mobil",
    "BAC": "Bank of America",
    "DIS": "Disney",
    # Indian Stocks (NSE)
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services TCS",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever HUL",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India SBI",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen Toubro",
    "AXISBANK.NS": "Axis Bank",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "ADANIENT.NS": "Adani Enterprises",
    # Crypto
    "BTC-USD": "Bitcoin BTC",
    "ETH-USD": "Ethereum ETH",
    "SOL-USD": "Solana SOL",
    "BNB-USD": "Binance BNB",
    "XRP-USD": "Ripple XRP",
    "ADA-USD": "Cardano ADA",
    "DOGE-USD": "Dogecoin DOGE",
    "AVAX-USD": "Avalanche AVAX",
    "DOT-USD": "Polkadot DOT",
    "MATIC-USD": "Polygon MATIC",
    # Forex
    "EURUSD=X": "EUR USD euro dollar",
    "GBPUSD=X": "GBP USD pound dollar",
    "USDJPY=X": "USD JPY dollar yen",
    "USDCHF=X": "USD CHF dollar franc",
    "AUDUSD=X": "AUD USD australian dollar",
    "USDCAD=X": "USD CAD dollar canadian",
    "NZDUSD=X": "NZD USD new zealand dollar",
    "USDINR=X": "USD INR dollar rupee indian",
    # Commodities
    "GC=F": "gold price",
    "SI=F": "silver price",
    "CL=F": "crude oil price",
    "NG=F": "natural gas price",
    "HG=F": "copper price",
    # Indices
    "^GSPC": "S&P 500 stock market",
    "^IXIC": "NASDAQ market",
    "^DJI": "Dow Jones market",
    "^VIX": "VIX volatility fear",
    "^TNX": "treasury yield bonds",
    "^NSEI": "NIFTY 50 Indian stock market",
    "^BSESN": "BSE SENSEX Indian stock market",
    "^NSEBANK": "NIFTY Bank Indian banking",
}

# Default RSS feeds for financial news
DEFAULT_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://www.investing.com/rss/news.rss",
]

# General financial RSS feeds (not ticker-specific)
GENERAL_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?region=IN&lang=en-IN",
]

# Indian financial news RSS feeds
INDIAN_RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
]


class NewsFetcher:
    """
    Fetches real financial news headlines from multiple sources.
    
    Usage:
        fetcher = NewsFetcher()
        headlines = fetcher.fetch_headlines("AAPL", lookback_hours=48)
        # Returns: [{"headline": "Apple...", "published": datetime, "source": "Yahoo Finance"}, ...]
    """

    def __init__(self):
        self.cache_dir = DATA_DIR / "news_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limit_delay = 1.0  # seconds between requests
        self._last_request_time = 0.0
        self._general_cache: Optional[List[Dict]] = None
        self._general_cache_time = 0.0

    def fetch_headlines(
        self,
        ticker: str,
        lookback_hours: int = None,
        max_headlines: int = None,
    ) -> List[Dict]:
        """
        Fetch recent headlines for a ticker.
        
        Args:
            ticker: Asset symbol (e.g., "AAPL", "BTC-USD")
            lookback_hours: How far back to look
            max_headlines: Maximum headlines to return
            
        Returns:
            List of {"headline": str, "published": datetime|None, "source": str}
        """
        lookback_hours = lookback_hours or getattr(cfg.features, 'news_lookback_hours', 48)
        max_headlines = max_headlines or getattr(cfg.features, 'news_max_headlines', 50)

        # Check cache first
        cached = self._load_cache(ticker)
        if cached is not None:
            return cached[:max_headlines]

        headlines = []

        # 1. Try RSS feeds (always available, free)
        rss_headlines = self._fetch_rss(ticker, lookback_hours)
        headlines.extend(rss_headlines)

        # 2. Try NewsAPI if key is configured
        api_key = getattr(cfg.features, 'newsapi_key', '')
        if api_key:
            api_headlines = self._fetch_newsapi(ticker, api_key, lookback_hours)
            headlines.extend(api_headlines)

        # 3. If no ticker-specific results, try general financial news
        if len(headlines) < 5:
            general = self._fetch_general_news()
            # Filter general news for ticker relevance
            search_term = TICKER_TO_COMPANY.get(ticker, ticker).lower()
            search_words = search_term.split()
            for h in general:
                headline_lower = h["headline"].lower()
                if any(w in headline_lower for w in search_words if len(w) > 2):
                    headlines.append(h)

        # Deduplicate by headline text
        seen = set()
        unique = []
        for h in headlines:
            key = h["headline"].strip().lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(h)
        headlines = unique

        # Sort by published date (most recent first)
        headlines.sort(
            key=lambda x: x.get("published") or datetime.min,
            reverse=True
        )

        # Limit
        headlines = headlines[:max_headlines]

        # Cache results
        if headlines:
            self._save_cache(ticker, headlines)
            logger.debug(f"  News: {len(headlines)} headlines for {ticker}")
        else:
            logger.debug(f"  News: No headlines found for {ticker}")

        return headlines

    def _fetch_rss(self, ticker: str, lookback_hours: int) -> List[Dict]:
        """Fetch headlines from RSS feeds."""
        headlines = []
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed. Run: pip install feedparser")
            return headlines

        # Determine region-specific RSS URLs
        is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")
        if is_indian:
            yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN"
            rss_urls = [yahoo_url] + INDIAN_RSS_FEEDS
        else:
            yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            rss_urls = [yahoo_url]

        for url in rss_urls:
            try:
                self._rate_limit()
                feed = feedparser.parse(url)
                cutoff = datetime.now() - timedelta(hours=lookback_hours)

                for entry in feed.entries:
                    published = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published = datetime(*entry.published_parsed[:6])
                        except Exception:
                            pass

                    # Skip old entries
                    if published and published < cutoff:
                        continue

                    title = entry.get('title', '').strip()
                    if title and len(title) > 10:
                        headlines.append({
                            "headline": title,
                            "published": published,
                            "source": "Yahoo Finance RSS" if "yahoo" in url else "Indian Financial RSS",
                        })
            except Exception as e:
                logger.debug(f"RSS fetch failed for {url}: {e}")

        return headlines

    def _fetch_newsapi(
        self, ticker: str, api_key: str, lookback_hours: int
    ) -> List[Dict]:
        """Fetch headlines from NewsAPI."""
        headlines = []
        try:
            import requests
        except ImportError:
            return headlines

        search_term = TICKER_TO_COMPANY.get(ticker, ticker)
        from_date = (datetime.now() - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%S")

        try:
            self._rate_limit()
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": search_term,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": 50,
                    "apiKey": api_key,
                },
                timeout=10,
            )

            if resp.status_code == 200:
                data = resp.json()
                for article in data.get("articles", []):
                    title = article.get("title", "").strip()
                    published = None
                    pub_str = article.get("publishedAt", "")
                    if pub_str:
                        try:
                            published = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).replace(tzinfo=None)
                        except Exception:
                            pass

                    if title and len(title) > 10 and title != "[Removed]":
                        headlines.append({
                            "headline": title,
                            "published": published,
                            "source": f"NewsAPI/{article.get('source', {}).get('name', 'unknown')}",
                        })
            else:
                logger.debug(f"NewsAPI returned {resp.status_code}: {resp.text[:200]}")

        except Exception as e:
            logger.debug(f"NewsAPI fetch failed: {e}")

        return headlines

    def _fetch_general_news(self) -> List[Dict]:
        """Fetch general financial news (cached for 30 min)."""
        if self._general_cache is not None and (time.time() - self._general_cache_time) < 1800:
            return self._general_cache

        headlines = []
        try:
            import feedparser
        except ImportError:
            return headlines

        for url in GENERAL_RSS_FEEDS:
            try:
                self._rate_limit()
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    title = entry.get('title', '').strip()
                    published = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published = datetime(*entry.published_parsed[:6])
                        except Exception:
                            pass
                    if title and len(title) > 10:
                        headlines.append({
                            "headline": title,
                            "published": published,
                            "source": "General Financial RSS",
                        })
            except Exception as e:
                logger.debug(f"General RSS failed: {e}")

        self._general_cache = headlines
        self._general_cache_time = time.time()
        return headlines

    # ── Caching ──────────────────────────────────────────────────────────────

    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace("=", "_").replace("^", "_").replace("-", "_")
        return self.cache_dir / f"news_{safe}.json"

    def _load_cache(self, ticker: str) -> Optional[List[Dict]]:
        path = self._cache_path(ticker)
        if not path.exists():
            return None

        # Cache valid for 1 hour
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > 1.0:
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Convert published strings back to datetime
            for h in data:
                if h.get("published"):
                    try:
                        h["published"] = datetime.fromisoformat(h["published"])
                    except Exception:
                        h["published"] = None
            return data
        except Exception:
            return None

    def _save_cache(self, ticker: str, headlines: List[Dict]):
        path = self._cache_path(ticker)
        try:
            serializable = []
            for h in headlines:
                serializable.append({
                    "headline": h["headline"],
                    "published": h["published"].isoformat() if h.get("published") else None,
                    "source": h.get("source", "unknown"),
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to cache news for {ticker}: {e}")

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()


# Module-level singleton
_news_fetcher = None


def get_news_fetcher() -> NewsFetcher:
    global _news_fetcher
    if _news_fetcher is None:
        _news_fetcher = NewsFetcher()
    return _news_fetcher


if __name__ == "__main__":
    fetcher = NewsFetcher()
    for ticker in ["AAPL", "BTC-USD", "GC=F"]:
        print(f"\n--- {ticker} ---")
        headlines = fetcher.fetch_headlines(ticker, lookback_hours=72)
        for h in headlines[:5]:
            pub = h["published"].strftime("%m/%d %H:%M") if h["published"] else "N/A"
            print(f"  [{pub}] {h['headline'][:80]}")
        print(f"  Total: {len(headlines)} headlines")
